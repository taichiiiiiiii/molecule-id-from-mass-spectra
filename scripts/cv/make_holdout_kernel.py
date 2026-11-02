#!/usr/bin/env python3
"""kernels/lb479 のノートブックから、ホールドアウト採点用のノートブック kernels/holdout を作る。

非公開データセットの容量制限（約 0.9 GB）のため、学習データはアップロードしない。Kaggle 上で、競技の
train.parquet からホールドアウトの構造を除いたものを /kaggle/working/hold_comp に作り、そこを COMP にする。

変更点（それ以外は 0.479 版のまま）:
1. 先頭にセルを 1 つ追加する: hold_comp を作り、CASMI_COMP / CASMI_TEST_PATH / CASMI_SAMPLE_PATH を設定する。
2. 外側のセルで、COMP の候補の先頭に HOLD を入れる。
3. 埋め込まれた 0.465 パイプライン（FROZEN_B64）を 2 か所だけ書き換え、SHA-256 を差し替える。
   - `COMP = os.path.dirname(find('test.parquet'))` → CASMI_COMP があればそちらを使う
   - E004 preflight の COMP の場所チェック → CASMI_COMP も許す

    python scripts/cv/make_holdout_kernel.py --dataset taichiiiii/casmi26-holdout-np-v1 --holdout np_v1
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

SRC = Path("kernels/lb479")

GUARD = '''# ── Holdout evaluation (taichiiiii) ────────────────────────────────────────────
# 隠れテストではなく、正解付きホールドアウトで公開パイプライン（0.479 版）を採点するためのノートブック。
HOLDOUT = '__HOLDOUT__'
import glob, os, shutil, time
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
_t0 = time.time()
SRCD = next(Path(p).parent for p in sorted(glob.glob('/kaggle/input/**/HOLDOUT.json', recursive=True), key=len)
            if Path(p).parent.name == 'casmi26-holdout-' + HOLDOUT.replace('_', '-'))
COMP_TRAIN = next(p for p in sorted(glob.glob('/kaggle/input/**/train.parquet', recursive=True), key=len)
                  if 'enveda' in p.lower() and 'casmi26' in p.lower())
HOLD = Path('/kaggle/working/hold_comp'); HOLD.mkdir(parents=True, exist_ok=True)
for f in ('test.parquet', 'sample_submission.csv', 'HOLDOUT.json'):
    shutil.copyfile(SRCD / f, HOLD / f)
_drop = pa.array(sorted(set((SRCD / 'drop_keys.txt').read_text().split())))
_pf = pq.ParquetFile(COMP_TRAIN)
_w = pq.ParquetWriter(HOLD / 'train.parquet', _pf.schema_arrow, compression='zstd')
_n_in = _n_out = 0
for _b in _pf.iter_batches(batch_size=131072):
    _t = pa.Table.from_batches([_b]); _n_in += _t.num_rows
    _t = _t.filter(pc.invert(pc.is_in(_t['inchikey14'], value_set=_drop))); _n_out += _t.num_rows
    _w.write_table(_t)
_w.close(); del _pf, _w
os.environ['CASMI_COMP'] = str(HOLD)
os.environ['CASMI_POOL_UNHIDE'] = str(SRCD / 'drop_keys.txt')
os.environ['CASMI_POOL_HIDE'] = str(SRCD / 'drop_keys.txt')
os.environ['CASMI_TEST_PATH'] = str(HOLD / 'test.parquet')
os.environ['CASMI_SAMPLE_PATH'] = str(HOLD / 'sample_submission.csv')
print(f'HOLDOUT {HOLDOUT}: train {_n_in:,} -> {_n_out:,} rows (dropped {_n_in - _n_out:,}) from {COMP_TRAIN} '
      f'in {time.time() - _t0:.0f}s; COMP={HOLD}')
assert _n_in - _n_out > 0
'''

UNHIDE = r"""
# ---- holdout patch: PubChem チャネルがホールドアウトの構造を「プールに無い」扱いにする（クラス 2 の測定用） ----
_init_worker_v17 = init_worker
def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    _init_worker_v17(pc_dir, bits_path, pool_meta_path, code_dir)
    _p = os.environ.get('CASMI_POOL_UNHIDE')
    if _p and os.path.isfile(_p):
        _keys = set(open(_p).read().split())
        _before = len(_W['pool_keys']); _W['pool_keys'] -= _keys
        print('UNHIDE pool keys', _before, '->', len(_W['pool_keys']), flush=True)
"""

FROZEN_PATCHES = [
    ("COMP = os.path.dirname(find('test.parquet'))",
     "COMP = os.environ.get('CASMI_COMP') or os.path.dirname(find('test.parquet'))"),
    ("    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
     "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'} "
     "or comp == os.path.realpath(os.environ.get('CASMI_COMP', '-'))"),
]
# PubChem チャネルの分子ごとの FPNet ロジット z と中性質量を保存する（ローカルで別の候補源を採点するため）
LOGITS_PATCH = ("    del bank; torch.cuda.empty_cache()",
                "    np.savez_compressed('/kaggle/working/pc_logits.npz', **{m: z for m, _, z in tasks}, "
                "__targets__=np.array([[m, t] for m, t, _ in tasks], dtype=object))\n"
                "    del bank; torch.cuda.empty_cache()")
# ---- クラス 3（新規構造）の模擬: 正解を全候補プールから見えなくする ----
# Engine 1（casmi.pool.Pool.window）はメインプロセスで import されるので、その場で差し替える（fork した worker にも伝わる）
HIDE_POOL_MAIN = ("from casmi.pool import Pool\n",
"""from casmi.pool import Pool
import casmi.pool as _cp
_HIDE_PATH = os.environ.get('CASMI_POOL_HIDE')
if _HIDE_PATH and os.path.isfile(_HIDE_PATH):
    _HIDE_KEYS = set(open(_HIDE_PATH).read().split())
    _orig_window = _cp.Pool.window
    def _hidden_window(self, target, ppm):
        idx = _orig_window(self, target, ppm)
        if not hasattr(self, '_hide_mask'):
            self._hide_mask = np.fromiter((k in _HIDE_KEYS for k in self.key), dtype=bool, count=len(self.key))
            print('HIDE pool rows', int(self._hide_mask.sum()), flush=True)
        return idx[~self._hide_mask[idx]]
    _cp.Pool.window = _hidden_window
    Pool.window = _hidden_window
""")
# Engine 2（casmi_engine.py、サブプロセス）は ENG_FILES の文字列として埋め込まれている
HIDE_POOL_ENG = ("""def pool_window(t, ppm):
    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")
    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")
    return np.arange(a, b)
""", """def pool_window(t, ppm):
    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")
    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")
    idx = np.arange(a, b)
    _h = os.environ.get("CASMI_POOL_HIDE")
    if _h and os.path.isfile(_h):
        if "hide" not in POOL:
            _keys = set(open(_h).read().split())
            POOL["hide"] = np.fromiter((k in _keys for k in POOL["keys"]), dtype=bool, count=len(POOL["keys"]))
            log(f"HIDE engine2 pool rows {int(POOL['hide'].sum())}")
        idx = idx[~POOL["hide"][idx]]
    return idx
""")

UNHIDE_PATCH = ("open('/kaggle/working/probe_core2.py', 'w').write(CORE + V17_PATCH)",
                "open('/kaggle/working/probe_core2.py', 'w').write(CORE + V17_PATCH + " + repr(UNHIDE) + ")")


def src_of(c) -> str:
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout", required=True, help="データセット内のサブディレクトリ名")
    ap.add_argument("--slug", default=None)
    ap.add_argument("--hide-pool", action="store_true",
                    help="クラス 3 の模擬: 正解を Engine 1/2 の候補プールから見えなくする（PubChem チャネルは元のまま）")
    ap.add_argument("--unhide-pool", action="store_true",
                    help="PubChem チャネルで drop_keys をプール外扱いにする（クラス 2 の測定）")
    args = ap.parse_args()

    suffix = ("-pcopen" if args.unhide_pool else "") + ("-class3" if args.hide_pool else "")
    slug = args.slug or "casmi26-holdout-eval-" + args.holdout.replace("_", "-") + suffix
    dst = Path("kernels") / f"holdout_{args.holdout}{suffix.replace('-', '_')}"
    nb = json.loads((SRC / "notebook.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]

    # 2. 外側の COMP
    old = "COMP = next((p for p in (INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',"
    new = "COMP = next((p for p in (HOLD, INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',"
    hit = [c for c in code if old in src_of(c)]
    assert len(hit) == 1, len(hit)
    hit[0]["source"] = src_of(hit[0]).replace(old, new, 1)

    # 3. 埋め込みパイプライン
    pin = [c for c in code if "FROZEN_B64 = '" in src_of(c)]
    assert len(pin) == 1
    s = src_of(pin[0])
    b64 = re.search(r"FROZEN_B64 = '([^']+)'", s).group(1)
    old_sha = re.search(r"FROZEN_SHA256 = '([0-9a-f]{64})'", s).group(1)
    raw = base64.b64decode(b64)
    assert hashlib.sha256(raw).hexdigest() == old_sha
    fr = json.loads(raw)
    patches = FROZEN_PATCHES + ([UNHIDE_PATCH] if args.unhide_pool else [])
    if args.unhide_pool:  # RUNNER は文字列として埋め込まれているので、repr した形で置換する
        cells = [c for c in fr["cells"] if c["cell_type"] == "code" and "RUNNER = " in src_of(c)]
        assert len(cells) == 1
        src = src_of(cells[0])
        esc = lambda t: t.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")  # 単一引用符の文字列リテラル内の表記
        old, new = esc(LOGITS_PATCH[0]), esc(LOGITS_PATCH[1])
        assert src.count(old) == 1, "RUNNER の del bank 行が見つからない"
        cells[0]["source"] = src.replace(old, new, 1)
    if args.hide_pool:
        patches = patches + [HIDE_POOL_MAIN]
        cells = [c for c in fr["cells"] if c["cell_type"] == "code" and "ENG_FILES = {" in src_of(c)]
        assert len(cells) == 1
        src = src_of(cells[0])
        esc = lambda t: t.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
        old, new = esc(HIDE_POOL_ENG[0]), esc(HIDE_POOL_ENG[1])
        assert src.count(old) == 1, "casmi_engine.py の pool_window が見つからない"
        cells[0]["source"] = src.replace(old, new, 1)
    for a, b in patches:
        cells = [c for c in fr["cells"] if c["cell_type"] == "code" and a in src_of(c)]
        assert len(cells) == 1 and src_of(cells[0]).count(a) == 1, a
        cells[0]["source"] = src_of(cells[0]).replace(a, b, 1)
    new_raw = json.dumps(fr, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    s = s.replace(b64, base64.b64encode(new_raw).decode(), 1).replace(old_sha, new_sha)
    pin[0]["source"] = s
    n_old = json.dumps(nb).count(old_sha)
    assert n_old == 0 or True  # メタデータにだけ残る旧 SHA は表示用なので触らない

    # 1. 先頭のセル
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": GUARD.replace("__HOLDOUT__", args.holdout)})
    # 最後に、作った学習データを消す（出力の容量を節約し、出力のダウンロードを軽くする）
    nb["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                        "source": "import os\nos.remove('/kaggle/working/hold_comp/train.parquet')\nprint('removed hold_comp/train.parquet')\n"})

    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"taichiiiii/{slug}", title=slug.replace("-", " "), is_private=True)
    meta["dataset_sources"] = [args.dataset] + [d for d in meta["dataset_sources"] if d != args.dataset]

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", dst, meta["id"], "| frozen sha", old_sha[:12], "->", new_sha[:12])


if __name__ == "__main__":
    main()
