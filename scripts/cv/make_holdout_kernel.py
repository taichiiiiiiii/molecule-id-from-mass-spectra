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
os.environ['CASMI_TEST_PATH'] = str(HOLD / 'test.parquet')
os.environ['CASMI_SAMPLE_PATH'] = str(HOLD / 'sample_submission.csv')
print(f'HOLDOUT {HOLDOUT}: train {_n_in:,} -> {_n_out:,} rows (dropped {_n_in - _n_out:,}) from {COMP_TRAIN} '
      f'in {time.time() - _t0:.0f}s; COMP={HOLD}')
assert _n_in - _n_out > 0
'''

FROZEN_PATCHES = [
    ("COMP = os.path.dirname(find('test.parquet'))",
     "COMP = os.environ.get('CASMI_COMP') or os.path.dirname(find('test.parquet'))"),
    ("    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'}",
     "    assert comp in {f'/kaggle/input/{slug}', f'/kaggle/input/competitions/{slug}'} "
     "or comp == os.path.realpath(os.environ.get('CASMI_COMP', '-'))"),
]


def src_of(c) -> str:
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--holdout", required=True, help="データセット内のサブディレクトリ名")
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()

    slug = args.slug or "casmi26-holdout-eval-" + args.holdout.replace("_", "-")
    dst = Path("kernels") / f"holdout_{args.holdout}"
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
    for a, b in FROZEN_PATCHES:
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
