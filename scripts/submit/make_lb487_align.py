#!/usr/bin/env python3
"""#127 診断: 0.487 写し（kernels/lb487、31 セル）に整列チャネル（自前・fold-safe）を足したノートを作る。

変えるのは 2 か所だけ（ほかのセルはバイト一致。tests/test_make_lb487_align.py で確かめる）:
1. 外側セル 5（設定）の直後に 1 セルを足す（ADDED_INDEX = 6）。中身:
   - `COMMIT_MODE = 'smoke'`（#77 と同じく commit で見える 12 分子が全段を通る。採点の再実行の動きには関わらない）
   - 段のコード（`scripts/submit/align0487_runner.py`、`scripts/train/align_speed_trial.py`、`scripts/engine2/base_fs.py`、
     PR #110 の `scripts/train/align_train.py`〈sha256 8e1e9dde…、git show で取り出す〉、学習の `align_config.json`）と、
     DreaMS（MIT、commit dbec3a0）の `dreams/**/*.py` と LICENSE の zip を base64 で持つ。どれも sha256 を照合してから書く
   - `ALIGN0487_MIX(idx, za, zb, zfun)`: runner を別のプロセスで走らせ（上限 3,600 秒 = 足す段 ≤ 1 時間の関門）、
     a_align を読み、a が分子の全候補で有限な分子だけ (za + zb + zfun(a)) / 3、ほかの分子は今と同じ 0.5 * za + 0.5 * zb。
     report は /kaggle/working/align0487_report.json。runner が失敗・`fp_mismatch` なら status を書いて例外で止まる
     （黙って 2 項に戻らない）
   - `_a0487_downstream`（IPython の post_run_cell）: 尾の枠（セル 16 の PARALLAX_DIAGNOSTICS）と空き枠（セル 21 の
     MERGE_REPORT）に入った外側の候補の数を report の `downstream` に足す（読むだけ。#127 固定すること (3)）
2. 外側セル 15 の `neural_score` の 1 行（OLD_LINE）を NEW_LINE に替える（同じセルのほかの行は変えない）

kernel-metadata.json（#127 本文 案 A″）: 元の 10 件の dataset_sources に公開の DreaMS の重みのデータセット
`ahmedberatozer/casmi26-dreams-ssl-readiness-v1` を足し、model_sources に private の Kaggle Models
`taichiiiii/casmi26-align0487/PyTorch/foldsafe-v1/1`（整列の model_best.pt だけ）を入れる。id は別 slug、id_no なし。

    python scripts/submit/make_lb487_align.py --align-config <outputs/align_train/align_config.json> \
        --dreams-root <data/ext/dreams/DreaMS-dbec3a0…> [--out-dir kernels/lb487_align]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_NB = ROOT / "kernels/lb487/casmi26-lb487-copy.ipynb"
SRC_META = ROOT / "kernels/lb487/kernel-metadata.json"
SRC_SHA256 = "380e2c820c2a47cfad268b75cdd0dfc564350b599b9b052a1915d292987113c5"
N_CELLS = 31
CONFIG_CELL = 5
ADDED_INDEX = 6
NEURAL_CELL = 15            # 元のノートの番号（新しいノートでは 16）
SLUG = "taichiiiii/casmi26-lb487-align"
DREAMS_DS = "ahmedberatozer/casmi26-dreams-ssl-readiness-v1"     # 公開、weights/ssl_model.ckpt（Zenodo 10997887 と同じバイト列）
ALIGN_MODEL_SRC = "taichiiiii/casmi26-align0487/PyTorch/foldsafe-v1/1"   # private の Kaggle Models（model_best.pt だけ）
ALIGN_TRAIN_REF = "origin/feat/109-align-local-train:scripts/train/align_train.py"
ALIGN_TRAIN_SHA256 = "8e1e9dde20add0f7a7cce911f36835ea448311f84ea2f8806275a3aaca1966ef"
DREAMS_COMMIT = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
OLD_LINE = ("        _a['neural_score'] = 0.5 * _z(_a['neural_score'].fillna(0.0)) + "
            "0.5 * _z(_b['neural_score'].fillna(0.0))\n")
NEW_LINE = ("        _a['neural_score'] = ALIGN0487_MIX(_a.index, _z(_a['neural_score'].fillna(0.0)), "
            "_z(_b['neural_score'].fillna(0.0)), _z)\n")
MARK = "# ── #127（診断）: 整列チャネル"

CELL_TEMPLATE = r'''# ── #127（診断）: 整列チャネル（自前・fold-safe の整列 MLP ＋ 凍結 DreaMS）を neural_score に足す ──────────────
# 変えるのはこのセルと、外側セル 15 の neural_score の 1 行だけ（Issue #127 の事前登録）。
#   新: neural_score = (z(sanpier v1) + z(v80) + z(a_align)) / 3。a_align が作れない分子は今と同じ 0.5*z + 0.5*z
#   a_align = cos(normalize(mean(normalize(g))), h)。g = 整列 MLP のスペクトル側（DreaMS の埋め込みから）、h = 分子側
# 重み（#127 本文 案 A″）: 整列の model_best.pt は private の Kaggle Models（taichiiiii/casmi26-align0487、自前・fold-safe）、
#   DreaMS の ssl_model.ckpt は公開データセット ahmedberatozer/casmi26-dreams-ssl-readiness-v1（runner の中で大きさ・md5・sha256 を照合）。
# Credit: DreaMS pre-trained weights — Roman Bushuiev et al. / Pluskal laboratory, "Self-supervised learning of molecular
#   representations from millions of tandem mass spectra using DreaMS", Nat Biotechnol 2025 (doi:10.1038/s41587-025-02663-3),
#   Zenodo 10997887, licensed CC-BY-4.0. DreaMS code: MIT (pluskal-lab/DreaMS dbec3a0).
COMMIT_MODE = 'smoke'   # #127: commit で見える 12 分子が全段を通る（#77 と同じ。採点の再実行の動きには関わらない）
import base64 as _a_b64, io as _a_io, zipfile as _a_zip, json as _a_json, hashlib as _a_hl
import subprocess as _a_sp, sys as _a_sys, time as _a_time
from pathlib import Path as _a_P
_A0487_FILES = __FILES__
_A0487_SHA256 = __SHA__
_A0487_DREAMS_ZIP = __DREAMS_ZIP__
_A0487_DREAMS_SHA256 = __DREAMS_SHA__
_A0487_DREAMS_COMMIT = __DREAMS_COMMIT__
_A0487_TIMEOUT_S = 3600   # 足す段 ≤ 1 時間（#127 の時間の関門）
_A0487_DIR = _a_P('/kaggle/tmp/align0487')


def _a0487_one(pattern, must):
    hits = sorted(p for p in _a_P('/kaggle/input').rglob(pattern) if must in str(p))
    if len(hits) != 1:
        raise RuntimeError(('#127 input not found exactly once', pattern, must, [str(h) for h in hits]))
    return hits[0]


def _a0487_model(root=_a_P('/kaggle/input')):
    # private の Kaggle Models の 2 通りの mount の形を固定の深さで探す（framework の段は * 、版の下に 0 段か 1 段）
    pats = ['models/*/casmi26-align0487/*/foldsafe-v1/*/model_best.pt',
            'models/*/casmi26-align0487/*/foldsafe-v1/*/*/model_best.pt',
            'casmi26-align0487/*/foldsafe-v1/*/model_best.pt',
            'casmi26-align0487/*/foldsafe-v1/*/*/model_best.pt']
    found = {}
    for pat in pats:                     # 同じ実体（symlink で 2 つの形に出る場合）は 1 つと数える
        for p in sorted(root.glob(pat)):
            found.setdefault(p.resolve(), p)
    hits = sorted(found.values())
    if len(hits) != 1:
        raise RuntimeError(('#127 align weights not found exactly once in Kaggle Models', [str(h) for h in hits]))
    print(f'#127 align weights (Kaggle Models): {hits[0]}', flush=True)
    return hits[0]


def _a0487_rss():
    import resource as _r
    return {k: round(_r.getrusage(w).ru_maxrss * 1024 / 1e6, 1)
            for k, w in (('self', _r.RUSAGE_SELF), ('children', _r.RUSAGE_CHILDREN))}


def _a0487_materialise():
    _A0487_DIR.mkdir(parents=True, exist_ok=True)
    for name, b64 in _A0487_FILES.items():
        data = _a_b64.b64decode(b64)
        h = _a_hl.sha256(data).hexdigest()
        if h != _A0487_SHA256[name]:
            raise RuntimeError(('#127 embedded file sha256 mismatch', name, h))
        (_A0487_DIR / name).write_bytes(data)
    z = _a_b64.b64decode(_A0487_DREAMS_ZIP)
    if _a_hl.sha256(z).hexdigest() != _A0487_DREAMS_SHA256:
        raise RuntimeError('#127 embedded DreaMS zip sha256 mismatch')
    with _a_zip.ZipFile(_a_io.BytesIO(z)) as zf:
        zf.extractall(_A0487_DIR / 'dreams_code')
    return _A0487_DIR / 'dreams_code' / f'DreaMS-{_A0487_DREAMS_COMMIT}'


def ALIGN0487_MIX(idx, za, zb, zfun):
    """外側セル 15 の新しい 1 行。idx = (molecule_id, smiles) の索引、za・zb = 既存の 2 項の z、zfun = _z。"""
    import numpy as _np
    import pandas as _pd
    t0 = _a_time.time()
    rep = dict(issue=127, status='started')
    rp = ROOT / 'align0487_report.json'

    def dump():
        rp.write_text(_a_json.dumps(rep, indent=1, default=str))

    dump()
    try:
        dreams_root = _a0487_materialise()
        pool_meta = _a0487_one('pool_meta.parquet', 'casmi26-v2-pool')
        casmi_chem = _a0487_one('chem.py', 'casmi26-v3-models/code/casmi')
        ckpt = _a0487_one('ssl_model.ckpt', 'casmi26-dreams-ssl-readiness-v1/weights')
        model = _a0487_model()
        rep.update(align_weights_path=str(model), dreams_ckpt_path=str(ckpt))
        cands = _pd.DataFrame({'molecule_id': idx.get_level_values(0).astype(str),
                               'smiles': idx.get_level_values(1).astype(str)})
        cands.to_parquet(_A0487_DIR / 'cands.parquet', index=False)
        import torch as _t
        dev = 'cuda' if _t.cuda.is_available() else 'cpu'
        cmd = [_a_sys.executable, str(_A0487_DIR / 'align0487_runner.py'), '--test', str(COMP / 'test.parquet'),
               '--cands', str(_A0487_DIR / 'cands.parquet'), '--out', str(_A0487_DIR / 'a_align.parquet'),
               '--report', str(ROOT / 'align0487_runner.json'), '--pool-dir', str(pool_meta.parent),
               '--casmi-code', str(casmi_chem.parent.parent), '--code-dir', str(_A0487_DIR),
               '--dreams-root', str(dreams_root), '--dreams-ckpt', str(ckpt), '--align-model', str(model),
               '--align-config', str(_A0487_DIR / 'align_config.json'), '--device', dev,
               '--expect-rdkit', '2026.03.3']
        rep['device'] = dev
        try:
            r = _a_sp.run(cmd, capture_output=True, text=True, timeout=_A0487_TIMEOUT_S)
            (ROOT / 'align0487_runner.log').write_text(r.stdout + '\n--- stderr ---\n' + r.stderr)
            rc = r.returncode
        except _a_sp.TimeoutExpired:
            rep['status'] = 'timeout'
            raise
        rr = _a_json.loads((ROOT / 'align0487_runner.json').read_text())
        rep['runner'] = {k: rr.get(k) for k in ('status', 'versions', 'input_sha256', 'check_d', 'candidates',
                                                 'spectra', 'molecules', 'unknown_adduct_spectra', 'out_sha256',
                                                 'seconds', 't_sha_s', 't_dreams_load_s', 't_keys_s', 't_fp_s',
                                                 't_h_s', 't_preproc_s', 't_embed_s', 't_g_s', 'dreams_ckpt',
                                                 'max_rss_mb', 'max_rss_mb_after_strip', 'error')}
        rep['max_rss_mb'] = _a0487_rss()
        try:
            rep['mem_total_mb'] = round(int(next(l for l in open('/proc/meminfo') if l.startswith('MemTotal')).split()[1]) * 1024 / 1e6, 1)
        except Exception:
            rep['mem_total_mb'] = None
        if rc != 0 or rr.get('status') != 'ok':
            rep['status'] = (rr.get('status') if rr.get('status') in ('fp_mismatch', 'weights_mismatch') else 'error')
            raise RuntimeError(('#127 align runner failed', rc, rr.get('status'), rr.get('error')))
        a = _pd.read_parquet(_A0487_DIR / 'a_align.parquet')
        if not (a.molecule_id.astype(str).tolist() == cands.molecule_id.tolist()
                and a.smiles.astype(str).tolist() == cands.smiles.tolist()):
            raise RuntimeError('#127 a_align rows differ from the candidates')
        a_ser = _pd.Series(a.a_align.values.astype(_np.float64), index=idx)
        fin = _pd.Series(_np.isfinite(a_ser.values), index=idx).groupby(level=0).all()
        ok = set(fin[fin].index)
        mask = _np.asarray(idx.get_level_values(0).isin(ok))
        base = 0.5 * za + 0.5 * zb
        out = base.copy()
        if mask.any():
            out[mask] = (za.values[mask] + zb.values[mask] + zfun(a_ser[mask]).values) / 3
        n_mol = int(len(fin))
        rep.update(status='ok', molecules=n_mol, three_term_molecules=len(ok), two_term_molecules=n_mol - len(ok),
                   two_term_frac=((n_mol - len(ok)) / n_mol) if n_mol else None, rows=int(len(idx)),
                   rows_three_term=int(mask.sum()),
                   base_sha256=_a_hl.sha256(_np.ascontiguousarray(base.values, dtype=_np.float64).tobytes()).hexdigest(),
                   new_sha256=_a_hl.sha256(_np.ascontiguousarray(out.values, dtype=_np.float64).tobytes()).hexdigest(),
                   embedded_sha256=_A0487_SHA256, dreams_zip_sha256=_A0487_DREAMS_SHA256,
                   seconds=round(_a_time.time() - t0, 1))
        dump()
        print(f"ALIGN0487 neural = (z(public) + z(ours) + z(align)) / 3 on {len(ok)}/{n_mol} molecules "
              f"(2-term on {n_mol - len(ok)}), {rep['seconds']}s", flush=True)
        return out
    except Exception as e:
        if rep.get('status') in ('started', 'ok'):
            rep['status'] = 'error'
        rep['error'] = repr(e)[:2000]
        rep['seconds'] = round(_a_time.time() - t0, 1)
        try:
            rep['max_rss_mb'] = _a0487_rss()
        except Exception:
            pass
        dump()
        raise


def _a0487_downstream(*_args, **_kw):
    """#127 固定すること (3) の記録: 尾の枠（外側セル 16 の PARALLAX_DIAGNOSTICS.tail_changed）と空き枠（外側セル 21 の
    MERGE_REPORT.outsider_fills）に入った外側の候補の数を report に足す。読むだけで、どの値も変えない（失敗しても無視）。"""
    try:
        rp = ROOT / 'align0487_report.json'
        if not rp.is_file():
            return
        rep = _a_json.loads(rp.read_text())
        down = dict(rep.get('downstream') or {})
        g = globals()
        pdg, mr = g.get('PARALLAX_DIAGNOSTICS'), g.get('MERGE_REPORT')
        if isinstance(pdg, dict):
            down['tail_slot'] = {k: pdg.get(k) for k in ('status', 'error', 'tail_changed', 'top1_changed',
                                                         'adjacent_swaps', 'changed_rows') if k in pdg}
        if isinstance(mr, dict):
            down['empty_slot'] = {k: mr.get(k) for k in ('status', 'outsider_fills', 'lotus_fills', 'dedup_swaps',
                                                         'old_leader_pinned') if k in mr}
        if down != (rep.get('downstream') or {}):
            rep['downstream'] = down
            rep['downstream_base'] = ('the 0.487 copy (two-term) is not run in this session; its counts come from a '
                                      'separate base run (Issue #127)')
            rp.write_text(_a_json.dumps(rep, indent=1, default=str))
    except Exception:
        pass


try:
    get_ipython().events.register('post_run_cell', _a0487_downstream)
except Exception:
    pass

print('#127 align channel cell ready (COMMIT_MODE =', COMMIT_MODE + ')')
'''


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dreams_zip(dreams_root: Path) -> bytes:
    """dreams/**/*.py と LICENSE だけの zip（中身は DreaMS-<commit>/ の下、日付は固定、名前順）。"""
    files = sorted([p for p in (dreams_root / "dreams").rglob("*.py")] + [dreams_root / "LICENSE"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            zi = zipfile.ZipInfo(f"DreaMS-{DREAMS_COMMIT}/" + p.relative_to(dreams_root).as_posix(), (2024, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
    return buf.getvalue()


def make_cell(files: dict[str, bytes], dz: bytes) -> str:
    b64 = {k: base64.b64encode(v).decode() for k, v in files.items()}
    sha = {k: sha256_bytes(v) for k, v in files.items()}
    s = CELL_TEMPLATE
    s = s.replace("__FILES__", repr(b64)).replace("__SHA__", repr(sha))
    s = s.replace("__DREAMS_ZIP__", repr(base64.b64encode(dz).decode())).replace("__DREAMS_SHA__", repr(sha256_bytes(dz)))
    s = s.replace("__DREAMS_COMMIT__", repr(DREAMS_COMMIT))
    return s


def to_source(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    return lines


def build(base_nb: dict, cell_text: str) -> dict:
    cells = base_nb["cells"]
    assert len(cells) == N_CELLS, len(cells)
    src15 = "".join(cells[NEURAL_CELL]["source"])
    assert src15.count(OLD_LINE) == 1, "the neural_score line must appear exactly once"
    new = json.loads(json.dumps(base_nb))
    new15 = src15.replace(OLD_LINE, NEW_LINE)
    new["cells"][NEURAL_CELL]["source"] = to_source(new15)
    added = {"cell_type": "code", "execution_count": None, "metadata": {"trusted": True}, "outputs": [],
             "source": to_source(cell_text)}
    new["cells"].insert(ADDED_INDEX, added)
    return new


def git_show(ref: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", ref], check=True, capture_output=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-config", required=True)
    ap.add_argument("--dreams-root", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "kernels/lb487_align"))
    args = ap.parse_args()
    raw = SRC_NB.read_bytes()
    assert sha256_bytes(raw) == SRC_SHA256, "kernels/lb487 notebook is not the 0.487 copy"
    at = git_show(ALIGN_TRAIN_REF)
    assert sha256_bytes(at) == ALIGN_TRAIN_SHA256, sha256_bytes(at)
    files = {
        "align0487_runner.py": (ROOT / "scripts/submit/align0487_runner.py").read_bytes(),
        "align_speed_trial.py": (ROOT / "scripts/train/align_speed_trial.py").read_bytes(),
        "base_fs.py": (ROOT / "scripts/engine2/base_fs.py").read_bytes(),
        "align_train.py": at,
        "align_config.json": Path(args.align_config).read_bytes(),
        "strip_dreams_ckpt.py": (ROOT / "scripts/ship/strip_dreams_ckpt.py").read_bytes(),
    }
    dz = dreams_zip(Path(args.dreams_root))
    cell = make_cell(files, dz)
    compile(cell, "added_cell", "exec")
    nb = build(json.loads(raw), cell)
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            if src.lstrip().startswith(("%", "!")):
                continue
            compile(src, f"cell{i}", "exec")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = json.loads(SRC_META.read_text())
    meta.pop("id_no", None)
    meta.update(id=SLUG, title="casmi26 lb487 align", code_file="notebook.ipynb")
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [DREAMS_DS]
    meta["model_sources"] = [ALIGN_MODEL_SRC]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    info = {"issue": 127, "base_sha256": SRC_SHA256, "added_index": ADDED_INDEX, "neural_cell_base": NEURAL_CELL,
            "embedded_sha256": {k: sha256_bytes(v) for k, v in files.items()}, "dreams_zip_sha256": sha256_bytes(dz),
            "dreams_commit": DREAMS_COMMIT, "align_train_ref": ALIGN_TRAIN_REF,
            "notebook_sha256": sha256_bytes((out / "notebook.ipynb").read_bytes())}
    (out / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
