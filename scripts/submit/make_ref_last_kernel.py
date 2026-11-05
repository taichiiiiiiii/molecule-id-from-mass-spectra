#!/usr/bin/env python3
"""
提出用ノートブック: 0.479 版（kernels/lb479）の最終 submission.csv に ref_last（Issue #22）を掛けるセルを 1 つ足す。

    python scripts/submit/make_ref_last_kernel.py [--slug casmi26-lb479-reflast]

足すセルは最終レイヤー（Variant M のマージ）の直後、表示用セルの前。失敗したら元の submission.csv のまま。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path("kernels/lb479")

CELL = r'''
# ---- Issue #22: candidates with their own reference spectrum go last when the library match is weak ----
import json, hashlib, traceback
REF_LAST_REPORT = dict(status='not run')
__REF_LAST_SOURCE__
try:
    import pyarrow.parquet as _pq
    _orig = (ROOT / 'submission.csv').read_bytes()
    _ref = set(_pq.read_table(COMP / 'train.parquet', columns=['inchikey14']).column(0).to_pylist())
    _lib = dict(LIB_MAX) if LIB_MAX else {}
    if not _lib:
        _nf = pd.read_parquet(ROOT / 'neural_candidate_features.parquet', columns=['molecule_id', 'library_max'])
        _lib = {str(m): float(v) for m, v in _nf.groupby('molecule_id', sort=False).library_max.max().items()}
    _keys_of = make_keys_of()
    _sub = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
    _rows, _n_ch, _n_top1, _n_weak = [], 0, 0, 0
    for _mid, _ans in _sub.itertuples(index=False):
        _c = [t for t in str(_ans).split(';') if t]
        _weak = float(_lib.get(str(_mid), 0.0)) < LIB_TAU
        _n_weak += _weak
        _new, _ch = ref_last(_c, _keys_of, _ref, _lib.get(str(_mid), 0.0))
        _n_ch += _ch; _n_top1 += bool(_c) and _new[0] != _c[0]
        _rows.append((_mid, ';'.join(_new)))
    _final = pd.DataFrame(_rows, columns=['molecule_id', 'smiles'])
    assert _final.molecule_id.tolist() == _sub.molecule_id.tolist()
    assert _final.smiles.str.split(';').map(len).between(1, 25).all()
    assert all(sorted(a.split(';')) == sorted(b.split(';')) for a, b in zip(_final.smiles, _sub.smiles))  # same set, new order
    _final.to_csv(ROOT / 'submission.csv', index=False)
    REF_LAST_REPORT = dict(status='ok', rows=len(_final), weak_library=_n_weak, lists_changed=_n_ch, top1_changed=_n_top1,
                           train_keys=len(_ref), lib_tau=LIB_TAU,
                           before_sha256=hashlib.sha256(_orig).hexdigest(),
                           after_sha256=hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest())
except Exception as _e:
    traceback.print_exc()
    try:
        (ROOT / 'submission.csv').write_bytes(_orig)
    except Exception:
        pass
    REF_LAST_REPORT = dict(status='failed -> unchanged submission', error=repr(_e))
(ROOT / 'ref_last_report.json').write_text(json.dumps(REF_LAST_REPORT, indent=2))
print('REF_LAST', json.dumps(REF_LAST_REPORT))
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="casmi26-lb479-reflast")
    args = ap.parse_args()
    nb = json.loads((SRC / "notebook.ipynb").read_text())
    code_idx = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    merge_i = [i for i in code_idx if "LIB_MAX = {}" in "".join(nb["cells"][i]["source"])]
    assert len(merge_i) == 1, merge_i
    src = (HERE / "ref_last.py").read_text()
    body = src.split('"""', 2)[2].replace("from __future__ import annotations\n", "")  # docstring と __future__ を除く
    cell = CELL.replace("__REF_LAST_SOURCE__", body)
    compile(cell, "ref_last_cell", "exec")
    nb["cells"].insert(merge_i[0] + 1, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                                        "source": cell})
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"taichiiiii/{args.slug}", title=args.slug.replace("-", " "), is_private=True)
    dst = Path("kernels") / args.slug.replace("-", "_")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", dst, meta["id"], "| inserted after code cell", merge_i[0])


if __name__ == "__main__":
    main()
