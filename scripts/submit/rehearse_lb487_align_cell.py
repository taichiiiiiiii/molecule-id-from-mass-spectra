#!/usr/bin/env python3
"""#127 手順 5 の続き（ローカル CPU、学習なし、v3 を見ない）: 生成したノートの足したセルを、/kaggle/input の代わりに
手元のパスを返すようにだけ差し替えて exec し、`ALIGN0487_MIX` を見えているテストの数分子で動かす（リハーサル）。

    PYTHONPATH=<scratch>/rdkit202603 .venv/bin/python scripts/submit/rehearse_lb487_align_cell.py \
        --root <元のフォルダ> --nb kernels/lb487_align/notebook.ipynb --work <scratch>/n1117n_rehearse \
        --cands <check_align0487_local.py が書いた cands.parquet> --test <同じく test_sub.parquet> \
        --dreams-state <dreams_ssl_state.pt> --align-model <model_best.pt>

確かめること:
- セルの中の埋め込みファイル・DreaMS の zip の sha256 照合と展開、runner の別プロセスでの実行（RDKit 2026.03.3 を期待）が通る
- 返り値が `align0487_runner.mix_rows`（式の参照）と一致する（最大の絶対差 ≤ 1e-12）。2 項に戻った分子の行は
  0.5 * za + 0.5 * zb とバイト一致
- report の status が ok
- 尾の枠・空き枠の記録の関数（`_a0487_downstream`）が、下流のセルの結果（偽物）を report の `downstream` に写す
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("root", "nb", "work", "cands", "test", "dreams-state", "align-model"):
        ap.add_argument("--" + k, required=True)
    args = ap.parse_args()
    import pandas as pd
    R, W = Path(args.root), Path(args.work)
    W.mkdir(parents=True, exist_ok=True)
    comp = W / "comp"; comp.mkdir(exist_ok=True)
    tp = comp / "test.parquet"
    if not tp.exists():
        tp.symlink_to(Path(args.test).resolve())
    nb = json.loads(Path(args.nb).read_text())
    cell = "".join(nb["cells"][6]["source"])
    assert cell.startswith("# ── #127（診断）")
    ns = {"ROOT": W, "COMP": comp, "pd": pd, "__name__": "rehearse"}
    exec(compile(cell, "added_cell", "exec"), ns)
    ns["_A0487_DIR"] = W / "align0487"
    local = {("pool_meta.parquet", "casmi26-v2-pool"): R / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet",
             ("chem.py", "casmi26-v3-models/code/casmi"): R / "data/ext/v4g/casmi26-v3-models/code/casmi/chem.py",
             ("dreams_ssl_state.pt", "casmi26-align0487-weights"): Path(args.dreams_state),
             ("model_best.pt", "casmi26-align0487-weights"): Path(args.align_model)}
    ns["_a0487_one"] = lambda pattern, must: local[(pattern, must)]
    c = pd.read_parquet(args.cands)
    idx = pd.MultiIndex.from_arrays([c.molecule_id.astype(str), c.smiles.astype(str)], names=["molecule_id", "smiles"])
    rng = np.random.default_rng(0)

    def _z(_s):   # 外側セル 15 の _z と同じ
        _g = _s.groupby(level=0)
        return (_s - _g.transform("mean")) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)

    za = _z(pd.Series(rng.normal(size=len(idx)), index=idx))
    zb = _z(pd.Series(rng.normal(size=len(idx)), index=idx))
    out = ns["ALIGN0487_MIX"](idx, za, zb, _z)
    # 尾の枠・空き枠の記録（post_run_cell の関数）: 下流のセルの結果の形の偽物を置いて 1 回呼ぶ。値は読むだけ
    ns["PARALLAX_DIAGNOSTICS"] = {"status": "ok", "tail_changed": 3, "top1_changed": 1, "adjacent_swaps": 2}
    ns["MERGE_REPORT"] = {"status": "ok", "outsider_fills": 5, "lotus_fills": 0}
    ns["_a0487_downstream"]()
    rep = json.loads((W / "align0487_report.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import align0487_runner as AR
    a = pd.read_parquet(W / "align0487" / "a_align.parquet").a_align.values
    ref, ok_m = AR.mix_rows(idx.get_level_values(0).values, za.values, zb.values, a, AR.z_by_molecule)
    two = ~np.asarray(idx.get_level_values(0).isin(ok_m))
    base = 0.5 * za.values + 0.5 * zb.values
    res = {"status": rep["status"], "molecules": rep.get("molecules"), "three_term": rep.get("three_term_molecules"),
           "two_term": rep.get("two_term_molecules"), "two_term_frac": rep.get("two_term_frac"),
           "max_abs_vs_mix_rows": float(np.max(np.abs(out.values - ref))),
           "two_term_rows": int(two.sum()), "two_term_rows_bytes_equal": bool(np.array_equal(out.values[two], base[two])),
           "runner_rdkit": (rep.get("runner") or {}).get("versions"), "check_d": (rep.get("runner") or {}).get("check_d"),
           "seconds": rep.get("seconds"), "runner_seconds": (rep.get("runner") or {}).get("seconds"),
           "downstream": rep.get("downstream"), "downstream_ok": rep.get("downstream") == {
               "tail_slot": {"status": "ok", "tail_changed": 3, "top1_changed": 1, "adjacent_swaps": 2},
               "empty_slot": {"status": "ok", "outsider_fills": 5, "lotus_fills": 0}}}
    (W / "rehearse.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str))


if __name__ == "__main__":
    main()
