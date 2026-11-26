#!/usr/bin/env python3
"""#244: dev 表の候補の並び（分子ごとの cand_key の順）が B8 の b8_features.json の L・Q と同じかを 20 集合で数える。
完了レビューの minor への対応: 各集合の記録の logits_sha256 = B8 の logits_sha256、np_score = #241 の pred_dev.parquet（L は
query_fold と SMILES、Q は query_fold と鍵で結ぶ。has_np = 0 の行は除く）も数える。"""
import numpy as np
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
import dump_dev as D  # noqa: E402


def main() -> None:
    tab = pd.read_parquet(ROOT / "outputs/local_eval/features/dev_b8.parquet",
                          columns=["set", "molecule_id", "cand_key", "cand_smiles", "is_tier", "query_fold", "np_score", "has_np"])
    rep = json.loads((ROOT / "docs/features/dev_b8_report.json").read_text())
    g = tab.groupby(["set", "molecule_id"], sort=False).cand_key.apply(list).to_dict()
    out = dict(sets={}, n_same=0, n_diff=0, n_missing=0)
    for h in D.DEV_SETS:
        j = json.loads((ROOT / f"outputs/local_eval/{h}/b8_features.json").read_text())
        same = diff = miss = 0
        for m in j["molecules"]:
            if m.get("missing_logits"):
                continue
            want = [k for k, _ in m["L"]] + [k for k, _ in m["Q"]]
            got = g.get((h, m["molecule_id"]))
            if got is None:
                miss += 1
            elif got == want:
                same += 1
            else:
                diff += 1
        out["sets"][h] = dict(same=same, diff=diff, missing=miss,
                              logits_sha256_eq_b8=rep["sets"][h]["logits_sha256"] == j["logits_sha256"])
        out["n_same"] += same; out["n_diff"] += diff; out["n_missing"] += miss
    out["n_table_molecules"] = len(g)
    out["logits_sha256_eq_b8_all"] = all(v["logits_sha256_eq_b8"] for v in out["sets"].values())
    p = pd.read_parquet(ROOT / "outputs/local_eval/np_likeness/pred_dev.parquet", columns=["source", "key", "smiles", "query_fold", "np_score"])
    pool = p[p.source == "pool"].drop_duplicates(["query_fold", "smiles"]).set_index(["query_fold", "smiles"]).np_score
    tier = p[p.source == "pubchem_b8"].drop_duplicates(["query_fold", "key"]).set_index(["query_fold", "key"]).np_score
    t = tab[tab.has_np == 1]
    L_, Q_ = t[t.is_tier == 0], t[t.is_tier == 1]
    xl = pool.reindex(pd.MultiIndex.from_arrays([L_.query_fold, L_.cand_smiles])).to_numpy()
    xq = tier.reindex(pd.MultiIndex.from_arrays([Q_.query_fold, Q_.cand_key])).to_numpy()
    def cmp(x, v):
        ok = np.isfinite(x)
        return dict(n=int(len(x)), n_joined=int(ok.sum()), n_close=int(np.isclose(x[ok], v[ok], rtol=1e-9, atol=1e-12).sum()),
                    max_abs_diff=float(np.abs(x[ok] - v[ok]).max()) if ok.any() else None)
    # pred_dev の Q は鍵ごとに SMILES 1 つ。表は候補自身の SMILES で予測するので、同じ SMILES の行でも比べる
    tier_s = p[p.source == "pubchem_b8"].drop_duplicates(["query_fold", "key", "smiles"]).set_index(["query_fold", "key", "smiles"]).np_score
    xqs = tier_s.reindex(pd.MultiIndex.from_arrays([Q_.query_fold, Q_.cand_key, Q_.cand_smiles])).to_numpy()
    out["np_vs_pred_dev"] = dict(L=cmp(xl, L_.np_score.to_numpy()), Q_by_key=cmp(xq, Q_.np_score.to_numpy()),
                                 Q_by_key_and_smiles=cmp(xqs, Q_.np_score.to_numpy()), n_has_np0=int((tab.has_np == 0).sum()))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
