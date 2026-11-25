#!/usr/bin/env python3
"""#244: dev 表の候補の並び（分子ごとの cand_key の順）が B8 の b8_features.json の L・Q と同じかを 20 集合で数える。"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
import dump_dev as D  # noqa: E402


def main() -> None:
    tab = pd.read_parquet(ROOT / "outputs/local_eval/features/dev_b8.parquet", columns=["set", "molecule_id", "cand_key", "is_tier"])
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
        out["sets"][h] = dict(same=same, diff=diff, missing=miss)
        out["n_same"] += same; out["n_diff"] += diff; out["n_missing"] += miss
    out["n_table_molecules"] = len(g)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
