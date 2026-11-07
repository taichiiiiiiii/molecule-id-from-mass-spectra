#!/usr/bin/env python3
"""
学習グループ（Issue #29）の点検: v3 / split.parquet との積集合が空、fold が 1 つだけ、hide_rule が v3 と同じ、drop_keys = 正解キー、
グループ間の重複なし。結果を outputs/ranker/GROUPS_CHECK.json に書き、1 つでも外れたら終了コード 1。

    python scripts/ranker/check_groups.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAMES = [f"rk_f{k}_c{c}" for k in range(1, 5) for c in (1, 2)]


def main() -> None:
    truth = lambda n: set(pd.read_csv(ROOT / f"outputs/holdouts/{n}/truth.csv").inchikey14)
    v3 = truth("v3_c1") | truth("v3_c2")
    split = set(pd.read_parquet(ROOT / "data/ext/foldsafe/split.parquet").inchikey14)
    fold = pd.read_parquet(ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet",
                           columns=["inchikey14", "fold_identity"]).drop_duplicates("inchikey14").set_index("inchikey14").fold_identity
    seen, rows, ok = set(), {}, True
    for n in NAMES:
        t = truth(n)
        k, c = int(n[4]), n[-1]
        rule = json.loads((ROOT / f"data/holdouts/{n}/hide_rule.json").read_text())
        ref = json.loads((ROOT / f"data/holdouts/v3_c{c}/hide_rule.json").read_text())
        drop = set((ROOT / f"data/holdouts/{n}/drop_keys.txt").read_text().split())
        r = dict(n=len(t), and_v3=len(t & v3), and_split=len(t & split), folds=sorted(int(x) for x in set(fold[sorted(t)])),
                 rule_same_as_v3=rule == ref, drop_equals_truth=drop == t, and_other_groups=len(t & seen))
        good = r["and_v3"] == 0 and r["and_split"] == 0 and r["folds"] == [k] and r["rule_same_as_v3"] and r["drop_equals_truth"] \
            and r["and_other_groups"] == 0
        r["ok"] = bool(good); ok &= good
        rows[n] = r; seen |= t
        print(n, json.dumps(r))
    out = ROOT / "outputs/ranker/GROUPS_CHECK.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(ok=bool(ok), groups=rows), indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
