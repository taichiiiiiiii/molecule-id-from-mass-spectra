"""scripts/research/b8_fold_counts.py の小さな道具のテスト（合成データ、Issue #216）。"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("b8_fold_counts", ROOT / "scripts/research/b8_fold_counts.py")
B8 = importlib.util.module_from_spec(spec)
sys.modules["b8_fold_counts"] = B8
spec.loader.exec_module(B8)


def test_split_labels_np_group_wins_over_fold():
    ts = pd.DataFrame(dict(inchikey14=["A", "B", "C", "D"], key=["A", "B", "C", "D"], libmask=[0, 1 << 9, 0, 0],
                           sid=[0, 1, 2, 3]))
    st = pd.DataFrame(dict(inchikey14=["A", "B", "C", "D"], identity_group=[1, 2, 2, 3], fold_identity=[1, 0, 1, 0]))
    lab = B8.split_labels(ts, st, 1)
    assert dict(zip(lab.inchikey14, lab.fold)) == {"A": "hold", "B": "np", "C": "np", "D": "train"}


def test_n_bal_disjoint_and_conservative():
    grp_of = {f"a{i}": f"g{i}" for i in range(10)}
    grp_of.update({f"b{i}": f"h{i}" for i in range(10)})
    grp_of.update({f"c{i}": f"e{i}" for i in range(10)})
    grp_of["b0"] = "g0"   # B の 1 つは A と同じ群 → B から外れる
    A = {f"a{i}" for i in range(5)}
    B = {f"b{i}" for i in range(10)}
    C = {f"c{i}" for i in range(10)}
    D = {f"e{i}" for i in range(10)}   # 群そのもの。C と同じ群 → (C∪D) の群は 10 → 2 型で 5
    grp = lambda x: x if x in D else grp_of.get(x)
    assert B8.n_bal(A, B, C, D, grp) == 5
    A2 = {f"a{i}" for i in range(10)}
    assert B8.n_bal(A2, B, C, D, grp) == 5   # ⌊10/2⌋ が効く


def test_feasible_rule():
    ok = B8.feasible({1: 60, 2: 60, 3: 60, 4: 60})
    assert ok["train_sums_leave_one_out"] == {1: 180, 2: 180, 3: 180, 4: 180} and ok["eval_sum"] == 240 and ok["feasible"]
    ng = B8.feasible({1: 20, 2: 20, 3: 20, 4: 20})
    assert not ng["ok_train"] and not ng["ok_eval"] and not ng["feasible"]
    ng2 = B8.feasible({1: 100, 2: 0, 3: 0, 4: 50})
    assert ng2["train_sums_leave_one_out"][1] == 50 and not ng2["ok_train"]


def test_overlap_counts_all_key_types():
    a = pd.DataFrame(dict(inchikey14=["A", "B"], key=["K1", "K2"]))
    b = pd.DataFrame(dict(inchikey14=["C", "D"], key=["K2", "K9"]))
    sc = {"A": "S1", "B": "S2", "C": "S3", "D": "S1"}
    assert B8.overlap_counts(a, b, sc) == dict(raw_inchikey14=0, canonical_key=1, scorer_key=1)
