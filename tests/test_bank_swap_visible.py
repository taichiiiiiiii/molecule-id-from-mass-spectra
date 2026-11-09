"""Issue #49: bank_swap_visible の比較関数（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bank_swap_visible", ROOT / "scripts/cv/bank_swap_visible.py")
bs = importlib.util.module_from_spec(_spec)
sys.modules["bank_swap_visible"] = bs
_spec.loader.exec_module(bs)


def test_compare_eng_jaccard_and_top1():
    ea = {"m1": {"keys": ["A", "B", "C"]}, "m2": {"keys": ["X", "Y"]}, "m3": {"keys": ["Q"]}}
    eb = {"m1": {"keys": ["B", "A", "D"]}, "m2": {"keys": ["X", "Y"]}}
    r = bs.compare_eng(ea, eb, top=3)
    assert r["n"] == 2 and r["only_a"] == ["m3"]
    assert r["jaccard"] == {"m1": 0.5, "m2": 1.0}
    assert r["top1_changed"] == 1 and r["top1_changed_ids"] == ["m1"]
    assert r["order_same"] == 1


def test_compare_pc_diff_definition():
    pa = {"m1": dict(S=2.0, fz_top=600.0, best_pool_fz=800.0, pc_keys=["A", "B"])}
    pb = {"m1": dict(S=1.5, fz_top=-5.0, best_pool_fz=20.0, pc_keys=["A", "C"])}
    r = bs.compare_pc(pa, pb)
    row = r["per_molecule"]["m1"]
    assert row["d_a"] == -200.0 and row["d_b"] == -25.0
    assert row["pc_jaccard"] == round(1 / 3, 4) and row["pc_top1_same"]
    assert r["median"]["pc_top1_same"] == 1


def test_merge_stats_from_log_takes_last_line():
    t = "x\nmerge stats {'untouched': 3, 'gentle': 1}\ny\nmerge stats {'untouched': 12, 'gentle': 0}\n"
    assert bs.merge_stats_from_log(t) == {"untouched": 12, "gentle": 0}
    assert bs.merge_stats_from_log("nothing") is None


def test_compare_sub_ranked_vs_single():
    sa = {"m1": ["a", "b"], "m2": ["c"], "m3": ["d", "e"]}
    sb = {"m1": ["a", "x"], "m2": ["c"], "m3": ["e", "d"]}
    r = bs.compare_sub(sa, sb)
    assert r["ranked_both"] == 2 and r["top1_same"] == 1
    assert r["single_a"] == 1 and r["single_both"] == 1
