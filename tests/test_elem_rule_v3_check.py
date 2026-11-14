"""Issue #90: elem_rule_v3_check の数え上げ（合成データ）と、採否が frag_v3_check.judge と同じであること。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/validate", "scripts/ranker", "scripts/local_eval", "scripts/cv"):
    sys.path.insert(0, str(ROOT / p))

import elem_rule_v3_check as ec  # noqa: E402
import frag_v3_check as fc  # noqa: E402


def key(s):
    return s


def test_count_molecule_unusual_above_truth():
    c = ec.count_molecule("CCO", ["OB(O)O", "CCO", "CCC"], ["CCO", "CCC", "OB(O)O"], key)
    assert c == dict(truth_unusual=False, ctrl_top25_has_unusual=True, unusual_above_truth=True, elem_changed_top25=True)


def test_count_molecule_truth_absent_or_below():
    c = ec.count_molecule("CCO", ["CCN", "OB(O)O"], ["CCN", "OB(O)O"], key)
    assert c["unusual_above_truth"] is False and c["elem_changed_top25"] is False and c["ctrl_top25_has_unusual"]
    c = ec.count_molecule("C[Se]C", ["CCO", "C[Se]C"], ["CCO", "C[Se]C"], key)
    assert c["truth_unusual"] and not c["unusual_above_truth"]


def test_visible_counts():
    v = ec.visible_counts({"a": ["OB(O)O", "CCO"], "b": ["CCO", "C[SiH](C)C"], "c": ["CCO"], "d": []})
    assert v == dict(n=4, top25_has_unusual=2, top1_unusual=1, n_candidates=5, n_unusual_candidates=2)


def test_judge_is_the_preregistered_one():
    assert ec.judge is fc.judge and ec.REQUIRED == ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
