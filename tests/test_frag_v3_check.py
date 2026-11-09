"""Issue #48: frag_v3_check の採否規則と rr の計算（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/validate"))
_spec = importlib.util.spec_from_file_location("frag_v3_check", ROOT / "scripts/cv/frag_v3_check.py")
fc = importlib.util.module_from_spec(_spec)
sys.modules["frag_v3_check"] = fc
_spec.loader.exec_module(fc)


def d(lo):
    return dict(diff=0.0, lo=lo, hi=lo + 0.1)


def test_adopt_when_all_noninferior_and_primary_win():
    r = fc.judge({"v3_c1": d(0.01), "v3_c2": d(0.0), "enveda_v1": d(-0.01), "v3_c2n": d(-0.02)})
    assert r["adopt"] and r["significant_win"] == ["v3_c1"] and not r["failing"]


def test_reject_when_secondary_fails():
    r = fc.judge({"v3_c1": d(0.05), "v3_c2": d(0.0), "enveda_v1": d(-0.021), "v3_c2n": d(0.0)})
    assert not r["adopt"] and r["failing"] == ["enveda_v1"]


def test_reject_when_only_secondary_wins():
    r = fc.judge({"v3_c1": d(0.0), "v3_c2": d(0.0), "enveda_v1": d(0.05), "v3_c2n": d(0.05)})
    assert not r["adopt"] and r["significant_win"] == []


def test_reject_when_required_missing():
    r = fc.judge({"v3_c1": d(0.05), "v3_c2": d(0.0), "enveda_v1": d(0.0)})
    assert not r["adopt"] and r["missing"] == ["v3_c2n"]


def test_per_molecule_rr_missing_prediction_is_zero():
    k1, k2 = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"
    rr = fc.per_molecule_rr({"m1": k1, "m2": k2}, {"m1": [k2, k1]})
    assert rr == {"m1": 0.5, "m2": 0.0}
