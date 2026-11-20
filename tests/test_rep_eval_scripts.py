"""#171 §3・§5: rep_v3_check.py（採否の機械判定と assert）と rep_dseen.py（Δ_dom）の純粋な関数。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

import rep_v3_check as RC  # noqa: E402
import rep_dseen as RD  # noqa: E402


def _d(lo):
    return dict(diff=0.0, lo=lo, hi=lo + 0.05)


# ---------------------------------------------------------------- 採否
def test_judge_adopts_only_with_all_noninferior_and_one_positive():
    ok = {h: _d(-0.01) for h in RC.REQUIRED} | {"v3_c2n": _d(0.001)}
    assert RC.judge(ok)["adopt"] is True
    assert RC.judge(ok)["lo_gt_0"] == ["v3_c2n"]


def test_judge_rejects_when_any_below_noninf():
    d = {h: _d(0.01) for h in RC.REQUIRED} | {"enveda_v1": _d(-0.0201)}
    j = RC.judge(d)
    assert j["adopt"] is False and j["failing"] == ["enveda_v1"]


def test_judge_boundary_values():
    # 下限ちょうど −0.02 は可、0 ちょうどは「> 0」ではない
    d = {h: _d(-0.02) for h in RC.REQUIRED}
    assert RC.judge(d)["adopt"] is False
    d["v3_c1"] = _d(0.0)
    assert RC.judge(d)["adopt"] is False
    d["v3_c1"] = _d(0.0001)
    assert RC.judge(d)["adopt"] is True


def test_judge_rejects_missing_set():
    d = {h: _d(0.05) for h in RC.REQUIRED if h != "v3_c2"}
    j = RC.judge(d)
    assert j["adopt"] is False and j["missing"] == ["v3_c2"]


def test_check_done_requires_all_five_and_same_commit():
    dones = {h: dict(issue=171, holdout=h, commit="abc") for h in RC.DONE_SETS}
    assert RC.check_done(dones) == "abc"
    with pytest.raises(AssertionError):
        RC.check_done({h: v for h, v in dones.items() if h != "e180_c2k"})
    bad = dict(dones)
    bad["v3_c2"] = dict(issue=171, holdout="v3_c2", commit="def")
    with pytest.raises(AssertionError):
        RC.check_done(bad)


def _rj(**kw):
    rj = dict(rep_drop_keys_sha256="S", rep_align_all_ok=True, stages=["score", "prior", "rep", "frag"],
              rep_audit={k: 0 for k in RC.LEAK_KEYS}, rep_counts=dict(molecules=200))
    rj.update(kw)
    return rj


def test_check_inputs_asserts():
    h = "v3_c1"
    good = dict(manifest_after_sha="S", dk_sha="S", ctrl_sha=RC.CTRL_SHA256[h], ctrl_mrr=0.85351, n_logit_molecules=200)
    assert RC.check_inputs(h, _rj(), **good)["molecules"] == 200
    for change in (dict(ctrl_sha="x"), dict(ctrl_mrr=0.85), dict(dk_sha="T"), dict(n_logit_molecules=199)):
        with pytest.raises(AssertionError):
            RC.check_inputs(h, _rj(), **(good | change))
    with pytest.raises(AssertionError):
        RC.check_inputs(h, _rj(rep_audit={k: (1 if k == "leak_rep_meta_lib" else 0) for k in RC.LEAK_KEYS}), **good)
    with pytest.raises(AssertionError):
        RC.check_inputs(h, _rj(rep_align_all_ok=False), **good)
    with pytest.raises(AssertionError):
        RC.check_inputs(h, _rj(stages=["score", "prior", "frag"]), **good)


def test_split_by_coverage():
    a = {"m1": 1.0, "m2": 0.5, "m3": 0.0, "m4": 1.0}
    b = {"m1": 0.5, "m2": 0.5, "m3": 1.0, "m4": 1.0}
    cov = {"m1": 3, "m2": 0, "m3": 0, "m4": None}
    s = RC.split_by_coverage(a, b, cov)
    assert s["has"] == dict(n=1, diff=0.5, better=1, worse=0)
    assert s["none"] == dict(n=2, diff=-0.5, better=0, worse=1)
    assert s["unreadable"]["n"] == 1


# ---------------------------------------------------------------- Δ_dom
def test_nanmax_all():
    assert RD.nanmax_all([math.nan, 1.0, None, 3.0]) == 3.0
    assert math.isnan(RD.nanmax_all([math.nan, None]))


def test_molecule_d_reasons_order():
    assert RD.molecule_d([dict(group="U", rep=math.nan, f=1.0)]) == (None, "no_candidates")
    assert RD.molecule_d([dict(group="O", rep=1.0, f=1.0), dict(group="S", rep=2.0, f=0.0)]) == (None, "no_U")
    assert RD.molecule_d([dict(group="U", rep=1.0, f=1.0), dict(group="S", rep=2.0, f=0.0)]) == (None, "no_O")
    # sd 0 は rep と MetFrag-lite の 2 つで判定
    assert RD.molecule_d([dict(group="U", rep=1.0, f=1.0), dict(group="O", rep=1.0, f=0.0)]) == (None, "sd0")
    assert RD.molecule_d([dict(group="U", rep=1.0, f=1.0), dict(group="O", rep=2.0, f=1.0)]) == (None, "sd0")


def test_molecule_d_values():
    c = [dict(group="U", rep=3.0, f=1.0), dict(group="O", rep=1.0, f=1.0), dict(group="O", rep=2.0, f=4.0),
         dict(group="S", rep=0.0, f=2.0), dict(group="U", rep=math.nan, f=9.0)]
    rec, why = RD.molecule_d(c)
    assert why is None and rec["n_cands"] == 4 and rec["n_U"] == 1 and rec["n_O"] == 2 and rec["n_S"] == 1
    r = np.array([3.0, 1.0, 2.0, 0.0])
    zr = (r - r.mean()) / r.std()
    assert rec["d_dom"]["rep"] == pytest.approx(zr[0] - zr[1:3].mean())
    assert rec["d_seen"]["rep"] == pytest.approx(zr[3] - zr[0])
    f = np.array([1.0, 1.0, 4.0, 2.0])
    zf = (f - f.mean()) / f.std()
    assert rec["d_dom"]["f"] == pytest.approx(zf[0] - zf[1:3].mean())


def test_band_and_class2_risk():
    assert RD.band(0.01, 0.2) == "higher_train"
    assert RD.band(-0.2, -0.01) == "lower_train"
    assert RD.band(-0.1, 0.1) == "none" and RD.band(0.0, 0.1) == "none"
    assert RD.class2_risk({"v3_c1": "none", "e180_c2k": "higher_train"}) is True
    assert RD.class2_risk({"v3_c1": "lower_train", "e180_c2k": "none"}) is False


def test_summarize_set_uses_raw_bounds():
    recs = {f"m{i}": dict(n_cands=3, n_S=0, n_U=1, n_O=2, d_dom=dict(rep=1.0 + 0.01 * i, f=0.0),
                          d_seen=dict(rep=None, f=None)) for i in range(10)}
    s = RD.summarize_set(recs, {"x": "no_O"})
    assert s["band"] == "higher_train" and s["excluded"]["no_O"] == 1 and s["delta_seen"] is None
    assert s["delta_dom"]["n"] == 10
