"""scripts/research/cross_instrument_sim.py の小さな道具（Issue #280）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))
import cross_instrument_sim as C  # noqa: E402


def test_inst_value_missing():
    for v in (None, float("nan"), np.nan, "", " ", "nan", "N/A", "None"):
        assert C.inst_value(v) is None
    assert C.inst_value("qTof") == "qTof"


def test_family_rules():
    exp = {"timsTOF": "timsTOF", "Orbitrap": "Orbitrap/FT", "ESI-QFT": "Orbitrap/FT", "LC-ESI-ITFT": "Orbitrap/FT",
           "Hybrid FT": "Orbitrap/FT", "Q-Exactive Plus": "Orbitrap/FT", "LC-ESI-HRMS-FT": "Orbitrap/FT",
           "LC-ESI-QEHF": "Orbitrap/FT", "QTOF": "TOF", "qtof": "TOF", "LC-ESI-ITTOF": "TOF",
           "Maxis II HD Q-TOF Bruker": "TOF", "impact HD": "TOF", "LC-ESI-QQ": "QQQ", "Quattro_QQQ:10eV": "QQQ",
           "Flow-injection QqQ/MS": "QQQ", "Linear Ion Trap": "IonTrap", "QIT": "IonTrap", "iontrap": "IonTrap",
           "LIT": "IonTrap", "Thermo LTQ": "IonTrap", "LC-ESI-Q": "other", "FAB-EBEB": "other"}
    for k, v in exp.items():
        assert C.family(k) == v, k
    assert C.family(float("nan")) is None and C.family(None) is None


def test_pick_structures_excludes_first_then_cuts():
    sets = {"D": {"a", "b"}, "A": {"a", "b"}, "B": {"a"}, "C": {"a", "c"}, "E": {"b", "c"}}
    pick, s = C.pick_structures(sets, {"A"}, 2)
    assert pick == ["C", "D"]
    assert s["n_eligible"] == 4 and s["n_eligible_excluded"] == 1 and s["n_excluded_before_cutoff"] == 1
    pick, _ = C.pick_structures(sets, set(), 2)
    assert pick == ["A", "C"]


def test_pairs_and_max_tables():
    sim = lambda a, b: {frozenset((1, 2)): 0.9, frozenset((1, 3)): 0.5, frozenset((2, 3)): 0.4}[frozenset((a, b))]
    groups = {("K", "[M+H]+"): [(10, "timsTOF", 1), (11, "timsTOF", 2), (12, "Orbitrap", 3)],
              ("K", "[M-H]-"): [(13, "Orbitrap", 1)]}
    p = C.pair_table(groups, sim)
    assert len(p) == 3 and sorted(p.sim) == [0.4, 0.5, 0.9]
    d = C.directed(p)
    assert len(d) == 6
    fam = {"timsTOF": "timsTOF", "Orbitrap": "Orbitrap/FT"}
    out = C.max_tables(d, fam)
    # スペクトルごと: 10 の同じ装置の最大 0.9、別 0.5。11: 0.9 / 0.4。12: 同じなし、別 max(0.5, 0.4) = 0.5
    assert out["same_raw"]["per_spectrum_max"]["n"] == 2 and out["same_raw"]["per_spectrum_max"]["q50"] == 0.9
    assert out["diff_raw"]["per_spectrum_max"]["n"] == 3
    assert out["tims_q__non_tims_r"]["per_group_max"] == C.summary([0.5])
    assert out["non_tims_q__tims_r"]["per_group_max"] == C.summary([0.5])
    pr = out["paired"]["same_raw_minus_diff_raw"]["per_spectrum"]
    assert pr["diff"]["n"] == 2 and abs(pr["diff"]["q50"] - 0.45) < 1e-9 and pr["share_b_ge_a"] == 0.0


def test_ref_instruments_after_hide():
    agg = pd.DataFrame(dict(inchikey14=["T1", "T1", "T2", "T3", "T4"],
                            inst=["timsTOF", "Orbitrap", "timsTOF", np.nan, "QTOF"],
                            ingest_lib=["enveda-180", "gnps", "enveda-180", "mona", "gnps"], n=[1, 1, 1, 1, 1]))
    keys = {"T1", "T2", "T3", "T4", "T5"}
    r = C.ref_instruments_after_hide(keys, {"T1", "T2"}, "enveda_libs", agg, {})
    assert r == dict(has_timsTOF=0, non_timsTOF_only=2, missing_inst_only=1, none=2)
    r = C.ref_instruments_after_hide(keys, set(), "all", agg, {})
    assert r == dict(has_timsTOF=2, non_timsTOF_only=1, missing_inst_only=1, none=1)
    r = C.ref_instruments_after_hide({"X"}, set(), "all", agg, {"X": {"T4"}})
    assert r["non_timsTOF_only"] == 1


def test_boot_median_ci_deterministic():
    v = [0.1, 0.2, 0.3, 0.9, 0.8]
    c = ["A", "A", "B", "C", "C"]
    a = C.boot_median_ci(v, c, n_boot=200, seed=0)
    b = C.boot_median_ci(v, c, n_boot=200, seed=0)
    assert a == b and a["n_clusters"] == 3 and a["median"] == 0.3
    assert a["ci95"][0] <= a["median"] <= a["ci95"][1]
    assert C.boot_median_ci([], []) == dict(n=0)
