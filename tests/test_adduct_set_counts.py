import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import adduct_set_counts as asc  # noqa: E402


def _df():
    return pd.DataFrame({
        "molecule_id": ["a", "a", "a", "b", "c", "c", "c", "c"],
        "adduct": ["[M+H]+", "[M+H]+", "[M-H]-", "[M+H]+", "[M+H]+", "[2M+Na]+", "[M+CH2O2-H]-", "[M+Na]+"],
        "instrument_type": ["timsTOF"] * 7 + ["Orbitrap"],
    })


def test_summarize_counts():
    s = asc.summarize(_df())
    assert s["n_molecules"] == 3 and s["n_spectra"] == 8
    assert s["n_adduct_types"] == {"1": 1, "2": 1, "3+": 1}
    assert s["ge2_frac"] == round(2 / 3, 4)
    assert dict(map(tuple, s["top_sets"]))["[M+H]+ [M-H]-"] == 1
    pats = dict(map(tuple, s["obs_patterns"]))
    assert pats == {"1100": 1, "1000": 1, "1011": 1}
    assert s["polarity"] == {"+": 1, "+-": 2}
    assert s["has_multimer_frac"] == round(1 / 3, 4)
    assert s["all_timstof_frac"] == round(2 / 3, 4)


def test_obs_pattern_order_and_polarity():
    assert asc.OBS == ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+")
    assert asc.obs_pattern({"[M-H]-", "[2M-H]-"}) == "0100"
    assert asc.polarity("[M+NH4]+") == "+" and asc.polarity("[M+Cl]-") == "-" and asc.polarity("x") == "?"


def test_same_formula_pairs():
    frm = {"k1": "C6H6", "k2": "C6H6", "k3": "C6H6", "k4": "C2H6O"}
    pat = {"k1": "1000", "k2": "1000", "k3": "1100", "k4": "1000"}
    r = asc.same_formula_pairs(frm, pat)
    assert r["formula_groups_ge2"] == 1 and r["formula_groups_with_ge2_patterns"] == 1
    assert r["pairs"] == 3 and r["pairs_diff_pattern"] == 2 and r["pairs_diff_frac"] == round(2 / 3, 4)


def test_main_end_to_end(tmp_path):
    hd = tmp_path / "holdouts"
    for h in asc.HOLDOUTS:
        (hd / h).mkdir(parents=True)
        _df().to_parquet(hd / h / "test.parquet")
    vis = _df().iloc[:4]
    vis.to_parquet(tmp_path / "test.parquet")
    out = tmp_path / "o.json"
    asc.main(["--holdout-dir", str(hd), "--test", str(tmp_path / "test.parquet"), "--out", str(out)])
    d = json.loads(out.read_text())
    assert set(d["sets"]) == set(asc.HOLDOUTS) | {"visible_test"}
    assert d["visible_vocab"] == ["[M+H]+", "[M-H]-"]
    v = d["sets_visible_vocab"]["v3_c1"]
    # c は [M+H]+ だけが残り、a は 2 種、b は 1 種
    assert v["n_molecules"] == 3 and v["n_adduct_types"] == {"1": 2, "2": 1, "3+": 0}
    assert v["molecules_dropped_no_vocab_spectra"] == 0
