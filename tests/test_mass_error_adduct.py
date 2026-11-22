"""#203: scripts/eda/mass_error_adduct.py の純粋な関数（rdkit・casmi を使わない）。"""
import importlib.util
import math
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "scripts/eda/mass_error_adduct.py"
_s = importlib.util.spec_from_file_location("mass_error_adduct", _p)
M = importlib.util.module_from_spec(_s)
_s.loader.exec_module(M)

MASSES = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.9949146196, "Na": 22.9897692809}


def test_parse_formula_basic_and_charge():
    glc = 6 * 12.0 + 12 * MASSES["H"] + 6 * MASSES["O"]
    assert math.isclose(M.parse_formula("C6H12O6", MASSES), glc, rel_tol=0, abs_tol=1e-9)
    # 末尾の電荷は無視する（[M]+ の陽イオンの式）
    choline = 5 * 12.0 + 14 * MASSES["H"] + MASSES["N"] + MASSES["O"]
    assert math.isclose(M.parse_formula("C5H14NO+", MASSES), choline, abs_tol=1e-9)
    assert math.isclose(M.parse_formula("C2H6O2", MASSES), M.parse_formula("C2H6O2-", MASSES), abs_tol=1e-12)


def test_parse_formula_rejects_bad():
    assert math.isnan(M.parse_formula(None, MASSES))
    assert math.isnan(M.parse_formula("", MASSES))
    assert math.isnan(M.parse_formula("C6H12Xx6", MASSES))   # 表に無い元素
    assert math.isnan(M.parse_formula("C6 H12", MASSES))     # 余分な文字
    assert math.isnan(M.parse_formula("[13C]H4", MASSES))


def test_ppm_and_group_stats():
    p = M.ppm_error([100.001, 99.999, 100.0], [100.0, 100.0, 100.0])
    assert np.allclose(p, [10.0, -10.0, 0.0], atol=1e-6)
    v = np.array([0.0] * 98 + [20.0, -30.0, np.nan])
    g = M.group_stats(v)
    assert g["n"] == 100 and g["n_out"] == 2 and math.isclose(g["frac_out"], 0.02)
    assert g["median"] == 0.0
    assert M.group_stats(np.array([np.nan]))["n"] == 0


def test_grouped_table_other_bucket():
    instr = ["T"] * 150 + ["T"] * 5 + ["Q"] * 3
    add = ["[M+H]+"] * 150 + ["[M+K]+"] * 5 + ["[M-H]-"] * 3
    ppm = np.zeros(len(instr))
    rows = M.grouped_table(instr, add, ppm, min_group=100)
    names = {(r["instrument"], r["adduct"]): r["n"] for r in rows}
    assert names[("T", "[M+H]+")] == 150 and names[("T", "その他")] == 5 and names[("Q", "その他")] == 3


def test_match_known_delta_signs_and_tol():
    Mx = 300.0
    assert M.match_known_delta(18.010565, Mx) == ["+H2O"]
    assert "-H2O" in M.match_known_delta(-18.010565 + 0.001, Mx)          # 3.3 ppm 以内
    assert M.match_known_delta(-18.010565 + 0.01, Mx) == []                 # 33 ppm はずれ
    assert "+Na-H" in M.match_known_delta(21.981942, Mx)
    assert M.match_multimer(600.0, 300.0) == ["2M"]
    assert M.match_multimer(150.0005, 300.0) == ["M/2"]


def test_nearest_errors_signed():
    fm = np.array([50.0, 100.0, 150.0])
    e = M.nearest_errors([101.0083, 151.0, 30.0], fm, [1.0073])
    assert math.isclose(e[0], 0.001, abs_tol=1e-9)
    assert math.isclose(e[1], -0.0073, abs_tol=1e-9)
    assert math.isclose(e[2], 30.0 - 1.0073 - 50.0, abs_tol=1e-9)
    assert np.isnan(M.nearest_errors([1.0], np.array([]), [1.0])).all()


def test_background_corrected_uniform_bg():
    # 真の一致 100 本は |e| ≤ 0.002、背景は 0〜0.05 に一様に 1000 本
    rng = np.random.default_rng(0)
    a = np.concatenate([rng.uniform(0, 0.002, 100), rng.uniform(0, 0.05, 1000)])
    r = M.background_corrected(a)
    assert abs(r["in_tol_corr"] - 100) < 25
    assert r["true_frac_outside_tol"] is not None and abs(r["true_frac_outside_tol"]) < 0.25
    assert len(r["hist"]) == 50 and sum(r["hist"]) == r["n_peaks_within_hi"]


def test_multi_adduct_consistency():
    mol = ["a", "a", "b", "b", "c"]
    add = ["[M+H]+", "[M-H]-", "[M+H]+", "[M+Na]+", "[M+H]+"]
    nm = [300.0, 300.0003, 300.0, 322.0, 200.0]
    r = M.multi_adduct_consistency(mol, add, nm)
    assert r["n_molecules"] == 3 and r["n_multi_adduct"] == 2 and r["n_consistent"] == 1


def test_molecule_half_width():
    table = {("timsTOF", "[M+H]+"): 4.1, ("timsTOF", "その他"): 5.5, ("qTof", "[M+H]+"): 1e6}
    assert M.molecule_half_width(["timsTOF", "timsTOF"], ["[M+H]+", "[M+K]+"], table) == 5.5
    assert M.molecule_half_width(["timsTOF", "qTof"], ["[M+H]+", "[M+H]+"], table) == 1e6
    assert math.isnan(M.molecule_half_width(["Orbitrap"], ["[M+H]+"], table))
