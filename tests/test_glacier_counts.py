"""Issue #242: scripts/bench/glacier_counts.py の純粋な関数のテスト（区分の規則・区間の和集合・組成）。"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/bench"))
import glacier_counts as G  # noqa: E402


def test_instr_family_and_token():
    assert G.instr_family("timsTOF") == 0
    assert G.instr_family("Orbitrap Exploris") == 1
    assert G.instr_family("LC-ESI-QFT") == 1
    assert G.instr_family("Q-TOF") == 2
    assert G.instr_family(None) == 5
    assert G.instr_family(float("nan")) == 5
    assert G.instr_token("Q Exactive Plus") == "Orbitrap"
    assert G.instr_token("timsTOF") == "QTOF"
    assert G.instr_token(None) == "QTOF"


def test_spectrum_ce_and_bucket():
    assert G.spectrum_ce("eV", "40", [40.0]) == 40.0
    assert G.spectrum_ce("eV", "-60,-40,-20", [20.0, 40.0, 60.0]) == 40.0
    assert G.spectrum_ce("NCE", "30,45", [12.0]) == 37.5
    assert math.isnan(G.spectrum_ce("eV", "", []))
    assert math.isnan(G.spectrum_ce(None, None, None))
    assert G.ce_bucket(37.5) == 40.0
    assert G.ce_bucket(32.4) == 30.0
    assert G.ce_bucket(1.0) == 5.0
    assert math.isnan(G.ce_bucket(float("nan")))


def test_spectrum_jobs_coverage():
    j = G.spectrum_jobs("[M+H]+", "positive", "Orbitrap", "eV", "35", [35.0], G.COV_R)
    assert j == {("[M+H]+", "Orbitrap", 35.0)}
    assert G.spectrum_jobs("[M+Na]+", "positive", "timsTOF", "eV", "20", [20.0], G.COV_R) == {("[M+Na]+", "QTOF", 20.0)}
    assert G.spectrum_jobs("[M+Na]+", "positive", "timsTOF", "eV", "20", [20.0], G.COV_H) == set()
    assert G.spectrum_jobs("[M-H]-", "negative", "timsTOF", "eV", "20", [20.0], G.COV_R) == set()
    assert G.spectrum_jobs("[M+NH4]+", "positive", "timsTOF", "eV", "20", [20.0], G.COV_R) == set()
    # CE が NaN → 20・40・60 の 3 予測（ランナーの 'sum'）
    j = G.spectrum_jobs("[M+H]+", "positive", "QTOF", "eV", "", [], G.COV_R)
    assert j == {("[M+H]+", "QTOF", 20.0), ("[M+H]+", "QTOF", 40.0), ("[M+H]+", "QTOF", 60.0)}


def test_union_length():
    assert G.union_length([]) == 0
    assert G.union_length([(0, 10)]) == 10
    assert G.union_length([(0, 10), (5, 15)]) == 15
    assert G.union_length([(0, 10), (10, 12)]) == 12
    assert G.union_length([(0, 3), (5, 8), (6, 7)]) == 6
    assert G.union_length([(4, 4), (2, 1)]) == 0


def test_heavy_comp_str():
    assert G.heavy_comp_str("CCO") == (("C", 2), ("O", 1))
    assert G.heavy_comp_str("c1ccccc1Cl") == (("C", 6), ("Cl", 1))
    assert G.heavy_comp_str("[NH4+]") == (("N", 1),)
    assert G.heavy_comp_str("[13CH3]O") == (("C", 1), ("O", 1))


def test_no_test_parquet_in_sample_inputs():
    """測定の入力は B8 の dev 集合（train 由来）・プール・PubChem 層だけから作る（競技の test は読まない）。"""
    src = (ROOT / "scripts/bench/glacier_counts.py").read_text()
    assert "data/test.parquet" not in src
    assert 'ROOT / "data/test.parquet"' not in src
