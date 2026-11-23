"""Issue #210: scripts/research/headroom.py の純粋な関数のテスト（実データは使わない）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import headroom as H  # noqa: E402


def test_heavy_comp_from_formula_drops_h_and_charge():
    assert H.heavy_comp_from_formula("C10H12N2O") == (("C", 10), ("N", 2), ("O", 1))
    assert H.heavy_comp_from_formula("C5H12NO+") == (("C", 5), ("N", 1), ("O", 1))
    assert H.heavy_comp_from_formula("C2H3O2-") == (("C", 2), ("O", 2))
    assert H.heavy_comp_from_formula("CH4O") == (("C", 1), ("O", 1))
    # 保護・脱保護で H だけ違う式は同じ組成
    assert H.heavy_comp_from_formula("C6H11O7-") == H.heavy_comp_from_formula("C6H12O7")
    assert H.heavy_comp_from_formula("H3NO") == (("N", 1), ("O", 1))
    assert H.heavy_comp_from_formula("") is None
    assert H.heavy_comp_from_formula(None) is None
    assert H.heavy_comp_from_formula("c1ccc") is None


def test_heavy_comp_from_smiles_matches_formula():
    pytest.importorskip("rdkit")
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    for smi in ["OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C[N+](C)(C)CC(=O)[O-]", "c1cc[nH]c1", "[2H]C([2H])([2H])O",
                "OC(=O)c1ccccc1Cl"]:
        f = CalcMolFormula(Chem.MolFromSmiles(smi))
        assert H.heavy_comp_from_smiles(smi) == H.heavy_comp_from_formula(f), smi
    assert H.heavy_comp_from_smiles("not a smiles((") is None


def test_rank_of():
    assert H.rank_of(1.0) == 1
    assert H.rank_of(1 / 3) == 3
    assert H.rank_of(1 / 25) == 25
    assert H.rank_of(0.0) is None


def test_classify_order_and_cases():
    assert H.classify(1, True, True, False) == "iv_top1"
    assert H.classify(7, True, True, False) == "iii_rank2_25"
    assert H.classify(None, True, True, True) == "ii_below25"
    # 窓に無い: プール（窓の外）が PubChem より先
    assert H.classify(None, False, True, True) == "i_pool_offwin"
    assert H.classify(None, False, False, True) == "i_pubchem"
    assert H.classify(None, False, False, False) == "i_none"
    with pytest.raises(AssertionError):
        H.classify(26, True, True, False)


def test_summarize_bounds_sum_to_one_and_counts():
    recs = [dict(cat="iv_top1", rr=1.0), dict(cat="iv_top1", rr=1.0), dict(cat="iii_rank2_25", rr=0.5),
            dict(cat="iii_rank2_25", rr=0.25), dict(cat="ii_below25", rr=0.0), dict(cat="i_pubchem", rr=0.0),
            dict(cat="i_none", rr=0.0), dict(cat="i_pool_offwin", rr=0.0)]
    s = H.summarize(recs)
    assert s["n"] == 8
    assert math.isclose(s["mrr_raw"], 2.75 / 8)
    assert s["top1_share"] == 0.25 and s["top1_share"] <= s["mrr"]
    c = s["cats"]
    assert c["iv_top1"]["n"] == 2 and c["iv_top1"]["bound_raw"] == 0.0
    assert math.isclose(c["iii_rank2_25"]["bound_raw"], (0.5 + 0.75) / 8)
    for k in ("ii_below25", "i_pubchem", "i_none", "i_pool_offwin"):
        assert c[k]["n"] == 1 and math.isclose(c[k]["bound_raw"], 1 / 8)
    assert sum(v["n"] for v in c.values()) == 8
    assert math.isclose(sum(v["bound_raw"] for v in c.values()) + s["mrr_raw"], 1.0)


def test_wilson_known_values():
    lo, hi = H.wilson(50, 100)
    assert math.isclose(lo, 0.4038, abs_tol=1e-4) and math.isclose(hi, 0.5962, abs_tol=1e-4)
    lo, hi = H.wilson(0, 400)
    assert math.isclose(lo, 0.0, abs_tol=1e-12) and math.isclose(hi, 0.00952, abs_tol=1e-4)
    lo, hi = H.wilson(400, 400)
    assert math.isclose(hi, 1.0, abs_tol=1e-12) and math.isclose(lo, 0.99048, abs_tol=1e-4)


def test_in_candidates_only_keys_same_composition():
    calls = []

    def fake_key(s):
        calls.append(s)
        return {"A": "KA", "B": "KB", "C": "KT"}.get(s)

    thc = (("C", 6), ("O", 6))
    other = (("C", 5), ("O", 6))
    # 組成の違う候補は to_key を呼ばない（同じキーは同じ組成なので漏れない）
    assert H.in_candidates(["A", "B", "C"], [other, other, thc], {"KT"}, thc, fake_key) is True
    assert calls == ["C"]
    calls.clear()
    assert H.in_candidates(["A", "B"], [thc, None], {"KT"}, thc, fake_key) is False
    assert calls == ["A", "B"]   # 組成の分からない候補も調べる
    calls.clear()
    # 正解の組成が分からなければ全部調べる
    assert H.in_candidates(["A", "C"], [other, other], {"KT"}, None, fake_key) is True
    assert calls == ["A", "C"]
