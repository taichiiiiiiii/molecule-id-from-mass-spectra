"""Issue #184: scripts/research/bde_coverage.py の純粋な関数のテスト（実データは使わない）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import bde_coverage as B  # noqa: E402


def test_parse_keys_sorts_elements_and_orders():
    assert B.parse_keys("O=C C#N Br-C") == {("C", "O", 2), ("C", "N", 3), ("Br", "C", 1)}


def test_table_sizes_are_fixed():
    # 出典の表から数えた項目数（docs/research/bde-table-source.md §1）。キーを足し引きしたら落ちる
    assert {k: len(v) for k, v in B.TABLES.items()} == {
        "openstax_t7_2": 68, "wikipedia_bde": 13, "huheey_libretexts": 58,
        "metfrag_default_keys": 76, "bdedb2_single": 24}
    assert all(o in (1, 2, 3) for t in B.TABLES.values() for _, _, o in t)


def test_benzene_kekule_and_aromatic_keys():
    pytest.importorskip("rdkit")
    k = B.mol_bond_keys("c1ccccc1")
    assert sorted(k["kekule"]) == [("C", "C", 1)] * 3 + [("C", "C", 2)] * 3
    assert k["aromatic"] == [("C", "C", "ar")] * 6
    assert B.mol_bond_keys("not a smiles") is None


def test_rules_r0_r1():
    tab = B.parse_keys("C-C C-O")
    assert B.covered(("C", "O", 1), tab, "R0")
    assert not B.covered(("C", "O", 2), tab, "R0")
    assert B.covered(("C", "O", 2), tab, "R1")          # 同じ元素の組の単結合で引く
    assert not B.covered(("C", "N", 1), tab, "R1")      # 組が無ければ R1 でも無い
    assert not B.covered(("C", "C", "ar"), tab, "R1")   # 芳香族の型は R1 でも引かない
    assert not B.covered(("C", "C", "other"), tab, "R1")


def test_coverage_counts_bonds_and_structures():
    tab = B.parse_keys("C-C C-O")
    structs = [{"kekule": [("C", "C", 1), ("C", "O", 1)], "aromatic": []},
               {"kekule": [("C", "C", 1), ("C", "N", 1), ("C", "N", 1)], "aromatic": []}]
    c = B.coverage(structs, tab, "kekule", "R0")
    assert (c["n_struct"], c["n_bonds"], c["n_bonds_covered"], c["n_struct_all_covered"]) == (2, 5, 3, 1)
    assert c["bond_frac"] == 0.6 and c["struct_frac"] == 0.5
    assert c["top_missing"] == [["C-N", 2]]
