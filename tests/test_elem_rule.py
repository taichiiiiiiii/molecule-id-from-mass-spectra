"""Issue #90: elem 規則（天然物らしくない元素を含む候補を後ろへ）の単体テスト。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

import elem_rule as E  # noqa: E402


def test_a_allowed_and_unusual_elements():
    for smi in ["c1ccccc1", "CCO", "Clc1ccccc1"]:          # ベンゼン・エタノール・クロロベンゼン
        assert E.has_unusual_element(smi) is False, smi
    for smi in ["C[SiH](C)C", "OB(O)O", "C[Se]CCC(N)C(=O)O"]:   # トリメチルシラン・ホウ酸・セレノメチオニン
        assert E.has_unusual_element(smi) is True, smi


def test_allowed_set_is_fixed():
    assert E.ALLOWED_ELEMENTS == {"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"}


def test_b_stable_partition():
    rows = [("a", "CCO"), ("b", "OB(O)O"), ("c", "c1ccccc1"), ("d", "C[SiH](C)C"),
            ("e", "Clc1ccccc1"), ("f", "C[Se]CCC(N)C(=O)O")]
    out = E.demote_unusual(rows, lambda r: r[1])
    assert [r[0] for r in out] == ["a", "c", "e", "b", "d", "f"]
    assert sorted(out) == sorted(rows)


def test_c_no_target_unchanged():
    rows = [(3, "CCO"), (1, "c1ccccc1"), (2, "Clc1ccccc1"), (0, "OP(=O)(O)O")]
    assert E.demote_unusual(rows, lambda r: r[1]) == rows
    assert E.demote_unusual([], lambda r: r) == []


def test_d_unreadable_smiles_is_false():
    for smi in ["not_a_smiles", "C1CC", ""]:
        assert E.has_unusual_element(smi) is False, smi
    rows = [("x", "OB(O)O"), ("y", "not_a_smiles")]
    assert [r[0] for r in E.demote_unusual(rows, lambda r: r[1])] == ["y", "x"]
