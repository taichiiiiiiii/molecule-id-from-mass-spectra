"""
Issue #90: 自前の規則 `elem`（定数・重み・しきい値なし）。

天然物らしくない元素（下の ALLOWED_ELEMENTS 以外）を 1 つでも含む候補を、含まない候補の後ろに回す。
安定な分割で、それぞれの中の順は変えない。RDKit で読めない SMILES は「対象外」（元素なしと同じ扱い）。
元素でない原子（ダミー原子 `*`、原子番号 0）は数えない。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Sequence, TypeVar

# 許可する元素（事前登録 #90。ここ 1 か所だけに書く）
ALLOWED_ELEMENTS = frozenset({"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"})

T = TypeVar("T")


@lru_cache(maxsize=None)
def has_unusual_element(smiles: str) -> bool:
    """ALLOWED_ELEMENTS 以外の元素を含めば True。読めない SMILES は False。"""
    if not smiles:
        return False
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(a.GetAtomicNum() > 0 and a.GetSymbol() not in ALLOWED_ELEMENTS for a in mol.GetAtoms())


def demote_unusual(rows: Sequence[T], smiles_of: Callable[[T], str]) -> list[T]:
    """対象元素を含む候補を後ろへ（安定な分割）。対象が 0 件なら同じ並びを返す。"""
    keep, back = [], []
    for r in rows:
        (back if has_unusual_element(smiles_of(r)) else keep).append(r)
    return keep + back
