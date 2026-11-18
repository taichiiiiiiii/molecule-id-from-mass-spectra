"""Issue #154: scripts/cv/msg_wild_repr_check.py の純粋な関数のテスト（合成の SMILES だけ）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))
pytest.importorskip("rdkit")

import msg_wild_repr_check as M  # noqa: E402


def test_imidic_and_tautomer_form():
    from rdkit import Chem
    amide = M.repr_features("CC(=O)Nc1ccccc1")
    imidic = M.repr_features("CC(O)=Nc1ccccc1")
    assert amide["imidic"] is False and imidic["imidic"] is True
    # InChI は移動する H を同じキーにする（だから表記の違いはキーでは見えない）
    k = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in ("CC(=O)Nc1ccccc1", "CC(O)=Nc1ccccc1")]
    assert k[0] == k[1]
    assert amide["taut_canon"] is True
    assert imidic["taut_canon"] is False


def test_rdkit_canonical_string_and_stereo():
    kek = M.repr_features("C1=CC=CC=C1O")
    aro = M.repr_features("Oc1ccccc1")
    assert kek["rdkit_canon_str"] is False and aro["rdkit_canon_str"] is True
    assert M.repr_features("C[C@H](N)C(=O)O")["stereo"] is True
    assert aro["stereo"] is False
    assert M.repr_features("not_a_smiles") is None


def test_rates_and_flags():
    t = [dict(rdkit_canon_str=True, taut_canon=False, imidic=True, stereo=False)] * 2
    d = [dict(rdkit_canon_str=True, taut_canon=True, imidic=False, stereo=False)] * 3 + [None]
    tr, dr = M.rates(t), M.rates(d)
    assert tr["n"] == 2 and dr["n"] == 3
    assert tr["imidic"] == 1.0 and dr["imidic"] == 0.0
    assert M.flag_diffs(tr, dr) == ["taut_canon", "imidic"]
    assert M.rates([])["imidic"] is None
    # taut_canon が None（正準化の失敗）の行は割合の分母から外す
    r = M.rates([dict(rdkit_canon_str=True, taut_canon=None, imidic=False, stereo=False),
                 dict(rdkit_canon_str=True, taut_canon=True, imidic=False, stereo=False)])
    assert r["taut_canon"] == 1.0 and r["n"] == 2
