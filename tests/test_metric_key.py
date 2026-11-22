"""#197 A1: 手元の採点（mrr25）の判定が公式の metric ノートの写しの判定と同じになることを、作り物のケースで確かめる。

metric の写し `scripts/validate/casmi_metric_official.py` は metric ノート（ipynb sha256 bf8535f0…09dd）の
唯一のセルをそのまま取り出したもの（書き換えない）。

既知の差（版を固定できない場合）: metric は RDKit 2026.03.3 に固定している。CI と手元の .venv は別の版で、
互変異性体の正準化の結果が版で変わりうる。ここでは同じプロセス（同じ版）で metric の関数と mrr25 を比べるので、
版の差はこのテストでは出ない。版の差の大きさは #197 に実データで記録している（metric_key_audit.py）。
"""
import hashlib
import importlib.util
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "scripts/validate/casmi_metric_official.py"
OFFICIAL_SHA256 = "c61d163f1bb768313fd0863822952f035f343924f37e3f5e4b5883ccef7e6201"   # 写しの .py の sha256


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mrr25 = _load("mrr25", ROOT / "scripts/validate/mrr25.py")

# (名前, 正解, 候補の並び)。正解はカンマ区切りで複数可（metric と同じ）
CASES = [
    # (i) 立体だけ違う
    ("stereo_only", "C[C@H](N)C(=O)O", ["CC(N)C(=O)O"]),
    ("stereo_flip", "C[C@@H](N)C(=O)O", ["CCO", "C[C@H](N)C(=O)O"]),
    ("stereo_ez", "C/C=C/C(=O)O", ["C/C=C\\C(=O)O"]),
    # (ii) 互変異性体
    ("tautomer_pyridone", "Oc1ccccn1", ["O=c1cccc[nH]1"]),
    ("tautomer_keto_enol", "CC(=O)CC(C)=O", ["CC(O)=CC(C)=O"]),
    # (iii) 塩・電荷が付いた形
    ("salt_sodium", "CC(=O)O", ["CC(=O)[O-].[Na+]"]),
    ("salt_hcl", "NCC(=O)O", ["NCC(=O)O.Cl"]),
    ("zwitterion", "NCC(=O)O", ["[NH3+]CC(=O)[O-]"]),
    ("protonated_amine", "CCN", ["CC[NH3+]"]),
    # (iv) 同じ InChIKey14 が重複したリスト（最初の一致だけが効く）
    ("dup_before_hit", "CCO", ["CC(=O)O", "OC(C)=O", "OCC", "CCO"]),
    ("dup_stereo_isomers", "CCCO", ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CCCO"]),
    # 立体中心が互変異性を縛る分子（metric の回帰テストの例: 修正前の手元は 0.0）
    ("stereo_pins_tautomer", "C=C(C)C1CCC2CCC(=O)C=C2C1", ["C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1"]),
    ("stereo_pins_tautomer_rev", "C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1", ["CCO", "C=C(C)C1CCC2CCC(=O)C=C2C1"]),
    # 正解が複数 SMILES
    ("multi_truth_second", "CCO,CCCO", ["CO", "CCCO"]),
    ("multi_truth_none", "CCO,CCCO", ["CO", "CCCCO"]),
    # 読めない候補は不一致
    ("unparseable_candidate", "CCO", ["not_a_smiles", "OCC"]),
]


@pytest.fixture(scope="module")
def official():
    pytest.importorskip("rdkit")
    pytest.importorskip("pandas")
    assert hashlib.sha256(OFFICIAL.read_bytes()).hexdigest() == OFFICIAL_SHA256
    mod = _load("casmi_metric_official", OFFICIAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # 版が 2026.03.3 でないときの UserWarning（上の既知の差）
        mod.install_dependencies()
    return mod


def _official_score(mod, truth, cands):
    import pandas as pd
    sol = pd.DataFrame({"molecule_id": ["m"], "smiles": [truth]})
    sub = pd.DataFrame({"molecule_id": ["m"], "smiles": [";".join(cands)]})
    return mod.score(sol, sub, "molecule_id", max_rank=25)


@pytest.mark.parametrize("name,truth,cands", CASES, ids=[c[0] for c in CASES])
def test_local_equals_official(official, name, truth, cands):
    local = mrr25.calculate_mrr25({"m": truth}, {"m": cands})
    assert local == pytest.approx(_official_score(official, truth, cands))


def test_expected_values_of_key_cases(official):
    # 値そのものも固定する（metric の docstring と同じ答え）
    assert mrr25.calculate_mrr25({"m": "C=C(C)C1CCC2CCC(=O)C=C2C1"}, {"m": ["C=C(C)[C@@H]1CCC2CCC(=O)C=C2C1"]}) == 1.0
    assert mrr25.calculate_mrr25({"m": "CCO,CCCO"}, {"m": ["CO", "CCCO"]}) == 0.5
    assert mrr25.calculate_mrr25({"m": "CCO"}, {"m": ["CC(=O)O", "OC(C)=O", "OCC", "CCO"]}) == pytest.approx(1 / 3)


def test_to_key_matches_official_canonicalizer(official):
    smiles = sorted({s for _, t, c in CASES for s in [*t.split(","), *c]})
    for s in smiles:
        assert mrr25.to_key(s) == official.canonicalize_smiles_to_ik14(s), s


def test_truth_keys_and_reciprocal_rank_accept_sets():
    A, B = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"
    assert mrr25.truth_keys(f"{A},{B}") == {A, B}
    assert mrr25.truth_keys("") == set()
    assert mrr25.reciprocal_rank({A, B}, ["CCCCCCCCCCCCCC", B]) == 0.5
    assert mrr25.reciprocal_rank(None, [A]) == 0.0
    assert mrr25.reciprocal_rank(A, [A]) == 1.0
