"""scripts/research/spectra_per_molecule.py の小さな道具のテスト（合成データ、Issue #276）。"""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("spectra_per_molecule", ROOT / "scripts/research/spectra_per_molecule.py")
SPM = importlib.util.module_from_spec(spec)
sys.modules["spectra_per_molecule"] = SPM
spec.loader.exec_module(SPM)


def test_summarize_counts():
    s = SPM.summarize([1, 3, 3, 5, 20])
    assert s["n_molecules"] == 5 and s["n_spectra"] == 32
    assert s["mean"] == 6.4 and s["median"] == 3.0 and s["max"] == 20
    assert s["share_1"] == 0.2 and s["share_le3"] == 0.6 and s["share_gt16"] == 0.2
    assert s["mean_capped16"] == round((1 + 3 + 3 + 5 + 16) / 5, 3)
    assert s["hist"] == {"1": 1, "3": 2, "5": 1, "20": 1}


def test_summarize_empty():
    assert SPM.summarize([]) == dict(n_molecules=0)


def test_main_on_synthetic_tree(tmp_path, monkeypatch):
    # 2 集合（型 a・d）と見える test を作り、型ごとの数が合うことを確かめる（train は読まない）
    for s, mols in {"b8_f1_a_t10": ["m1", "m1", "m2"], "b8_f1_d_t10": ["x", "y", "y", "y"]}.items():
        d = tmp_path / "data/holdouts" / s
        d.mkdir(parents=True)
        pd.DataFrame(dict(molecule_id=mols)).to_parquet(d / "test.parquet")
    table = tmp_path / "t.parquet"
    pd.DataFrame(dict(set=["b8_f1_a_t10", "b8_f1_d_t10", "b8_f1_a_t10"])).to_parquet(table)
    test = tmp_path / "test.parquet"
    pd.DataFrame(dict(molecule_id=["q1", "q1", "q2"], adduct=["[M+H]+", "[M-H]-", "[M+H]+"],
                      ionization_mode=[1, -1, 1])).to_parquet(test)
    out = tmp_path / "o.json"
    monkeypatch.setattr(SPM, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--table", str(table), "--test", str(test), "--no-train", "--out", str(out)])
    SPM.main()
    j = json.loads(out.read_text())
    assert j["dev_sets"] == ["b8_f1_a_t10", "b8_f1_d_t10"]
    assert j["dev_all"]["n_molecules"] == 4 and j["dev_all"]["n_spectra"] == 7
    assert j["dev_by_type"]["a"]["mean"] == 1.5 and j["dev_by_type"]["d"]["median"] == 2.0
    assert j["visible_test"]["n_molecules"] == 2 and j["visible_test_modes_per_molecule"] == {"2": 1, "1": 1}
    assert "train_per_inchikey14" not in j
