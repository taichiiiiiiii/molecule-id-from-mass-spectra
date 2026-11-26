import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("per_mol_rr", ROOT / "scripts/local_eval/per_mol_rr.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

A, B, C = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"


def test_values_and_mean_match_metric():
    truth = {"m1": A, "m2": B, "m3": C}
    preds = {"m1": [A], "m2": [C, B], "m3": []}
    rows = P.per_molecule_rr(truth, preds)
    assert rows == [("m1", 1.0), ("m2", 0.5), ("m3", 0.0)]


def test_missing_prediction_is_zero():
    assert P.per_molecule_rr({"m": A}, {}) == [("m", 0.0)]


def test_too_many_candidates_rejected():
    M = P._mrr25()
    with pytest.raises(M.SubmissionError):
        P.per_molecule_rr({"m": A}, {"m": [B] * 26})


def test_cli_writes_csv(tmp_path):
    t = tmp_path / "truth.csv"
    t.write_text(f"molecule_id,inchikey\nm1,{A}\nm2,{B}\n")
    s = tmp_path / "sub.csv"
    s.write_text(f"molecule_id,smiles\nm1,{C};{A}\nm2,{C}\n")
    o = tmp_path / "rr.csv"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/local_eval/per_mol_rr.py"), "--truth", str(t),
                        "--pred", str(s), "--out", str(o)], capture_output=True, text=True, check=True)
    assert '"mrr25": 0.25' in r.stdout
    with open(o) as f:
        assert list(csv.reader(f)) == [["molecule_id", "rr"], ["m1", "0.5"], ["m2", "0.0"]]


def test_empty_truth_rejected():
    with pytest.raises(ValueError):
        P.per_molecule_rr({}, {})


def test_cli_rejects_extra_ids(tmp_path):
    t = tmp_path / "truth.csv"
    t.write_text(f"molecule_id,inchikey\nm1,{A}\n")
    s = tmp_path / "sub.csv"
    s.write_text(f"molecule_id,smiles\nm1,{A}\nm9,{B}\n")
    r = subprocess.run([sys.executable, "-O", str(ROOT / "scripts/local_eval/per_mol_rr.py"), "--truth", str(t),
                        "--pred", str(s), "--out", str(tmp_path / "rr.csv")], capture_output=True, text=True)
    assert r.returncode != 0 and "truth に無い" in r.stderr
