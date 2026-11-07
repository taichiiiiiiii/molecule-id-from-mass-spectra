"""Issue #34: 提出経路（scripts/submit/ranker_submission.py）の部品。"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ranker_submission", ROOT / "scripts/submit/ranker_submission.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["ranker_submission"] = rs
_spec.loader.exec_module(rs)
sys.path.insert(0, str(ROOT / "scripts/validate"))
from mrr25 import validate_submission  # noqa: E402


def test_write_submission_fallbacks(tmp_path):
    out = tmp_path / "s.csv"
    c = rs.write_submission(out, ["a", "b", "c"], {"a": ["C"] * 30}, {"b": ["N", "O"]})
    assert c == dict(ranker=1, fallback_score_prior=1, fallback_cco=1)
    rows = list(csv.reader(open(out)))
    got = validate_submission(rows, expected_ids={"a", "b", "c"})
    assert len(got["a"]) == 25 and got["b"] == ["N", "O"] and got["c"] == ["CCO"]


def test_prepare_submission_hides_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    test, sample = tmp_path / "t.parquet", tmp_path / "s.csv"
    pd.DataFrame(dict(molecule_id=["m1"])).to_parquet(test)
    pd.DataFrame(dict(molecule_id=["m1", "m2"], smiles="CCO")).to_csv(sample, index=False)
    rs.prepare("x", test, sample, None)
    rule = json.loads((tmp_path / "data/holdouts/x/hide_rule.json").read_text())
    assert rule["rule"] == "all" and rule["pool"] == "keep"
    assert (tmp_path / "data/holdouts/x/drop_keys.txt").read_text().split() == []
    truth = pd.read_csv(tmp_path / "outputs/holdouts/x/truth.csv")
    assert list(truth.molecule_id) == ["m1", "m2"]


def test_prepare_like_copies_holdout(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    src = tmp_path / "data/holdouts/h"; src.mkdir(parents=True)
    for f in ("test.parquet", "drop_keys.txt", "hide_rule.json", "sample_submission.csv"):
        (src / f).write_text(f)
    (tmp_path / "outputs/holdouts/h").mkdir(parents=True)
    (tmp_path / "outputs/holdouts/h/truth.csv").write_text("molecule_id,inchikey14,smiles\n")
    rs.prepare("h2", None, None, "h")
    assert (tmp_path / "data/holdouts/h2/hide_rule.json").read_text() == "hide_rule.json"
    assert (tmp_path / "outputs/holdouts/h2/truth.csv").is_file()


def test_fill_and_validate_fills_empty_and_missing(tmp_path):
    p = tmp_path / "fb.csv"
    p.write_text("molecule_id,smiles\na,\nb,N;O\n")
    c = rs.fill_and_validate(p, ["a", "b", "c"])
    got = validate_submission(list(csv.reader(open(p))), expected_ids={"a", "b", "c"})
    assert got["a"] == ["CCO"] and got["b"] == ["N", "O"] and got["c"] == ["CCO"] and c["fallback_cco"] == 2


def test_clear_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(rs, "ROOT", tmp_path)
    for d in ("outputs/local_eval/x", "outputs/holdout_eval/x_fold0", "outputs/ranker/features"):
        (tmp_path / d).mkdir(parents=True)
    (tmp_path / "outputs/local_eval/x/pubchem_fold0.json").write_text("{}")
    (tmp_path / "outputs/ranker/features/x.parquet").write_text("")
    rs.clear_caches("x")
    assert not (tmp_path / "outputs/local_eval/x").exists() and not (tmp_path / "outputs/ranker/features/x.parquet").exists()
