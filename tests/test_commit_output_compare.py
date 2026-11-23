"""Issue #212: scripts/research/commit_output_compare.py のテスト（合成データのみ）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import commit_output_compare as C  # noqa: E402


def _write(p: Path, rows: list[tuple[str, str]]) -> Path:
    p.write_text("molecule_id,smiles\n" + "".join(f"{m},{s}\n" for m, s in rows))
    return p


def test_placeholder_matches_commit_filler():
    cands = C.PLACEHOLDER.split(";")
    assert len(cands) == 25
    assert cands[0] == "CO" and cands[1] == "CCO" and cands[-1] == "C" * 25 + "O"
    assert C.is_placeholder(cands)
    assert not C.is_placeholder(cands[:-1])


def test_identical_files(tmp_path):
    rows = [("m1", "A;B;C"), ("m2", C.PLACEHOLDER)]
    pa = _write(tmp_path / "a.csv", rows)
    pb = _write(tmp_path / "b.csv", rows)
    r = C.compare_files(pa, pb)
    assert r["sha256_equal"] and r["ids_equal"] and r["row_order_equal"]
    assert r["n_identical"] == 2 and r["n_differ"] == 0
    assert r["n_placeholder_a"] == 1 and r["n_real"] == 1 and r["n_real_identical"] == 1
    assert r["differ_top1_same_rate"] is None


def test_differences_counted(tmp_path):
    pa = _write(tmp_path / "a.csv", [("m1", "A;B;C"), ("m2", "X;Y"), ("m3", "P;Q"), ("m4", C.PLACEHOLDER)])
    pb = _write(tmp_path / "b.csv", [("m1", "A;C;B"), ("m2", "Y;X;Z"), ("m3", "P;Q"), ("m5", "K")])
    r = C.compare_files(pa, pb)
    assert not r["sha256_equal"]
    assert not r["ids_equal"] and r["only_a"] == ["m4"] and r["only_b"] == ["m5"]
    assert r["n_common"] == 3 and r["n_identical"] == 1 and r["n_differ"] == 2
    assert r["differ_ids"] == ["m1", "m2"]
    # m1: 1 位同じ・集合同じ、m2: 1 位違う・集合違う
    assert r["differ_top1_same"] == 1 and r["differ_top1_same_rate"] == 0.5
    assert r["differ_set_same"] == 1
    # 共通候補の順位差: m1 A0 B1 C1, m2 X1 Y1 -> [0,1,1,1,1]
    assert r["differ_common_rank_absdiff_mean"] == pytest.approx(0.8)
    assert r["differ_common_rank_absdiff_max"] == 1


def test_duplicate_id_raises(tmp_path):
    pa = _write(tmp_path / "a.csv", [("m1", "A"), ("m1", "B")])
    with pytest.raises(ValueError):
        C.read_submission(pa)


def test_strip_time_and_diff_paths():
    a = {"wall_s": 1.0, "x": {"seconds": 3, "n": 1}, "timings": {"1": 2}}
    b = {"wall_s": 9.0, "x": {"seconds": 7, "n": 1}}
    assert C.strip_time(a) == {"x": {"n": 1}}
    assert C.diff_paths(C.strip_time(a), C.strip_time(b)) == []
    assert C.diff_paths({"x": {"n": 1}, "y": 2}, {"x": {"n": 2}}) == ["x.n", "y"]


def _p5(path: Path, data: dict) -> Path:
    import json
    path.write_text(json.dumps({"wall_s": 1.0, "molecules_data": data}))
    return path


def test_compare_dirs(tmp_path):
    import json
    da, db = tmp_path / "a", tmp_path / "b"
    (da / "variants").mkdir(parents=True)
    (db / "variants").mkdir(parents=True)
    rows = [("m1", C.PLACEHOLDER)]
    for d in (da, db):
        _write(d / "submission.csv", rows)
        _write(d / "variants" / "submission_V2.csv", rows)
    (da / "x.log").write_text("1")
    (db / "x.log").write_text("2")
    (da / "only_a.json").write_text("{}")
    (da / "p5tail_audit.json").write_text(json.dumps({"total_s": 1, "selection": {"inserted": 49}}))
    (db / "p5tail_audit.json").write_text(json.dumps({"total_s": 2, "selection": {"inserted": 48}}))
    m_same = {"base": [["A", "K", 0, 0]], "gen": [["K", "A", 1]], "called": True}
    _p5(da / "p5tail_e1.json", {"m1": m_same, "m2": {"base": [["A", "K", 0, 0], ["B", "L", 1, 0]], "gen": []}})
    _p5(db / "p5tail_e1.json", {"m1": m_same, "m2": {"base": [["B", "L", 1, 0], ["A", "K", 0, 0]], "gen": []}})
    r = C.compare_dirs(da, db)
    files = {x["file"]: x for x in r["files"]}
    assert files["submission.csv"]["equal"] and files["variants/submission_V2.csv"]["equal"]
    assert not files["x.log"]["equal"] and files["only_a.json"]["sha256_b"] is None
    assert set(r["submissions"]) == {"submission.csv", "variants/submission_V2.csv"}
    assert r["json"]["p5tail_audit.json"] == {"equal_without_time": False, "diff_paths": ["selection.inserted"]}
    p = r["p5tail_e1"]
    assert p["ids_equal"] and p["n_identical"] == 1
    assert p["differ"]["m2"] == {"fields": ["base"], "base_order_equal": False, "base_set_equal": True, "gen_set_equal": True}


def test_main_writes_report(tmp_path):
    rows = [("m1", "A;B")]
    pa = _write(tmp_path / "a.csv", rows)
    pb = _write(tmp_path / "b.csv", rows)
    out = tmp_path / "r.json"
    assert C.main([str(pa), str(pb), "--out", str(out)]) == 0
    assert '"sha256_equal": true' in out.read_text()
