"""Issue #178: scripts/research/ddom_metfrag_ci.py の合成データのテスト。実データは読まない。"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

import ddom_metfrag_ci as M  # noqa: E402
import dseen as D  # noqa: E402


def _rows(spec: dict[str, list[float]]):
    return [dict(holdout=h, molecule_id=f"m_{i:05d}", d_dom_metfrag=repr(float(v))) for h, vs in spec.items()
            for i, v in enumerate(vs)]


def test_band_thresholds_match_rep_dseen():
    assert M.band(0.01, 0.5) == "higher_train"
    assert M.band(-0.5, -0.01) == "lower_train"
    assert M.band(-0.1, 0.1) == "none"
    assert M.band(0.0, 0.3) == "none"      # lo = 0 は 0 を含む
    assert M.band(-0.3, 0.0) == "none"


def test_per_set_values_rejects_duplicates():
    rows = _rows({"a": [1.0, 2.0]}) + [dict(holdout="a", molecule_id="m_00000", d_dom_metfrag="3.0")]
    with pytest.raises(AssertionError):
        M.per_set_values(rows)


def test_check_means_uses_4dp_rounding_and_n():
    vals = M.per_set_values(_rows({"a": [0.1, 0.2, 0.30004], "b": [1.0, 2.0]}))
    ref = {"a": {"d_dom": {"metfrag": {"mean": 0.2, "n": 3}}},
           "b": {"d_dom": {"metfrag": {"mean": 1.5, "n": 3}}}}   # b は n が違う
    assert M.check_means(vals, ref, ["a", "b"]) == ["b"]
    ref["a"]["d_dom"]["metfrag"]["mean"] = 0.2001
    assert M.check_means(vals, ref, ["a"]) == ["a"]
    ref["c"] = {"d_dom": {"metfrag": {"mean": 0.0, "n": 1}}}   # 表に無い集合は食い違いとして返す
    assert M.check_means(vals, ref, ["c"]) == ["c"]


def test_summarize_equals_dseen_bootstrap_ci_order_independent():
    rng = np.random.default_rng(1)
    v = rng.normal(0.2, 1.0, 50)
    rows = _rows({"a": list(v)})
    s1 = M.summarize(M.per_set_values(rows), ["a"])["a"]
    s2 = M.summarize(M.per_set_values(rows[::-1]), ["a"])["a"]   # 行の順に依らない（molecule_id の昇順）
    b = D.bootstrap_ci({f"m_{i:05d}": float(x) for i, x in enumerate(v)})
    assert s1 == s2
    assert (s1["lo_raw"], s1["hi_raw"], s1["mean_raw"]) == (b["lo"], b["hi"], b["mean"])
    assert s1["n"] == 50 and s1["positive"] + s1["negative"] == 50


def test_main_exits_2_on_mean_mismatch(tmp_path):
    spec = {h: [0.1, 0.3] for h in M.SETS + M.REFERENCE_SETS}
    p = tmp_path / "per.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["holdout", "molecule_id", "d_dom_metfrag"])
        w.writeheader()
        w.writerows(_rows(spec))
    ref = {h: {"d_dom": {"metfrag": {"mean": 0.2, "n": 2}}} for h in spec}
    rj = tmp_path / "ref.json"
    rj.write_text(json.dumps(ref))
    out = tmp_path / "out.json"
    cmd = [sys.executable, str(ROOT / "scripts/research/ddom_metfrag_ci.py"), "--csv", str(p), "--ref-json", str(rj),
           "--out", str(out)]
    assert subprocess.run(cmd, capture_output=True).returncode == 0
    got = json.loads(out.read_text())
    assert got["means_match_ref_json"] is True and got["v3_c1"]["mean"] == 0.2
    ref["v3_c2"]["d_dom"]["metfrag"]["mean"] = 0.2001
    rj.write_text(json.dumps(ref))
    out.unlink()
    r = subprocess.run(cmd, capture_output=True)
    assert r.returncode == M.MISMATCH_EXIT and not out.exists()
