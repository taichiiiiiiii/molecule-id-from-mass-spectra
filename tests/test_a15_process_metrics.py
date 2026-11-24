import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("a15", ROOT / "scripts/research/a15_process_metrics.py")
a15 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a15)

T = a15.parse_ts


def test_union_overlap_and_clip():
    iv = [(T("2026-01-01T00:00:00Z"), T("2026-01-01T02:00:00Z")),
          (T("2026-01-01T01:00:00Z"), T("2026-01-01T03:00:00Z")),  # 重なり
          (T("2026-01-01T05:00:00Z"), T("2026-01-01T06:00:00Z"))]
    assert a15.union_hours(iv) == pytest.approx(4.0)
    # window で切る
    assert a15.union_hours(iv, T("2026-01-01T02:30:00Z"), T("2026-01-01T05:30:00Z")) == pytest.approx(1.0)


def test_union_rejects_reversed():
    with pytest.raises(ValueError):
        a15.union_hours([(T("2026-01-01T02:00:00Z"), T("2026-01-01T01:00:00Z"))])


def _toy():
    return {
        "window": {"start": "2026-01-01T00:00:00Z", "end": "2026-01-02T00:00:00Z"},
        "training_intervals": [
            {"name": "a", "own_design": True, "start": "2025-12-31T23:00:00Z", "end": "2026-01-01T06:00:00Z", "source": "x"},
            {"name": "b", "own_design": False, "start": "2026-01-01T05:00:00Z", "end": "2026-01-01T12:00:00Z", "source": "x"},
        ],
        "experiments": [
            {"issue": 1, "title": "t", "start": "2026-01-01T00:00:00Z", "conclusion": "2026-01-01T10:00:00Z",
             "conclusion_ref": "r", "compute": [{"start": "2026-01-01T01:00:00Z", "end": "2026-01-01T03:00:00Z", "source": "x"}]},
        ],
        "work_breakdown": {"_rule": "", "experiment": [1], "research": [2, 3], "repro_audit": [4]},
    }


def test_compute_toy():
    r = a15.compute(_toy())
    assert r["window"]["hours"] == pytest.approx(24.0)
    e = r["a_experiments"][0]
    assert e["total_h"] == pytest.approx(10.0) and e["compute_h"] == pytest.approx(2.0)
    assert e["procedure_h"] == pytest.approx(8.0)
    assert r["b_training"]["train_h"] == pytest.approx(12.0)  # 0〜12 時（前日の分は切る）
    assert r["b_training"]["own_design_h"] == pytest.approx(6.0)
    assert r["c_breakdown"]["n_issues"] == 4
    assert r["c_breakdown"]["experiment"]["share"] == pytest.approx(0.25)


def test_duplicate_category_rejected():
    d = _toy()
    d["work_breakdown"]["research"] = [1, 2]
    with pytest.raises(ValueError):
        a15.compute(d)


def test_real_inputs_parse():
    # 実際の入力が壊れていないこと（値そのものは文書と Issue に記録）
    d = json.loads((ROOT / "docs/research/schedule/a15_inputs.json").read_text())
    r = a15.compute(d)
    assert r["a_summary"]["n"] == 5
    assert r["c_breakdown"]["n_issues"] == 39
    assert 0 < r["b_training"]["share"] < 1
