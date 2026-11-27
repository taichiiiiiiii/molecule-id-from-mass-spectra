"""Issue #290（項目 6）: add_glacier_match.py の --which holdout の期待の数と parts の名前（実データは使わない）。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import add_glacier_match as A  # noqa: E402
import add_neighbour_ref as NR  # noqa: E402


def test_expect_dev_unchanged():
    e = A.expect_for("dev")
    assert e["sets"] is None and e["n_mol"] == 3901 and e["n_rows"] == 430979


def test_expect_holdout_from_dump_records(tmp_path):
    rows = dict(zip(NR.HOLDOUT_SETS, (10, 20, 30, 40)))
    for s, n in rows.items():
        (tmp_path / f"{s}.json").write_text(json.dumps({"holdout": s, "n_rows": n}))
    e = A.expect_for("holdout", tmp_path)
    assert tuple(e["sets"]) == NR.HOLDOUT_SETS
    assert e["n_mol"] == 1358 and e["n_rows"] == 100


def test_expect_holdout_missing_record_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        A.expect_for("holdout", tmp_path)


def test_parts_names():
    # dev は #259/#268 と同じ名前のまま（既存の parts を再利用できる）、holdout は別の置き場
    assert A.parts_name("dev", False) == "parts_gl_dev"
    assert A.parts_name("dev", True) == "parts_gl_dev_tier"
    assert A.parts_name("holdout", False) == "parts_gl_holdout"
    assert A.parts_name("holdout", True) == "parts_gl_holdout_tier"


def test_script_reads_no_label_column():
    # holdout は値を開かない: 正解のラベルの列名を読まない
    src = (ROOT / "scripts/features/add_glacier_match.py").read_text()
    for name in ("is_truth", "\"label\"", "truth.csv", "\"y\""):
        assert name not in src, name


def test_cli_has_which():
    out = subprocess.run([sys.executable, str(ROOT / "scripts/features/add_glacier_match.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "--which" in out.stdout
