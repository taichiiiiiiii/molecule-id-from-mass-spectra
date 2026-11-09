"""Issue #52: extlib_coverage.coverage の数え方（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("extlib_coverage", ROOT / "scripts/cv/extlib_coverage.py")
ec = importlib.util.module_from_spec(_spec)
sys.modules["extlib_coverage"] = ec
_spec.loader.exec_module(ec)


def test_coverage_counts():
    ext = {"A", "B", "C", "D"}
    instr = pd.Series(["LC-ESI-Orbitrap", "timsTOF Pro", "LC-ESI-QTOF", "LC-ESI-Orbitrap"])
    r = ec.coverage(ext, instr, train_keys={"A", "B", "X"}, hold={"A", "B"}, tims_keys={"B", "X"}, public_q_keys={"A"})
    assert (r["n_struct"], r["in_train"], r["not_in_train"]) == (4, 2, 2)
    assert r["not_in_train_tims_query"] == 0
    assert r["tims_spectra"] == 1 and r["n_spectra"] == 4
    assert (r["hold"], r["hold_tims"], r["hold_public_query"]) == (2, 1, 1)
    assert r["instrument"]["LC-ESI-Orbitrap"] == 2
