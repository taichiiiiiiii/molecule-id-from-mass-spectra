"""Issue #36 案 A（PubChem 入場門）と差し替え試験 (ii) の合成データテスト。"""
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
_spec = importlib.util.spec_from_file_location("apply_eval", ROOT / "scripts/ranker/apply_eval.py")
AE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AE)


def _table():
    # m1: lib_max 0.95（門が閉じる）、m2: lib_max 0.5（門が開く）、m3: 参照なし（lib_max 0）
    return pd.DataFrame(dict(
        molecule_id=["m1"] * 3 + ["m2"] * 3 + ["m3"] * 2,
        key=["A", "B", "Q", "C", "D", "R", "E", "S"],
        smiles=["a", "b", "q", "c", "d", "r", "e", "s"],
        src_pubchem=[0, 0, 1, 0, 0, 1, 0, 1],
        lib_sim=[0.95, 0.3, np.nan, 0.5, np.nan, np.nan, np.nan, np.nan],
        lib_max=[0.95] * 3 + [0.5] * 3 + [0.0] * 2,
        label=[0, 1, 0, 0, 0, 1, 1, 0],
        s=[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0],
    ))


def test_gate_tau_matches_run_py_constant():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    m = re.search(r"^PC_LIB_TAU, REL_TH, PROMOTE_REL = ([0-9.]+),", src, re.M)
    assert m and float(m.group(1)) == AE.GATE_TAU == 0.9


def test_gate_removes_pubchem_only_where_closed():
    df = _table()
    g = AE.gate_pubchem(df)
    assert g.key.tolist() == ["A", "B", "C", "D", "R", "E", "S"]          # m1 の Q だけ消える
    import train_eval as TE
    before, after = TE.ranked_lists(df, "s"), TE.ranked_lists(g, "s")
    assert after["m1"] == ["b", "a"]                                      # 閉じた分子は残りをランカー順
    assert after["m2"] == before["m2"] and after["m3"] == before["m3"]    # 開いた分子は並びが変わらない


def test_gate_boundary_is_inclusive():
    df = _table()
    df.loc[df.molecule_id == "m2", "lib_max"] = 0.9                       # lib_max == 0.9 は閉じる（run.py と同じ >=）
    assert "R" not in AE.gate_pubchem(df).key.tolist()


def test_swap_truth_only_pool_truth_and_lib_max():
    df = _table()
    sw, n_sw, n_pc = AE.swap_truth(df)
    assert n_sw == 2 and n_pc == 1                                        # m1, m3 の正解はプール、m2 は PubChem
    assert sw.loc[sw.key == "B", "lib_sim"].item() == 0.95
    assert sw.loc[sw.key == "E", "lib_sim"].item() == 0.95
    assert (sw[sw.molecule_id == "m3"].lib_max == 0.95).all()             # 0 → 0.95
    assert (sw[sw.molecule_id == "m1"].lib_max == 0.95).all()             # max(0.95, 0.95)
    pd.testing.assert_frame_equal(sw[sw.molecule_id == "m2"], df[df.molecule_id == "m2"])   # PubChem 由来は対象外
    assert df.loc[df.key == "B", "lib_sim"].item() == 0.3 and np.isnan(df.loc[df.key == "E", "lib_sim"].item())  # 元の表は不変


def test_top1_rate():
    df = _table()
    assert AE.top1_rate(df, "s", 3) == round(1 / 3, 4)                   # m2 だけ正解が 1 位（R, s=3）
    assert AE.top1_rate(AE.gate_pubchem(df), "s", 3) == round(2 / 3, 4)   # m1 は Q が消えて B が 1 位
    assert AE.top1_rate(df, "s", 6) == round(1 / 6, 4)                    # 分母は全分子数
