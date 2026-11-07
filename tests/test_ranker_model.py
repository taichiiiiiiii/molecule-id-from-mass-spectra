import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
import ranker_features as FT  # noqa: E402
import train_eval as TE  # noqa: E402


def test_molecule_rows_features_do_not_use_label():
    pool = [(5.0, "P1", "CCO"), (3.0, "P2", "CCN")]
    pc = [("CCC", 6.0, "Q1")]
    sim = {"P2": 0.8}
    a = FT.molecule_rows("m", pool, pc, sim, {"P1": 1.0, "P2": 2.0}, {"CCO": 0.5}, truth_key="Q1")
    b = FT.molecule_rows("m", pool, pc, sim, {"P1": 1.0, "P2": 2.0}, {"CCO": 0.5}, truth_key="P1")
    fa = pd.DataFrame(a)[FT.FEATURES]; fb = pd.DataFrame(b)[FT.FEATURES]
    pd.testing.assert_frame_equal(fa, fb)               # 正解が変わっても特徴量は同じ
    assert [r["label"] for r in a] == [0, 0, 1] and [r["label"] for r in b] == [1, 0, 0]


def test_molecule_rows_values():
    pool = [(5.0, "P1", "CCO"), (3.0, "P2", "CCN")]
    pc = [("CCC", 6.0, "Q1"), ("CCCC", 4.0, "Q2")]
    rows = pd.DataFrame(FT.molecule_rows("m", pool, pc, {"P2": 0.8}, {"P1": 1.0, "P2": 2.0}, {}, None))
    assert rows.fz_rank_src.tolist() == [1, 2, 1, 2]
    assert rows.src_pubchem.tolist() == [0, 0, 1, 1]
    assert rows.delta_pc.unique().tolist() == [1.0]     # PubChem 最良 6 − プール最良 5
    assert rows.lib_max.unique().tolist() == [0.8]
    assert math.isnan(rows.lib_sim[0]) and rows.lib_sim[1] == 0.8 and rows.lib_sim[2:].isna().all()
    assert rows.prior[:2].tolist() == [1.0, 2.0] and rows.prior[2:].isna().all()
    assert rows.label.sum() == 0


def test_molecule_rows_no_pubchem():
    rows = pd.DataFrame(FT.molecule_rows("m", [(1.0, "P1", "C")], [], {}, {}, {}, "P1"))
    assert rows.delta_pc.isna().all() and rows.lib_max.tolist() == [0.0] and rows.label.tolist() == [1]


def test_ranked_lists_sorts_and_dedups_by_key():
    df = pd.DataFrame(dict(molecule_id=["m"] * 4, key=["A", "B", "A", "C"], smiles=["a", "b", "a2", "c"], s=[1.0, 3.0, 2.0, 3.0]))
    assert TE.ranked_lists(df, "s") == {"m": ["b", "c", "a2"]}   # 同点は元の並び、A は高い方の 1 件だけ


def test_bootstrap_diff_identical_is_zero():
    a = {f"m{i}": 1.0 / (i % 5 + 1) for i in range(50)}
    r = TE.bootstrap_diff(a, dict(a))
    assert r == dict(diff=0.0, lo=0.0, hi=0.0, better=0, worse=0)


def test_bootstrap_diff_sign():
    a = {f"m{i}": 1.0 for i in range(40)}
    b = {f"m{i}": (0.0 if i < 20 else 1.0) for i in range(40)}
    r = TE.bootstrap_diff(a, b)
    assert r["diff"] == 0.5 and r["lo"] > 0 and r["better"] == 20 and r["worse"] == 0


def test_params_fixed_as_in_issue():
    assert TE.PARAMS["objective"] == "lambdarank" and TE.N_ESTIMATORS == 500
    assert {k: TE.PARAMS[k] for k in ("num_leaves", "learning_rate", "min_data_in_leaf", "feature_fraction", "bagging_fraction",
                                      "bagging_freq", "seed", "lambdarank_truncation_level")} == \
        dict(num_leaves=31, learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
             seed=0, lambdarank_truncation_level=25)
    assert not set(TE.TRAIN) & set(TE.EVAL)


def test_adopt_rule_32():
    from train_eval import adopt
    ok = dict(lo=-0.01, hi=0.05, diff=0.02)
    win = dict(lo=0.01, hi=0.05, diff=0.03)
    bad = dict(lo=-0.03, hi=0.01, diff=-0.01)
    assert adopt({"a": {"ranker_minus_rule479": ok}, "b": {"ranker_minus_rule479": win}})["adopt"] is True
    assert adopt({"a": {"ranker_minus_rule479": ok}, "b": {"ranker_minus_rule479": ok}})["adopt"] is False   # 有意な改善なし
    r = adopt({"a": {"ranker_minus_rule479": bad}, "b": {"ranker_minus_rule479": win}})
    assert r["adopt"] is False and r["failing"] == ["a"]
    assert adopt({"a": {"ranker_minus_rule479": win}, "b": {}})["adopt"] is False                            # 比較できない評価がある
