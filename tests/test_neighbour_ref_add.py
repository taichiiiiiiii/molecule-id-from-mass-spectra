"""scripts/features/add_neighbour_ref.py の単体テスト（Issue #261、群 N1）。合成データだけ。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/features", "scripts/local_eval", "scripts/ranker", "scripts/validate"):
    sys.path.insert(0, str(ROOT / p))

import add_neighbour_ref as AN  # noqa: E402
import lambdarank_oof as LR  # noqa: E402


def toy_table() -> pd.DataFrame:
    return pd.DataFrame(dict(set=["s1"] * 3 + ["s2"] * 2, molecule_id=["m1", "m1", "m2", "m3", "m3"],
                             cand_key=list("ABCDE"), fz=[0.1, 0.2, 0.3, 0.4, 0.5]))


def toy_parts() -> dict:
    p1 = pd.DataFrame(dict(set=["s1"] * 3, molecule_id=["m1", "m1", "m2"], cand_key=list("ABC"),
                           nbr_ms_mean=[0.5, np.nan, np.nan], nbr_tani_mean=[0.7, np.nan, 0.6],
                           nbr_n=np.array([10, 0, 10], np.int16)))      # m2 は問い合わせなし（ms だけ NaN）
    p2 = pd.DataFrame(dict(set=["s2"] * 2, molecule_id=["m3", "m3"], cand_key=list("DE"),
                           nbr_ms_mean=[0.1, 0.2], nbr_tani_mean=[0.3, 0.4], nbr_n=np.array([10, 4], np.int16)))
    return {"s1": p1, "s2": p2}


def test_add_columns_values_and_counts():
    df = toy_table()
    out, checks = AN.add_columns(df, toy_parts())
    pd.testing.assert_frame_equal(out[df.columns.tolist()], df)          # 元の列はそのまま
    np.testing.assert_array_equal(out.nbr_tani_mean.to_numpy(), [0.7, np.nan, 0.6, 0.3, 0.4])
    np.testing.assert_array_equal(out.nbr_ms_mean.to_numpy(), [0.5, np.nan, np.nan, 0.1, 0.2])
    assert "nbr_n" not in out.columns
    c1, c2 = checks
    assert c1["n_nbr0"] == 1 and c1["n_ms_nan_with_nbr"] == 1 and c1["n_nan"] == dict(nbr_ms_mean=2, nbr_tani_mean=1)
    assert c2["n_lt_k"] == 1 and c2["n_nbr0"] == 0


def test_add_columns_works_when_sets_are_interleaved():
    """集合の行が表の中で混ざっていても、集合の中の順で入る。"""
    df = toy_table().iloc[[0, 3, 1, 4, 2]].reset_index(drop=True)
    out, _ = AN.add_columns(df, toy_parts())
    assert out.set_index("cand_key").loc[list("ABCDE"), "nbr_tani_mean"].tolist()[0] == 0.7
    np.testing.assert_array_equal(out.set_index("cand_key").loc[list("ABCDE"), "nbr_tani_mean"].to_numpy(),
                                  [0.7, np.nan, 0.6, 0.3, 0.4])


def test_add_columns_stops_on_mismatch():
    df = toy_table()
    parts = toy_parts()
    bad = {k: v.copy() for k, v in parts.items()}
    bad["s1"].loc[0, "cand_key"] = "Z"                                   # 鍵の順が違う
    with pytest.raises(ValueError):
        AN.add_columns(df, bad)
    with pytest.raises(ValueError):
        AN.add_columns(df, {"s1": parts["s1"]})                          # 集合が足りない
    with pytest.raises(ValueError):
        AN.add_columns(df.assign(nbr_ms_mean=0.0), parts)                # 既に列がある
    bad = {k: v.copy() for k, v in parts.items()}
    bad["s1"].loc[1, "nbr_tani_mean"] = 0.5                              # 近傍 0 件なのに値
    with pytest.raises(ValueError):
        AN.add_columns(df, bad)
    bad = {k: v.copy() for k, v in parts.items()}
    bad["s2"].loc[0, "nbr_tani_mean"] = np.nan                           # 近傍があるのに NaN
    with pytest.raises(ValueError):
        AN.add_columns(df, bad)


def test_n1_registered_in_extra_allowed():
    iss, cols = LR.EXTRA_ALLOWED["n1"]
    assert iss == "#261" and tuple(cols) == AN.NEW_COLS == ("nbr_ms_mean", "nbr_tani_mean")
    feats = LR.resolve_features(("prior_unverified_pop",), ("cc", "cc_z") + AN.NEW_COLS)
    assert feats[-2:] == AN.NEW_COLS
    with pytest.raises(LR.HarnessError):
        LR.resolve_features((), ("nbr_n",))                              # nbr_n は登録しない（表にも入れない）


def test_dev_sets_are_the_20_b8_sets():
    assert len(AN.DEV_SETS) == 20 and AN.DEV_SETS[0] == "b8_f1_a_t10" and AN.DEV_SETS[-1] == "b8_f4_e_t10"
