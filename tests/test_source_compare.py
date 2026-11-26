"""scripts/features/add_source_compare.py の単体テスト（Issue #266、群 N2）。合成データだけ。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/features", "scripts/local_eval", "scripts/ranker", "scripts/validate"):
    sys.path.insert(0, str(ROOT / p))

import add_source_compare as SC  # noqa: E402
import lambdarank_oof as LR  # noqa: E402
import relativize as RZ  # noqa: E402
import run as R  # noqa: E402


def synth_pool(n: int = 400, seed: int = 0, dup_every: int = 5) -> R.Pool:
    rng = np.random.default_rng(seed)
    mass = np.sort(rng.uniform(299.99, 300.01, n))
    keys = [f"K{i // 2 if i % dup_every == 0 else i:05d}" for i in range(n)]   # 一部の鍵は重複（同じ鍵の別の行）
    meta = pd.DataFrame(dict(key=keys, smiles=[f"C{i}" for i in range(n)], mass=mass, pid=np.arange(n)))
    return R.Pool(meta.sample(frac=1.0, random_state=1).reset_index(drop=True), None)


def test_tier_window_count_matches_brute_force():
    rng = np.random.default_rng(3)
    mass = np.sort(rng.uniform(100, 900, 50_000))
    for t in rng.uniform(100, 900, 200).tolist() + [float(mass[17]), float(mass[-1]), 50.0, 1000.0]:
        tol = t * SC.PPM * 1e-6
        brute = int(((mass >= t - tol) & (mass <= t + tol)).sum())
        assert SC.tier_window_count(mass, t) == brute


def test_tier_window_bounds_same_as_pubchem_tier():
    """pubchem.Tier.window と同じ境界（窓の端の値ちょうどを含む）。"""
    t = 300.0
    tol = t * SC.PPM * 1e-6
    mass = np.array([t - tol - 1e-9, t - tol, t, t + tol, t + tol + 1e-9])
    assert SC.tier_window_count(mass, t) == 3


def test_pool_window_counts_visible_distinct_keys_and_hides_truth():
    pool = synth_pool()
    t = 300.0
    idx = pool.window(t)
    allk = np.unique(pool.key[idx])
    hidden = {str(allk[0]), str(allk[3]), "NOT_IN_POOL"}   # 型 c・e: drop_keys（scorer の鍵）
    vk = SC.pool_window_keys(pool, t, hidden)
    assert len(vk) == len(allk) - 2
    assert not (set(vk.tolist()) & hidden)                  # 隠した正解は数えない
    assert set(vk.tolist()) == set(pool.key[R.visible_window(pool, t, hidden)].tolist())
    assert len(SC.pool_window_keys(pool, t, set())) == len(allk)   # 隠さない集合は窓の鍵そのもの


def test_pool_window_lt60_equals_visible_pool_rows():
    """窓の重複を除いた鍵が 60 未満なら、run.py の規則（visible_window → score → dedup TOPN）で作る行の数と一致。"""
    for n in (20, 50, 70, 200):
        pool = synth_pool(n=n, seed=n)
        t = 300.0
        hidden = {str(pool.key[pool.window(t)][0])}
        idx = R.visible_window(pool, t, hidden)
        rows = sorted(((float(np.sin(i)), int(i)) for i in idx), key=lambda r: -r[0])   # 点の代わり
        lst = R.dedup(rows, pool, R.TOPN)
        nv = len(SC.pool_window_keys(pool, t, hidden))
        if nv < R.TOPN:
            assert len(lst) == nv
        else:
            assert len(lst) == R.TOPN


def toy_table() -> pd.DataFrame:
    # 分子 m1: プール 3・層 2、m2: プールだけ 2、m3: 層だけ 2（別の集合）
    return pd.DataFrame(dict(
        set=["s1"] * 7 + ["s2"] * 2,
        molecule_id=["m1"] * 5 + ["m2"] * 2 + ["m3"] * 2,
        is_tier=np.array([0, 1, 0, 1, 0, 0, 0, 1, 1], np.int8),
        fz=[3.0, 5.0, 1.0, 2.0, 2.0, 4.0, 1.0, 0.5, 0.7],
        tani_pred=[0.3, 0.5, 0.1, 0.2, 0.2, 0.4, 0.1, 0.05, 0.07],
        cand_key=list("ABCDEFGHI")))


def toy_counts() -> pd.DataFrame:
    return pd.DataFrame(dict(set=["s1", "s1", "s2"], molecule_id=["m1", "m2", "m3"],
                             n_tier_window=[100, 5, 0], n_pool_window=[3, 2, 0]))


def test_source_gaps_and_nan_when_one_side_missing():
    df = toy_table()
    g = SC.source_gaps(df)
    # m1: プールの最大 3、層の最大 5
    assert g["src_gap_fz"][:5] == pytest.approx([-2.0] * 5)
    assert g["opp_gap_fz"][:5] == pytest.approx([3 - 5, 5 - 3, 1 - 5, 2 - 3, 2 - 5])
    assert g["src_gap_tani"][0] == pytest.approx(0.3 - 0.5)
    # m2（層なし）・m3（プールなし）: 差は NaN
    for c in SC.GAP_COLS:
        assert np.isnan(g[c][5:]).all(), c


def test_source_relative_groups_split_by_is_tier():
    df = toy_table()
    r = SC.source_relative(df)
    m1_pool = np.array([3.0, 1.0, 2.0])
    exp = RZ.relativize(m1_pool)
    assert r["fz_src_dmax"][[0, 2, 4]] == pytest.approx(exp["dmax"])
    assert r["fz_src_z"][[0, 2, 4]] == pytest.approx(exp["z"])
    assert r["fz_src_softmax"][[0, 2, 4]].sum() == pytest.approx(1.0)
    assert r["fz_src_softmax"][[1, 3]].sum() == pytest.approx(1.0)       # 層の 2 行は別の群
    assert r["fz_src_dmax"][1] == 0.0 and r["fz_src_dmax"][3] == pytest.approx(-3.0)
    assert r["tani_pred_src_softmax"][[7, 8]].sum() == pytest.approx(1.0)
    # 群は set ごとにも分かれる（同じ molecule_id が別の集合にあっても混ぜない）
    df2 = df.copy(); df2.loc[7:, "molecule_id"] = "m1"
    r2 = SC.source_relative(df2)
    assert r2["fz_src_softmax"][[7, 8]].sum() == pytest.approx(1.0)
    assert r2["fz_src_softmax"][[1, 3]].sum() == pytest.approx(1.0)


def test_add_columns_keeps_rows_order_and_values():
    df = toy_table()
    out = SC.add_columns(df, toy_counts())
    assert len(out) == len(df)
    assert list(out.columns) == list(df.columns) + list(SC.NEW_COLS)
    for c in df.columns:
        assert out[c].equals(df[c])
    assert out["log_n_tier_window"].tolist() == pytest.approx(np.log1p([100] * 5 + [5] * 2 + [0] * 2).tolist())
    assert out["log_n_pool_window"].tolist() == pytest.approx(np.log1p([3] * 5 + [2] * 2 + [0] * 2).tolist())
    assert not np.isinf(out[list(SC.NEW_COLS)].to_numpy(np.float64)).any()
    # 並べ替えた表でも、行ごとの値は同じ（行の順に依らない）
    perm = np.random.default_rng(0).permutation(len(df))
    out2 = SC.add_columns(df.iloc[perm].reset_index(drop=True), toy_counts())
    pd.testing.assert_frame_equal(out2, out.iloc[perm].reset_index(drop=True))


def test_add_columns_stops_on_missing_counts_or_existing_column():
    df = toy_table()
    with pytest.raises(ValueError):
        SC.add_columns(df, toy_counts().iloc[:2])
    with pytest.raises(ValueError):
        SC.add_columns(df.assign(src_gap_fz=0.0), toy_counts())


def test_n2_registered_in_extra_allowed():
    iss, cols = LR.EXTRA_ALLOWED["n2"]
    assert iss == "#266" and tuple(cols) == SC.NEW_COLS and len(cols) == 16
    feats = LR.resolve_features(("prior_unverified_pop",), ("cc", "cc_z") + SC.NEW_COLS)
    assert feats[-16:] == SC.NEW_COLS


def test_unregistered_extra_name_still_stops():
    with pytest.raises(LR.HarnessError):
        LR.resolve_features((), ("src_gap_fz", "src_gap_mf"))      # 登録外の名前
    with pytest.raises(LR.HarnessError):
        LR.check_columns(list(LR.ID_COLS) + list(LR.FEATURES) + ["n_tier_raw"], ())   # 表に知らない列
