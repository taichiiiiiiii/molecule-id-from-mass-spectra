"""#244: 候補群の相対化（式 (8)）と特徴表の書き出しの部品の単体テスト（合成データ）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import relativize as RZ  # noqa: E402


def test_relativize_five_values():
    v = np.array([1.0, 2.0, 4.0])
    r = RZ.relativize(v)
    mu, sd = 7 / 3, np.std(v)
    assert np.allclose(r["dmean"], v - mu)
    assert np.allclose(r["z"], (v - mu) / sd)
    assert np.allclose(r["dmax"], v - 4.0)
    e = np.exp(v)
    assert np.allclose(r["softmax"], e / e.sum())
    assert np.allclose(r["sd"], sd)
    assert set(r) == set(RZ.SUFFIXES)


def test_relativize_sigma_floor():
    v = np.array([0.50, 0.51, 0.52])   # σ ≈ 0.0082 < 0.02
    r = RZ.relativize(v)
    assert np.std(v) < RZ.SIGMA_FLOOR == 0.02
    assert np.allclose(r["z"], (v - v.mean()) / 0.02)


def test_relativize_single_candidate():
    r = RZ.relativize([3.7])
    assert r["dmean"].tolist() == [0.0] and r["z"].tolist() == [0.0] and r["dmax"].tolist() == [0.0]
    assert r["softmax"].tolist() == [1.0] and r["sd"].tolist() == [0.0]


def test_relativize_softmax_large_values_stable():
    r = RZ.relativize([1000.0, 999.0])
    assert np.isfinite(r["softmax"]).all()
    assert np.isclose(r["softmax"].sum(), 1.0)
    assert np.isclose(r["softmax"][0], 1 / (1 + np.exp(-1.0)))


def test_relativize_rejects_nonfinite():
    with pytest.raises(AssertionError):
        RZ.relativize([1.0, np.nan])


def test_add_relative_groups_independent_and_order_kept():
    df = pd.DataFrame(dict(g=["a", "b", "a", "b", "c"], v=[1.0, 10.0, 3.0, 30.0, 5.0]))
    out = RZ.add_relative(df, ["g"], ["v"])
    assert out.g.tolist() == df.g.tolist()
    assert np.allclose(out.v_dmean, [-1.0, -10.0, 1.0, 10.0, 0.0])
    assert np.allclose(out.v_dmax, [-2.0, -20.0, 0.0, 0.0, 0.0])
    assert out.v_softmax[4] == 1.0 and out.v_sd[4] == 0.0
    assert np.allclose(out.v_z[:4], [-1.0, -1.0, 1.0, 1.0])


def test_tanimoto_pred():
    pytest.importorskip("rdkit")
    import dump_features as DF
    z = np.array([0.0, 0.0, 100.0, -100.0])   # p = 0.5, 0.5, 1, 0
    f = np.array([1, 0, 1, 0], np.float32)
    # Σpf = 1.5、Σp = 2、Σf = 2 → 1.5 / 2.5
    assert np.isclose(DF.tanimoto_pred(f, z), 1.5 / 2.5)
    assert DF.tanimoto_pred(np.zeros(2), np.array([-1e3, -1e3])) == 0.0


def test_set_meta_dev_names():
    pytest.importorskip("rdkit")
    import dump_features as DF
    assert DF.set_meta("b8_f3_e_t10") == ("e", 3)
    assert DF.set_meta("b8_f1_a_t10") == ("a", 1)


def test_q_refs_hide_rules():
    pytest.importorskip("rdkit")
    import dump_features as DF
    import library as L
    rows = pd.DataFrame(dict(inchikey14=["K1", "K1", "K2"], ingest_lib=["enveda-180", "gnps", "gnps"],
                             ms2_mzs=[np.array([100.0, 150.0])] * 3, ms2_normalized_intensities=[np.array([1.0, 0.5])] * 3))
    orig = L._prep
    L._prep = lambda mz, it: (tuple(mz), tuple(it))   # 前処理は別のテストの範囲。ここでは行の数だけを見る
    try:
        all_ = DF.q_refs("x", rows, dict(rule="all"), {"K1"})
        env = DF.q_refs("x", rows, dict(rule="enveda_libs"), {"K1"})
    finally:
        L._prep = orig
    assert set(all_) == {"K2"}                          # (b)(c)(d): 隠す鍵の行は全部隠す
    assert len(env["K1"]) == 1 and len(env["K2"]) == 1  # (a)(e): enveda 系の行だけ隠す
    assert DF.q_refs("x", rows.iloc[:0], dict(rule="all"), set()) == {}


def test_excluded_columns_listed():
    pytest.importorskip("rdkit")
    import dump_features as DF
    for c in ("src", "train_sid", "np_coconut", "used_b8", "msg_overlap", "in_train", "label"):
        assert c in DF.EXCLUDED
    assert not set(DF.ID_COLS + DF.BASE) & set(DF.EXCLUDED)


def test_run_dump_requires_control_stages():
    pytest.importorskip("rdkit")
    import run as R
    import inspect
    sig = inspect.signature(R.run)
    assert sig.parameters["dump_features"].default is None and sig.parameters["dump_opts"].default is None
    with pytest.raises(AssertionError, match="dump-features"):
        R.run("x", ["score", "prior"], dump_features="/nonexistent/x.parquet")
