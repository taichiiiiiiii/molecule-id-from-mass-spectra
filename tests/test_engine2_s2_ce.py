"""#180: S2 の入力の CE の欠けを学習（train_fp3.py 65 行、欠け → 25.0）とそろえる修正と、非有限のロジットで止める assert。

torch の要らないテスト（pv_fp を偽のモジュールに差し替える）と、torch があれば本物の FPNet で
「CE が None でも全ビット有限、25.0 を渡したときとバイト一致」を確かめるテスト。
"""
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_engine2_build_tables as T  # noqa: E402

BT, E = T.BT, T.E


# ------------------------------------------------------------------------------------------------
# s2_ce: 非有限なら 25.0、有限ならそのまま（parse_ce と同じ値）
# ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("v", [None, [None], [np.nan], np.nan, float("nan"), [None, None], [10.0, None]])
def test_s2_ce_missing_is_training_fill(v):
    assert not np.isfinite(E.parse_ce(v))     # 前提: parse_ce は欠けで例外にならず nan を返す（#180 の原因）
    assert BT.s2_ce(v) == 25.0 == BT.S2_CE_MISSING


@pytest.mark.parametrize("v", [[], np.array([]), "x"])
def test_s2_ce_parse_ce_fallback_kept(v):
    """空・読めない値は parse_ce がもともと 25.0 を返す（変わらない）。"""
    assert E.parse_ce(v) == 25.0 and BT.s2_ce(v) == 25.0


@pytest.mark.parametrize("v,ref", [(35.0, 35.0), ([10.0, 30.0], 20.0), ([-20.0], 20.0), (0.0, 0.0), (np.float32(17.5), 17.5)])
def test_s2_ce_finite_unchanged(v, ref):
    assert BT.s2_ce(v) == E.parse_ce(v) == ref


# ------------------------------------------------------------------------------------------------
# s2_logits: 欠けの行に 25.0 を渡す、非有限のロジットで assert（偽の pv_fp）
# ------------------------------------------------------------------------------------------------
def _fake_pv_fp(monkeypatch, out):
    seen = {}

    def molecule_logits(models, specs, prec, adduct, instr, ce, mode):
        seen["ce"] = np.array(ce, copy=True)
        return out(ce) if callable(out) else out
    mod = types.ModuleType("pv_fp")
    mod.molecule_logits = molecule_logits
    monkeypatch.setitem(sys.modules, "pv_fp", mod)
    return seen


def _inputs(ce_values):
    P, L, add, I, rng = T._setup()
    rows = np.flatnonzero(L["row_p"] == 7)[:len(ce_values)]
    assert len(rows) == len(ce_values)
    n = len(L["prec"])
    ce = np.array([30.0] * n + [None], dtype=object)[:-1]
    for r, v in zip(rows, ce_values):
        ce[r] = v
    meta = dict(mode=np.where(L["pol"] == 1, "positive", "negative").astype(object), ce=ce,
                instr=np.array(["Orbitrap"] * n, dtype=object))
    return L, meta, rows, add[rows]


def test_s2_logits_fills_missing_ce_with_25(monkeypatch):
    seen = _fake_pv_fp(monkeypatch, lambda ce: np.asarray(ce, np.float64).mean() + np.zeros(6930))
    L, meta, rows, add = _inputs([None, 40.0, [np.nan]])
    z = BT.s2_logits(("S", [], "cpu"), L, meta, rows, add)
    assert seen["ce"].dtype == np.float32 and np.array_equal(seen["ce"], np.array([25.0, 40.0, 25.0], np.float32))
    assert z.dtype == np.float32 and np.isfinite(z).all() and np.allclose(z, 30.0)


def test_s2_logits_finite_ce_passed_as_parse_ce(monkeypatch):
    """CE がそろっている分子では、修正前（parse_ce をそのまま）と同じ値をモデルに渡す。"""
    seen = _fake_pv_fp(monkeypatch, np.ones(6930))
    vals = [[10.0, 30.0], 35.0, [-45.0]]
    L, meta, rows, add = _inputs(vals)
    BT.s2_logits(("S", [], "cpu"), L, meta, rows, add)
    assert np.array_equal(seen["ce"], np.array([E.parse_ce(v) for v in vals], np.float32))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_s2_logits_asserts_on_nonfinite_logits(monkeypatch, bad):
    z = np.zeros(6930); z[123] = bad
    _fake_pv_fp(monkeypatch, z)
    L, meta, rows, add = _inputs([35.0, 20.0])
    with pytest.raises(AssertionError, match="S2 のロジットに非有限"):
        BT.s2_logits(("S", [], "cpu"), L, meta, rows, add)


def test_s2_logits_none_passthrough(monkeypatch):
    _fake_pv_fp(monkeypatch, None)
    L, meta, rows, add = _inputs([None])
    assert BT.s2_logits(("S", [], "cpu"), L, meta, rows, add) is None


# ------------------------------------------------------------------------------------------------
# 表を書く直前: build（fp_source='s2'）と save_logits で非有限の Z を止める
# ------------------------------------------------------------------------------------------------
def test_build_s2_asserts_before_rows(monkeypatch):
    L, meta, I, queries, fold, calls = T._synthetic_build_inputs(monkeypatch)
    meta = dict(meta, instr=np.array(["Orbitrap"] * len(L["prec"]), dtype=object))
    n = [0]

    def fake_s2(models, L_, meta_, rows, add):
        n[0] += 1
        z = np.zeros(6930, np.float32)
        if n[0] == 3:
            z[:] = np.nan
        return z
    monkeypatch.setattr(BT, "s2_logits", fake_s2)
    with pytest.raises(AssertionError, match="S2 のロジットに非有限"):
        BT.build(E, L, meta, I, queries, fold, None, None, 1, log=lambda *a: None, fp_source="s2",
                 s2_models=("S2", [], "cpu"))


def test_save_logits_asserts_on_nonfinite(tmp_path):
    ok = [("gnps", "A" * 14, 1, np.zeros(8, np.float32)), ("riken", "B" * 14, 2, np.ones(8, np.float32))]
    p = BT.save_logits(str(tmp_path), ok)
    assert np.load(p)["Z"].shape == (2, 8)
    bad = ok + [("mona", "C" * 14, 3, np.full(8, np.nan, np.float32))]
    with pytest.raises(AssertionError, match="非有限"):
        BT.save_logits(str(tmp_path / "bad"), bad)
    assert not os.path.exists(tmp_path / "bad" / "own_fp_logits.npz")


# ------------------------------------------------------------------------------------------------
# 本物の FPNet（torch があるときだけ）: CE None → 25.0 を渡したときとバイト一致、全ビット有限
# ------------------------------------------------------------------------------------------------
def test_real_fpnet_missing_ce_equals_25(monkeypatch):
    torch = pytest.importorskip("torch")
    E_ = BT.base_fs.engine()  # noqa: F841
    import pv_fp
    torch.manual_seed(0)
    net = pv_fp.FPNet(6930, d=32, layers=1, heads=4, drop=0.0).eval()
    L, meta, rows, add = _inputs([None, 40.0, [None]])
    z = BT.s2_logits(([net], [], "cpu"), L, meta, rows, add)
    assert np.isfinite(z).all()
    meta25 = dict(meta, ce=meta["ce"].copy())
    for r in rows:
        if meta25["ce"][r] is None or isinstance(meta25["ce"][r], list):
            meta25["ce"][r] = 25.0
    assert np.array_equal(z, BT.s2_logits(([net], [], "cpu"), L, meta25, rows, add))
    # 修正前の入力（parse_ce の nan をそのまま）なら全ビット NaN になる（#165 の症状の再現）
    specs = BT.query_specs(L, rows)
    z_old = pv_fp.molecule_logits(([net], [], "cpu"), specs, L["prec"][rows].astype(np.float32), list(add.astype(str)),
                                  ["Orbitrap"] * len(rows), np.array([E.parse_ce(v) for v in meta["ce"][rows]], np.float32),
                                  np.where(L["pol"][rows] == 1, 1.0, -1.0))
    assert np.isnan(z_old).all()
