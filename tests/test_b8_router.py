"""B8 振り分け器（#216）の単体テスト（合成データ）。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))

import b8_fit_eval as FE  # noqa: E402
import b8_build_sets as BS  # noqa: E402
import b8_features as BF  # noqa: E402


def test_sign_constraints_hold():
    rng = np.random.default_rng(1)
    n = 400
    X = rng.normal(size=(n, 3))
    # 真の向き: 列 0 は −（なのに + に制約）、列 1 は +（なのに − に制約）、列 2 は自由
    s = -2 * X[:, 0] + 2 * X[:, 1] + 1.5 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-s))).astype(int)
    m = FE.fit_lr(X, y, [+1, -1, 0])
    assert m["w"][0] >= 0 and m["w"][1] <= 0
    assert m["w"][0] == pytest.approx(0, abs=1e-8) and m["w"][1] == pytest.approx(0, abs=1e-8)
    assert m["w"][2] > 0.5
    m2 = FE.fit_lr(X, y, [-1, +1, 0])
    assert m2["w"][0] < -0.5 and m2["w"][1] > 0.5
    p = FE.predict(m2, X)
    assert ((p > 0) & (p < 1)).all()


def test_compose_ties_l_first_dedup_25():
    L = [[f"k{i}", f"L{i}"] for i in range(1, 31)]
    Q = [[f"q{i}", f"Q{i}"] for i in range(1, 31)]
    # p = 0.5: L の r 番目と Q の r 番目は同点 → L が先
    out = FE.compose(L, Q, 0.5)
    assert out[:4] == ["L1", "Q1", "L2", "Q2"] and len(out) == 25
    # 重複キー: Q の 1 位が L の 1 位と同じキー → 飛ばす
    Q2 = [["k1", "dup"]] + Q
    out = FE.compose(L, Q2, 0.5)
    assert "dup" not in out and out[:3] == ["L1", "L2", "Q1"]
    assert len(set(out)) == len(out) == 25
    # p = 1 なら L だけ（Q の重みは 0、L の最後より下）
    assert FE.compose(L, Q, 1.0) == [s for _, s in L[:25]]
    assert FE.compose(L, Q, 0.0) == [s for _, s in Q[:25]]
    assert FE.compose([], [], 0.3) == []


def test_decide_refuses_missing_type():
    ok = dict(n=800, mean=0.01, lo=0.001, hi=0.02)
    per = {t: dict(ok) for t in FE.ALL_TYPES}
    assert FE.decide(per, "r1_and_s6")["adopt"] is True
    per2 = {t: dict(ok) for t in ("a", "b", "c", "d")}
    d = FE.decide(per2, "r1_and_s6")
    assert d["adopt"] is False and d["refused"] and d["missing"] == ["e"]
    per3 = dict(per); per3["c"] = dict(n=0, mean=0, lo=0, hi=0)
    assert FE.decide(per3, "r1_and_s6")["refused"]
    # (a) の下限 ≤ 0 → 不採用、(b) の平均 < −0.02 → 不採用、下限 < −0.02 → 不採用
    per4 = dict(per); per4["a"] = dict(n=800, mean=0.01, lo=0.0, hi=0.02)
    assert FE.decide(per4, "r1_and_s6")["adopt"] is False
    per5 = dict(per); per5["b"] = dict(n=800, mean=-0.021, lo=-0.03, hi=0)
    assert FE.decide(per5, "r1_and_s6")["adopt"] is False
    per6 = dict(per); per6["e"] = dict(n=800, mean=0.0, lo=-0.0201, hi=0.02)
    assert FE.decide(per6, "r1_and_s6")["adopt"] is False
    with pytest.raises(AssertionError):
        FE.decide(per, "s6_only")


def test_impute_rules():
    feats = ("conf_Q", "d", "gap_L")
    tr = np.array([[0.5, 1.0, 2.0], [np.nan, np.nan, math.inf], [0.2, 3.0, -1.0], [0.1, -math.inf, -math.inf]])
    st = FE.impute_stats(tr, feats)
    X = FE.impute(tr, st, feats)
    assert X[1, 0] == 0.0          # |Q| = 0 → 0
    assert X[1, 1] == 3.0          # d の欠け → 学習側の d の最大
    assert X[1, 2] == 2.0 and X[3, 2] == -1.0   # ±inf → 有限の最大・最小
    assert X[3, 1] == 1.0          # d の −inf → 有限の最小
    assert np.isfinite(X).all()


def test_boot_ci_deterministic():
    d = np.random.default_rng(3).normal(0.01, 0.1, 800)
    assert FE.boot_ci(d) == FE.boot_ci(d)
    lo, hi = FE.boot_ci(d)
    assert lo < d.mean() < hi


def test_draw_same_seed_same_set():
    c = [f"K{i:04d}" for i in range(500)]
    a = BS.draw(c, 200, BS.seed_of(1, "a"))
    assert a == BS.draw(list(reversed(c)), 200, BS.seed_of(1, "a"))   # 入力の順に依らない（sorted）
    assert a == sorted(a) and len(set(a)) == 200
    assert a != BS.draw(c, 200, BS.seed_of(1, "e"))
    assert BS.draw(c[:50], 200, 7) == sorted(c[:50])                   # 足りなければ全数
    assert BS.seed_of(3, "e") == 20261124 + 304 and BS.ORDER == ("a", "e", "b", "c", "d")


def test_candidates_exclude_pending_and_hit():
    class Ex:
        grp_of = {"A": 1, "B": 2, "C": 3}

        def hit(self, keys):
            return {"C"} & set(keys)
    got = BS.candidates("a", ["A", "B", "C"], [], [], Ex(), pending_groups={2})
    assert got == ["A"]


def test_pubchem_cache_sha_guard(tmp_path):
    lg = tmp_path / "pc_logits.npz"
    lg.write_bytes(b"one")
    sha = BF.check_cache_sha(tmp_path, lg)
    assert (tmp_path / BF.SHA_NAME).read_text().strip() == sha
    assert BF.check_cache_sha(tmp_path, lg) == sha           # 同じなら通る
    lg.write_bytes(b"two")
    with pytest.raises(SystemExit):
        BF.check_cache_sha(tmp_path, lg)                      # 食い違い → 止まる
    d2 = tmp_path / "x"; d2.mkdir()
    (d2 / "pubchem_fold0.json").write_text("{}")
    with pytest.raises(SystemExit):
        BF.check_cache_sha(d2, lg)                            # 印なしのキャッシュ → 止まる


def test_feature_signs_match_prereg():
    assert BF.FEATURES == ("conf_L", "gap_L", "conf_Q", "gap_Q", "d", "lib", "log_nwin_pool", "log_nwin_pc")
    assert BF.SIGNS == dict(conf_L=1, gap_L=1, conf_Q=-1, gap_Q=-1, d=1, lib=1, log_nwin_pool=0, log_nwin_pc=0)
    assert FE.TRAIN_TYPES == ("a", "b", "c", "d") and FE.C_REG == 1.0 and FE.N_BOOT == 2000
