"""Issue #208: scripts/cv/adoption_power.py のテスト（作り物の sd・n・δ だけ）。"""
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))

import adoption_power as A  # noqa: E402


def test_phi_known_values():
    assert A.phi(0.0) == pytest.approx(0.5)
    assert A.phi(-1.96) == pytest.approx(0.0249979, abs=1e-6)
    assert A.phi(1.96) == pytest.approx(0.9750021, abs=1e-6)


def test_type_prob_known_values():
    # se = 0.1/√100 = 0.01。primary: Φ(δ/se − 1.96)
    assert A.type_prob(0.0, 0.1, 100, True) == pytest.approx(A.phi(-1.96))       # 第 1 種の過誤 2.5%
    assert A.type_prob(0.0196, 0.1, 100, True) == pytest.approx(0.5)             # δ = 1.96·se で 50%
    assert A.type_prob(0.028, 0.1, 100, True) == pytest.approx(A.phi(0.84))      # δ = 2.8·se で約 80%
    # 非 primary: Φ((δ + 0.02)/se − 1.96)。δ=0 で Φ(2 − 1.96)
    assert A.type_prob(0.0, 0.1, 100, False) == pytest.approx(A.phi(0.04))
    # sd = 0 は L = δ で決まる
    assert A.type_prob(0.0, 0.0, 100, False) == 1.0
    assert A.type_prob(0.0, 0.0, 100, True) == 0.0


def test_adopt_prob_is_product_and_type1_rate():
    sds = {"C1": 0.1, "enveda": 0.2, "C2": 0.0}
    ns = {"C1": 100, "enveda": 400, "C2": 50}
    d0 = {t: 0.0 for t in sds}
    p, per = A.adopt_prob(d0, sds, ns)
    # δ = 0: C1 は 2.5%、enveda は se = 0.01 → Φ(0.04)、C2 は sd 0 で必ず通る
    assert per["C1"] == pytest.approx(A.phi(-1.96))
    assert per["enveda"] == pytest.approx(A.phi(0.04))
    assert per["C2"] == 1.0
    assert p == pytest.approx(A.phi(-1.96) * A.phi(0.04))
    assert p <= 0.025  # 第 1 種の過誤の率は C1 単独の 2.5% を超えない


def test_mc_matches_analytic():
    sds = {"C1": 0.15, "enveda": 0.18, "v3_c2n": 0.25}
    ns = {"C1": 394, "enveda": 373, "v3_c2n": 394}
    for d in (0.0, 0.02, 0.05):
        dl = {"C1": d, "enveda": 0.0, "v3_c2n": 0.0}
        p, _ = A.adopt_prob(dl, sds, ns)
        pmc = A.adopt_prob_mc(dl, sds, ns, n_sim=100_000, seed=0)
        assert pmc == pytest.approx(p, abs=4 * math.sqrt(max(p * (1 - p), 1e-6) / 100_000) + 1e-4)


def test_sd_by_type_filters_unchanged_and_stale():
    mde = {
        "v3_c1": {"variants": {"a": {"sd_delta": 0.1, "n_changed": 5}, "b": {"sd_delta": 0.3, "n_changed": 3},
                               "z": {"sd_delta": 0.0, "n_changed": 0}, "c": {"sd_delta": 0.2, "n_changed": 4}},
                  "stale_excluded": []},
        "v3_c2": {"variants": {"x": {"sd_delta": 0.0, "n_changed": 0}}, "stale_excluded": ["y"]},
        "enveda_v1": {"variants": {"a": {"sd_delta": 0.5, "n_changed": 1}, "s": {"sd_delta": 9.0, "n_changed": 9}},
                      "stale_excluded": ["s"]},
    }
    r = A.sd_by_type(mde)
    assert r["C1"]["median"] == 0.2 and r["C1"]["min"] == 0.1 and r["C1"]["max"] == 0.3
    assert r["C1"]["n_variants"] == 3
    assert r["C2"] is None            # 値が無い型は未計算
    assert r["v3_c2n"] is None        # 集合が無い
    assert r["enveda"]["values"] == [0.5]


def test_build_scenarios_and_c2_zero():
    mde = {
        "v3_c1": {"variants": {"a": {"sd_delta": 0.15, "n_changed": 5, "diff": 99.0}}, "stale_excluded": []},
        "enveda_v1": {"variants": {"a": {"sd_delta": 0.18, "n_changed": 5}}, "stale_excluded": []},
        "v3_c2n": {"variants": {"a": {"sd_delta": 0.25, "n_changed": 5}}, "stale_excluded": []},
        "v3_c2": {"variants": {"a": {"sd_delta": 0.0, "n_changed": 0}}, "stale_excluded": []},
    }
    res = A.build(mde, n_sim=2000)
    assert len(res["table"]["median"]["rows"]) == len(A.GUARDS) * 2 * len(A.DELTAS)
    rows = {(r["scenario"], r["delta"]): r for r in res["table"]["median"]["rows"] if r["rule"] == "lower"}
    assert len(rows) == 2 * len(A.DELTAS)
    assert res["types"]["C2"]["status"] == "未計算"
    assert "C2" not in rows[("S1", 0.02)]["per_type"]
    # S2 は enveda・v3_c2n にも効果があるので S1 以上
    for d in A.DELTAS:
        assert rows[("S2", d)]["p_adopt"] >= rows[("S1", d)]["p_adopt"] - 1e-12
    # 検出力は δ について単調
    ps = [rows[("S1", d)]["p_adopt"] for d in A.DELTAS]
    assert ps == sorted(ps)
    # diff は計算に入らない（入れても結果が同じ）
    mde["v3_c1"]["variants"]["a"]["diff"] = -99.0
    again = [r for r in A.build(mde, n_sim=10)["table"]["median"]["rows"]
             if r["rule"] == "lower" and r["scenario"] == "S1" and r["delta"] == 0.02]
    assert again[0]["p_adopt"] == rows[("S1", 0.02)]["p_adopt"]


def test_point_guard_known_values_and_mc():
    # 提案 R1: C1 以外は平均の差 ≥ −0.02 → Φ((δ + 0.02)/se)。C1 は変えない
    assert A.type_prob(0.0, 0.1, 100, False, "point") == pytest.approx(A.phi(2.0))
    assert A.type_prob(-0.02, 0.1, 100, False, "point") == pytest.approx(0.5)
    assert A.type_prob(0.0, 0.1, 100, True, "point") == pytest.approx(A.phi(-1.96))
    sds = {"C1": 0.15, "enveda": 0.18}
    ns = {"C1": 394, "enveda": 373}
    dl = {"C1": 0.02, "enveda": 0.0}
    p, _ = A.adopt_prob(dl, sds, ns, "point")
    assert A.adopt_prob_mc(dl, sds, ns, n_sim=100_000, seed=0, guard="point") == pytest.approx(p, abs=0.006)
    # 第 1 種の過誤（全型 δ=0）は C1 の 2.5% を超えない
    p0, _ = A.adopt_prob({"C1": 0.0, "enveda": 0.0}, sds, ns, "point")
    assert p0 <= 0.025


def test_c2_type_lower_bound_never_below_floor():
    # 対照 rr が全分子 0 の集合（C2 型、n=197）: Δ ∈ [0,1] のどんな並びでも正規近似の下限は −0.02 以上
    n = 197
    worst = 0.0
    for k in range(0, n + 1):            # k 分子が Δ=1、残り 0（分散が最大になる形）
        d = [1.0] * k + [0.0] * (n - k)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        worst = min(worst, m - A.Z * sd / math.sqrt(n))
    assert worst > A.FLOOR
    assert worst == pytest.approx(-0.0049, abs=0.001)
