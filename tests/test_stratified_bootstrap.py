"""stratified_bootstrap の単体テスト（合成の分子ごとの逆数順位）。"""
import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts", "ranker"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "validate"))
spec = importlib.util.spec_from_file_location("stratified_bootstrap", os.path.join(ROOT, "scripts", "cv", "stratified_bootstrap.py"))
SB = importlib.util.module_from_spec(spec); spec.loader.exec_module(SB)
from train_eval import bootstrap_diff  # noqa: E402


def _sets(sizes, shift, seed=1):
    rng = np.random.default_rng(seed)
    a, b = {}, {}
    for s, n in sizes.items():
        rr = rng.choice([0.0, 0.2, 0.5, 1.0], size=n)
        a[s] = {f"{s}_{i:03d}": float(min(1.0, r + shift[s] * (i % 2))) for i, r in enumerate(rr)}
        b[s] = {f"{s}_{i:03d}": float(r) for i, r in enumerate(rr)}
    return a, b


def test_single_set_equals_bootstrap_diff():
    a, b = _sets({"v3_c1": 57}, {"v3_c1": 0.3})
    r = SB.stratified_bootstrap_diff(a, b)
    d = bootstrap_diff(a["v3_c1"], b["v3_c1"])
    assert (r["diff"], r["lo"], r["hi"], r["better"], r["worse"]) == (d["diff"], d["lo"], d["hi"], d["better"], d["worse"])
    assert r["per_set"]["v3_c1"]["lo"] == d["lo"] and r["n_boot"] == 2000 and r["seed"] == 0


def test_equal_weight_not_pooled():
    # 小さい集合だけが大きく動く: 等重みの E は分子をまとめた平均より大きい
    a, b = _sets({"v3_c1": 200, "v3_c2n": 20}, {"v3_c1": 0.0, "v3_c2n": 1.0})
    r = SB.stratified_bootstrap_diff(a, b)
    per = [np.mean([a[s][m] - b[s][m] for m in a[s]]) for s in ("v3_c1", "v3_c2n")]
    pooled = np.mean([a[s][m] - b[s][m] for s in a for m in a[s]])
    assert r["diff"] == round(float(np.mean(per)), 4)
    assert r["diff"] > pooled + 0.05
    assert r["lo"] <= r["diff"] <= r["hi"]


def test_resampling_within_each_set_keeps_zero_set_at_zero():
    # 差が全部 0 の集合が混ざっても、その集合の寄与は毎回 0（集合をまたいで引かない）
    a, b = _sets({"v3_c1": 50, "v3_c2": 50}, {"v3_c1": 0.4, "v3_c2": 0.0})
    r = SB.stratified_bootstrap_diff(a, b)
    r1 = SB.stratified_bootstrap_diff({"v3_c1": a["v3_c1"]}, {"v3_c1": b["v3_c1"]})
    # v3_c2 の差は 0 なので E の区間は v3_c1 だけの区間のちょうど半分（同じ seed、v3_c1 が先に引かれる）
    assert r["lo"] == pytest.approx(r1["lo"] / 2, abs=1e-4) and r["hi"] == pytest.approx(r1["hi"] / 2, abs=1e-4)


def test_exclusion_and_order_and_errors():
    a, b = _sets({"v3_c1": 30, "enveda_v1": 30}, {"v3_c1": 0.2, "enveda_v1": 0.2})
    ex = {"enveda_v1": {"enveda_v1_000", "enveda_v1_001"}}
    r = SB.stratified_bootstrap_diff(a, b, exclude=ex)
    assert r["per_set"]["enveda_v1"]["n"] == 28 and r["per_set"]["enveda_v1"]["n_excluded"] == 2
    assert r["sets"] == ["v3_c1", "enveda_v1"]          # SETS の順
    # 再現性
    assert SB.stratified_bootstrap_diff(a, b, exclude=ex) == r
    with pytest.raises(ValueError):
        SB.stratified_bootstrap_diff(a, {"v3_c1": b["v3_c1"]})
    bad = {k: dict(v) for k, v in b.items()}; bad["v3_c1"].pop("v3_c1_000")
    with pytest.raises(ValueError):
        SB.stratified_bootstrap_diff(a, bad)
    with pytest.raises(ValueError):
        SB.stratified_bootstrap_diff(a, b, exclude={"v3_c1": {"nope"}})
    with pytest.raises(ValueError):
        SB.stratified_bootstrap_diff(a, b, exclude={"v3_c9": set()})
