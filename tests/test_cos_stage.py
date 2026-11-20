"""Issue #140: cos 段（cos(σ(z), fp)）と cos_v3_check の合成データのテスト。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "tests"))
import run as R  # noqa: E402


def _pool(n):
    meta = pd.DataFrame(dict(key=[f"K{i}" for i in range(n)], smiles=[f"S{i}" for i in range(n)],
                             mass=100.0 + np.arange(n) * 1e-4, pid=np.arange(n)))
    return R.Pool(meta, np.zeros(n))


def _fpc(vecs):
    return lambda smi: vecs.get(smi)


# ---------------------------------------------------------------- 式

def test_cos_matches_formula_and_sigmoid_is_stable():
    z = np.array([2.0, -1.0, 0.5, -800.0, 800.0])
    p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    assert np.allclose(R.sigmoid(z), p) and np.all(np.isfinite(R.sigmoid(z)))
    fp = np.array([1, 0, 1, 1, 0], np.float32)
    rows = R.cos_window(np.arange(1), _pool(1), z, _fpc({"S0": fp}))
    f64 = fp.astype(np.float64)
    want = float(f64 @ p / (np.linalg.norm(f64) * np.linalg.norm(p) + 1e-12))
    assert rows == [(pytest.approx(want, rel=1e-12), 0)]


def test_all_zero_fingerprint_scores_zero_and_none_is_skipped():
    z = np.array([-3.0, -2.0, -4.0])        # 内積では全 0（0 点）が負の点の候補より上に来る
    vecs = {"S0": np.array([1, 1, 0], np.float32), "S1": np.zeros(3, np.float32), "S2": None}
    cos = R.cos_window(np.arange(3), _pool(3), z, _fpc(vecs))
    dot = R.score_window(np.arange(3), _pool(3), z.astype(np.float32), _fpc(vecs))
    assert [i for _, i in cos] == [0, 1] and cos[1][0] == 0.0        # 全 0 は 0 点で最下位、None は飛ばす
    assert [i for _, i in dot] == [1, 0]                                # 対照（内積）は全 0 が上


def test_cos_scale_invariance():
    rng = np.random.default_rng(0)
    z = rng.normal(size=16)
    vecs = {f"S{i}": (rng.random(16) < 0.4).astype(np.float32) for i in range(8)}
    a = R.cos_window(np.arange(8), _pool(8), z, _fpc(vecs))
    b = R.cos_window(np.arange(8), _pool(8), z, _fpc({k: 3.5 * v for k, v in vecs.items()}))   # 指紋を定数倍
    assert [i for _, i in a] == [i for _, i in b]
    assert np.allclose([s for s, _ in a], [s for s, _ in b], rtol=1e-12)
    # ‖p‖ は分子ごとに一定: 点は (fp·p)/‖fp‖ に比例し、順位はそれと同じ
    p = R.sigmoid(z)
    ref = sorted(range(8), key=lambda i: -(vecs[f"S{i}"] @ p / max(np.linalg.norm(vecs[f"S{i}"]), 1e-300)))
    assert [i for _, i in a] == ref


def test_more_on_bits_not_favoured_example():
    # A は 2 ビット（z=4,4）、B は 6 ビット（z=5,5,5,-2,-2,-2）。内積は B（9 > 8）、cos は A（1.389 > 1.363）/‖p‖
    z = np.array([4, 4, 5, 5, 5, -2, -2, -2], np.float64)
    A = np.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32)
    B = np.array([0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    vecs = {"S0": A, "S1": B}
    dot = R.score_window(np.arange(2), _pool(2), z.astype(np.float32), _fpc(vecs))
    cos = R.cos_window(np.arange(2), _pool(2), z, _fpc(vecs))
    assert [i for _, i in dot] == [1, 0]
    assert [i for _, i in cos] == [0, 1]


# ---------------------------------------------------------------- check_stages

def test_check_stages_cos():
    R.check_stages(["cos", "prior", "frag"])
    R.check_stages(["cos"])
    R.check_stages(["score", "prior", "frag"])                 # cos を指定しなければ今までどおり
    for bad in (["cos", "score", "prior", "frag"], ["prior", "frag"]):
        with pytest.raises(AssertionError):
            R.check_stages(bad)
    for other in ("pmi", "nbz", "joint", "pubchem", "sanpier", "fragnn", "fragnnblind", "elem", "lib", "reflast"):
        with pytest.raises(AssertionError):
            R.check_stages(["cos", other, "prior", "frag"])
        with pytest.raises(AssertionError):
            R.run("v3_c1", ["cos", other, "prior"])   # 重い読み込みの前に止まる


# ---------------------------------------------------------------- run（合成データ、端から端まで）

def test_run_end_to_end_cos_prior_frag(tmp_path, monkeypatch):
    import test_fragnn_stage as TF
    root = TF._synthetic_tree(tmp_path, monkeypatch)
    smi = TF.SMI
    z = np.array([4, 4, 5, 5, 5, -2, -2, -2], np.float32)
    A = np.array([1, 1, 0, 0, 0, 0, 0, 0], np.float32)
    B = np.array([0, 0, 1, 1, 1, 1, 1, 1], np.float32)
    # 分子 a の窓 = 候補 0..3。正解は候補 3（c1ccccc1O）= A。候補 0 = B、1 = 全 0、2 = 指紋なし
    fps = {smi[0]: B, smi[1]: np.zeros(8, np.float32), smi[2]: None, smi[3]: A, smi[4]: A, smi[5]: A}

    class FakeFP:
        def __init__(self, bits):
            self.bits = bits

        def __call__(self, s):
            return fps.get(s)
    monkeypatch.setattr(R, "FPCache", FakeFP)
    zdir = root / "outputs/holdout_eval/syn_fold0"
    np.savez(zdir / "pc_logits.npz", __targets__=np.array([["a", "100.0003"], ["b", "200.0005"]], object), a=z, b=z)
    base = R.run("syn", ["score", "prior", "frag"])
    new = R.run("syn", ["cos", "prior", "frag"])
    t0, t1 = TF._top(root, "score+prior+frag"), TF._top(root, "cos+prior+frag")
    assert t0["a"] == [smi[0], smi[3], smi[1]]       # 内積: B(9) > A(8) > 全 0(0)
    assert t1["a"] == [smi[3], smi[0], smi[1]]       # cos: A > B > 全 0
    assert base["mrr25"] < new["mrr25"] and new["stages"] == ["cos", "prior", "frag"]


# ---------------------------------------------------------------- cos_v3_check

def test_cos_v3_check_uses_frag_judge_and_conditions():
    import cos_v3_check as C
    import frag_v3_check as F
    assert C.judge is F.judge and C.NEW_TAG == "cos+prior+frag" and C.CTRL_TAG == "score+prior+frag"
    d = {h: dict(lo=-0.01) for h in F.REQUIRED}
    d["v3_c1"] = dict(lo=0.001); d["v3_c2"] = dict(lo=0.0)
    assert C.judge(d)["adopt"] and C.conditions(d) == dict(C1=True, C2=True)
    d["enveda_v1"] = dict(lo=-0.021)
    assert not C.judge(d)["adopt"]
    d.pop("enveda_v1")
    assert not C.judge(d)["adopt"]                    # 欠けたら採用しない
    assert C.conditions({"v3_c2": dict(lo=-0.03)}) == dict(C1=False, C2=False)


def test_top1_changed():
    import cos_v3_check as C
    a = {"m1": ["CCO", "CCC"], "m2": ["CCN"], "m3": []}
    b = {"m1": ["CCC", "CCO"], "m2": ["CCN", "CCO"], "m3": []}
    assert C.top1_changed(a, b, ["m1", "m2", "m3"]) == 1
