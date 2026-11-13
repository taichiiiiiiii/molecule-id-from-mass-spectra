"""Issue #97: pmi 段（候補の点 fp @ (z − z̄)、z̄ は問い合わせロジットの平均）の合成データでのテスト。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import run as R  # noqa: E402

# 4 ビットの合成指紋。候補 3 は全ビット 1
FPS = {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0], "c": [0, 0, 1, 1], "d": [1, 1, 1, 1]}


def make_pool():
    meta = pd.DataFrame(dict(key=["KA", "KB", "KC", "KD"], smiles=list(FPS), mass=[100.0] * 4, pid=[0, 1, 2, 3]))
    return R.Pool(meta, None)


def fpc(smi):
    return np.asarray(FPS[smi], np.float32)


class FakeNpz:
    """np.load の NpzFile と同じ .files / [] を持つ入れ物。"""
    def __init__(self, d):
        self.d = d
        self.files = list(d)

    def __getitem__(self, k):
        return self.d[k]


def ranked(pool, z, zbar):
    rows = R.score_window(pool.window(100.0), pool, R.stage_z(z, zbar), fpc)
    return rows


def test_e_pmi_with_joint_or_pubchem_stops():
    for other in ("joint", "pubchem"):
        with pytest.raises(AssertionError):
            R.check_stages(["score", "pmi", "prior", other])
        with pytest.raises(AssertionError):
            R.run("v3_c1", ["score", "pmi", other])   # 重い読み込みの前に止まる
    R.check_stages(["score", "pmi", "prior", "frag"])     # 併用しなければ通る
    R.check_stages(["score", "prior", "pubchem"])         # pmi が無ければ今までどおり


def test_a_zero_zbar_keeps_score_order():
    pool = make_pool()
    z = np.array([0.5, 2.0, -1.0, 0.3], np.float32)
    base = R.score_window(pool.window(100.0), pool, z, fpc)
    got = ranked(pool, z, np.zeros(4))
    assert [i for _, i in got] == [i for _, i in base]
    assert np.allclose([s for s, _ in got], [s for s, _ in base])


def test_b_identical_queries_give_all_ties_stable():
    pool = make_pool()
    z = np.array([0.5, 2.0, -1.0, 0.3], np.float32)
    zf = FakeNpz({"__targets__": np.array([["m1", 100.0], ["m2", 100.0]], object), "m1": z, "m2": z.copy()})
    zbar = R.query_mean(zf)
    assert np.allclose(zbar, z)
    got = ranked(pool, z, zbar)
    assert all(s == 0.0 for s, _ in got)
    assert [i for _, i in got] == [0, 1, 2, 3]            # 同点は元の窓の順のまま（安定）


def test_c_all_ones_candidate_drops_by_zbar_hand_example():
    pool = make_pool()
    # 問い合わせ 2 つ: z̄ = (0, 3, 0, 1)。ビット 1 はどの問い合わせにも高い（hub ビット）
    z1 = np.array([1.0, 3.0, 0.0, 1.0], np.float32)
    z2 = np.array([-1.0, 3.0, 0.0, 1.0], np.float32)
    zf = FakeNpz({"__targets__": np.array([["m1", 100.0], ["m2", 100.0]], object), "m1": z1, "m2": z2})
    zbar = R.query_mean(zf)
    assert np.allclose(zbar, [0.0, 3.0, 0.0, 1.0])
    raw = {pool.smiles[i]: s for s, i in R.score_window(pool.window(100.0), pool, z1, fpc)}
    pmi = {pool.smiles[i]: s for s, i in ranked(pool, z1, zbar)}
    # 手計算: 生の点 a=1, b=3, c=1, d=5 → d が 1 位。補正後 a=1, b=0, c=0, d=1
    assert raw == {"a": 1.0, "b": 3.0, "c": 1.0, "d": 5.0}
    assert pmi == {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}
    assert raw["d"] - pmi["d"] == pytest.approx(float(np.ones(4) @ zbar))   # 全ビット 1 の候補は Σz̄ = 4 下がる
    order = [pool.smiles[i] for _, i in ranked(pool, z1, zbar)]
    assert order[:2] == ["a", "d"]                       # 生の 1 位 d が a と同点に下がり、元の順で a が先


def test_d_without_pmi_score_input_is_unchanged():
    z = np.array([0.5, 2.0, -1.0, 0.3], np.float64)
    out = R.stage_z(z, None)
    assert out.dtype == np.float32 and np.array_equal(out, z.astype(np.float32))   # 今までの zf[mid].astype(float32) と同じ
    pool = make_pool()
    assert R.score_window(pool.window(100.0), pool, R.stage_z(z, None), fpc) == \
        R.score_window(pool.window(100.0), pool, z.astype(np.float32), fpc)


def test_d_run_source_only_uses_zbar_when_pmi():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    assert 'zbar = query_mean(zf) if "pmi" in stages else None' in src
    assert src.count("stage_z(zf[mid], zbar)") == 1


def test_check_script_rule():
    spec = importlib.util.spec_from_file_location("pmi_v3_check", ROOT / "scripts/cv/pmi_v3_check.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.NEW_TAG == "score+pmi+prior+frag" and m.CTRL_TAG == "score+prior+frag"
    assert m.REQUIRED == ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
    ok = {h: dict(lo=0.0) for h in m.REQUIRED}
    assert m.judge(ok)["adopt"] is False                                    # v3_c1 の下限 > 0 が要る
    assert m.judge({**ok, "v3_c1": dict(lo=0.001)})["adopt"] is True
    assert m.judge({**ok, "v3_c1": dict(lo=0.01), "v3_c2": dict(lo=0.5)})["adopt"] is True
    assert m.judge({**ok, "v3_c1": dict(lo=0.0), "v3_c2": dict(lo=0.5)})["adopt"] is False   # v3_c2 の勝ちは根拠にしない
    assert m.judge({**ok, "v3_c1": dict(lo=0.01), "v3_c2n": dict(lo=-0.0201)})["adopt"] is False
    assert m.judge({**ok, "v3_c1": dict(lo=0.01), "enveda_v1": dict(lo=-0.02)})["adopt"] is True   # −0.02 ちょうどは可
    miss = {h: dict(lo=0.1) for h in m.REQUIRED if h != "enveda_v1"}
    assert m.judge(miss)["adopt"] is False and m.judge(miss)["missing"] == ["enveda_v1"]
