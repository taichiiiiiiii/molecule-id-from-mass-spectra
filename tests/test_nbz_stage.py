"""Issue #103: nbz 段（相互最近傍の問い合わせのロジットを s で混ぜる）の合成データでのテスト。"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import run as R  # noqa: E402

# 4 ビットの合成指紋。候補 d は全ビット 1
FPS = {"a": [1, 0, 0, 0], "b": [0, 1, 0, 0], "c": [0, 0, 1, 1], "d": [1, 1, 1, 1]}


def make_pool():
    meta = pd.DataFrame(dict(key=["KA", "KB", "KC", "KD"], smiles=list(FPS), mass=[100.0] * 4, pid=[0, 1, 2, 3]))
    return R.Pool(meta, None)


def fpc(smi):
    return np.asarray(FPS[smi], np.float32)


def order(pool, z):
    return [pool.smiles[i] for _, i in R.score_window(pool.window(100.0), pool, z, fpc)]


def stage_input(zs, pairs, mid):
    """run() のメインループと同じ選び方: つながりがあれば z'、無ければ元の z の float32。"""
    zz = R.nbz_mix(zs, pairs)
    return zz[mid] if mid in zz else np.asarray(zs[mid]).astype(np.float32)


Z1 = np.array([2.0, 0.0, 0.0, 0.0], np.float32)
Z2 = np.array([0.0, 4.0, 0.0, 0.0], np.float32)
Z3 = np.array([0.5, 2.0, -1.0, 0.3], np.float32)


def test_a_no_connection_keeps_score_order():
    pool = make_pool()
    zs = {"m1": Z1, "m2": Z2, "m3": Z3}
    assert R.nbz_mix(zs, []) == {}
    for m, z in zs.items():
        got = R.score_window(pool.window(100.0), pool, stage_input(zs, [], m), fpc)
        assert got == R.score_window(pool.window(100.0), pool, z.astype(np.float32), fpc)
    # 組はあっても、m3 はつながりが無いので元の z のまま
    zz = R.nbz_mix(zs, [("m1", "m2", 0.7)])
    assert set(zz) == {"m1", "m2"}
    assert np.array_equal(stage_input(zs, [("m1", "m2", 0.7)], "m3"), Z3)
    # 片端のロジットが無い組は使わない
    assert R.nbz_mix({"m1": Z1}, [("m1", "m2", 0.7)]) == {}


def test_b_zero_similarity_leaves_z_unchanged():
    pool = make_pool()
    zs = {"m1": Z1, "m2": Z2}
    zz = R.nbz_mix(zs, [("m1", "m2", 0.0)])
    assert np.array_equal(zz["m1"], Z1) and np.array_equal(zz["m2"], Z2)
    assert order(pool, zz["m1"]) == order(pool, Z1)


def test_c_hand_example():
    pool = make_pool()
    zs = {"m1": Z1, "m2": Z2}
    zz = R.nbz_mix(zs, [("m1", "m2", 1.0)])
    # 手計算: s = 1 → z1' = (z1 + z2)/2 = (1, 2, 0, 0)、z2' = (z2 + z1)/2 = 同じ（元の z で混ぜ、反復しない）
    assert np.allclose(zz["m1"], [1.0, 2.0, 0.0, 0.0]) and np.allclose(zz["m2"], [1.0, 2.0, 0.0, 0.0])
    assert zz["m1"].dtype == np.float32
    # 生の点 a=2, b=0, c=0, d=2 → a, d（同点は窓の順）, b, c。混ぜた後 a=1, b=2, c=0, d=3 → d, b, a, c
    assert order(pool, Z1) == ["a", "d", "b", "c"]
    assert order(pool, zz["m1"]) == ["d", "b", "a", "c"]
    # s = 0.5: z1' = ((2,0,0,0) + 0.5·(0,4,0,0)) / 1.5 = (4/3, 4/3, 0, 0)
    zz = R.nbz_mix(zs, [("m1", "m2", 0.5)])
    assert np.allclose(zz["m1"], [4 / 3, 4 / 3, 0.0, 0.0])
    assert np.allclose(zz["m2"], [2 / 3, 8 / 3, 0.0, 0.0])


def test_d_without_nbz_run_path_is_unchanged():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    # nbz が無ければ nbz_z = {} のまま、score_window には今までどおり zf[mid].astype(np.float32) が渡る
    assert 'if "nbz" in stages:' in src
    assert src.count("z = nbz_z[mid] if mid in nbz_z else zf[mid].astype(np.float32)") == 1
    assert src.count("nbz_z, nbz_diag = {}, {}") == 1


def test_e_forbidden_combinations_stop():
    for other in ("joint", "pubchem", "sanpier", "lib", "reflast"):
        with pytest.raises(AssertionError):
            R.check_stages(["score", "nbz", "prior", other])
        with pytest.raises(AssertionError):
            R.run("v3_c1", ["score", "nbz", other])   # 重い読み込みの前に止まる
    R.check_stages(["score", "nbz", "prior", "frag"])     # 併用しなければ通る
    R.check_stages(["score", "prior", "pubchem"])         # nbz が無ければ今までどおり


def test_crosscheck_with_101_json(tmp_path):
    p = tmp_path / "qn.json"
    p.write_text(json.dumps(dict(pairs=[dict(a="m1", b="m2", s=0.5)])))
    assert R.nbz_crosscheck([("m1", "m2", 0.5)], p) == "match"
    with pytest.raises(AssertionError):
        R.nbz_crosscheck([("m1", "m3", 0.5)], p)
    with pytest.raises(AssertionError):
        R.nbz_crosscheck([("m1", "m2", 0.6)], p)
    assert R.nbz_crosscheck([], tmp_path / "none.json") == "no_json"


def test_check_script_rule():
    spec = importlib.util.spec_from_file_location("nbz_v3_check", ROOT / "scripts/cv/nbz_v3_check.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.NEW_TAG == "score+nbz+prior+frag" and m.CTRL_TAG == "score+prior+frag"
    assert m.REQUIRED == ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
    ok = {h: dict(lo=0.0) for h in m.REQUIRED}
    assert m.judge(ok)["adopt"] is False                                    # v3_c1 の下限 > 0 が要る
    assert m.judge({**ok, "v3_c1": dict(lo=0.001)})["adopt"] is True
    assert m.judge({**ok, "v3_c1": dict(lo=0.0), "v3_c2": dict(lo=0.5)})["adopt"] is False   # v3_c2 の勝ちは根拠にしない
    assert m.judge({**ok, "v3_c1": dict(lo=0.01), "v3_c2n": dict(lo=-0.0201)})["adopt"] is False
    assert m.judge({**ok, "v3_c1": dict(lo=0.01), "enveda_v1": dict(lo=-0.02)})["adopt"] is True   # −0.02 ちょうどは可
    miss = {h: dict(lo=0.1) for h in m.REQUIRED if h != "enveda_v1"}
    assert m.judge(miss)["adopt"] is False and m.judge(miss)["missing"] == ["enveda_v1"]
    sp = m.split_summary({"x": 1.0, "y": 0.5, "z": 0.2}, {"x": 0.5, "y": 0.5, "z": 0.25}, {"x", "z"})
    assert sp["connected"] == dict(n=2, diff=0.225, better=1, worse=1)
    assert sp["unconnected"] == dict(n=1, diff=0.0, better=0, worse=0)
