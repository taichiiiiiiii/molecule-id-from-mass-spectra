"""#74 判断 (a) の run1/run2 一致判定（check_tables.run_agreement）の単体テスト（合成データ）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "engine2"))
from check_tables import FP_COLS, run_agreement  # noqa: E402


def _write(d, sim, rt, z):
    o = os.path.join(d, "own")
    os.makedirs(o, exist_ok=True)
    np.savez(os.path.join(o, "own_sim_rank_rows.npz"), **sim)
    np.savez(os.path.join(o, "own_rank_train.npz"), **rt)
    np.savez(os.path.join(o, "own_fp_logits.npz"), **z)


def _base(rng):
    sim = dict(X=rng.normal(size=(20, 51)).astype(np.float32), Y=np.arange(20) % 2, G=np.arange(20) // 2)
    rt = dict(X=rng.normal(size=(30, 31)).astype(np.float32), Y=(np.arange(30) % 3 == 0).astype(np.float32),
              G=np.arange(30) // 3, M=(np.arange(30) % 2).astype(np.float32))
    z = dict(Z=rng.normal(size=(5, 8)).astype(np.float32), set=np.array(["a"] * 5), ik=np.array(list("ABCDE")), fold=np.zeros(5, int))
    return sim, rt, z


def _copy(x):
    return {k: v.copy() for k, v in x.items()}


def test_identical_agree(tmp_path):
    sim, rt, z = _base(np.random.default_rng(0))
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim, rt, z)
    assert run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))["agree"]


def test_fp_rounding_agrees(tmp_path):
    sim, rt, z = _base(np.random.default_rng(1))
    rt2, z2 = _copy(rt), _copy(z)
    rt2["X"][:, FP_COLS] += 2e-6            # FP 由来の列の丸めの差は許す（z で判定）
    z2["Z"] = (z2["Z"].astype(np.float64) * (1 + 3e-6)).astype(np.float32)
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim, rt2, z2)
    res = run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))
    assert res["agree"] and res["rank_train_nonfp_cols_equal"]


def test_nonfp_col_diff_disagrees(tmp_path):
    sim, rt, z = _base(np.random.default_rng(2))
    rt2 = _copy(rt)
    nonfp = [j for j in range(31) if j not in FP_COLS][0]
    rt2["X"][0, nonfp] += 1e-7
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim, rt2, z)
    assert not run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))["agree"]


def test_sim_diff_disagrees(tmp_path):
    sim, rt, z = _base(np.random.default_rng(3))
    sim2 = _copy(sim)
    sim2["X"][3, 0] += 1e-7
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim2, rt, z)
    assert not run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))["agree"]


def test_z_beyond_tolerance_disagrees(tmp_path):
    sim, rt, z = _base(np.random.default_rng(4))
    z2 = _copy(z)
    z2["Z"][0, 0] += 1e-3
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim, rt, z2)
    assert not run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))["agree"]


def test_z_molecule_order_disagrees(tmp_path):
    sim, rt, z = _base(np.random.default_rng(5))
    z2 = _copy(z)
    z2["ik"] = z2["ik"][::-1].copy()
    _write(tmp_path / "r1", sim, rt, z); _write(tmp_path / "r2", sim, rt, z2)
    assert not run_agreement(str(tmp_path / "r1"), str(tmp_path / "r2"))["agree"]
