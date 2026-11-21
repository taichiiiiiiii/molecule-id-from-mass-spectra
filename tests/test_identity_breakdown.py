"""identity_breakdown の単体テスト（合成データ、(i)〜(iv) を 1 つずつ起こす 4 例 + 全部通る 1 例）。"""
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "identity_breakdown", os.path.join(HERE, "..", "scripts", "engine2", "identity_breakdown.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)


def _dz(raw):
    raw = np.asarray(raw, dtype=np.float64)
    d = raw - raw.max(); z = (raw - raw.mean()) / raw.std()
    return d, z


def test_pass_case():
    d, z = _dz([5.0, 3.0, 1.0, -2.0, 0.5])
    t, f = B.classify_group(d, z)
    assert t == "pass" and f["tie_rows"] == 0 and not f["reversal"]


def test_type_i_max_not_zero():
    d, z = _dz([5.0, 3.0, 1.0, -2.0])
    t, f = B.classify_group(d - 0.5, z)
    assert t == "i" and f["i"] and not f["ii"]


def test_type_ii_flag_positive_row():
    d, z = _dz([5.0, 3.0, 1.0, -2.0])
    d = d.copy(); d[1] = 0.3          # d > tol の行（max(d) も 0 でなくなるので (i) の旗も立つ）
    t, f = B.classify_group(d, z)
    assert f["ii"] and f["i"] and t == "i"


def test_type_iii_tie_only():
    # d が同値の 2 行で z が違う。_rank_norm(d) の任意の順が z と逆になると不成立
    d = np.array([0.0, -1.0, -1.0, -3.0]); z = np.array([1.5, 0.1, 0.2, -1.8])
    t, f = B.classify_group(d, z)
    assert t == "iii" and f["tie_rows"] == 2 and not f["reversal"]


def test_type_iv_strict_reversal():
    d = np.array([0.0, -2.0, -1.0, -3.0]); z = np.array([1.5, 0.5, 0.1, -1.8])   # z 2 位の d が 3 位より小さい
    t, f = B.classify_group(d, z)
    assert t == "iv" and f["reversal"]


def test_tie_rows_ignore_equal_z():
    d = np.array([0.0, -1.0, -1.0, -2.0, -2.0]); z = np.array([2.0, 0.5, 0.5, -1.0, -1.5])
    assert B.tie_rows_z_differs(d, z) == 2   # 塊 (-1,-1) は z が同じ → 数えない、(-2,-2) は数える


def _table(groups):
    X = np.zeros((sum(len(d) for d, _ in groups), 31), np.float32); G = []; rows = 0
    for g, (d, z) in enumerate(groups):
        n = len(d); X[rows:rows + n, B.COL_D] = d; X[rows:rows + n, B.COL_Z] = z
        X[rows:rows + n, B.COL_RANK] = np.argsort(np.argsort(-z, kind="mergesort")) / max(1, n - 1)
        G += [g] * n; rows += n
    G = np.array(G, np.int64); M = np.zeros(len(G)); Y = np.zeros(len(G))
    return X, Y, G, M


def test_breakdown_verdicts():
    ok = [_dz(np.arange(10.0) * (k + 1)) for k in range(30)]
    tie = (np.array([0.0, -1.0, -1.0, -3.0]), np.array([1.5, 0.1, 0.2, -1.8]))
    rev = (np.array([0.0, -2.0, -1.0, -3.0]), np.array([1.5, 0.5, 0.1, -1.8]))
    X, Y, G, M = _table(ok)
    r = B.breakdown(X, G, M); assert r["verdict"] == "pass" and r["n_fail"] == 0
    X, Y, G, M = _table(ok + [rev])
    r = B.breakdown(X, G, M); assert r["verdict"] == "stop" and r["types"]["iv"] == 1
    # (iii) だけ: 塊の行 2 / 304 = 0.66% ≤ 1% → 通過（_rank_norm の順の運で検査が通ることもあるので n_fail は問わない）
    X, Y, G, M = _table(ok + [tie])
    r = B.breakdown(X, G, M)
    assert r["tie_rows_z_differs"] == 2 and r["types"]["iv"] == 0 and r["verdict"] == "pass"
    # 塊の行が 1% を超えると (iii) だけでも停止
    X, Y, G, M = _table(ok[:3] + [tie])
    r = B.breakdown(X, G, M)
    assert r["tie_frac"] > 0.01 and r["verdict"] == "stop"


def test_main_writes_json(tmp_path):
    X, Y, G, M = _table([_dz(np.arange(8.0)) for _ in range(4)])
    Y[::8] = 1
    p = tmp_path / "t.npz"; np.savez(p, X=X, Y=Y, G=G, M=M)
    out = tmp_path / "o.json"
    B.main(["--table", f"a={p}", "--out", str(out)])
    j = json.load(open(out))
    assert j["tables"]["a"]["verdict"] == "pass" and j["tables"]["a"]["stats"]["n_groups_G"] == 4
    assert j["tables"]["a"]["stats"]["fp_top1_rate_M0_pos"] == 0.0
