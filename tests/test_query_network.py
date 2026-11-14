"""Issue #101: 問い合わせ同士の相互最近傍（scripts/cv/query_network.py）の純粋な関数を合成スペクトルで確かめる。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("query_network", ROOT / "scripts/cv/query_network.py")
qn = importlib.util.module_from_spec(_spec)
sys.modules["query_network"] = qn
_spec.loader.exec_module(qn)


def spec(peaks, mode=1):
    """合成スペクトル: ピーク位置の集合だけを持つ。"""
    return dict(peaks=frozenset(peaks), mode=mode)


def jacc(a, b):
    """合成の類似度（対称）。ピーク集合の Jaccard。"""
    u = a["peaks"] | b["peaks"]
    return len(a["peaks"] & b["peaks"]) / len(u) if u else 0.0


def test_a_one_sided_pair_is_excluded():
    # A と B はよく似る（相互最近傍）。C は B に一番似るが、B の最近傍は A → C→B は片思い
    specs = {"A": [spec({1, 2, 3, 4})], "B": [spec({1, 2, 3, 5})], "C": [spec({3, 5, 9, 10})]}
    ids = list(specs)
    S = qn.sim_matrix(ids, specs, jacc)
    nn = qn.nearest(ids, S)
    assert [ids[j] for j in nn] == ["B", "A", "B"]
    assert [(ids[i], ids[j]) for i, j in qn.mutual_pairs(ids, S)] == [("A", "B")]


def test_b_different_ionisation_modes_are_not_compared():
    # A（正）と B（負）は同じピークでも比べない → s 未定義、つながらない。C（正）とだけ比べる
    specs = {"A": [spec({1, 2, 3}, 1)], "B": [spec({1, 2, 3}, -1)], "C": [spec({1, 7, 8}, 1)]}
    ids = list(specs)
    S = qn.sim_matrix(ids, specs, jacc)
    assert np.isnan(S[0, 1]) and np.isnan(S[1, 2])
    assert S[0, 2] == jacc(specs["A"][0], specs["C"][0])
    nn = qn.nearest(ids, S)
    assert nn[1] is None
    assert [(ids[i], ids[j]) for i, j in qn.mutual_pairs(ids, S)] == [("A", "C")]
    # 片方が両モードを持つときは同じモードの組だけ使う
    s2 = qn.pair_sim([spec({1, 2, 3}, -1), spec({5, 6}, 1)], [spec({1, 2, 3}, 1)], jacc)
    assert s2 == 0.0


def test_c_ties_are_broken_by_molecule_id():
    # m_b と m_c は m_a に同じだけ似る → m_a の最近傍は id の小さい m_b。m_b・m_c 同士は似ない
    a, b, c = spec({1, 2, 3, 4}), spec({1, 2, 3, 10}), spec({1, 2, 3, 20})
    # 並びを id 順と逆にしても結果は id 順で決まる
    specs = {"m_c": [c], "m_a": [a], "m_b": [b]}
    ids = list(specs)
    S = qn.sim_matrix(ids, specs, jacc)
    S[0, 2] = S[2, 0] = 0.0                  # m_b と m_c を似なくする
    nn = qn.nearest(ids, S)
    assert ids[nn[ids.index("m_a")]] == "m_b"
    pairs = [(ids[i], ids[j]) for i, j in qn.mutual_pairs(ids, S)]
    assert pairs == [("m_a", "m_b")] or pairs == [("m_b", "m_a")]


def test_pair_sim_uses_max_of_both_directions():
    def asym(x, y):
        return 0.9 if min(x["peaks"]) == 1 else 0.2
    assert qn.pair_sim([spec({1})], [spec({2})], asym) == 0.9


def test_tanimoto_and_spectrum_key():
    assert qn.tanimoto(np.array([1, 1, 0]), np.array([1, 0, 1])) == 1 / 3
    assert qn.tanimoto(np.zeros(3), np.zeros(3)) == 0.0
    k1 = qn.spectrum_key("[M+H]+", 100.1, [1.0, 2.0], [0.5, 1.0])
    assert k1 == qn.spectrum_key("[M+H]+", 100.1, np.array([1.0, 2.0]), np.array([0.5, 1.0]))
    assert k1 != qn.spectrum_key("[M+H]+", 100.1, [1.0, 2.0], [0.5, 0.9])
