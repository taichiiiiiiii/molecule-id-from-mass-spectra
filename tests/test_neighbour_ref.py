"""scripts/features/neighbour_ref.py の単体テスト（Issue #261）。合成データだけ（casmi・実データを読まない）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import neighbour_ref as NR  # noqa: E402

NB = 16


def fp(on):
    v = np.zeros(NB, np.uint8)
    v[list(on)] = 1
    return v


def packed(rows):
    return np.packbits(np.array(rows, np.uint8), axis=1)


def dot_sim(q, r):
    """合成の「スペクトル」はスカラー。類似度 = 1 − |q − r|。"""
    return 1.0 - abs(q - r)


def test_hidden_refs_not_used():
    # 構造 0 = 候補と同じ指紋（隠す正解）、1・2 = 近い、3 = 遠い。正準の鍵 C0..C3、C0 には raw の別名 R0b もある
    skeys = np.array(["C0", "C1", "C2", "C3"])
    sidx = {k: i for i, k in enumerate(skeys)}
    raw2canon = {"R0": "C0", "R0b": "C0", "R1": "C1", "R2": "C2", "R3": "C3"}
    lib = [fp([0, 1, 2, 3]), fp([0, 1, 2]), fp([0, 1]), fp([10, 11])]
    # train の行: (raw, ingest_lib, スペクトル)
    rows = pd.DataFrame(dict(raw=["R0", "R0b", "R0", "R1", "R1", "R2", "R3"],
                             lib=["enveda-180", "gnps", "mona", "gnps", "enveda-180", "gnps", "gnps"],
                             spec=[1.0, 1.0, 1.0, 0.2, 0.9, 0.4, 0.0]))
    canon = rows.raw.map(raw2canon).to_numpy()
    hidden = NR.drop_canon({"R0"}, raw2canon)
    assert hidden == {"C0"}
    cand = np.array([fp([0, 1, 2, 3])])
    q = [0.95]

    # rule = all: C0 の行は別名 R0b も含めすべて隠れ、C0 は近傍に入らない
    keep = NR.keep_rows(canon, rows.lib.to_numpy(), hidden, "all")
    assert keep.tolist() == [False, False, False, True, True, True, True]
    avail = np.zeros(4, bool)
    avail[[sidx[c] for c in canon[keep]]] = True
    idx, tani = NR.topk_neighbours(cand, packed(lib), avail, k=10, nbits=NB, ref_chunk=2, cand_block=1)
    js = [j for j in idx[0] if j >= 0]
    assert 0 not in js and js == [1, 2, 3]
    acc = {}
    need = {int(j): {"m"} for j in idx[0] if j >= 0}
    # 行はすべて渡し、隠す行は keep で落とす（流し読みと同じ形）
    NR.accumulate(acc, [sidx[c] for c in canon], keep, rows.spec.tolist(), need, {"m": q}, dot_sim)
    assert all(j != 0 for (_m, j) in acc)
    f = NR.molecule_features(idx, tani, True, NR.struct_scores(acc, "m"))
    exp_ms = np.mean([np.mean([dot_sim(0.95, 0.2), dot_sim(0.95, 0.9)]), dot_sim(0.95, 0.4), dot_sim(0.95, 0.0)])
    assert f.nbr_ms_mean[0] == pytest.approx(exp_ms)
    assert f.nbr_tani_mean[0] == pytest.approx(np.mean([3 / 4, 2 / 4, 0.0]))
    assert int(f.nbr_n[0]) == 3

    # rule = enveda_libs: C0 は enveda 以外の行（R0b gnps, R0 mona）で残り、enveda の行だけ隠れる。C1 は隠さない
    keep = NR.keep_rows(canon, rows.lib.to_numpy(), hidden, "enveda_libs")
    assert keep.tolist() == [False, True, True, True, True, True, True]
    avail = np.zeros(4, bool)
    avail[[sidx[c] for c in canon[keep]]] = True
    idx, tani = NR.topk_neighbours(cand, packed(lib), avail, k=2, nbits=NB, ref_chunk=3, cand_block=1)
    assert idx[0].tolist() == [0, 1] and tani[0].tolist() == pytest.approx([1.0, 0.75])
    acc = {}
    need = {int(j): {"m"} for j in idx[0] if j >= 0}
    NR.accumulate(acc, [sidx[c] for c in canon], keep, rows.spec.tolist(), need, {"m": q}, dot_sim)
    assert acc[("m", 0)][1] == 2         # 隠した enveda の行は使わない（C0 の 3 行のうち 2 行）
    assert ("m", 3) not in acc           # 近傍に無い構造の行は使わない
    f = NR.molecule_features(idx, tani, True, NR.struct_scores(acc, "m"))
    assert f.nbr_ms_mean[0] == pytest.approx(np.mean([dot_sim(0.95, 1.0),
                                                      np.mean([dot_sim(0.95, 0.2), dot_sim(0.95, 0.9)])]))
    # 参照の残らない構造が近傍に入れば止める
    with pytest.raises(AssertionError):
        NR.molecule_features(np.array([[0, 3]]), np.array([[1.0, 0.0]]), True, {0: 1.0})
    with pytest.raises(ValueError):
        NR.keep_rows(canon, rows.lib.to_numpy(), hidden, "nope")


def test_zero_neighbours_nan():
    lib = packed([fp([0, 1]), fp([2])])
    cand = np.array([fp([0, 1]), fp([5])])
    idx, tani = NR.topk_neighbours(cand, lib, np.zeros(2, bool), k=10, nbits=NB)
    assert (idx == -1).all() and np.isnan(tani).all()
    f = NR.molecule_features(idx, tani, True, {})
    assert f.nbr_ms_mean.isna().all() and f.nbr_tani_mean.isna().all() and (f.nbr_n == 0).all()
    # 近傍はあるが問い合わせのスペクトルが無い: 照合は NaN、Tanimoto の平均は値あり
    idx, tani = NR.topk_neighbours(cand, lib, np.ones(2, bool), k=10, nbits=NB)
    acc = {}
    NR.accumulate(acc, [0, 1], [True, True], [0.1, 0.2], {0: {"m"}, 1: {"m"}}, {"m": []}, dot_sim)
    assert acc == {}
    f = NR.molecule_features(idx, tani, False, NR.struct_scores(acc, "m"))
    assert f.nbr_ms_mean.isna().all() and f.nbr_tani_mean.notna().all() and (f.nbr_n == 2).all()


def test_topk_matches_brute_force_with_ties():
    rng = np.random.default_rng(0)
    lib = (rng.random((57, NB)) < 0.3).astype(np.uint8)
    lib[10] = lib[3]                     # 同点（番号の小さい 3 が先）
    cand = (rng.random((9, NB)) < 0.3).astype(np.uint8)
    cand[0] = lib[3]
    avail = rng.random(57) < 0.8
    avail[[3, 10]] = True
    idx, tani = NR.topk_neighbours(cand, packed(lib), avail, k=5, nbits=NB, ref_chunk=7, cand_block=4)
    for i in range(len(cand)):
        inter = (lib.astype(int) & cand[i]).sum(1)
        uni = lib.sum(1) + cand[i].sum() - inter
        t = np.where(uni > 0, inter / np.where(uni > 0, uni, 1), 0.0)
        order = [j for j in sorted(range(57), key=lambda j: (-t[j], j)) if avail[j]][:5]
        assert idx[i].tolist() == order
        assert tani[i] == pytest.approx(t[order])
    assert idx[0][:2].tolist() == [3, 10]
