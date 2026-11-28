"""scripts/features/fragment_stats.py の単体テスト（Issue #296、群 H3 pfs_*）。合成の分子だけ（frag_rescore があれば一致も見る）。"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/features"))
import fragment_stats as P  # noqa: E402

pytest.importorskip("rdkit")
A, B = P.ALPHA, P.BETA


def _fake_frag():
    """frag_rescore の必要な部分だけの偽物（[M+H]+ の担い手 1 つ、dh = 0 だけ、質量は整数で見やすく）。"""
    from rdkit import Chem
    F = types.SimpleNamespace()
    F.H = 1.0
    F.DEFAULT_CFG = dict(tol=0.005, h_shifts=(0,))
    mass = {"C": 12.0, "O": 16.0, "N": 14.0}

    def mol_graph(smi):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        w = np.array([mass[a.GetSymbol()] + a.GetTotalNumHs() for a in m.GetAtoms()], np.float64)
        ea = np.array([b.GetBeginAtomIdx() for b in m.GetBonds()], np.int64)
        eb = np.array([b.GetEndAtomIdx() for b in m.GetBonds()], np.int64)
        return w, ea, eb, len(Chem.GetMolFrags(m))

    F.mol_graph = mol_graph
    F._carriers = lambda adduct, mode: (1.0,)
    return F


F0 = _fake_frag()


# ---------------------------------------------------------------- 1. 列挙の数
def test_enumeration_counts_ethanol_and_ring():
    fr = P.enumerate_fragments("CCO", F0)
    # 原子の集合: {012} k0、{0} {12} {01} {2} k1、{1} k2 → 6
    assert len(fr.mass) == 6
    got = sorted(zip(np.round(fr.mass, 3).tolist(), fr.k.tolist(), fr.name))
    assert got == sorted([(46.0, 0, "CCO"), (15.0, 1, "C"), (31.0, 1, "CO"), (29.0, 1, "CC"),
                          (17.0, 1, "O"), (14.0, 2, "C")])
    ring = P.enumerate_fragments("C1CC1", F0)
    # 1 本切りでは割れない。2 本切り 3 通り × (1 原子, 2 原子) → 6 + 分子そのもの 1
    assert len(ring.mass) == 7 and sorted(ring.k.tolist()) == [0] + [2] * 6
    only1 = P.enumerate_fragments("CCO", F0, max_bonds_2=1)   # 結合 2 > 1 なので 1 本切りだけ
    assert len(only1.mass) == 5 and set(only1.k.tolist()) == {0, 1}
    assert P.enumerate_fragments("not_a_smiles", F0) is None
    assert np.all(np.diff(fr.mass) >= 0)                     # 質量の昇順


def test_masses_match_frag_rescore_when_available():
    frag_dir = ROOT / "data/ext/frag"
    if not (frag_dir / "frag_rescore.py").exists():
        pytest.skip("data/ext/frag が無い（CI）")
    sys.path.insert(0, str(frag_dir))
    import frag_rescore as fr
    for s in ["OC(=O)c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "OCC1OC(O)C(O)C(O)C1O", "[Na+].CC(=O)[O-]"]:
        mine = P.enumerate_fragments(s, fr)
        lv = fr.fragment_levels(s)
        assert set(np.round(mine.mass, 5)) == set(np.round(np.concatenate(lv), 5))
        # 質量ごとの最少の切断数も同じ
        for k, arr in enumerate(lv):
            for x in np.round(arr, 5):
                assert mine.k[np.round(mine.mass, 5) == x].min() == k


# ---------------------------------------------------------------- 数え上げの選び方
def test_count_spectrum_picks_fewest_cleavages_then_name():
    fr = P.enumerate_fragments("CCO", F0)
    # ピーク 16.0 = 断片 15（{0}, k1, "C"）+ 1、ピーク 15.0 = {1}（k2）+ 1、ピーク 18 = "O"+1、ピーク 99 = 説明なし
    spec = dict(mz=np.array([16.0, 15.0, 18.0, 99.0]), it=np.array([1.0, 0.5, 0.2, 0.1]), adduct="[M+H]+", mode=1)
    b, fi = P.count_spectrum(fr, spec, F0)
    assert len(b) == 3
    names = [fr.name[i] for i in fi]
    ks = [int(fr.k[i]) for i in fi]
    assert sorted(zip(b.tolist(), names, ks)) == [(1500, "C", 2), (1600, "C", 1), (1800, "O", 1)]
    assert P.count_spectrum(fr, dict(mz=np.array([]), it=np.array([])), F0)[0].size == 0


# ---------------------------------------------------------------- 2. own-out
def _table():
    key = lambda b, f: (b << P.FID_BITS) | f   # noqa: E731
    # 構造 0: bin 100 の断片 1 を 3 回、構造 1: 同じ鍵 2 回 + bin 100 の断片 2 を 1 回、構造 2: bin 200 の断片 1 を 4 回
    rows = [(0, 0, key(100, 1))] * 3 + [(1, 1, key(100, 1))] * 2 + [(1, 1, key(100, 2))] + [(2, 0, key(200, 1))] * 4
    return P.CountTable.from_rows(rows), key


def test_own_out_subtracts_own_contribution():
    tab, key = _table()
    sc = tab.for_set(np.ones(len(tab.key), bool))
    bins = np.array([100])
    peak = np.array([0])
    keys = np.array([key(100, 1)])
    lt_all, seen_all = sc.theta_peaks(bins, peak, keys, None)
    # 全部: N = 5、S = 6、|F| = 2
    assert lt_all[0] == pytest.approx(math.log((5 + A) / (6 + 2 * A + B)))
    own = sc.own({0})
    lt_own, _ = sc.theta_peaks(bins, peak, keys, own)
    # 構造 0 を引く: N = 2、S = 3、|F| = 2
    assert lt_own[0] == pytest.approx(math.log((2 + A) / (3 + 2 * A + B)))
    own1 = sc.own({1})
    lt1, seen1 = sc.theta_peaks(bins, np.array([0]), np.array([key(100, 2)]), own1)
    # 構造 1 を引く: 断片 2 は N = 0 で F_m から外れる → (0 + α)/(3 + α·(1 + 1) + β)、見ていない
    assert lt1[0] == pytest.approx(math.log(A / (3 + 2 * A + B))) and not seen1[0]
    assert seen_all[0]
    # 除外（keep）で外した構造は own でも引かない（負にならない）
    keep = tab.struct != 0
    sc2 = tab.for_set(keep)
    lt2, _ = sc2.theta_peaks(bins, peak, keys, sc2.own({0}))
    assert lt2[0] == pytest.approx(math.log((2 + A) / (3 + 2 * A + B)))


# ---------------------------------------------------------------- 3. θ の式
def test_theta_formula_bot_and_unseen_bin():
    tab, key = _table()
    sc = tab.for_set(np.ones(len(tab.key), bool))
    bins = np.array([100, 100, 300, 300])
    # ピーク 0: 断片 1 と 2 が合う → θ の大きい断片 1（N 5）、ピーク 1: 合う断片なし（⊥）、
    # ピーク 2: 学習に無い bin で合う断片あり、ピーク 3: 学習に無い bin で ⊥
    peak = np.array([0, 0, 2])
    keys = np.array([key(100, 2), key(100, 1), key(300, 7)])
    lt, seen = sc.theta_peaks(bins, peak, keys, None)
    assert lt[0] == pytest.approx(math.log((5 + A) / (6 + 2 * A + B)))
    assert lt[1] == pytest.approx(math.log(B / (6 + 2 * A + B)))
    assert lt[2] == pytest.approx(math.log(A / (A + B)))     # ≈ 0.038
    assert lt[3] == pytest.approx(0.0)                       # θ_m⊥ = β/β
    assert seen.tolist() == [True, False, False, False]
    # 学習に無い名前（鍵 −1）は N = 0 で合ったことにする
    lt_u, _ = sc.theta_peaks(np.array([100]), np.array([0]), np.array([-1]), None)
    assert lt_u[0] == pytest.approx(math.log(A / (6 + 3 * A + B)))


# ---------------------------------------------------------------- 4. 欠けの扱い
def test_missing_inputs():
    tab, _ = _table()
    sc = tab.for_set(np.ones(len(tab.key), bool))
    fr = P.enumerate_fragments("CCO", F0)
    fids = np.full(len(fr.mass), -1, np.int64)
    out = P.molecule_features([], [(fr, fids), (None, None)], sc, [None, None], ["C2H6O", None], 0.005)
    for c in P.COLUMNS:
        assert out[c].tolist() == [0.0, 0.0]                  # 問い合わせにピークが無い → 0
    p = P.prep(dict(mz=np.array([1.005, 16.0]), it=np.array([1.0, 0.25]), adduct="[M+H]+", mode=1), F0)
    out = P.molecule_features([p], [(fr, fids), (None, None)], sc, [None, None], ["C2H6O", None], 0.005)
    # 読めない候補は全ピーク ⊥: bin 100 → β/(6 + 2α + β)、bin 1600 は学習に無い → log 1 = 0
    exp_bad = math.log(B / (6 + 2 * A + B))
    assert out["pfs_ll"][1] == pytest.approx(exp_bad)
    assert out["pfs_ll_per_peak"][1] == pytest.approx(exp_bad / 2)
    assert out["pfs_seen_frac"][1] == 0.0
    assert out["pfs_ll_fdmax"][1] == 0.0                      # 式が None → その候補だけの群
    # 候補 CCO: ピーク 16.0 に断片 "C"(15)+1 が合う（学習に無い名前・無い bin）→ α/(α+β)、ピーク 1.005 は ⊥
    assert out["pfs_ll"][0] == pytest.approx(exp_bad + math.log(A / (A + B)))
    assert np.all(np.isfinite(np.concatenate([out[c] for c in P.COLUMNS])))
    assert P.prep(dict(mz=np.array([1.0]), it=np.array([0.0])), F0) is None


def test_fdmax_same_formula_only():
    v = np.array([-3.0, -1.0, -5.0, -2.0])
    got = P.fdmax(v, ["C2H6O", "C2H6O", "C3H8", None])
    assert got.tolist() == [-2.0, 0.0, 0.0, 0.0]


def test_subset_keys_deterministic():
    import add_fragment_stats as AF
    keys = np.array([f"K{i:05d}" for i in range(100)])
    a, b = AF.subset_keys(keys, 10), AF.subset_keys(keys, 10)
    assert a.tolist() == b.tolist() and len(set(a)) == 10 and list(a) == sorted(a)
    assert AF.subset_keys(keys, 0).tolist() == keys.tolist()


# ---------------------------------------------------------------- 5. #296 の直し: 名前の hash・塊ごとの集約が旧（名前の辞書）と同じ値
_TRAIN = ["CCO", "CC(=O)O", "OCC(O)CO", "CCN", "c1ccccc1O", "CC(C)O", "NCC(=O)O", "COC"]
_CANDS = ["CCO", "OCC", "CC(=O)O", "OC(C)C", "CCCO", "NCCO", "c1ccccc1O", "COC", "bad_smiles"]


def _spectra_of(smi, rng):
    """合成の参照: 断片の質量 + 1（担い手）にいくつかの雑音のピーク。"""
    fr = P.enumerate_fragments(smi, F0)
    mz = np.concatenate([rng.choice(fr.mass, size=min(4, len(fr.mass)), replace=False) + 1.0, rng.uniform(5, 80, 2)])
    return [dict(mz=mz, it=rng.uniform(0.1, 1.0, len(mz)), adduct="[M+H]+", mode=1) for _ in range(2)]


def _old_table(specs):
    """ed16a3c の build_counts と同じ: 名前 → 番号の辞書（列挙の順に setdefault）、鍵 = (bin << 40) | 番号。"""
    fid: dict = {}
    frs = {s: P.enumerate_fragments(s, F0) for s in _TRAIN}
    gid = {s: np.array([fid.setdefault(nm, len(fid)) for nm in fr.name], np.int64) for s, fr in frs.items()}
    rows = []
    for si, (s, env, sp) in enumerate(specs):
        b, fi = P.count_spectrum(frs[s], sp, F0)
        rows += [(_TRAIN.index(s), env, int(k)) for k in (b << P.FID_BITS) | gid[s][fi]]
    return P.CountTable.from_rows(rows), fid


def _new_table(specs, chunk):
    frs = {s: P.enumerate_fragments(s, F0, keep_names=False) for s in _TRAIN}
    assert all(fr.name is None for fr in frs.values())
    order = sorted(range(len(_TRAIN)), key=lambda i: _TRAIN[i])
    parts = []
    for c0 in range(0, len(order), chunk):
        chunk_s = {_TRAIN[i] for i in order[c0:c0 + chunk]}
        S, E, B, H = [], [], [], []
        for s, env, sp in specs:
            if s not in chunk_s:
                continue
            b, fi = P.count_spectrum(frs[s], sp, F0)
            S.append(np.full(len(b), _TRAIN.index(s)))
            E.append(np.full(len(b), env))
            B.append(b)
            H.append(frs[s].h[fi])
        if S:
            parts.append(P.aggregate_records(np.concatenate(S), np.concatenate(E), np.concatenate(B), np.concatenate(H)))
    cat = [np.concatenate([p[i] for p in parts]) for i in range(5)]
    return P.CountTable.from_hashed(*cat)


def _features(tab, ids_of, query, own_struct):
    sc = tab.for_set(np.ones(len(tab.key), bool))
    cands, owns, forms = [], [], []
    for i, s in enumerate(_CANDS):
        fr = P.enumerate_fragments(s, F0)
        cands.append((None, None) if fr is None else (fr, ids_of(fr)))
        owns.append(sc.own(own_struct[i]) if own_struct[i] else None)
        forms.append(P.formula_of(s))
    return P.molecule_features(query, cands, sc, owns, forms, 0.005)


def test_hashed_counts_equal_name_dictionary():
    rng = np.random.default_rng(0)
    specs = [(s, int(j % 2), sp) for j, s in enumerate(_TRAIN) for sp in _spectra_of(s, rng)]
    old, fid = _old_table(specs)
    rq = np.random.default_rng(1)
    query = [P.prep(sp, F0) for s in ("CCO", "CC(=O)O", "c1ccccc1O") for sp in _spectra_of(s, rq)]
    own_struct = [{_TRAIN.index(s)} if s in _TRAIN else set() for s in _CANDS]
    f_old = _features(old, lambda fr: np.array([fid.get(nm, -1) for nm in fr.name], np.int64), query, own_struct)
    for chunk in (1, 2, len(_TRAIN)):
        new, vocab = _new_table(specs, chunk)
        assert len(new.key) == len(old.key) and new.cnt.sum() == old.cnt.sum()
        f_new = _features(new, lambda fr: P.vocab_ids(vocab, fr.h), query, own_struct)
        for c in P.COLUMNS:
            assert np.array_equal(f_old[c], f_new[c]), (chunk, c)
    assert np.any(f_old["pfs_seen_frac"] > 0)          # 合成の参照で「見た」ピークがある（空の比べではない）


def test_name_hash_and_aggregate():
    h = P.name_hash(["C", "CO", "C"])
    assert h.dtype == np.int64 and h[0] == h[2] and h[0] != h[1]
    assert P.name_hash(["CCO"])[0] == P.name_hash(["CCO"])[0]
    s, e, b, hh, c = P.aggregate_records([1, 0, 1, 1], [0, 0, 0, 1], [5, 5, 5, 5], [7, 7, 7, 7])
    assert s.tolist() == [0, 1, 1] and e.tolist() == [0, 0, 1] and c.tolist() == [1, 2, 1]
    assert [len(x) for x in P.aggregate_records([], [], [], [])] == [0] * 5
    vocab = np.array([-5, 3, 9], np.int64)
    assert P.vocab_ids(vocab, np.array([3, 4, -5, 10])).tolist() == [1, -1, 0, -1]
    assert P.vocab_ids(np.zeros(0, np.int64), np.array([1])).tolist() == [-1]
    tab, voc = P.CountTable.from_hashed(np.array([0, 1]), np.array([0, 1]), np.array([2, 3]), np.array([9, -5]),
                                        np.array([4, 1]))
    assert voc.tolist() == [-5, 9] and tab.key.tolist() == [(2 << P.FID_BITS) | 1, (3 << P.FID_BITS) | 0]
    with pytest.raises(ValueError):
        P.CountTable.from_hashed(np.array([0]), np.array([0]), np.array([-1]), np.array([1]), np.array([1]))
