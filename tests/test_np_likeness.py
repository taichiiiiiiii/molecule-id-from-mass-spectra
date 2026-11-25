"""Issue #240: scripts/features/np_likeness.py の単体テスト（作り物のデータだけ。実データは読まない）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/features"))
import np_likeness as N  # noqa: E402

SMI = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "OC1C(O)C(O)C(CO)OC1O", "CCN(CC)CC", "C1CCCCC1",
       "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "O=C1C=CC(=O)C=C1", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
       "C[C@H](N)C(=O)O", "OCC(O)CO", "Clc1ccc(Cl)cc1"]


def _struct():
    return pd.DataFrame(dict(inchikey14=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"],
                             scorer_key=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBX", "CCCCCCCCCCCCCC"],
                             identity_group=["10", "11", "12"], fold_identity=[1, 2, 1]))


def _rows(raw, canon, sk, grp, compat=None):
    n = len(raw)
    return pd.DataFrame(dict(raw_key=raw, canon_key=canon, scorer_key=sk, identity_group=grp,
                             compat=compat if compat is not None else [True] * n))


def test_fold_exclusion_by_bucket():
    # train の構造に当たる行は fold_identity、当たらない行は hash の bucket。bucket != k の行には fold k の構造が無い
    st = _struct()
    keys = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "ZZZZZZZZZZZZZZ", "CCCCCCCCCCCCCC"]
    tf = [1, 2, -1, 1]
    b = N.assign_bucket(keys, tf)
    assert b[0] == 1 and b[1] == 2 and b[3] == 1 and b[2] == N.hash_bucket("ZZZZZZZZZZZZZZ")
    rows = _rows(keys, keys, [k if t >= 0 else None for k, t in zip(keys, tf)],
                 ["10", "11", None, "12"])
    fk = N.fold_keysets(st, 1)
    ov = N.assert_no_fold_overlap(rows[b != 1], fk, 1)
    assert all(v == 0 for v in ov.values())
    with pytest.raises(AssertionError):
        N.assert_no_fold_overlap(rows, fk, 1)


@pytest.mark.parametrize("col,val", [("raw_key", "AAAAAAAAAAAAAA"), ("canon_key", "CCCCCCCCCCCCCC"),
                                     ("scorer_key", "AAAAAAAAAAAAAA"), ("identity_group", "12")])
def test_each_of_4_keys_detected(col, val):
    rows = _rows(["QQQQQQQQQQQQQQ"], ["QQQQQQQQQQQQQQ"], [None], [None])
    rows.loc[0, col] = val
    with pytest.raises(AssertionError):
        N.assert_no_fold_overlap(rows, N.fold_keysets(_struct(), 1), 1)


def test_uncanonicalised_compatible_row_is_refused():
    rows = _rows(["QQQQQQQQQQQQQQ"], [""], [None], [None], compat=[True])
    with pytest.raises(AssertionError):
        N.assert_no_fold_overlap(rows, N.fold_keysets(_struct(), 1), 1)
    rows2 = _rows(["QQQQQQQQQQQQQQ"], [""], [None], [None], compat=[False])
    N.assert_no_fold_overlap(rows2, N.fold_keysets(_struct(), 1), 1)


def test_pred_fold_mismatch_stops():
    ok = pd.DataFrame(dict(query_fold=[1, 2, 3, 4], model_fold=[1, 2, 3, 4]))
    N.check_pred_folds(ok)
    with pytest.raises(AssertionError):
        N.check_pred_folds(pd.DataFrame(dict(query_fold=[1, 2], model_fold=[1, 3])))
    with pytest.raises(AssertionError):
        N.check_pred_folds(pd.DataFrame(dict(query_fold=[0, 1], model_fold=[0, 1])))


def test_in_window():
    c = np.array([300.0, 100.0])
    m = np.array([100.0, 100.0009, 100.0011, 300.0029, 300.0031, 200.0])
    assert N.in_window(m, c).tolist() == [True, True, False, True, False, False]
    assert N.in_window(m, np.array([])).sum() == 0


def test_mass_matched_draw_deterministic_and_per_bin():
    tier = np.sort(np.random.default_rng(1).uniform(150, 160, 5000))
    pos = np.array([150.2, 150.7, 151.1, 155.5, 155.6, 155.7, 155.9])
    a = N.mass_matched_draw(pos, tier, seed=7)
    b = N.mass_matched_draw(pos, tier, seed=7)
    assert a.keys() == b.keys() == {150, 151, 155}
    for k in a:
        assert np.array_equal(a[k][0], b[k][0])
        assert len(a[k][0]) == int(np.ceil(a[k][1] * N.OVERSAMPLE))
        assert (np.floor(tier[a[k][0]]) == k).all() and len(set(a[k][0])) == len(a[k][0])
    binof = np.array([150, 150, 150, 155, 155])
    keep = N.keep_first_per_bin(binof, np.array([False, True, True, True, True]), {150: 1, 155: 1})
    assert keep.tolist() == [False, True, False, True, False]


def test_featurize_and_same_seed_same_prediction():
    X, ok = N.featurize(SMI * 20 + ["not_a_smiles"])
    assert X.shape == (len(SMI) * 20 + 1, N.NBITS + len(N.DESC_NAMES)) and ok[:-1].all() and not ok[-1]
    # 立体を消すので L-アラニン と ラセミの表記は同じ特徴
    a, _ = N.featurize(["C[C@H](N)C(=O)O", "CC(N)C(=O)O"])
    assert (a[0] != a[1]).nnz == 0
    y = np.array(([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0] * 20) + [0])
    X = X[np.where(ok)[0]]; y = y[ok]
    p1 = N.train_model(X, y).predict(X)
    p2 = N.train_model(X, y).predict(X)
    assert np.array_equal(p1, p2)


def test_auc():
    assert N.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert N.auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_wl_and_comp_hash_tautomers_compatible():
    # 互変異性体（2-ヒドロキシピリジン / 2-ピリドン）は組成も WL も同じ
    s1, s2 = "Oc1ccccn1", "O=c1cccc[nH]1"
    assert N.comp_hash(N.heavy_comp_str(s1)) == N.comp_hash(N.heavy_comp_str(s2))
    assert N.wl_hash(s1) == N.wl_hash(s2) != 0


def test_multi_fold_rows_never_trained():
    st = pd.DataFrame(dict(inchikey14=["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"], scorer_key=["BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"],
                           identity_group=["1", "2"], fold_identity=[1, 3]))
    mp = N.struct_map(st)
    assert mp["BBBBBBBBBBBBBB"][0] == N.MULTI and mp["AAAAAAAAAAAAAA"][0] == 1
    b = N.assign_bucket(["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"], [mp["BBBBBBBBBBBBBB"][0], 1])
    assert b[0] == N.MULTI
    for k in N.FOLDS:
        assert not N.train_mask(b, k)[0]
