"""scripts/research/np_subset_counts.py の小さな道具のテスト（合成データ、Issue #300）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("np_subset_counts", ROOT / "scripts/research/np_subset_counts.py")
NS = importlib.util.module_from_spec(spec)
sys.modules["np_subset_counts"] = NS
spec.loader.exec_module(NS)


def test_is_np_uses_canonical_key_only():
    coco = {"CANON00000000A"}
    assert NS.is_np(["CANON00000000A", "RAW0000000000A", None, ""], coco).tolist() == [True, False, False, False]


def test_msg_hit_raw_or_canonical():
    ms = dict(raw_tv={"RAWHIT00000000"}, key_tv={"KEYHIT00000000"})
    raw = ["RAWHIT00000000", "XXXXXXXXXXXXXX", "YYYYYYYYYYYYYY", "ZZZZZZZZZZZZZZ"]
    canon = ["Q", "KEYHIT00000000", None, "RAWHIT00000000"]   # 正準の鍵は MSG の正準の鍵とだけ比べる
    assert NS.msg_hit(raw, canon, ms).tolist() == [True, True, False, False]


def test_set_exclusion_raw_canonical_scorer_group():
    import holdout_expand as HE
    st = pd.DataFrame(dict(inchikey14=["T1", "A1", "T2", "A2", "G1", "G2", "F"],
                           scorer_key=["T1", "A1", "T2", "S2", "G1", "G2", "F"],
                           identity_group=[1, 2, 3, 3, 4, 4, 5]))
    ts_key = {"T1": "T1", "A1": "C1", "T2": "T2", "A2": "A2", "G1": "G1", "G2": "G2", "F": "F"}
    truth = pd.DataFrame(dict(inchikey14=["T1", "ZZ"], smiles=["s1", "s_c1"]))
    to_key = {"s1": "T1", "s_c1": "C1"}.get        # 正解の SMILES の正準の鍵 C1 = A1 の正準の鍵
    e_raw, e_key, e_grp = HE.exclusion_sets([truth], [{"G1"}], to_key, ts_key, st)
    sc = dict(zip(st.inchikey14, st.scorer_key))
    grp = dict(zip(st.inchikey14, st.identity_group))
    hit = HE.excluded(set(st.inchikey14), e_raw, e_key, e_grp, ts_key, sc, grp)
    # T1 生、A1 正準（別名）、G1 drop_keys、G2 同じ群。T2・A2（群 3）と F は残る
    assert hit == {"T1", "A1", "G1", "G2"}


def _table():
    return pd.DataFrame(dict(
        tims=[True, True, True, True, True, False, True],
        np=[True, True, True, True, True, True, False],
        mass_ok=[True, True, True, True, False, True, True],
        msg=[False, True, False, False, False, False, False],
        excl=[False, False, True, False, False, False, False],
        public=[True, True, True, False, True, True, True],
        group=[1, 2, 3, 4, 5, 6, 7]))


def test_stage_counts_and_prime_split():
    c = NS.stage_counts(_table())
    assert (c["n1_tims_np"], c["n2_mass"], c["n3_no_msg"], c["n4_no_sets"]) == (5, 4, 3, 2)
    assert (c["drop_msg"], c["drop_sets"], c["a_prime"], c["d_prime"]) == (1, 1, 1, 2)
    assert c["n1_tims_nonnp"] == 1 and c["n4_groups"] == 2


def test_counts_per_fold_with_split_labels():
    import b8_fold_counts as B8F
    ts = pd.DataFrame(dict(inchikey14=list("ABCDE"), key=list("ABCDE"), libmask=[0, 0, 0, 1 << 9, 0], sid=range(5)))
    st = pd.DataFrame(dict(inchikey14=list("ABCDE"), identity_group=[1, 2, 3, 4, 5], fold_identity=[0, 0, 1, 1, 2]))
    t = pd.DataFrame(dict(inchikey14=list("ABCDE"), tims=True, np=True, mass_ok=True, msg=False, excl=False,
                          public=True, group=[1, 2, 3, 4, 5]))
    got = {}
    for k in (0, 1, 2):
        lab = dict(zip(*B8F.split_labels(ts, st, k)[["inchikey14", "fold"]].T.values))
        got[k] = NS.stage_counts(t[t.inchikey14.map(lab) == "hold"])["d_prime"]
    assert got == {0: 2, 1: 1, 2: 1}   # D は np（enveda-np-examples）なので fold 1 の hold に入らない


def test_decide_rule():
    assert NS.decide(149, 400) == dict(n_a_prime=149, n_d_prime_pool=400, make_a_prime=False, d_prime_remainder=400,
                                       make_d_prime=True, add=True)
    d = NS.decide(150, 299)
    assert d["make_a_prime"] and d["d_prime_remainder"] == 149 and not d["make_d_prime"]
    d = NS.decide(300, 400)
    assert d["d_prime_remainder"] == 200 and d["make_d_prime"]
    assert not NS.decide(10, 100)["add"]


def test_breakdown_hides_small_cells():
    n = 40
    df = pd.DataFrame(dict(type=["a"] * n + ["b"] * 5, np=[False] * 35 + [True] * 5 + [True] * 5,
                           new=np.r_[np.ones(35), np.zeros(10)], ref=np.zeros(45)))
    for t in "cde":
        df = pd.concat([df, pd.DataFrame(dict(type=[t], np=[False], new=[0.0], ref=[0.0]))], ignore_index=True)
    out = NS.breakdown(df, "new", "ref")
    assert out["a"]["np"] == dict(n=5)
    assert out["a"]["non_np"]["n"] == 35 and out["a"]["non_np"]["delta"] == 1.0
    assert out["a"]["all"]["n"] == 40 and set(out["a"]["all"]) >= {"mrr", "lo", "hi"}
    assert out["b"]["np"] == dict(n=5)


def test_seed_mean_aligns_by_molecule():
    f1 = pd.DataFrame(dict(set=["s", "s"], molecule_id=["m1", "m2"], type=["a", "a"], rr=[1.0, 0.0]))
    f2 = pd.DataFrame(dict(set=["s", "s"], molecule_id=["m2", "m1"], type=["a", "a"], rr=[0.5, 0.0]))
    m = NS.seed_mean([f1, f2])
    assert dict(zip(m.molecule_id, m.rr)) == {"m1": 0.5, "m2": 0.25}
