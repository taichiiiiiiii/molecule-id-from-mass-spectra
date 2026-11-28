"""H3 の全 train・dev 20 集合・当てはめ exp022〜024 の部品（Issue #296）。合成のデータだけ。"""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/features", "scripts/ranker", "scripts/validate"):
    sys.path.insert(0, str(ROOT / p))

H3 = ("pfs_ll", "pfs_ll_per_peak", "pfs_seen_frac", "pfs_ll_fdmax")


def test_harness_h3_registered():
    import lambdarank_oof as LR
    import fragment_stats as P
    iss, cols = LR.EXTRA_ALLOWED["h3"]
    assert iss == "#296" and cols == H3 == P.COLUMNS
    assert len(set(LR.EXTRA_ALL)) == len(LR.EXTRA_ALL)          # 群の間で列が重ならない
    rep = LR.check_columns(list(LR.ID_COLS) + list(LR.FEATURES) + list(H3), H3)
    assert rep["missing"] == [] and rep["extra"] == [] and rep["n_features"] == len(LR.FEATURES) + 4


def test_run_h3_features_are_version_a_plus_4():
    import lambdarank_oof as LR
    sh = (ROOT / "scripts/ranker/run_h3.sh").read_text()
    va = (ROOT / "scripts/ranker/run_version_a_combo.sh").read_text()
    var = lambda txt, n: re.search(rf'^{n}="([^"]*)"', txt, re.M).group(1)  # noqa: E731
    gl, n1, h3 = var(sh, "GL"), var(sh, "N1"), var(sh, "H3")
    assert gl == var(va, "GL") and n1 == var(va, "N1") and var(sh, "EXCLUDE") == "prior_unverified_pop"
    base = LR.resolve_features("prior_unverified_pop", f"cc,cc_z,{gl},{n1}")
    new = LR.resolve_features("prior_unverified_pop", f"cc,cc_z,{gl},{n1},{h3}")
    assert len(base) == 81 and len(new) == 85 and new[:81] == base and tuple(new[81:]) == H3
    assert re.findall(r'"(exp02[2-4]) ([0-2])"', sh) == [("exp022", "0"), ("exp023", "1"), ("exp024", "2")]
    assert "dev_b8_cc_gl2_n1_h3.parquet" in sh and "start_gate.sh" in sh


def test_h3_aux_pick_and_rows():
    import h3_aux as A
    by_t = {"a": [f"s\tm{i}" for i in range(6)], "c": ["s\tx"], "e": []}
    got = A.pick(by_t)
    exp_a = sorted(by_t["a"], key=lambda g: hashlib.sha256(g.encode()).hexdigest())[:4]
    assert got == exp_a + ["s\tx"]                              # 足りない型は埋めない
    df = pd.DataFrame(dict(set=["s"] * 5, molecule_id=["m1", "m1", "m1", "m2", "m2"],
                           cand_key=list("ABCDE"), cand_smiles=["C"] * 5))
    score = np.array([0.5, 0.9, 0.9, 0.1, 0.2])
    y = np.array([1, 0, 0, 0, 0])
    rows = A.per_molecule_rows(df, score, y)
    assert rows["s\tm1"] == (1, 0, 1)                            # 同点は row の順、正解の行、正解でない最上位
    assert rows["s\tm2"] == (4, None, 4)
    assert A.sign(0.0) == "0" and A.sign(-1e-9) == "-" and A.sign(2.0) == "+"


def test_add_fragment_stats_struct_keep_equals_record_keep():
    """集合の除外を構造ごとに求めて記録に引く形が、記録ごとに keep_rows を呼ぶ旧の形と同じ。"""
    import neighbour_ref as NR
    canon = np.array(["K1", "K2", "K3", "K4"], dtype=object)
    fold = np.array([1, 2, 1, 3])
    rec_s = np.array([0, 0, 1, 2, 3, 3, 1])
    rec_env = np.array([True, False, True, False, True, False, False])
    for rule in ("all", "enveda_libs"):
        hidden = {"K2", "K4"}
        old = NR.keep_rows(canon[rec_s], np.where(rec_env, "enveda-180", "other"), hidden, rule) & (fold[rec_s] != 1)
        ke = NR.keep_rows(canon, np.full(4, "enveda-180", dtype=object), hidden, rule) & (fold != 1)
        ko = NR.keep_rows(canon, np.full(4, "other", dtype=object), hidden, rule) & (fold != 1)
        new = np.where(rec_env, ke[rec_s], ko[rec_s])
        assert new.tolist() == old.tolist(), rule


def test_counts_cache_roundtrip(tmp_path):
    import add_fragment_stats as AF
    import fragment_stats as P
    tab, vocab = P.CountTable.from_hashed(np.array([0, 2]), np.array([0, 1]), np.array([3, 4]), np.array([11, -2]),
                                          np.array([5, 1]))
    p = tmp_path / "c.npz"
    AF.save_counts(p, tab, vocab, dict(subset=2, use_raw_sha256="x"))
    t2, v2, info = AF.load_counts(p)
    assert info == dict(subset=2, use_raw_sha256="x") and v2.tolist() == vocab.tolist()
    for a in ("struct", "env", "key", "cnt"):
        assert getattr(t2, a).tolist() == getattr(tab, a).tolist()
    assert not (tmp_path / "c.tmp.npz").exists()
