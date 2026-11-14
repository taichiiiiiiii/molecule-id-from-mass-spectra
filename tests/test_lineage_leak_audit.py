"""lineage_leak_audit の単体テスト（合成のキー集合で積の数え方と §2.1 の判定を確かめる）。"""
import importlib.util
import os
from pathlib import Path

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("lineage_leak_audit", os.path.join(HERE, "..", "scripts", "cv", "lineage_leak_audit.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)

T = {"v3_c1": {"m1": ("AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA"), "m2": ("BBBBBBBBBBBBBB", "BBBBBBBBBBBBBX"), "m3": ("CCCCCCCCCCCCCC", None)},
     "v3_c2": {"n1": ("DDDDDDDDDDDDDD", "DDDDDDDDDDDDDD"), "n2": ("EEEEEEEEEEEEEE", "EEEEEEEEEEEEEE")}}


def test_overlap_counts_either_key_and_denominator():
    r = A.overlap(T, {"AAAAAAAAAAAAAA", "BBBBBBBBBBBBBX", "ZZZZZZZZZZZZZZ"})
    assert r["v3_c1"] == dict(n=2, denom=3, frac=0.6667, n_ik14=1, n_scorer=2)
    assert r["v3_c2"]["n"] == 0 and r["v3_c2"]["denom"] == 2
    # scorer 側だけのキー集合も足せる
    r2 = A.overlap(T, set(), {"EEEEEEEEEEEEEE"})
    assert r2["v3_c2"]["n"] == 1 and r2["v3_c2"]["n_ik14"] == 1 and r2["v3_c2"]["n_scorer"] == 1   # どちらも target ∪ target_scorer


def test_hit_mids_and_overlap_after():
    h = A.hit_mids(T, {"CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD"})
    assert h == {"v3_c1": {"m3"}, "v3_c2": {"n1"}}
    assert A.overlap_after(T, {"CCCCCCCCCCCCCC", "DDDDDDDDDDDDDD"}, {"v3_c1": {"m3"}, "v3_c2": set()}) == {"v3_c1": 0, "v3_c2": 1}
    assert A.overlap_after(T, set(), None) is None


def test_c_rule_threshold_and_missing():
    big = {s: {f"{s}_{i}": (f"K{i:013d}", None) for i in range(100)} for s in ("v3_c1", "v3_c2")}
    comp = {"x": {"v3_c1": {f"v3_c1_{i}" for i in range(5)}, "v3_c2": set()},
            "y": {"v3_c1": {"v3_c1_0"}, "v3_c2": {"v3_c2_1"}}}
    r = A.c_rule(big, comp, required=("x",))
    assert r["per_set"]["v3_c1"]["c"] == 0.05 and r["per_set"]["v3_c1"]["n"] == 5   # 和集合（重複は 1 回）
    assert r["decision"] == "usable"                                               # c ≤ 5% は使える
    comp["x"]["v3_c1"].add("v3_c1_99")
    assert A.c_rule(big, comp, required=("x",))["decision"] == "not_usable"                         # 6% は使えない
    comp["x"]["v3_c1"].discard("v3_c1_99"); comp["z"] = None
    r = A.c_rule(big, comp, required=("x",))
    assert r["decision"] == "usable" and r["missing_components"] == ["z"] and r["c_is_lower_bound"]   # 規則に無い成分は判定を変えない
    r = A.c_rule(big, comp, required=("z",))
    assert r["decision"] == "not_usable" and r["required_not_exact"] == ["z"]     # fragnet（required）の一覧が無い → 使えない
    r = A.c_rule(big, comp, required=("x",), approximate=("x",))
    assert r["decision"] == "not_usable" and r["required_not_exact"] == ["x"]     # 近似だけでも「一覧を作れない」側


def test_kept_after_hide_rules():
    rows = [("gnps", "K1"), ("enveda-180", "K1"), ("enveda-180", "K2"), ("mona", "K3"), ("riken", "K9")]
    drop = {"v3_c1": {"K1", "K2"}, "v3_c2": {"K3"}}
    kept = A.kept_after_hide(rows, drop, {"v3_c1": "enveda_libs", "v3_c2": "all"})
    assert kept == {"v3_c1": {"K1"}, "v3_c2": set()}        # enveda 系だけ消すと K1 は gnps の行で残る、all は全部消す
    with pytest.raises(ValueError):
        A.kept_after_hide(rows, drop, {"v3_c1": "pool", "v3_c2": "all"})


def test_kept_after_hide_scorer_key_aliases():
    # K3 の構造が train には別の ik14（A3、scorer のキーは K3）でも載っている（tautomer の別名）
    rows = [("gnps", "K3"), ("riken", "A3"), ("mona", "A3"), ("enveda-180", "A1"), ("gnps", "A1")]
    canon = {"A3": "K3", "A1": "K1"}
    drop = {"v3_c1": {"K1"}, "v3_c2": {"K3"}}
    rules = {"v3_c1": "enveda_libs", "v3_c2": "all"}
    st: dict = {}
    kept = A.kept_after_hide(rows, drop, rules, canon=canon, stats=st)
    assert kept["v3_c2"] == {"A3", "K3"}                     # ik14 だけで落とすと別名の行が残る（#114 レビューの指摘 1）
    assert st["v3_c2"] == {"K3": {"riken": 1, "mona": 1}}
    T2 = {"v3_c2": {"n1": ("K3", "K3")}}
    assert A.overlap(T2, kept["v3_c2"])["v3_c2"]["n"] == 1   # 正解のキーで数えると 1
    fixed = A.kept_after_hide(rows, drop, rules, canon=canon, scorer_drop=True)
    assert fixed["v3_c2"] == set()                           # scorer のキーでも落とせば 0
    assert fixed["v3_c1"] == {"A1", "K1"}                    # enveda_libs は scorer のキーでも enveda 系の行だけ落とす
    # drop_canon を渡すと、drop_keys に無い正解の scorer のキーでも見張る
    k2 = A.kept_after_hide([("gnps", "B9")], {"v3_c2": {"K3"}}, {"v3_c2": "all"}, canon={"B9": "S9"},
                           drop_canon={"v3_c2": {"K3", "S9"}})
    assert k2["v3_c2"] == {"B9", "S9"}


def test_feature_dependency_and_grep(tmp_path: Path):
    feats = ["lib_max", "ap", "is_gen", "analog_struct__st_rank", "derivation__dp_hit", "fragnet__fn_rank", "model_views__fzD_z", "fz"]
    d = A.feature_dependency(feats)
    assert d["n_features"] == 8 and d["counts"]["analog_engine"] == 1 and d["counts"]["generate"] == 1
    assert d["n_train_table_dependent"] == 4 and d["n_trained_family"] == 2 and d["counts"]["library_spectra"] == 1
    f = tmp_path / "c99.py"; f.write_text("x = 1\ntfp = np.load('work/train_fp_sel.npy')\nE = Engine(L, P, tfp, EngineCfg(generate=True))\n")
    g = A.grep_usages([f], base=tmp_path)
    assert [(x["file"], x["line"]) for x in g] == [("c99.py", 2), ("c99.py", 3)]


def test_not_computable_shape():
    r = A.not_computable("無い", path="x")
    assert r == dict(status="not_computable", reason="無い", path="x")
