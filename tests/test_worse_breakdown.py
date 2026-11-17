"""Issue #120: worse_breakdown.py（悪くなった分子の 1 位の内訳）の合成データのテスト。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import worse_breakdown as WB  # noqa: E402


def test_worse_molecules_strictly_lower_and_missing_is_zero():
    ctl = {"m1": 1.0, "m2": 0.5, "m3": 0.0, "m4": 0.25, "m5": 0.5}
    new = {"m1": 0.5, "m2": 0.5, "m3": 0.0, "m4": 1.0}          # m5 は新に無い → 0
    assert WB.worse_molecules(new, ctl) == ["m1", "m5"]


def test_rank_of_first_hit_and_out_of_top_k():
    key = lambda s: s.upper()  # noqa: E731
    assert WB.rank_of("B", ["a", "b", "b"], key) == 2
    assert WB.rank_of("Z", ["a", "b"], key) == 0
    assert WB.rank_of(None, ["a"], key) == 0
    assert WB.rank_of("C", ["a", "b", "c"], key, top_k=2) == 0


# 合成のキー: 生のキー = 小文字の SMILES、正準化後のキー = 先頭 1 文字（tautomer の違いを潰すつもり）
RAW = lambda s: s.lower() if s else None  # noqa: E731
CANON = lambda s: s[0].upper() if s else None  # noqa: E731
FORM = {"tA": "C6H6", "x1": "C6H6", "y1": "C7H8", "q": None}


def _cls(truth, top1, T_raw=(), T_canon=(), tr_raw=(), tr_canon=()):
    return WB.classify(truth, top1, formula_fn=lambda s: FORM.get(s), raw_fn=RAW, canon_fn=CANON,
                       T_raw=set(T_raw), T_canon=set(T_canon), tr_raw=set(tr_raw), tr_canon=set(tr_canon))


def test_membership_any_is_raw_or_canon():
    m = WB.membership("x1", RAW, CANON, {"x1"}, set())
    assert (m["raw"], m["canon"], m["any"]) == (True, False, True)
    m = WB.membership("x1", RAW, CANON, set(), {"X"})
    assert (m["raw"], m["canon"], m["any"]) == (False, True, True)
    m = WB.membership("x1", RAW, CANON, {"zz"}, {"Z"})
    assert m["any"] is False
    m = WB.membership("x1", lambda s: None, lambda s: None, {None}, {None})   # None はどの集合にも入らない
    assert m["any"] is False


def test_classify_formula_and_sets():
    r = _cls("tA", "x1", T_canon={"X"}, tr_raw={"x1"})
    assert r["formula"] == "same" and r["in_T"]["any"] and r["in_train"]["any"]
    assert r["in_T"]["raw"] is False and r["in_T"]["canon"] is True
    r = _cls("tA", "y1")
    assert r["formula"] == "diff" and not r["in_T"]["any"] and not r["in_train"]["any"]
    # 1 位が無い・読めない → なし（集合は見ない）
    assert _cls("tA", None)["formula"] == WB.NONE_CELL
    assert _cls("tA", "q")["formula"] == WB.NONE_CELL
    assert "in_T" not in _cls("tA", None)
    # 正解そのものの所属も記録する
    assert _cls("tA", "y1", T_raw={"ta"})["truth_in_T"]["raw"] is True


def test_table_counts_eight_cells_and_none_per_mode():
    recs = [
        _cls("tA", "x1", T_canon={"X"}, tr_raw={"x1"}),   # same, T in（canon だけ）, train in（raw だけ）
        _cls("tA", "y1", T_raw={"y1"}),                   # diff, T in（raw だけ）, train out
        _cls("tA", "y1"),                                 # diff, out, out
        _cls("tA", None),                                 # なし
    ]
    t = WB.table(recs, "any")
    assert t["n"] == 4 and t["none"] == 1 and t["same"] == 1 and t["diff"] == 2
    assert t["cells"]["same|Tin|trainin"] == 1
    assert t["cells"]["diff|Tin|trainout"] == 1
    assert t["cells"]["diff|Tout|trainout"] == 1
    assert sum(t["cells"].values()) + t["none"] == t["n"] and len(t["cells"]) == 8
    raw = WB.table(recs, "raw")
    assert raw["cells"]["same|Tout|trainin"] == 1 and raw["cells"]["diff|Tin|trainout"] == 1
    canon = WB.table(recs, "canon")
    assert canon["cells"]["same|Tin|trainout"] == 1 and canon["cells"]["diff|Tout|trainout"] == 2


def test_formula_rdkit_isomers_and_unreadable():
    import pytest
    pytest.importorskip("rdkit")
    assert WB.formula_rdkit("CCO") == WB.formula_rdkit("COC") == "C2H6O"
    assert WB.formula_rdkit("CCCO") != WB.formula_rdkit("CCO")
    assert WB.formula_rdkit("not_a_smiles(") is None
    assert WB.formula_rdkit(None) is None


def test_targets_match_preregistration():
    assert [(n, h, t, e) for n, h, t, e in WB.TARGETS] == [
        ("fragnn_enveda_v1", "enveda_v1", "score+fragnn+prior+frag", 19),
        ("fragnn_v3_c2n", "v3_c2n", "score+fragnn+prior+frag", 26),
        ("align_v3_c2n", "v3_c2n", "score+align+prior+frag", 41),
    ]
    assert WB.CTRL_TAG == "score+prior+frag"
