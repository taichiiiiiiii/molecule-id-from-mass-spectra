"""#120 の記録用の診断 2 種（iso_rank_diag・trainstruct_bias）の合成データのテスト。実データは読まない。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))

import iso_rank_diag as IR  # noqa: E402
import trainstruct_bias as TB  # noqa: E402

NAN = math.nan
FORM = {"A1": "C6H6", "A2": "C6H6", "A3": "C6H6", "B1": "C7H8", "X": None}


def fml(smi):
    return FORM.get(smi)


def cand(**kw):
    """key → aggregate_by_key と同じ形の辞書。"""
    out = {}
    for k, (smi, s, f, n, nb) in kw.items():
        out[k] = dict(rep_row=0, smiles=smi, rows=[0], smiles_all=[smi], s=s, f=f, n=n, nb=nb)
    return out


# ------------------------------------------------------------------ (i) G の作り方
def test_aggregate_max_and_representative():
    rows = [dict(key="K1", row=5, smiles="A2", s=1.0, n=0.2, nb=None),
            dict(key="K1", row=3, smiles="A1", s=0.5, n=0.9, nb=0.1),
            dict(key="K2", row=4, smiles="B1", s=2.0, n=float("nan"), nb=float("inf"))]
    a = IR.aggregate_by_key(rows)
    assert a["K1"]["smiles"] == "A1" and a["K1"]["rep_row"] == 3      # Pool の並びで最初の行
    assert a["K1"]["s"] == 1.0 and a["K1"]["n"] == 0.9 and a["K1"]["nb"] == 0.1   # キーの中の最大
    assert a["K1"]["smiles_all"] == ["A1", "A2"]
    assert math.isnan(a["K2"]["n"]) and math.isnan(a["K2"]["nb"])     # NaN・inf は点なし
    assert math.isnan(a["K1"]["f"])                                     # f は後で G の中だけ計算する


def test_group_same_formula_only():
    c = cand(T=("A1", 1, 1, 1, 1), I=("A2", 1, 1, 1, 1), O=("B1", 1, 1, 1, 1), Z=("X", 1, 1, 1, 1))
    G, info = IR.build_group(c, "T", fml)
    assert G == ["T", "I"] and info["g_size"] == 2 and info["truth_formula"] == "C6H6"
    assert info["n_formula_none"] == 1


def test_group_exclusions():
    c = cand(T=("A1", 1, 1, 1, 1), O=("B1", 1, 1, 1, 1))
    assert IR.build_group(c, "Q", fml)[1]["excluded"] == "truth_absent"
    assert IR.build_group(c, None, fml)[1]["excluded"] == "truth_absent"
    assert IR.build_group(c, "T", fml)[1]["excluded"] == "g_lt2"
    c2 = cand(T=("X", 1, 1, 1, 1), I=("A2", 1, 1, 1, 1))
    assert IR.build_group(c2, "T", fml)[1]["excluded"] == "truth_formula_none"


def test_group_formula_conflict_counted():
    c = cand(T=("A1", 1, 1, 1, 1), I=("A2", 1, 1, 1, 1))
    c["I"]["smiles_all"] = ["A2", "B1"]
    assert IR.build_group(c, "T", fml)[1]["n_formula_conflict"] == 1


# ------------------------------------------------------------------ (i) 同点・点なし
def test_pessimistic_ties_and_missing():
    assert IR.pessimistic_rank([0.5, 0.5, 0.1], 0) == 2          # 同点は正解を最後に
    assert IR.pessimistic_rank([0.5, 0.9, 0.5, 0.5], 0) == 4
    assert IR.pessimistic_rank([0.3, NAN, 0.1], 0) == 1          # 点なしは下
    assert IR.pessimistic_rank([NAN, 0.1, NAN], 0) == 3          # 正解に点なし → 最下位
    assert IR.pessimistic_rank([0.2, 0.3, NAN], 0) == 2


def test_group_ranks_per_score():
    c = cand(T=("A1", 0.9, 0.1, 0.5, NAN), I=("A2", 0.2, 0.4, 0.5, 0.3), J=("A3", 0.1, 0.3, 0.1, 0.2))
    r = IR.group_ranks(c, ["T", "I", "J"], "T")
    assert r == dict(s=1, f=3, n=2, nb=3)


# ------------------------------------------------------------------ bootstrap の seed 固定
def test_bootstrap_seed_fixed_and_matches_manual():
    rng = np.random.default_rng(123)
    a = {f"m{i}": float(rng.random()) for i in range(50)}
    b = {f"m{i}": float(rng.random()) for i in range(50)}
    r1, r2 = IR.bootstrap(a, b), IR.bootstrap(a, b)
    assert r1 == r2
    mids = sorted(a)
    x = np.array([a[m] - b[m] for m in mids])
    bs = x[np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))].mean(1)
    assert r1["lo"] == round(float(np.percentile(bs, 2.5)), 4) and r1["hi"] == round(float(np.percentile(bs, 97.5)), 4)


def test_summarize_and_reading():
    per = {f"m{i}": dict(s=2, f=2, n=1, nb=3, g_size=4) for i in range(30)}
    s = IR.summarize(per, {"x": "truth_absent", "y": "g_lt2", "z": "g_lt2"})
    assert s["n"] == 30 and s["excluded"] == dict(truth_absent=1, truth_formula_none=0, g_lt2=2)
    assert s["mean_rr"]["C_fragnn"] == 1.0 and s["mean_rr"]["A_fpnet"] == 0.5
    assert s["top1_rate"]["C_fragnn"] == 1.0 and s["g_size_median"] == 4.0
    r = IR.read_rules(s)
    assert r == dict(C_minus_D="spectrum_contributes", C_minus_B="C_better", C_minus_A="C_better")
    per2 = {f"m{i}": dict(s=1, f=1, n=2, nb=1, g_size=3) for i in range(30)}
    r2 = IR.read_rules(IR.summarize(per2, {}))
    assert r2 == dict(C_minus_D="not_shown_spectral", C_minus_B="C_worse", C_minus_A="C_worse")
    assert IR.read_rules(IR.summarize({}, {})) == dict(status="no_molecules")


def test_reading_inconclusive():
    per = {f"m{i}": dict(s=(1 if i % 2 else 2), f=1, n=(2 if i % 2 else 1), nb=1, g_size=2) for i in range(40)}
    r = IR.read_rules(IR.summarize(per, {}))
    assert r["C_minus_A"] == "inconclusive"


# ------------------------------------------------------------------ 追加の数え上げ
def test_classify_and_flip():
    assert IR.classify_top1("T", ["T", "U"], "C6H6", ["C6H6"]) == "hit"
    assert IR.classify_top1("T", ["U"], "C6H6", ["C6H6"]) == "a"
    assert IR.classify_top1("T", ["U", "T"], "C6H6", ["C7H8"]) == "b"
    assert IR.classify_top1("T", ["U", "T"], "C6H6", ["C6H6"]) == "c"
    assert IR.classify_top1(None, ["U"], None, [None]) == "a"
    ctl = dict(m1="c", m2="c", m3="hit", m4="hit", m5="a")
    new = dict(m1="hit", m2="c", m3="hit", m4="c", m5="hit")
    assert IR.flip_counts(ctl, new) == dict(control_c=2, control_hit=2, c_to_hit=1, hit_lost=1)


# ------------------------------------------------------------------ (ii) T の作り方
def test_train_struct_rows_excludes_validation():
    struct_rows = np.array([10, 20, 30, 40, 50, 60])
    pos_struct = np.array([0, 0, 1, 4, 5])      # スペクトルごとの正解（struct_rows の添字）
    is_val = np.array([False, False, False, True, False])
    flat_tr = np.array([2, 1])                   # 学習の負例（struct_rows の添字）
    got = TB.train_struct_rows(struct_rows, pos_struct, is_val, flat_tr)
    assert got.tolist() == [10, 20, 30, 60]      # 40（内部検証の正解）と 50（どこにも無い）は入らない


def test_train_struct_keys_from_files(tmp_path):
    import pandas as pd
    np.savez(tmp_path / "prep.npz", struct_rows=np.array([1, 3, 4]), pos_struct=np.array([0, 2]),
             is_val=np.array([False, True]), flat_tr=np.array([1]))
    pd.DataFrame(dict(key=["k0", "k1", "k2", "k3", "k4"], smiles=list("abcde"))).to_parquet(tmp_path / "m.parquet")
    T, info = TB.train_struct_keys(tmp_path / "prep.npz", tmp_path / "m.parquet")
    assert T == {"k1", "k3"} and info["n_rows"] == 2
    assert TB.key_mismatch(info["smiles"], info["keys"], lambda s: {"b": "k1", "d": "kX"}[s]) == 1


# ------------------------------------------------------------------ (ii) z の差
def test_molecule_bias_z_difference():
    c = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 6.0, "e": NAN, "f": None}
    T = {"a", "b", "e"}
    d, info = TB.molecule_bias(c, T)
    v = np.array([1.0, 2.0, 3.0, 6.0]); z = (v - v.mean()) / v.std(ddof=0)
    assert d == pytest.approx(z[:2].mean() - z[2:].mean())
    assert info == dict(n_scored=4, n_in=2, n_out=2)       # 点なしは数えない


def test_molecule_bias_exclusions():
    assert TB.molecule_bias({"a": 1.0, "b": NAN}, {"a"})[1]["excluded"] == "few_scored"
    assert TB.molecule_bias({"a": 1.0, "b": 1.0}, {"a"})[1]["excluded"] == "sd0"
    assert TB.molecule_bias({"a": 1.0, "b": 2.0}, {"a", "b"})[1]["excluded"] == "one_group"
    assert TB.molecule_bias({"a": 1.0, "b": 2.0}, set())[1]["excluded"] == "one_group"


def test_bias_summary_seed_and_reading():
    rng = np.random.default_rng(7)
    d = {f"m{i}": float(rng.normal(-0.3, 0.2)) for i in range(60)}
    infos = {m: dict(n_in=3, n_out=5) for m in d}
    infos["x"] = dict(excluded="one_group")
    s1, s2 = TB.summarize(d, infos), TB.summarize(d, infos)
    assert s1 == s2 and s1["n"] == 60 and s1["excluded"] == {"one_group": 1}
    x = np.array([d[m] for m in sorted(d)])
    bs = x[np.random.default_rng(0).integers(0, len(x), size=(2000, len(x)))].mean(1)
    assert s1["hi"] == round(float(np.percentile(bs, 97.5)), 4)
    assert TB.reading(s1) == "lower_train"
    d2 = {f"m{i}": float(rng.normal(0.0, 0.2)) for i in range(60)}
    assert TB.reading(TB.summarize(d2, {m: dict(n_in=1, n_out=1) for m in d2})) == "record_only"
    assert TB.reading(TB.summarize({}, {})) == "record_only"
