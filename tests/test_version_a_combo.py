"""#282: 表の列の結合・3 seed 平均の差と関門・N1 の反事実の置き換えの単体テスト（合成データだけ）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/features"))

import feature_groups as FG  # noqa: E402
import join_columns as JC  # noqa: E402
import n1_counterfactual as CF  # noqa: E402


# ---- 表の結合 ----
def _base():
    return pd.DataFrame(dict(set=["s1", "s1", "s2", "s2"], molecule_id=["m1", "m1", "m2", "m2"],
                             cand_key=["A", "B", "A", "C"], x=[1.0, 2.0, 3.0, 4.0]))


def _src():
    # 順を入れ替え、余分な行を 1 つ
    return pd.DataFrame(dict(set=["s2", "s1", "s2", "s1", "s9"], molecule_id=["m2", "m1", "m2", "m1", "m9"],
                             cand_key=["C", "B", "A", "A", "Z"], n1=[40.0, 20.0, 30.0, 10.0, 99.0]))


def test_join_keeps_order_and_maps_by_key():
    out, rec = JC.join_columns(_base(), _src(), ["n1"])
    assert list(out["n1"]) == [10.0, 20.0, 30.0, 40.0]
    assert out[["set", "molecule_id", "cand_key", "x"]].equals(_base())
    assert rec["n_rows"] == 4 and rec["n_missing"] == 0 and rec["n_src_unused"] == 1


def test_join_missing_key_stops():
    src = _src().iloc[1:]   # (s2, m2, C) を消す
    with pytest.raises(JC.JoinError, match="結合の欠け"):
        JC.join_columns(_base(), src, ["n1"])


def test_join_duplicate_key_stops():
    src = pd.concat([_src(), _src().iloc[:1]])
    with pytest.raises(JC.JoinError, match="重複"):
        JC.join_columns(_base(), src, ["n1"])
    b = pd.concat([_base(), _base().iloc[:1]])
    with pytest.raises(JC.JoinError, match="重複"):
        JC.join_columns(b, _src(), ["n1"])


def test_join_inf_nan_and_existing_column_stop():
    s = _src()
    s.loc[1, "n1"] = np.inf
    with pytest.raises(JC.JoinError, match="inf"):
        JC.join_columns(_base(), s, ["n1"])
    s.loc[1, "n1"] = np.nan
    with pytest.raises(JC.JoinError, match="NaN"):
        JC.join_columns(_base(), s, ["n1"])
    with pytest.raises(JC.JoinError, match="既にある"):
        JC.join_columns(_base().assign(n1=0.0), _src(), ["n1"])


# ---- 3 seed 平均の差 ----
MOLS = [("s", f"m{i}") for i in range(10)]
TYPES = ["a", "b", "c", "d", "e"] * 2


def _run(tmp: Path, name: str, seed: int, rr, features_sha="F", eval_sha="E"):
    d = tmp / name
    d.mkdir()
    meta = dict(seed=seed, params=dict(seed=seed), table_sha256="T", features_sha256=features_sha, rows="all",
                num_boost_round=100, lightgbm="4.7.0", exclude_features=["p"], extra_features=["cc"],
                train_types=list("abcd"), n_rows=10, exclude_train_molecules=dict(sha256="X"),
                eval_molecules=dict(sha256=eval_sha))
    (d / "meta.json").write_text(json.dumps(meta))
    pd.DataFrame(dict(set=[m[0] for m in MOLS], molecule_id=[m[1] for m in MOLS], type=TYPES, rr=rr,
                      rr_ctrl=0.0, has_rows=True)).to_csv(d / "per_molecule.csv", index=False)
    return d


def test_seed_mean_values_and_gate(tmp_path):
    base = np.array([1.0, 0.5, 0.0, 1.0, 0.2] * 2)
    ctrl = [_run(tmp_path, f"k{s}", s, base) for s in (0, 1, 2)]
    # 候補: 型 c だけ seed ごとに +0.3 / 0 / 0 → 平均の差 +0.1、型 e は −0.06 / −0.03 / 0 → −0.03（< −0.02）
    adds = [dict(c=0.3, e=-0.06), dict(c=0.0, e=-0.03), dict(c=0.0, e=0.0)]
    cand = []
    for s, ad in zip((0, 1, 2), adds):
        rr = base.copy()
        for i, t in enumerate(TYPES):
            rr[i] += ad.get(t, 0.0)
        cand.append(_run(tmp_path, f"c{s}", s, rr, features_sha="G"))
    r = FG.seed_mean_compare(cand, ctrl)
    assert r["types"]["c"]["delta"] == pytest.approx(0.1)
    assert r["types"]["e"]["delta"] == pytest.approx(-0.03)
    assert r["types"]["a"]["delta"] == pytest.approx(0.0)
    assert r["types"]["c"]["cand_mrr_by_seed"] == pytest.approx([0.3, 0.0, 0.0])
    assert r["types"]["e"]["gate_pass"] is False and r["types"]["c"]["gate_pass"] is True
    assert r["gate_pass_all"] is False
    assert r["types"]["a"]["n"] == 2


def test_seed_mean_gate_uses_unrounded_value(tmp_path):
    base = np.full(10, 0.5)
    ctrl = [_run(tmp_path, f"k{s}", s, base) for s in (0, 1, 2)]
    rr = base.copy()
    rr[[4, 9]] -= 0.02004   # 型 e の差 −0.02004（丸めると −0.0200 だが関門は満たさない）
    cand = [_run(tmp_path, f"c{s}", s, rr) for s in (0, 1, 2)]
    r = FG.seed_mean_compare(cand, ctrl)
    assert r["types"]["e"]["gate_pass"] is False


def test_seed_mean_missing_or_wrong_seed_stops(tmp_path):
    base = np.full(10, 0.5)
    ctrl = [_run(tmp_path, f"k{s}", s, base) for s in (0, 1, 2)]
    cand2 = [_run(tmp_path, f"c{s}", s, base) for s in (0, 1)]
    with pytest.raises(RuntimeError, match="2 本"):
        FG.seed_mean_compare(cand2, ctrl)
    dup = cand2 + [_run(tmp_path, "c1b", 1, base)]
    with pytest.raises(RuntimeError, match="seed の組"):
        FG.seed_mean_compare(dup, ctrl)


def test_seed_mean_mismatched_config_or_molecules_stop(tmp_path):
    base = np.full(10, 0.5)
    ctrl = [_run(tmp_path, f"k{s}", s, base) for s in (0, 1, 2)]
    bad = [_run(tmp_path, "c0", 0, base), _run(tmp_path, "c1", 1, base, features_sha="Z"), _run(tmp_path, "c2", 2, base)]
    with pytest.raises(RuntimeError, match="features_sha256"):
        FG.seed_mean_compare(bad, ctrl)
    ev = [_run(tmp_path, f"e{s}", s, base, eval_sha="OTHER") for s in (0, 1, 2)]
    with pytest.raises(RuntimeError, match="eval_molecules"):
        FG.seed_mean_compare(ev, ctrl)
    cand = [_run(tmp_path, f"m{s}", s, base) for s in (0, 1, 2)]
    pm = pd.read_csv(cand[2] / "per_molecule.csv").iloc[:-1]
    pm.to_csv(cand[2] / "per_molecule.csv", index=False)
    with pytest.raises(RuntimeError, match="分子の集合"):
        FG.seed_mean_compare(cand, ctrl)


# ---- N1 の反事実 ----
def test_replace_truth_with_decoy_median():
    df = pd.DataFrame(dict(set="s", molecule_id=["m1"] * 4 + ["m2"] * 2 + ["m3"],
                           nbr_ms_mean=[0.9, 0.1, 0.2, 0.6, 0.8, 0.3, 0.7]))
    y = np.array([1, 0, 0, 0, 0, 1, 1])
    target = np.array([True] * 6 + [True])
    out, n_rep, n_skip = CF.replace_truth_with_decoy_median(df, y, target)
    assert out["nbr_ms_mean"].tolist() == [0.2, 0.1, 0.2, 0.6, 0.8, 0.8, 0.7]   # m3 は偽物が無いので変えない
    assert (n_rep, n_skip) == (2, 1)
    assert df["nbr_ms_mean"].tolist()[0] == 0.9   # 元は変えない


def test_replace_only_target_molecules():
    df = pd.DataFrame(dict(set="s", molecule_id=["m1", "m1", "m2", "m2"], nbr_ms_mean=[0.9, 0.1, 0.8, 0.2]))
    y = np.array([1, 0, 1, 0])
    out, n_rep, _ = CF.replace_truth_with_decoy_median(df, y, np.array([False, False, True, True]))
    assert out["nbr_ms_mean"].tolist() == [0.9, 0.1, 0.2, 0.2] and n_rep == 1
