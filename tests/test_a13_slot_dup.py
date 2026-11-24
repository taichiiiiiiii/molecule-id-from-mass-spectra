"""#230: A13（25 枠の重複）の数え方と run.py --dump-ranks の深い順位。"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import a13_slot_dup as A  # noqa: E402

_spec = importlib.util.spec_from_file_location("local_eval_run_a13", ROOT / "scripts/local_eval/run.py")
le = importlib.util.module_from_spec(_spec)
sys.modules["local_eval_run_a13"] = le
_spec.loader.exec_module(le)

# 作り物: 立体違い 2 つ・互変異性体 2 つ・別の分子
S_R, S_S = "C[C@@H](O)CCN", "C[C@H](O)CCN"          # 立体だけ違う（同じ答え）
T_OH, T_NH = "Oc1ccccn1", "O=c1cccc[nH]1"           # 2-ヒドロキシピリジン / 2-ピリドン（互変異性体、同じ答え）
OTHER = "CCCCCCO"                                    # 別の分子


def test_synthetic_keys_match_metric_units():
    assert A.to_key(S_R) == A.to_key(S_S)
    assert A.to_key(T_OH) == A.to_key(T_NH)
    assert len({A.to_key(S_R), A.to_key(T_OH), A.to_key(OTHER)}) == 3


def test_dup_flags_none_never_duplicates():
    assert A.dup_flags(["a", "b", "a", None, None, "b"]) == [False, False, True, False, False, True]


def test_molecule_stats_dup_above_truth():
    cands = [S_R, S_S, T_OH, T_NH, OTHER]          # 正解 OTHER は 5 位、上に重複 2
    st = A.molecule_stats(cands, A.truth_keys(OTHER))
    assert st["dup_top"] == 2 and st["rank"] == 5 and st["dup_above"] == 2 and st["rank_dedup"] == 3
    assert st["found"] is True and st["none_top"] == 0


def test_molecule_stats_no_dup_above_when_truth_first():
    st = A.molecule_stats([OTHER, S_R, S_S], A.truth_keys(OTHER))
    assert st["dup_top"] == 1 and st["rank"] == 1 and st["dup_above"] == 0


def test_molecule_stats_enters_top_k_after_dedup():
    # k=3: 上位 3 = S_R, S_S, T_OH（重複 1）。正解 OTHER は 5 位 → 重複除去で 4 位（まだ外）、k=4 なら入る
    cands = [S_R, S_S, T_OH, T_NH, OTHER]
    st3 = A.molecule_stats(cands, A.truth_keys(OTHER), k=3)
    assert st3["rank"] == 5 and st3["rank_dedup"] == 3   # 26 位以降の重複（T_NH）も数える
    s = A.summarize([dict(molecule_id="m", **st3)], k=3)
    assert s["mol_enter_top25_after_dedup"] == 1 and s["mol_dup_above_truth"] == 0 and s["truth_deep_only"] == 1


def test_truth_absent_and_submission_only():
    st = A.molecule_stats([S_R, S_S], A.truth_keys(OTHER))
    assert st["found"] is False and st["rank"] is None
    st2 = A.molecule_stats([S_R, S_S, T_OH], set())
    assert st2["found"] is None and st2["dup_top"] == 1
    s = A.summarize([dict(molecule_id="a", **st2), dict(molecule_id="b", **A.molecule_stats([OTHER], set()))])
    assert s["mol_with_dup"] == 1 and s["mol_dup_above_truth"] is None
    lo, hi = s["mol_with_dup_wilson95"]
    assert 0 < lo < 0.5 < hi < 1


def test_wilson_known_value():
    lo, hi = A.wilson(400, 400)
    assert abs(lo - 0.9905) < 1e-4 and hi == 1.0


def test_cli_rank_file_and_sub_check(tmp_path):
    ranks = tmp_path / "r.csv"
    pd.DataFrame(dict(molecule_id=["m1"] * 3 + ["m2"] * 2, rank=[1, 2, 3, 1, 2], pool_row=range(5), key=list("abcde"),
                      smiles=[S_R, S_S, OTHER, T_OH, T_NH])).to_csv(ranks, index=False)
    truth = tmp_path / "t.csv"
    pd.DataFrame(dict(molecule_id=["m1", "m2"], smiles=[OTHER, "CCCCCCCCN"])).to_csv(truth, index=False)
    sub = tmp_path / "s.csv"
    pd.DataFrame(dict(molecule_id=["m1", "m2"], smiles=[f"{S_R};{S_S};{OTHER}", f"{T_OH};{T_NH}"])).to_csv(sub, index=False)
    cmd = [sys.executable, str(ROOT / "scripts/research/a13_slot_dup.py"), "--set", "x", "--ranks", str(ranks),
           "--truth", str(truth), "--sub", str(sub), "--out-dir", str(tmp_path / "o")]
    out = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stdout)
    assert out["mol_with_dup"] == 2 and out["dup_total"] == 2 and out["mol_dup_above_truth"] == 1 and out["truth_absent"] == 1
    bad = tmp_path / "b.csv"
    pd.DataFrame(dict(molecule_id=["m1", "m2"], smiles=[f"{S_S};{S_R};{OTHER}", f"{T_OH};{T_NH}"])).to_csv(bad, index=False)
    r = subprocess.run(cmd[:-6] + ["--truth", str(truth), "--sub", str(bad), "--out-dir", str(tmp_path / "o")],
                       capture_output=True, text=True)
    assert r.returncode == 2


def make_pool():
    meta = pd.DataFrame(dict(key=["K1", "K2", "K2", "K3", "K4"], smiles=["a", "b", "b2", "c", "d"],
                             mass=[100.0, 100.0, 100.0, 100.0, 100.0], pid=[0, 1, 2, 3, 4]))
    return le.Pool(meta, np.zeros(5))


def test_deep_list_head_then_score_order_without_repeated_keys(tmp_path):
    p = make_pool()
    ki = {k: i for i, k in reversed(list(enumerate(p.key)))}
    score_rows = [(9.0, ki["K2"]), (8.0, ki["K2"] + 1), (7.0, ki["K1"]), (6.0, ki["K4"]), (5.0, ki["K3"])]
    head = [(1.0, ki["K1"]), (0.5, ki["K2"])]       # prior・frag の後の並び（上位 TOPN）
    deep = le.deep_list(head, score_rows, p)
    assert [p.key[i] for _, i in deep] == ["K1", "K2", "K4", "K3"]
    out = tmp_path / "d.csv"
    le.write_ranks(out, {"m": deep}, p)
    d = pd.read_csv(out)
    assert d["rank"].tolist() == [1, 2, 3, 4] and d["key"].tolist() == ["K1", "K2", "K4", "K3"]


def test_dump_ranks_refuses_pubchem():
    import pytest
    with pytest.raises(ValueError, match="dump-ranks"):
        le.run("x", ["score", "pubchem"], dump_ranks="y.csv")
