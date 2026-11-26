"""#255 dev の台帳の行の作り方（列の除外・追加、PubChem 層の行の除外、分子の一覧、dirty の印）の単体テスト（合成データだけ）。"""
from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))

import evaluate as EV  # noqa: E402
import lambdarank_oof as LR  # noqa: E402

_spec = importlib.util.spec_from_file_location("_h247", ROOT / "tests/test_lambdarank_harness.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


def tier_table(seed: int = 0):
    """合成の表に is_tier を付ける: 各分子の後ろ 3 候補を Q（is_tier = 1）。型 b・c の各集合の最初の分子は全候補を Q
    （プールの行が 0 の分子）にし、型 c の 2 番目の分子は正解の候補だけを Q にする。"""
    df, truth = H.synth_table(n_mol=5, n_cand=8, seed=seed)
    df = df.copy()
    pos = df.groupby(["set", "molecule_id"]).cumcount().to_numpy()
    df["is_tier"] = (pos >= 5).astype(float)
    for s in df["set"].unique():
        t = s.split("_")[2]
        mids = sorted(df.loc[df.set == s, "molecule_id"].unique())
        if t in ("b", "c"):
            df.loc[(df.set == s) & (df.molecule_id == mids[0]), "is_tier"] = 1.0
        if t == "c":
            m = mids[1]
            sel = (df.set == s) & (df.molecule_id == m)
            df.loc[sel, "is_tier"] = (df.loc[sel, "cand_smiles"] == truth[s][m]).astype(float)
    return df, truth


def _write(tmp_path, df, truth, name="t.parquet"):
    p = tmp_path / name
    df.to_parquet(p, index=False)
    ev = tmp_path / "ev"
    if not ev.exists():
        H.write_eval_root(ev, df, truth)
    return p, ev


# 1. 外した列が学習に入らない
def test_excluded_features_not_in_model(tmp_path):
    df, truth = H.synth_table(n_mol=4)
    p, ev = _write(tmp_path, df, truth)
    feats = LR.resolve_features("prior_unverified_pop", ())
    assert "prior_unverified_pop" not in feats and len(feats) == 62
    with pytest.raises(LR.HarnessError):
        LR.resolve_features("not_a_feature", ())
    with pytest.raises(LR.HarnessError):
        LR.resolve_features((), "unknown_extra")
    out = tmp_path / "o" / "e1"
    meta = LR.run_fit(p, ev, out, seed=0, approval="test", exclude_features="prior_unverified_pop")
    assert meta["features"] == list(feats) and meta["features_sha256"] == LR.features_sha256(feats)
    assert meta["features_sha256"] != LR.features_sha256()
    for k in LR.FOLDS:
        names = (out / f"model_fold{k}.txt").read_text().split("feature_names=")[1].split("\n")[0].split()
        assert names == list(feats) and "prior_unverified_pop" not in names


# 2. 登録外の余分な列で止まる／求めた追加の列が無ければ止まる／登録済みで求めていない列は止めず学習にも入らない
def test_extra_columns_registry(tmp_path):
    df, truth = H.synth_table(n_mol=4)
    rng = np.random.default_rng(1)
    dfc = df.assign(cc=rng.normal(size=len(df)), cc_z=rng.normal(size=len(df)))
    assert set(LR.EXTRA_ALLOWED) >= {"cc", "glacier"} and {"cc", "cc_z"} <= set(LR.EXTRA_ALL)
    with pytest.raises(LR.HarnessError, match="列が定義と一致しない"):
        LR.check_columns(dfc.assign(zzz=1.0).columns, ("cc", "cc_z"))
    with pytest.raises(LR.HarnessError, match="列が定義と一致しない"):
        LR.check_columns(df.columns, ("cc",))          # 求めた cc が表に無い
    rep = LR.check_columns(dfc.columns)                # 登録済みで求めていない → 止めない
    assert rep["ignored_registered"] == ["cc", "cc_z"]
    with pytest.raises(LR.HarnessError, match="入れない列"):
        LR.check_columns(dfc.assign(msg_overlap=0).columns, ("cc", "cc_z"))
    p, ev = _write(tmp_path, dfc, truth)
    m0 = LR.run_fit(p, ev, tmp_path / "o" / "e0", seed=0, approval="test")
    assert "cc" not in m0["features"] and m0["ignored_registered_columns"] == ["cc", "cc_z"]
    m1 = LR.run_fit(p, ev, tmp_path / "o" / "e1", seed=0, approval="test", extra_features="cc,cc_z")
    assert m1["features"][-2:] == ["cc", "cc_z"] and len(m1["features"]) == 65 and m1["ignored_registered_columns"] == []
    names = (tmp_path / "o" / "e1" / "model_fold1.txt").read_text().split("feature_names=")[1].split("\n")[0].split()
    assert names[-2:] == ["cc", "cc_z"]


# 3. pool: is_tier = 1 の行が学習・予測に 0、行の無い分子は rr = 0、n は truth の数のまま。all で行の無い分子は止まる
def test_pool_rows_and_zero_rr(tmp_path):
    df, truth = tier_table()
    p, ev = _write(tmp_path, df, truth)
    out = tmp_path / "o" / "pool"
    meta = LR.run_fit(p, ev, out, seed=0, approval="test", rows="pool")
    oof = pd.read_parquet(out / "oof.parquet")
    tab = pd.read_parquet(p)
    key = ["set", "molecule_id", "cand_key"]
    joined = oof.merge(tab[key + ["is_tier"]], on=key, how="left", validate="one_to_one")
    assert len(oof) == int((tab.is_tier == 0).sum()) and (joined.is_tier == 0).all()
    assert meta["rows"] == "pool" and meta["n_rows"] == len(oof) and meta["n_rows_table"] == len(tab)
    assert meta["n_molecules_without_rows_by_type"] == {"b": 4, "c": 4}
    per_mol = EV.build_per_molecule(oof, ev, rows_mode="pool", sets=meta["sets"])
    n_truth = sum(len(v) for v in truth.values())
    assert len(per_mol) == n_truth
    no = per_mol[~per_mol.has_rows]
    assert len(no) == 8 and (no.rr == 0).all() and set(no.type) == {"b", "c"}
    # 型 c の 2 番目の分子は正解が Q だけ → pool では外れ（rr = 0）
    for s in [x for x in truth if "_c_" in x]:
        m = sorted(truth[s])[1]
        assert float(per_mol[(per_mol.set == s) & (per_mol.molecule_id == m)].rr.iloc[0]) == 0.0
    res = EV.evaluate_types(per_mol)
    assert {t: res[t]["n"] for t in LR.TYPES} == {t: 20 for t in LR.TYPES}
    with pytest.raises(LR.HarnessError):
        EV.build_per_molecule(oof, ev, rows_mode="all", sets=meta["sets"])
    ecsv = tmp_path / "experiments.csv"
    EV.main(["--exp-dir", str(out), "--eval-root", str(ev), "--experiments-csv", str(ecsv), "--note", "syn"])
    row = list(csv.DictReader(open(ecsv)))[0]
    assert row["note"].startswith("syn; rows=pool")
    assert all(int(row[f"n_{t}"]) == 20 for t in LR.TYPES)


# 4. 学習から外した分子が学習の群に 0、評価の分子が一覧と一致
def test_exclude_train_and_eval_molecules(tmp_path):
    df, truth = H.synth_table(n_mol=5)
    p, ev = _write(tmp_path, df, truth)
    g = sorted(set(LR.group_ids(df)))
    excl = g[::3]
    ex_csv = tmp_path / "excl.csv"
    pd.DataFrame([x.split("\t") for x in excl], columns=["set", "molecule_id"]).to_csv(ex_csv, index=False)
    ev_ids = [x for x in g if x not in set(excl)]
    ev_csv = tmp_path / "eval.csv"
    pd.DataFrame([x.split("\t") for x in ev_ids], columns=["set", "molecule_id"]).to_csv(ev_csv, index=False)
    for k in LR.FOLDS:
        train, pred = LR.fold_masks(df, k, exclude_train=set(excl))
        gg = LR.group_ids(df)
        assert not (set(gg[train]) & set(excl))
        assert set(gg[pred]) == set(gg[df.query_fold.to_numpy() == k])   # 予測には残る
    out = tmp_path / "o" / "ex"
    meta = LR.run_fit(p, ev, out, seed=0, approval="test", exclude_train_molecules=ex_csv, eval_molecules=ev_csv)
    assert meta["exclude_train_molecules"]["n"] == len(excl) and meta["eval_molecules"]["n"] == len(ev_ids)
    assert meta["exclude_train_molecules"]["sha256"] == LR.sha256_file(ex_csv)
    ecsv = tmp_path / "experiments.csv"
    EV.main(["--exp-dir", str(out), "--eval-root", str(ev), "--experiments-csv", str(ecsv)])
    pm = pd.read_csv(out / "per_molecule.csv", dtype={"molecule_id": str})
    assert set(pm.set + "\t" + pm.molecule_id) == set(ev_ids)
    row = list(csv.DictReader(open(ecsv)))[0]
    assert sum(int(row[f"n_{t}"]) for t in LR.TYPES) == len(ev_ids)
    assert LR.sha256_file(ev_csv) in row["note"] and LR.sha256_file(ex_csv) in row["note"]
    bad = tmp_path / "bad.csv"
    pd.DataFrame([["b8_f1_a_t10", "nope"]], columns=["set", "molecule_id"]).to_csv(bad, index=False)
    with pytest.raises(LR.HarnessError):
        LR.run_fit(p, ev, tmp_path / "o" / "bad1", seed=0, approval="test", eval_molecules=bad)
    with pytest.raises(LR.HarnessError):
        LR.run_fit(p, ev, tmp_path / "o" / "bad2", seed=0, approval="test", exclude_train_molecules=bad)


# 5. dirty の印（台帳だけの変更なら付かない）と CSV の行末 LF
def test_dirty_mark_and_lf(tmp_path):
    r = tmp_path / "repo"
    (r / "docs/experiments").mkdir(parents=True)
    run = lambda *a: subprocess.run(["git", "-C", str(r), *a], check=True, capture_output=True, text=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (r / "a.py").write_text("x = 1\n")
    (r / EV.LEDGER_REL).write_text("h\n")
    run("add", "-A")
    run("commit", "-qm", "init")
    sha = run("rev-parse", "HEAD").stdout.strip()
    assert EV.git_commit_mark(r) == sha
    (r / EV.LEDGER_REL).write_text("h\nrow\n")
    assert EV.git_commit_mark(r) == sha
    (r / "a.py").write_text("x = 2\n")
    assert EV.git_commit_mark(r) == sha + "-dirty"
    run("checkout", "-q", "a.py")
    (r / "new.py").write_text("y = 1\n")
    assert EV.git_commit_mark(r) == sha + "-dirty"
    ecsv = tmp_path / "e.csv"
    EV.append_experiment(ecsv, {c: "1" for c in EV.EXP_COLUMNS} | {"exp_id": "a"})
    EV.append_experiment(ecsv, {c: "1" for c in EV.EXP_COLUMNS} | {"exp_id": "b"})
    b = ecsv.read_bytes()
    assert b"\r" not in b and b.count(b"\n") == 3
    assert b"\r" not in (ROOT / EV.LEDGER_REL).read_bytes()
