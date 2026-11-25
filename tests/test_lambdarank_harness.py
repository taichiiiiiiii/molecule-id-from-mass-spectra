"""#247 LambdaRank の学習・評価コードの単体テスト（合成データだけ。実データの表は読まない）。"""
from __future__ import annotations

import csv
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
from mrr25 import calculate_mrr25, read_submission, read_truth  # noqa: E402

SMILES = [f"{'C' * n}{x}" for n in range(1, 13) for x in ("O", "N", "Cl", "F", "S")]  # 60 個の別の構造


def synth_table(n_mol: int = 6, n_cand: int = 8, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """B8 と同じ形（20 集合 = fold 1〜4 × 型 a〜e）の合成の特徴表と、集合ごとの正解。

    正解の構造は fold ごとに別の範囲から取る（fold の間で重ならない）。正解の候補は fz が少し高い。
    """
    rng = np.random.default_rng(seed)
    rows, truth = [], {}
    for k in LR.FOLDS:
        pool = SMILES[(k - 1) * 15:k * 15]
        for t in LR.TYPES:
            s = f"b8_f{k}_{t}_t10"
            truth[s] = {}
            for m in range(n_mol):
                mid = f"m{k}{t}_{m:03d}"
                cands = list(rng.choice(pool, size=n_cand, replace=False))
                ti = int(rng.integers(n_cand))
                truth[s][mid] = cands[ti]
                for j, smi in enumerate(cands):
                    r = {"set": s, "type": t, "molecule_id": mid, "query_fold": k, "model_fold": k, "np_model_fold": k,
                         "cand_key": f"K{smi}", "cand_smiles": smi}
                    for f in LR.FEATURES:
                        r[f] = float(rng.normal())
                    r["fz"] = float(rng.normal() + (2.0 if j == ti else 0.0))
                    r["ctrl_rank"] = j + 1
                    rows.append(r)
    df = pd.DataFrame(rows)[list(LR.ID_COLS) + list(LR.FEATURES)]
    return df, truth


def write_eval_root(root: Path, df: pd.DataFrame, truth: dict) -> None:
    """集合ごとの truth.csv と対照の submission（ctrl_rank の順）。"""
    for s, sub in df.groupby("set"):
        d = root / s
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "truth.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["molecule_id", "inchikey14", "smiles"])
            for mid, smi in truth[s].items():
                w.writerow([mid, "X", smi])
        with open(d / "submission_score+prior+frag.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["molecule_id", "smiles"])
            for mid, g in sub.groupby("molecule_id", sort=True):
                w.writerow([mid, ";".join(g.sort_values("ctrl_rank").cand_smiles)])


def truth_by_group(truth: dict) -> dict:
    return {f"{s}\t{m}": t for s, d in truth.items() for m, t in d.items()}


# 1. 学習と予測に同じ分子がまたがらない
def test_no_molecule_in_both_train_and_pred():
    df, _ = synth_table()
    g = LR.group_ids(df)
    for k in LR.FOLDS:
        train, pred = LR.fold_masks(df, k)
        assert train.any() and pred.any()
        assert not (set(g[train]) & set(g[pred]))
        assert not (train & pred).any()
    # 1 つの分子が 2 つの fold にまたがる表は止める
    bad = df.copy()
    i = bad.index[(bad.molecule_id == bad.molecule_id.iloc[0])][0]
    bad.loc[i, ["query_fold", "model_fold", "np_model_fold"]] = 2
    with pytest.raises(LR.HarnessError):
        LR.check_rows(bad)


# 2. fold の境界が入力の fold 列と一致し、型 (e) は学習に入らない
def test_fold_boundaries_match_input_column():
    df, truth = synth_table()
    qf = df.query_fold.to_numpy()
    for k in LR.FOLDS:
        train, pred = LR.fold_masks(df, k)
        assert np.array_equal(pred, qf == k)
        assert np.array_equal(train, (qf != k) & (df.type != "e").to_numpy())
    y = LR.make_labels(df, truth_by_group(truth))
    _, _, info = LR.fit_predict_oof(df, y, seed=0)
    for k in LR.FOLDS:
        assert info[k]["n_pred_rows"] == int((qf == k).sum())
        assert info[k]["n_train_rows"] == int(((qf != k) & (df.type != "e").to_numpy()).sum())
    # fold の値が 1〜4 でない・model_fold が違う表は止める
    bad = df.copy()
    bad.loc[bad.query_fold == 4, ["query_fold", "model_fold", "np_model_fold"]] = 0
    with pytest.raises(LR.HarnessError):
        LR.check_rows(bad)
    bad = df.copy()
    bad.loc[0, "model_fold"] = 3
    with pytest.raises(LR.HarnessError):
        LR.check_rows(bad)


# 3. 同じ seed で同じ点
def test_same_seed_same_scores():
    df, truth = synth_table()
    y = LR.make_labels(df, truth_by_group(truth))
    assert int(y.sum()) == df.groupby(["set", "molecule_id"]).ngroups  # 分子ごとに正解 1 つ
    a, _, _ = LR.fit_predict_oof(df, y, seed=0)
    b, _, _ = LR.fit_predict_oof(df, y, seed=0)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert LR.lgb_params(7) == {"objective": "lambdarank", "seed": 7, "verbose": -1}
    assert LR.NUM_BOOST_ROUND == 100


# 4. 型が 1 つでも欠けたら評価が拒む
def test_evaluate_rejects_missing_type():
    pm = pd.DataFrame(dict(type=list("abcde") * 4, rr=np.linspace(0, 1, 20), rr_ctrl=np.linspace(1, 0, 20)))
    res = EV.evaluate_types(pm)
    assert set(res) == set("abcde")
    for t in "abcde":
        assert res[t]["delta_lo"] <= res[t]["delta"] <= res[t]["delta_hi"]
    assert EV.evaluate_types(pm) == res  # bootstrap の seed は固定
    with pytest.raises(LR.HarnessError):
        EV.evaluate_types(pm[pm.type != "c"])


# 5. 入れない列が表にあれば止まる（列の欠け・知らない列も止まる）
@pytest.mark.parametrize("col", ["used_b8", "msg_overlap", "y", "is_truth", "src", "in_train"])
def test_forbidden_column_stops(col, tmp_path):
    df, truth = synth_table(n_mol=2)
    assert LR.check_columns(df.columns)["n_features"] == 63 and len(df.columns) == 71
    bad = df.assign(**{col: 0})
    with pytest.raises(LR.HarnessError, match="入れない列"):
        LR.check_columns(bad.columns)
    p = tmp_path / "t.parquet"
    bad.to_parquet(p, index=False)
    write_eval_root(tmp_path / "ev", df, truth)
    with pytest.raises(LR.HarnessError):
        LR.check_only(p, tmp_path / "ev")
    with pytest.raises(LR.HarnessError):
        LR.check_columns(df.drop(columns=["fz_z"]).columns)
    with pytest.raises(LR.HarnessError):
        LR.check_columns(df.assign(new_feat=1.0).columns)


# 6. MRR@25 が metric の実装と一致する（端から端まで: check-only → 当てはめ → 評価 → experiments.csv）
def test_mrr_matches_metric_end_to_end(tmp_path):
    df, truth = synth_table()
    tab = tmp_path / "dev.parquet"
    df.to_parquet(tab, index=False)
    ev = tmp_path / "ev"
    write_eval_root(ev, df, truth)
    rep = LR.check_only(tab, ev)
    assert rep["columns"]["missing"] == [] and rep["rows"]["n_molecules"] == 20 * 6
    with pytest.raises(SystemExit):
        LR.main(["--table", str(tab), "--eval-root", str(ev), "--exp-id", "x", "--out-root", str(tmp_path / "o")])
    out = tmp_path / "o" / "e000"
    LR.run_fit(tab, ev, out, seed=0, approval="test")
    oof = pd.read_parquet(out / "oof.parquet")
    per_mol = EV.build_per_molecule(oof, ev)
    lists = EV.ranked_lists(oof)
    for s in sorted(truth):
        sub = tmp_path / f"sub_{s}.csv"
        with open(sub, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["molecule_id", "smiles"])
            for mid in truth[s]:
                w.writerow([mid, ";".join(lists[(s, mid)])])
        t = read_truth(ev / s / "truth.csv")
        got = per_mol[per_mol.set == s]
        assert got.rr.mean() == pytest.approx(calculate_mrr25(t, read_submission(sub)), abs=1e-12)
        ctrl = read_submission(ev / s / "submission_score+prior+frag.csv")
        assert got.rr_ctrl.mean() == pytest.approx(calculate_mrr25(t, ctrl), abs=1e-12)
    res = EV.evaluate_types(per_mol)
    assert per_mol.rr.mean() > per_mol.rr_ctrl.mean() + 0.1  # 合成データでは正解の fz を高くしてある（ラベルが学習に届く）
    for t in LR.TYPES:
        g = per_mol[per_mol.type == t]
        assert res[t]["mrr"] == pytest.approx(g.rr.mean())
    ecsv = tmp_path / "experiments.csv"
    EV.main(["--exp-dir", str(out), "--eval-root", str(ev), "--experiments-csv", str(ecsv), "--note", "synthetic"])
    rows = list(csv.DictReader(open(ecsv)))
    assert len(rows) == 1 and list(rows[0]) == EV.EXP_COLUMNS and rows[0]["exp_id"] == "e000"
    with pytest.raises(LR.HarnessError):  # 同じ exp_id は足さない
        EV.main(["--exp-dir", str(out), "--eval-root", str(ev), "--experiments-csv", str(ecsv)])
    with pytest.raises(LR.HarnessError):  # 実験の出力は上書きしない
        LR.run_fit(tab, ev, out, seed=0, approval="test")


def test_committed_experiments_csv_header():
    with open(ROOT / "docs/experiments/experiments.csv", newline="") as f:
        assert next(csv.reader(f)) == EV.EXP_COLUMNS
