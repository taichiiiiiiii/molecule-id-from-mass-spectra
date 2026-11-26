#!/usr/bin/env python3
"""
LambdaRank の OOF の点の評価（Issue #247）。型ごと（(a)〜(e)。(a) = C1 型、(c) = C2 型）の MRR@25、
対照 `score,prior,frag` との差、差の 95% 区間（分子単位の対応のある bootstrap 2,000 回、seed 0、百分位）。
型が 1 つでも欠けたら拒む。結果は `<exp-dir>/eval.json` と `docs/experiments/experiments.csv` に 1 行。

    python scripts/ranker/evaluate.py --exp-dir outputs/ranker/lambdarank/<exp_id> [--note "..."]

rr は `mrr25.reciprocal_rank`（metric と同じキー）。分子の候補の並びは OOF の点の降順（同点は表の行の順）、
`cand_key` で重複を除いた上位 25 の SMILES。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from lambdarank_oof import TYPES, HarnessError, set_files  # noqa: E402

K = 25
N_BOOT = 2000
BOOT_SEED = 0
EXPERIMENTS_CSV = ROOT / "docs/experiments/experiments.csv"
BASE_COLS = ["exp_id", "date_utc", "git_commit", "table_sha256", "n_molecules", "n_rows", "n_features",
             "features_sha256", "model", "seed", "train_types"]
TYPE_COLS = ["n", "mrr", "ctrl", "delta", "delta_lo", "delta_hi"]
EXP_COLUMNS = BASE_COLS + [f"{c}_{t}" for t in TYPES for c in TYPE_COLS] + ["note"]


def ranked_lists(oof: pd.DataFrame, k: int = K) -> dict[tuple[str, str], list[str]]:
    """分子ごとに score の降順（同点は row の順）、cand_key で重複を除いた上位 k の SMILES。"""
    d = oof.sort_values(["set", "molecule_id", "score", "row"], ascending=[True, True, False, True], kind="mergesort")
    out: dict[tuple[str, str], list[str]] = {}
    for (s, mid), g in d.groupby(["set", "molecule_id"], sort=False):
        seen, lst = set(), []
        for key, smi in zip(g["cand_key"], g["cand_smiles"]):
            if key in seen:
                continue
            seen.add(key)
            lst.append(smi)
            if len(lst) >= k:
                break
        out[(s, mid)] = lst
    return out


def per_molecule_rr(lists: dict[tuple[str, str], list[str]], truth: dict[tuple[str, str], str]) -> dict:
    """正解のある分子ごとの rr（候補の無い分子は 0）。"""
    from mrr25 import reciprocal_rank, truth_keys
    return {m: reciprocal_rank(truth_keys(t), lists.get(m, [])[:K]) for m, t in truth.items()}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """mean(a − b) の 95% 区間（分子を復元抽出、百分位）。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def evaluate_types(per_mol: pd.DataFrame, required=TYPES) -> dict:
    """per_mol: 列 type, rr, rr_ctrl（1 行 1 分子）。型が 1 つでも欠けたら拒む。"""
    present = set(per_mol["type"].astype(str))
    missing = [t for t in required if t not in present]
    if missing:
        raise HarnessError(f"評価に要る型が欠けている: {missing}")
    extra = sorted(present - set(required))
    if extra:
        raise HarnessError(f"知らない型がある: {extra}")
    out = {}
    for t in required:
        g = per_mol[per_mol["type"] == t]
        lo, hi = paired_bootstrap(g["rr"].to_numpy(), g["rr_ctrl"].to_numpy())
        out[t] = dict(n=int(len(g)), mrr=float(g["rr"].mean()), ctrl=float(g["rr_ctrl"].mean()),
                      delta=float((g["rr"] - g["rr_ctrl"]).mean()), delta_lo=lo, delta_hi=hi)
    return out


def build_per_molecule(oof: pd.DataFrame, eval_root: Path, rows_mode: str = "all", eval_ids: set | None = None,
                       sets: dict | None = None) -> pd.DataFrame:
    """OOF の点と対照の CSV から、分子ごとの rr と rr_ctrl（列 has_rows: OOF に行があるか）。

    - 評価の一覧（eval_ids: "set\tmolecule_id" の集合、#255）が無ければ、評価の分子 = 各集合の truth の全分子
    - rows_mode = "all": OOF の分子 ⊇ 評価の分子でなければ止める（一覧が無ければ今までどおり一致を求める）
    - rows_mode = "pool": 行の無い分子は rr = 0（外れ）として数える（評価から消さない）
    - OOF の分子 ⊆ truth でなければ止める。eval_ids ⊆ truth でなければ止める
    sets（集合 → 型、meta.json の sets）があれば、その集合を全部評価する（OOF に行の無い集合も）。"""
    from mrr25 import read_submission, read_truth
    if rows_mode not in ("all", "pool"):
        raise HarnessError(f"rows_mode が不明: {rows_mode}")
    lists = ranked_lists(oof)
    oof_types = oof.groupby("set", sort=True)["type"].unique()
    for s, ts in oof_types.items():
        if len(ts) != 1:
            raise HarnessError(f"{s}: 型が 1 つでない")
    if sets is None:
        sets = {str(s): str(ts[0]) for s, ts in oof_types.items()}
    for s, ts in oof_types.items():
        if str(s) not in sets or sets[str(s)] != str(ts[0]):
            raise HarnessError(f"{s}: OOF の集合・型が meta の sets と一致しない")
    oof_ids = set((oof["set"].astype(str) + "\t" + oof["molecule_id"].astype(str)).unique())
    all_truth: set = set()
    rows = []
    for s in sorted(sets):
        f = set_files(eval_root, s)
        truth = read_truth(f["truth"])
        all_truth |= {f"{s}\t{m}" for m in truth}
        have = {m for m in truth if f"{s}\t{m}" in oof_ids}
        targets = [m for m in truth if eval_ids is None or f"{s}\t{m}" in eval_ids]
        if eval_ids is None and rows_mode == "all" and have != set(truth):
            raise HarnessError(f"{s}: 正解の分子の一覧と OOF の分子の一覧が一致しない")
        missing = [m for m in targets if m not in have]
        if missing and rows_mode != "pool":
            raise HarnessError(f"{s}: 評価する分子のうち OOF に行の無い分子が {len(missing)}（pool のときだけ rr = 0 で数える）")
        ctrl = read_submission(f["control"])
        tt = {(s, m): truth[m] for m in targets}
        rr = per_molecule_rr(lists, tt)   # 行の無い分子は lists に無い → rr = 0
        rr_c = per_molecule_rr({(s, m): c for m, c in ctrl.items()}, tt)
        for m in targets:
            rows.append(dict(set=s, molecule_id=m, type=sets[s], rr=rr[(s, m)], rr_ctrl=rr_c[(s, m)],
                             has_rows=bool(m in have)))
    if oof_ids - all_truth:
        raise HarnessError(f"OOF に truth に無い分子が {len(oof_ids - all_truth)}")
    if eval_ids is not None:
        if eval_ids - all_truth:
            raise HarnessError(f"評価の一覧に truth に無い分子が {len(eval_ids - all_truth)}")
        if len(rows) != len(eval_ids):
            raise HarnessError("評価の分子の数が一覧と一致しない")
    return pd.DataFrame(rows)


LEDGER_REL = "docs/experiments/experiments.csv"


def git_commit_mark(root: Path = ROOT) -> str:
    """HEAD の sha。作業ツリーに台帳以外の変更（未追跡を含む）があれば末尾に -dirty（#255、#248 の minor 6）。"""
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=True).stdout.strip()
    st = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                        check=True).stdout
    changed = [ln[3:] for ln in st.splitlines() if ln.strip()]
    dirty = [c for c in changed if c.strip('"') != LEDGER_REL]
    return sha + ("-dirty" if dirty else "")


def auto_note(meta: dict) -> str:
    """台帳の note に足す実験の条件（#255）。"""
    parts = [f"rows={meta.get('rows', 'all')}"]
    if meta.get("exclude_features"):
        parts.append("exclude=" + ",".join(meta["exclude_features"]))
    if meta.get("extra_features"):
        parts.append("extra=" + ",".join(meta["extra_features"]))
    for k in ("exclude_train_molecules", "eval_molecules"):
        r = meta.get(k)
        if r:
            parts.append(f"{k}_sha256={r['sha256']}(n={r['n']})")
    return "; ".join(parts)


def experiment_row(exp_id: str, meta: dict, res: dict, note: str, git_commit: str, n_molecules: int) -> dict:
    row = dict(exp_id=exp_id, date_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), git_commit=git_commit,
               table_sha256=meta["table_sha256"], n_molecules=n_molecules, n_rows=meta["n_rows"],
               n_features=len(meta["features"]), features_sha256=meta["features_sha256"],
               model=f"lightgbm {meta['lightgbm']} lambdarank defaults, num_boost_round={meta['num_boost_round']}",
               seed=meta["seed"], train_types="".join(meta["train_types"]), note=note)
    for t, v in res.items():
        for c in TYPE_COLS:
            row[f"{c}_{t}"] = v[c] if c == "n" else f"{v[c]:.4f}"
    return row


def append_experiment(path: Path, row: dict) -> None:
    """experiments.csv に 1 行足す（消さない）。ヘッダーが違えば止める。同じ exp_id があれば止める。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="") as f:
            r = csv.reader(f)
            header = next(r)
            ids = [x[0] for x in r if x]
        if header != EXP_COLUMNS:
            raise HarnessError("experiments.csv のヘッダーが書式と違う")
        if row["exp_id"] in ids:
            raise HarnessError(f"exp_id {row['exp_id']} は既にある")
        mode = "a"
    else:
        mode = "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXP_COLUMNS, lineterminator="\n")   # #255: 行末は LF
        if mode == "w":
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in EXP_COLUMNS})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LambdaRank の OOF の評価（#247）")
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--experiments-csv", default=str(EXPERIMENTS_CSV))
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    d = Path(a.exp_dir)
    meta = json.loads((d / "meta.json").read_text())
    oof = pd.read_parquet(d / "oof.parquet")
    if not np.isfinite(oof["score"].to_numpy()).all():
        raise HarnessError("OOF の点に NaN/inf がある")
    eval_ids = None
    if meta.get("eval_molecules"):
        m = pd.read_csv(meta["eval_molecules"]["path"], dtype=str)
        eval_ids = set(m["set"] + "\t" + m["molecule_id"])
        from lambdarank_oof import sha256_file
        if sha256_file(Path(meta["eval_molecules"]["path"])) != meta["eval_molecules"]["sha256"]:
            raise HarnessError("評価の一覧のファイルが当てはめの時と違う（sha256）")
    rows_mode = meta.get("rows", "all")
    per_mol = build_per_molecule(oof, Path(a.eval_root), rows_mode=rows_mode, eval_ids=eval_ids, sets=meta.get("sets"))
    res = evaluate_types(per_mol)
    commit = git_commit_mark()
    note = "; ".join(x for x in (a.note, auto_note(meta)) if x)
    row = experiment_row(d.name, meta, res, note, commit, int(len(per_mol)))
    no_rows = {t: int((~per_mol.loc[per_mol.type == t, "has_rows"]).sum()) for t in TYPES}
    (d / "eval.json").write_text(json.dumps(dict(exp_id=d.name, by_type=res, n_molecules=int(len(per_mol)),
                                                 n_boot=N_BOOT, boot_seed=BOOT_SEED, git_commit=commit,
                                                 rows=rows_mode, n_molecules_without_rows=no_rows, note=note),
                                            indent=2, ensure_ascii=False))
    per_mol.to_csv(d / "per_molecule.csv", index=False)
    append_experiment(Path(a.experiments_csv), row)
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
