#!/usr/bin/env python3
"""
ランカー（model_c2p.txt、Issue #32 / #34）で提出ファイルを作る。

評価と同じコード経路をそのまま通す（複製しない）: test.parquet を「隠す物が無いホールドアウト」として
data/holdouts/<name>/ に置き（hide_rule = all・drop_keys 空・pool keep、truth.csv はダミー）、
  1. scripts/cv/local_fold0_logits.py   （z = fpnet_fold0、評価と同じモデル）
  2. scripts/local_eval/run.py --stages score,prior,lib,pubchem   （候補・参照・PubChem 窓のキャッシュ）
  3. scripts/local_eval/run.py --stages score,prior               （予備の並び）
  4. scripts/ranker/ranker_features.py
  5. LightGBM で並べ替え → キーで重複除去した上位 25（train_eval.ranked_lists と同じ）
予備: 特徴量の無い分子は score+prior の並び、それも空なら 'CCO'。

    python scripts/submit/ranker_submission.py --test data/test.parquet --sample data/sample_submission.csv --out outputs/submit/submission.csv
    # 経路の一致確認（#34 手順 3）: v3_c1 の問い合わせ・隠し方・正解を写した別名ホールドアウトで同じ処理を通し、特徴量表を比べる
    python scripts/submit/ranker_submission.py --like v3_c1 --name chk_v3_c1 --out outputs/submit/chk_v3_c1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
ENVEDA = ["enveda-180", "enveda-np-examples"]
FALLBACK = "CCO"


def clear_caches(name: str) -> None:
    """同じ名前で別の test を流したときに古いキャッシュ（参照・PubChem 窓・ロジット・特徴量）を使わないよう消す。"""
    for d in (ROOT / f"outputs/local_eval/{name}", ROOT / f"outputs/holdout_eval/{name}_fold0"):
        if d.exists():
            shutil.rmtree(d)
    f = ROOT / f"outputs/ranker/features/{name}.parquet"
    if f.exists():
        f.unlink()


def fill_and_validate(path: Path, ids: list[str]) -> dict:
    """予備の提出ファイル用: 空の行を CCO で埋め、id を揃え、形式を確かめて上書きする。"""
    from mrr25 import validate_submission
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    got = {r[0]: r[1] for r in rows[1:] if len(r) == 2}
    lists = {m: [x for x in got.get(m, "").split(";") if x] for m in ids}
    counts = write_submission(path, ids, {}, lists)
    with open(path, newline="") as f:
        validate_submission(list(csv.reader(f)), expected_ids=set(ids))
    return counts


def prepare(name: str, test: Path | None, sample: Path | None, like: str | None) -> None:
    hd, td = ROOT / f"data/holdouts/{name}", ROOT / f"outputs/holdouts/{name}"
    hd.mkdir(parents=True, exist_ok=True); td.mkdir(parents=True, exist_ok=True)
    if like:
        src = ROOT / f"data/holdouts/{like}"
        for f in ("test.parquet", "drop_keys.txt", "hide_rule.json", "sample_submission.csv"):
            if (src / f).is_file():
                shutil.copy(src / f, hd / f)
        shutil.copy(ROOT / f"outputs/holdouts/{like}/truth.csv", td / "truth.csv")
        return
    shutil.copy(test, hd / "test.parquet")
    shutil.copy(sample, hd / "sample_submission.csv")
    (hd / "drop_keys.txt").write_text("")
    (hd / "hide_rule.json").write_text(json.dumps(dict(rule="all", enveda_libs=ENVEDA, pool="keep", note="提出用: 何も隠さない"), indent=2))
    ids = pd.read_csv(sample).molecule_id
    pd.DataFrame(dict(molecule_id=ids, inchikey14="", smiles=FALLBACK)).to_csv(td / "truth.csv", index=False)  # ダミー（MRR は無意味）


def step(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True)


def build_lists(name: str, model_path: Path) -> tuple[dict, pd.DataFrame]:
    import lightgbm as lgb
    from ranker_features import FEATURES
    from train_eval import ranked_lists
    ft = pd.read_parquet(ROOT / f"outputs/ranker/features/{name}.parquet")
    model = lgb.Booster(model_file=str(model_path))
    ft["ranker"] = model.predict(ft[FEATURES])
    return ranked_lists(ft, "ranker"), ft


def write_submission(out: Path, ids: list[str], lists: dict, fallback: dict) -> dict:
    n_rank = n_fb = n_cco = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
        for m in ids:
            lst = lists.get(m) or []
            if lst:
                n_rank += 1
            else:
                lst = fallback.get(m) or []
                n_fb += bool(lst)
            if not lst:
                lst, n_cco = [FALLBACK], n_cco + 1
            w.writerow([m, ";".join(lst[:25])])
    return dict(ranker=n_rank, fallback_score_prior=n_fb, fallback_cco=n_cco)


def diagnostics(ft: pd.DataFrame) -> dict:
    """異常検知用（判断には使わない）: 1 位の出所、lib_max の分位点、lib_sim の NaN 割合、fz 上位の分布。"""
    top = ft.sort_values(["molecule_id", "ranker"], ascending=[True, False]).groupby("molecule_id").head(1)
    mol = ft.groupby("molecule_id").agg(lib_max=("lib_max", "first"), fz_max=("fz", "max"))
    q = lambda s: {p: round(float(s.quantile(p / 100)), 3) for p in (10, 50, 90)}
    return dict(molecules=int(ft.molecule_id.nunique()), top1_pubchem=round(float(top.src_pubchem.mean()), 3),
                lib_max_q=q(mol.lib_max), lib_max_ge_0_7=round(float((mol.lib_max >= 0.7).mean()), 3),
                lib_sim_nan=round(float(ft.lib_sim.isna().mean()), 3), fz_max_q=q(mol.fz_max))


def main() -> None:
    from mrr25 import read_submission, validate_submission
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=ROOT / "data/test.parquet")
    ap.add_argument("--sample", type=Path, default=ROOT / "data/sample_submission.csv")
    ap.add_argument("--like", default=None, help="既存ホールドアウトの写しで経路を確かめる")
    ap.add_argument("--name", default="sub_test")
    ap.add_argument("--model", type=Path, default=ROOT / "outputs/ranker/model_c2p.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/submit/submission.csv")
    ap.add_argument("--skip-steps", action="store_true", help="1〜4 を飛ばす（キャッシュ済みのとき）")
    args = ap.parse_args()
    if not args.skip_steps:
        clear_caches(args.name)
    prepare(args.name, args.test, args.sample, args.like)
    if not args.skip_steps:
        step(["scripts/cv/local_fold0_logits.py", "--holdout", args.name])
        step(["scripts/local_eval/run.py", "--holdout", args.name, "--stages", "score,prior,lib,pubchem"])
        step(["scripts/local_eval/run.py", "--holdout", args.name, "--stages", "score,prior"])
        step(["scripts/ranker/ranker_features.py", "--holdout", args.name])
    lists, ft = build_lists(args.name, args.model)
    fb = read_submission(ROOT / f"outputs/local_eval/{args.name}/submission_score+prior.csv")
    ids = list(pd.read_csv(ROOT / f"data/holdouts/{args.name}/sample_submission.csv").molecule_id)
    counts = write_submission(args.out, ids, lists, fb)
    with open(args.out, newline="") as f:
        validate_submission(list(csv.reader(f)), expected_ids=set(ids))
    rep = dict(name=args.name, out=str(args.out), counts=counts)
    try:   # 診断は判断に使わない。失敗しても検証済みの提出ファイルを壊さない
        rep["diagnostics"] = diagnostics(ft)
    except Exception as e:  # noqa: BLE001
        rep["diagnostics_error"] = repr(e)
    if args.like:
        from mrr25 import calculate_mrr25, read_truth
        rep["mrr25"] = round(calculate_mrr25(read_truth(ROOT / f"outputs/holdouts/{args.name}/truth.csv"), read_submission(args.out)), 4)
    (args.out.with_suffix(".json")).write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
