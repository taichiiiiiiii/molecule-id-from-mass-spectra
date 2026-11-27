#!/usr/bin/env python3
"""
Issue #284: 提出用の LambdaRank を lightgbm の 2 つの版で読んだとき、予測と上位 25 の順が同じかの点検。

段 1 `predict`（版ごとに別プロセス）: モデルと表の全行で `Booster(model_file).predict(X)` を出し npz に保存する。
  X は e2e_timing.py と同じ `tab[features].to_numpy(np.float64)`。`--expect-version` と `lightgbm.__version__`
  が違えば止まる（4.6.0 は `PYTHONPATH=<scratch>/lgbm460` を先頭に置いて呼ぶ）。

    PYTHONPATH=<scratch>/lgbm460 .venv/bin/python scripts/kaggle/lgbm_version_check.py predict \
        --expect-version 4.6.0 --model <ranker_submit.txt> --features <features.json> --table <dev_b8_cc.parquet> \
        --out <scratch>/pred_460.npz

段 2 `compare`: 2 つの npz を比べる。最大の絶対差、完全一致の行の割合、分子ごとの上位 25 の順
  （`evaluate.ranked_lists` = 提出と同じ: score 降順・同点は行の順・cand_key で重複を除く）の完全一致の割合。
  判定（Issue #284 で事前に固定）: 全分子で上位 25 の順が一致すれば「同じ」、1 分子でも違えば「違う」。

    .venv/bin/python scripts/kaggle/lgbm_version_check.py compare --a <pred_470.npz> --b <pred_460.npz> \
        --table <dev_b8_cc.parquet> --out <report.json>

補足 `replay`（Issue #284 のコメントで事前登録。本体の判定は変えない）: Kaggle のカーネルが出した特徴の表
  （features.parquet + parts_cc/sub_test.parquet）で手元の版の予測を出し、e2e_timing.py と同じ並べ方
  （ranked_lists → write_submission）の上位 25 を、同じ版の Kaggle の submission.csv と id ごとに比べる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
sys.path.insert(0, str(ROOT / "scripts/validate"))

K = 25
KEY_COLS = ["set", "molecule_id", "cand_key", "cand_smiles"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def predict_scores(model: Path, feats: list[str], tab: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb
    bst = lgb.Booster(model_file=str(model))
    if list(bst.feature_name()) != list(feats):
        raise ValueError("モデルの特徴の一覧が features.json と違う")
    X = tab[list(feats)].to_numpy(np.float64)
    if not np.isfinite(X).all():
        raise ValueError("特徴に NaN/inf")
    sc = np.asarray(bst.predict(X), dtype=np.float64)
    if sc.shape != (len(tab),) or not np.isfinite(sc).all():
        raise ValueError("予測の形が違うか、非有限の値がある")
    return sc


def compare_scores(keys: pd.DataFrame, sa: np.ndarray, sb: np.ndarray, k: int = K) -> dict:
    """同じ表の行に対する 2 つの予測を比べる。判定は全分子で上位 k の順が一致すれば「同じ」。"""
    from evaluate import ranked_lists
    sa = np.asarray(sa, dtype=np.float64)
    sb = np.asarray(sb, dtype=np.float64)
    n = len(keys)
    if sa.shape != (n,) or sb.shape != (n,):
        raise ValueError(f"予測の長さが表の行数 {n} と違う: {sa.shape} {sb.shape}")
    if not (np.isfinite(sa).all() and np.isfinite(sb).all()):
        raise ValueError("予測に非有限の値がある")
    missing = [c for c in KEY_COLS if c not in keys.columns]
    if missing:
        raise ValueError(f"表に列が無い: {missing}")
    base = keys[KEY_COLS].reset_index(drop=True).copy()
    base.insert(0, "row", np.arange(n))
    la = ranked_lists(base.assign(score=sa), k)
    lb = ranked_lists(base.assign(score=sb), k)
    if set(la) != set(lb):
        raise ValueError("分子の集合が違う")
    diff_mols = sorted(m for m in la if la[m] != lb[m])
    top1_diff = sum(1 for m in la if la[m][:1] != lb[m][:1])
    ad = np.abs(sa - sb)
    n_mol = len(la)
    return dict(
        n_rows=int(n), n_molecules=int(n_mol), k=int(k),
        max_abs_diff=float(ad.max()) if n else 0.0,
        n_rows_exact_equal=int((sa == sb).sum()), frac_rows_exact_equal=float((sa == sb).mean()) if n else 1.0,
        n_molecules_topk_identical=int(n_mol - len(diff_mols)),
        frac_molecules_topk_identical=float((n_mol - len(diff_mols)) / n_mol) if n_mol else 1.0,
        n_molecules_top1_differs=int(top1_diff),
        differing_molecules=[list(m) for m in diff_mols[:50]],
        verdict="同じ" if not diff_mols else "違う",
    )


def read_submission(p: Path) -> dict[str, list[str]]:
    import csv
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    if rows[0] != ["molecule_id", "smiles"]:
        raise ValueError(f"submission の見出しが違う: {rows[0]}")
    out = {}
    for mid, smi in rows[1:]:
        if mid in out:
            raise ValueError(f"id が重複: {mid}")
        out[mid] = smi.split(";")
    return out


def compare_submissions(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict:
    """id ごとに上位の SMILES の並びが完全一致か（補足の点検、Issue #284 のコメントで事前登録）。"""
    if set(a) != set(b):
        raise ValueError("id の集合が違う")
    diff = sorted(m for m in a if a[m] != b[m])
    first_pos = {}
    for m in diff[:50]:
        la, lb = a[m], b[m]
        first_pos[m] = next((i for i in range(max(len(la), len(lb))) if la[i:i + 1] != lb[i:i + 1]), None)
    n = len(a)
    return dict(n_ids=n, n_ids_identical=n - len(diff), frac_ids_identical=(n - len(diff)) / n if n else 1.0,
                differing_ids_first_pos=first_pos, verdict="同じ" if not diff else "違う")


def cmd_replay(a) -> int:
    """Kaggle が出した特徴の表で手元の版の予測を出し、e2e_timing.py と同じ並べ方の上位 25 を Kaggle の submission と比べる。"""
    import importlib.util
    import lightgbm as lgb
    if lgb.__version__ != a.expect_version:
        raise SystemExit(f"lightgbm の版が {a.expect_version} でない: {lgb.__version__}")
    spec = importlib.util.spec_from_file_location("e2e_timing", ROOT / "scripts/kaggle/e2e_timing.py")
    E2E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E2E)
    from evaluate import ranked_lists
    feats = json.loads(Path(a.features).read_text())
    tab = pd.concat([pd.read_parquet(a.table), pd.read_parquet(a.parts)], axis=1)
    sc = predict_scores(Path(a.model), feats, tab)
    oof = tab[KEY_COLS].copy()
    oof.insert(0, "row", np.arange(len(tab)))
    oof["score"] = sc
    lists = {mid: lst for (_s, mid), lst in ranked_lists(oof, E2E.K).items()}
    kag = read_submission(Path(a.submission))
    ids = list(kag)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"出力先が既にある（上書きしない）: {out}")
    out.mkdir(parents=True)
    counts = E2E.write_submission(out / "submission_local.csv", ids, lists)
    rep = compare_submissions(read_submission(out / "submission_local.csv"), kag)
    rep.update(lightgbm_local=lgb.__version__, counts=counts, n_rows=int(len(tab)),
               model_sha256=sha256_file(Path(a.model)), kaggle_submission_sha256=sha256_file(Path(a.submission)),
               local_submission_sha256=sha256_file(out / "submission_local.csv"),
               table_sha256=sha256_file(Path(a.table)), parts_sha256=sha256_file(Path(a.parts)))
    (out / "replay.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False))
    return 0


def cmd_predict(a) -> int:
    import lightgbm as lgb
    if lgb.__version__ != a.expect_version:
        raise SystemExit(f"lightgbm の版が {a.expect_version} でない: {lgb.__version__}（{lgb.__file__}）")
    t0 = time.time()
    feats = json.loads(Path(a.features).read_text())
    tab = pd.read_parquet(a.table, columns=list(feats))
    sc = predict_scores(Path(a.model), feats, tab)
    meta = dict(lightgbm=lgb.__version__, lightgbm_file=lgb.__file__, model_sha256=sha256_file(Path(a.model)),
                table_sha256=sha256_file(Path(a.table)), features_json_sha256=sha256_file(Path(a.features)),
                n_rows=int(len(tab)), n_features=len(feats), secs=round(time.time() - t0, 1))
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"出力先が既にある（上書きしない）: {out}")
    np.savez(out, score=sc, meta=json.dumps(meta))
    print(json.dumps(meta, ensure_ascii=False))
    return 0


def _load(p: str) -> tuple[np.ndarray, dict]:
    z = np.load(p, allow_pickle=False)
    return z["score"], json.loads(str(z["meta"]))


def cmd_compare(a) -> int:
    sa, ma = _load(a.a)
    sb, mb = _load(a.b)
    for key in ("model_sha256", "table_sha256", "features_json_sha256", "n_rows"):
        if ma[key] != mb[key]:
            raise SystemExit(f"2 つの予測の {key} が違う: {ma[key]} vs {mb[key]}")
    tsha = sha256_file(Path(a.table))
    if tsha != ma["table_sha256"]:
        raise SystemExit(f"表の sha256 が予測のときと違う: {tsha}")
    keys = pd.read_parquet(a.table, columns=KEY_COLS)
    rep = compare_scores(keys, sa, sb)
    rep.update(a=dict(path=a.a, **ma), b=dict(path=a.b, **mb))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in rep if k not in ("differing_molecules", "a", "b")}, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--expect-version", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--table", required=True)
    c.add_argument("--out", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--expect-version", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--features", required=True)
    r.add_argument("--table", required=True, help="Kaggle の features.parquet")
    r.add_argument("--parts", required=True, help="Kaggle の parts_cc/sub_test.parquet")
    r.add_argument("--submission", required=True, help="同じカーネルの版の submission.csv")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return dict(predict=cmd_predict, compare=cmd_compare, replay=cmd_replay)[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
