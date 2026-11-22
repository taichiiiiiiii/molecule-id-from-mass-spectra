#!/usr/bin/env python3
"""A1（#197）: 手元の採点キーと公式 metric のキーの違いを数える。

2 段に分ける（RDKit の版を PYTHONPATH で選ぶため、キーの計算は別プロセス）:

- `keys`: 集合の正解と候補の SMILES ごとに次を計算して npz に保存する。
    metric = 公式 metric ノートの写し `casmi_metric_official.py`（書き換えない）の `canonicalize_smiles_to_ik14`
    comp   = 同じ写しの `heavy_atom_composition`（metric の前段の絞り込み）
    legacy = 修正前の `mrr25.to_key` の手順（立体を外さずに互変異性体の正準化、失敗したら元の分子）
  あわせて、各集合の分子ごとの点を metric の `score()` そのもので計算する（1 分子ずつ呼ぶ）。
- `compare`: 2 つの版の npz から、キーの不一致の数・MRR が変わる分子の数・対照の値・25 件の中の重複を表にする。

使い方（RDKit 2026.03.3 を scratch に入れた場合）:
    PYTHONPATH=<scratch>/rdkit202603 python scripts/validate/metric_key_audit.py keys --out a.npz
    python scripts/validate/metric_key_audit.py keys --out b.npz
    python scripts/validate/metric_key_audit.py compare --pinned a.npz --venv b.npz --out audit.json
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOP_K = 25

# 集合名 -> (正解 CSV, 候補 CSV)。正解が無いもの（提出の commit 出力）は A13 の重複だけ数える
SETS = {
    "v3_c1": ("outputs/holdouts/v3_c1/truth.csv", "outputs/local_eval/v3_c1/submission_score+prior+frag.csv"),
    "v3_c2": ("outputs/holdouts/v3_c2/truth.csv", "outputs/local_eval/v3_c2/submission_score+prior+frag.csv"),
    "enveda_v1": ("outputs/holdouts/enveda_v1/truth.csv", "outputs/local_eval/enveda_v1/submission_score+prior+frag.csv"),
    "v3_c2n": ("outputs/holdouts/v3_c2n/truth.csv", "outputs/local_eval/v3_c2n/submission_score+prior+frag.csv"),
    "lb495_v1_commit": (None, "outputs/lb495/v1/submission.csv"),
}


def load_official():
    """metric ノートの写しを import する（書き換えない）。RDKit は呼んだ時点の sys.path のものを使う。"""
    p = ROOT / "scripts/validate/casmi_metric_official.py"
    spec = importlib.util.spec_from_file_location("casmi_metric_official", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.install_dependencies()
    return mod


def legacy_key(smiles: str) -> str | None:
    """修正前の mrr25.to_key（origin/main 9060ddd の 38〜60 行）と同じ手順。比較のためだけに残す。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    s = smiles.strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def read_truth(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["molecule_id"].strip(): r["smiles"] for r in csv.DictReader(f)}


def read_preds(path: Path) -> dict[str, list[str]]:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["molecule_id"].strip()] = [c.strip() for c in (r["smiles"] or "").split(";") if c.strip()]
    return out


def parse_extra(items: list[str] | None) -> dict:
    """--extra name=path（正解の無い提出 CSV。A13 の重複だけ数える）。"""
    out = {}
    for it in items or []:
        name, path = it.split("=", 1)
        out[name] = (None, path)
    return out


def load_sets(root: Path = ROOT, sets: dict = SETS, extra: dict | None = None):
    data = {}
    for name, (tp, pp) in {**sets, **(extra or {})}.items():
        truth = read_truth(root / tp) if tp else None
        data[name] = (truth, read_preds(root / pp))
    return data


def official_rr(M, truth: dict[str, str], preds: dict[str, list[str]]) -> dict[str, float]:
    """metric の score() を 1 分子ずつ呼んだ点。候補が無い分子は 0（metric も提出に無い分子は 0）。"""
    import pandas as pd

    out = {}
    for mid, t in truth.items():
        c = preds.get(mid, [])
        if not c:
            out[mid] = 0.0
            continue
        sol = pd.DataFrame({"molecule_id": [mid], "smiles": [t]})
        sub = pd.DataFrame({"molecule_id": [mid], "smiles": [";".join(c)]})
        out[mid] = M.score(sol, sub, "molecule_id", max_rank=TOP_K)
    return out


def cmd_keys(args) -> None:
    import rdkit

    M = load_official()
    data = load_sets(extra=parse_extra(args.extra))
    assert not any("," in t for truth, _ in data.values() if truth for t in truth.values()), "正解にカンマ区切りがある"
    smiles = sorted({s for truth, preds in data.values()
                     for s in ([*truth.values()] if truth else []) + [c for cs in preds.values() for c in cs]})
    metric = [M.canonicalize_smiles_to_ik14(s) or "" for s in smiles]
    comp = [json.dumps(M.heavy_atom_composition(s)) for s in smiles]
    legacy = [legacy_key(s) or "" for s in smiles]
    rr = {}
    full = {}
    for name, (truth, preds) in data.items():
        if truth is None:
            continue
        r = official_rr(M, truth, preds)
        rr[name] = r
        # 全体を 1 回で採点した値（1 分子ずつの平均と一致するかの確認）
        import pandas as pd
        sol = pd.DataFrame({"molecule_id": list(truth), "smiles": list(truth.values())})
        sub = pd.DataFrame({"molecule_id": [m for m in preds if preds[m]],
                            "smiles": [";".join(preds[m]) for m in preds if preds[m]]})
        full[name] = M.score(sol, sub, "molecule_id", max_rank=TOP_K)
    np.savez_compressed(args.out, smiles=np.array(smiles, dtype=object), metric=np.array(metric, dtype=object),
                        comp=np.array(comp, dtype=object), legacy=np.array(legacy, dtype=object),
                        meta=np.array(json.dumps(dict(rdkit=rdkit.__version__, rdkit_file=rdkit.__file__,
                                                      rr=rr, full=full)), dtype=object))
    print(json.dumps(dict(rdkit=rdkit.__version__, n_smiles=len(smiles), full=full)))


def first_hit(tkeys: set, keys: list) -> float:
    for r, k in enumerate(keys[:TOP_K], 1):
        if k and k in tkeys:
            return 1.0 / r
    return 0.0


def dup_stats(preds: dict[str, list[str]], key: dict[str, str]) -> dict:
    """25 件の中で、前に出た候補と同じキー（metric の上で同じ答え）になる候補の数。"""
    n_dup = n_rows = n_unkeyed = 0
    for cs in preds.values():
        seen, d = set(), 0
        for c in cs[:TOP_K]:
            k = key.get(c)
            if not k:
                n_unkeyed += 1
                continue
            if k in seen:
                d += 1
            seen.add(k)
        n_dup += d
        n_rows += d > 0
    return dict(dup_slots=n_dup, rows_with_dup=n_rows, rows=len(preds),
                slots=sum(min(len(c), TOP_K) for c in preds.values()), unkeyed_slots=n_unkeyed)


def compare(pinned: dict, venv: dict, data: dict) -> dict:
    """pinned / venv: {smiles: (metric, legacy)} の辞書と meta。"""
    out = dict(rdkit_pinned=pinned["meta"]["rdkit"], rdkit_venv=venv["meta"]["rdkit"], sets={})
    P, V = pinned["keys"], venv["keys"]
    off = {s: P[s][0] for s in P}            # 公式（metric の手順 × 固定の版）
    cur = {s: V[s][1] for s in V}            # 今の手元（修正前の手順 × .venv の版）
    new = {s: V[s][0] for s in V}            # 修正後の手元（metric の手順 × .venv の版）
    for name, (truth, preds) in data.items():
        cand = {c for cs in preds.values() for c in cs[:TOP_K]}
        res = {}
        if truth is not None:
            tsm = set(truth.values())
            for lab, sm in (("truth", tsm), ("candidates", cand)):
                n = len(sm)
                res[f"{lab}_unique_smiles"] = n
                res[f"{lab}_cur_ne_official"] = sum(cur[s] != off[s] for s in sm)
                res[f"{lab}_procedure_only"] = sum(V[s][1] != V[s][0] for s in sm)   # 同じ版で手順だけ
                res[f"{lab}_version_only"] = sum(V[s][0] != P[s][0] for s in sm)     # 同じ手順で版だけ
                res[f"{lab}_new_ne_official"] = sum(new[s] != off[s] for s in sm)
                res[f"{lab}_official_none"] = sum(not off[s] for s in sm)
                res[f"{lab}_cur_ne_official_frac"] = round(res[f"{lab}_cur_ne_official"] / n, 5) if n else None

            # 今の手元: 正解の文字列を 1 つの SMILES として扱う。修正後: metric と同じくカンマで分ける
            # （v3 の正解にカンマは無いことを keys の段で assert している）
            rr_cur = {m: first_hit({cur[t]} - {""}, [cur[c] for c in preds.get(m, [])]) for m, t in truth.items()}
            rr_new = {m: first_hit({new[t]} - {""}, [new[c] for c in preds.get(m, [])]) for m, t in truth.items()}
            rr_off = pinned["meta"]["rr"][name]
            res.update(
                n=len(truth),
                mrr_cur=round(float(np.mean(list(rr_cur.values()))), 4),
                mrr_official=round(float(np.mean(list(rr_off.values()))), 4),
                mrr_official_full_call=round(pinned["meta"]["full"][name], 4),
                mrr_new_local=round(float(np.mean(list(rr_new.values()))), 4),
                molecules_rr_changed_cur_vs_official=sorted(m for m in truth if abs(rr_cur[m] - rr_off[m]) > 1e-12),
                molecules_rr_changed_new_vs_official=sorted(m for m in truth if abs(rr_new[m] - rr_off[m]) > 1e-12),
            )
        res["dup_official"] = dup_stats(preds, off)
        res["dup_cur"] = dup_stats(preds, cur)
        out["sets"][name] = res
    return out


def _load_npz(p) -> dict:
    z = np.load(p, allow_pickle=True)
    keys = {s: (m, l) for s, m, l in zip(z["smiles"], z["metric"], z["legacy"])}
    return dict(keys=keys, meta=json.loads(str(z["meta"])))


def cmd_compare(args) -> None:
    res = compare(_load_npz(args.pinned), _load_npz(args.venv), load_sets(extra=parse_extra(args.extra)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for name, r in res["sets"].items():
        print(name, {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    k = sp.add_parser("keys")
    k.add_argument("--out", required=True)
    k.add_argument("--extra", action="append", help="name=path の提出 CSV（正解なし、重複だけ数える）")
    c = sp.add_parser("compare")
    c.add_argument("--pinned", required=True, help="RDKit 2026.03.3 で作った npz")
    c.add_argument("--venv", required=True, help=".venv の RDKit で作った npz")
    c.add_argument("--out", required=True)
    c.add_argument("--extra", action="append")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    {"keys": cmd_keys, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
