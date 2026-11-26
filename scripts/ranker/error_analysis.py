#!/usr/bin/env python3
"""exp003 の OOF の誤り分析（Issue #257、計画 2026-11-26 昼の項目 2、playbook §5・F1）。定義は #257 の本文で値を見る前に固定。

- 正解の順位: exp003 = OOF の点の降順（同点は行の順）で cand_key の重複を除いた並びの、正解のキー（`mrr25.to_key`）を
  持つ最初の候補の位置。対照 = `ctrl_rank`（score,prior,frag の上位 60、61 = 居ない）の正解の行の最小値。補助 = `q_rank`
  （PubChem 層の f·z 順）。並びに居なければ ∞
- 型ごとに 下がった（exp 順位 > 基準）/ 上がった（<）/ 変わらない（=、両方 ∞ を含む）を数える（合計 = 型の分子数を assert）
- 20 件: 型 a・b・d は対照との比べで、型 c・e は補助（q_rank）との比べで「下がった」分子を molecule_id の sha1 の昇順に
  4 件ずつ（足りなければ全部、他の型から補わない）
- 各件: 1 位の候補と正解の特徴の差 (1 位 − 正解) を全行の標準偏差で割った絶対値の上位 5 列、分子式が同じか、候補源、
  その分子の fold のモデルの pred_contrib の差の上位 5 列（補助）

dev の値は開発用で、ここから採否もハイパラも決めない。holdout の表は読まない（dev_b8 の 20 集合だけ）。

    python scripts/ranker/error_analysis.py --exp-dir outputs/local_eval/ranker/lambdarank/exp003 \\
        --out outputs/local_eval/error_analysis/exp003.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))

TYPES = ("a", "b", "c", "d", "e")
BASE_FOR_CASES = {"a": "ctrl", "b": "ctrl", "c": "q", "d": "ctrl", "e": "q"}   # #257: c・e は対照に正解が居ないので q
MISSING_RANK = 61
N_PER_TYPE = 4
TOP_N_COLS = 5


def sha1_key(molecule_id: str) -> str:
    return hashlib.sha1(str(molecule_id).encode()).hexdigest()


def heavy_composition(smiles: str) -> tuple | None:
    """重原子の組成（H を除く元素ごとの数、全断片）。to_key が同じなら必ず同じになる必要条件（立体除去・互変異性の
    正準化は重原子を変えない）。読めなければ None（= 比べの対象から外さない）。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None
    c: dict[str, int] = {}
    for a in m.GetAtoms():
        s = a.GetSymbol()
        if s != "H":
            c[s] = c.get(s, 0) + 1
    return tuple(sorted(c.items()))


def make_labels_fast(tab: pd.DataFrame, truth_by_group: dict[str, str]) -> np.ndarray:
    """lambdarank_oof.make_labels と同じラベル（to_key が正解のキーの集合に入れば 1）。重原子の組成が正解のどれとも違う
    候補は to_key を呼ばずに 0（必要条件の前ふるい）。呼び出し側で当てはめの n_pos との一致を確かめる。"""
    from mrr25 import to_key, truth_keys
    g = (tab["set"].astype(str) + "\t" + tab["molecule_id"].astype(str)).to_numpy()
    smis = tab["cand_smiles"].astype(str).to_numpy()
    comp_cache: dict[str, tuple | None] = {}

    def comp(s):
        if s not in comp_cache:
            comp_cache[s] = heavy_composition(s)
        return comp_cache[s]

    tinfo: dict[str, tuple[set, set]] = {}
    y = np.zeros(len(tab), dtype=np.int32)
    for i, (gid, smi) in enumerate(zip(g, smis)):
        if gid not in tinfo:
            t = truth_by_group.get(gid)
            ts = [x for x in (t or "").split(",") if x.strip()]
            tinfo[gid] = (truth_keys(t), {comp(x.strip()) for x in ts})
        keys, comps = tinfo[gid]
        if not keys:
            continue
        c = comp(smi)
        if c is not None and None not in comps and c not in comps:
            continue
        if to_key(smi) in keys:
            y[i] = 1
    return y


def truth_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """df: 列 set, molecule_id, type, row, score, cand_key, label, ctrl_rank, q_rank（1 行 1 候補）。
    分子ごとの rank_exp・rank_ctrl・rank_q（∞ = 居ない）、1 位の行 top_row、正解の行 truth_row（exp の並びで最初）。"""
    d = df.sort_values(["set", "molecule_id", "score", "row"], ascending=[True, True, False, True], kind="mergesort")
    out = []
    for (s, mid), g in d.groupby(["set", "molecule_id"], sort=True):
        first = ~g["cand_key"].duplicated(keep="first").to_numpy()
        gu = g[first]
        lab = gu["label"].to_numpy() == 1
        pos = np.flatnonzero(lab)
        rank_exp = float(pos[0] + 1) if len(pos) else np.inf
        tr = g[g["label"].to_numpy() == 1]
        rc = tr["ctrl_rank"].min() if len(tr) else MISSING_RANK
        rq = tr["q_rank"].min() if len(tr) else MISSING_RANK
        out.append(dict(set=s, molecule_id=mid, type=str(g["type"].iloc[0]), rank_exp=rank_exp,
                        rank_ctrl=np.inf if rc >= MISSING_RANK else float(rc),
                        rank_q=np.inf if rq >= MISSING_RANK else float(rq),
                        top_row=int(gu["row"].iloc[0]), truth_row=int(gu["row"].iloc[pos[0]]) if len(pos) else -1))
    return pd.DataFrame(out)


def classify(rank_new: np.ndarray, rank_base: np.ndarray) -> np.ndarray:
    """'down'（new > base）/ 'up'（<）/ 'same'（=、両方 ∞ を含む）。"""
    rank_new, rank_base = np.asarray(rank_new, float), np.asarray(rank_base, float)
    return np.where(rank_new > rank_base, "down", np.where(rank_new < rank_base, "up", "same"))


def count_by_type(per: pd.DataFrame, base: str) -> dict:
    c = classify(per["rank_exp"], per[f"rank_{base}"])
    out = {}
    for t in TYPES:
        m = (per["type"] == t).to_numpy()
        r = {k: int((c[m] == k).sum()) for k in ("down", "up", "same")}
        r["n"] = int(m.sum())
        assert r["down"] + r["up"] + r["same"] == r["n"], (t, r)
        out[t] = r
    return out


def select_cases(per: pd.DataFrame, n_per_type: int = N_PER_TYPE, base_for=BASE_FOR_CASES) -> pd.DataFrame:
    """型ごとに base_for[t] との比べで「下がった」分子を molecule_id の sha1 の昇順で先頭 n_per_type 件。
    入力の行の順に依らない（同じ sha1 は set で並べる）。"""
    rows = []
    for t in TYPES:
        g = per[per["type"] == t]
        down = g[classify(g["rank_exp"], g[f"rank_{base_for[t]}"]) == "down"].copy()
        down = down[np.isfinite(down["rank_exp"].to_numpy())]   # 正解が候補にある分子だけ
        down["sha1"] = down["molecule_id"].map(sha1_key)
        down = down.sort_values(["sha1", "set"], kind="mergesort").head(n_per_type)
        down["base"] = base_for[t]
        rows.append(down)
    return pd.concat(rows, ignore_index=True)


def top_diffs(x_top: np.ndarray, x_truth: np.ndarray, names, sd: np.ndarray, k: int = TOP_N_COLS) -> list[dict]:
    """(1 位 − 正解) / sd の絶対値の上位 k 列（sd = 0 の列は生の差）。同点は列の順。"""
    diff = np.asarray(x_top, float) - np.asarray(x_truth, float)
    sc = np.where(sd > 0, np.abs(diff) / np.where(sd > 0, sd, 1.0), np.abs(diff))
    order = sorted(range(len(names)), key=lambda j: (-sc[j], j))[:k]
    return [dict(col=names[j], top=float(x_top[j]), truth=float(x_truth[j]), diff=float(diff[j]),
                 diff_sd=float(sc[j])) for j in order]


def formula(smiles: str) -> str | None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles)
    return CalcMolFormula(m) if m is not None else None


def run(exp_dir: Path, eval_root: Path) -> dict:
    import lightgbm as lgb
    import lambdarank_oof as LR
    meta = json.loads((exp_dir / "meta.json").read_text())
    feats = list(meta["features"])
    tab = pd.read_parquet(meta["table"])
    if LR.sha256_file(Path(meta["table"])) != meta["table_sha256"]:
        raise LR.HarnessError("表の sha256 が当てはめの時と違う")
    if meta.get("rows", "all") != "all":
        raise LR.HarnessError("rows=all の実験だけを扱う")
    oof = pd.read_parquet(exp_dir / "oof.parquet")
    if len(oof) != len(tab) or not (oof["set"].values == tab["set"].values).all() \
            or not (oof["molecule_id"].values == tab["molecule_id"].values).all() \
            or not (oof["cand_key"].values == tab["cand_key"].values).all():
        raise LR.HarnessError("OOF と表の行が揃っていない")
    truth = LR.load_truth(tab, eval_root)
    y = make_labels_fast(tab, truth)
    if int(y.sum()) != int(meta["n_pos"]):
        raise LR.HarnessError(f"ラベルの数が当てはめと違う: {int(y.sum())} != {meta['n_pos']}")
    df = tab[["set", "molecule_id", "type", "cand_key", "cand_smiles", "ctrl_rank", "q_rank", "is_tier",
              "has_ref", "query_fold"]].copy()
    df["row"] = oof["row"].to_numpy()
    df["score"] = oof["score"].to_numpy()
    df["label"] = y
    per = truth_ranks(df)
    counts = {"ctrl": count_by_type(per, "ctrl"), "q": count_by_type(per, "q")}
    cases = select_cases(per)
    X = tab[feats].to_numpy(dtype=np.float64)
    sd = X.std(axis=0)
    models = {k: lgb.Booster(model_file=str(exp_dir / f"model_fold{k}.txt")) for k in LR.FOLDS}
    recs = []
    for _, c in cases.iterrows():
        it, iu = int(c.top_row), int(c.truth_row)
        k = int(df["query_fold"].iloc[it])
        contrib = models[k].predict(X[[it, iu]], pred_contrib=True)   # 最後の列は bias
        cd = contrib[0, :-1] - contrib[1, :-1]
        corder = sorted(range(len(feats)), key=lambda j: (-abs(cd[j]), j))[:TOP_N_COLS]
        ft, fu = formula(df["cand_smiles"].iloc[it]), formula(df["cand_smiles"].iloc[iu])
        recs.append(dict(
            type=c.type, set=c.set, molecule_id=c.molecule_id, sha1=c.sha1, base=c.base,
            rank_exp=float(c.rank_exp), rank_ctrl=float(c.rank_ctrl), rank_q=float(c.rank_q),
            top=dict(smiles=df["cand_smiles"].iloc[it], formula=ft, source="Q" if df["is_tier"].iloc[it] else "L",
                     has_ref=int(df["has_ref"].iloc[it]), score=float(df["score"].iloc[it])),
            truth=dict(smiles=df["cand_smiles"].iloc[iu], formula=fu, source="Q" if df["is_tier"].iloc[iu] else "L",
                       has_ref=int(df["has_ref"].iloc[iu]), score=float(df["score"].iloc[iu])),
            same_formula=bool(ft is not None and ft == fu),
            top_feature_diffs=top_diffs(X[it], X[iu], feats, sd),
            top_contrib_diffs=[dict(col=feats[j], diff=float(cd[j])) for j in corder]))
    # 下がった分子全体の要約（20 件の外の全数。対照 = 型ごとの抜き出しの基準と同じ）
    summary = {}
    for t in TYPES:
        g = per[per["type"] == t]
        base = BASE_FOR_CASES[t]
        dn = g[(classify(g["rank_exp"], g[f"rank_{base}"]) == "down") & np.isfinite(g["rank_exp"].to_numpy())]
        if len(dn):
            ft = [formula(s) for s in df["cand_smiles"].to_numpy()[dn["top_row"].to_numpy()]]
            fu = [formula(s) for s in df["cand_smiles"].to_numpy()[dn["truth_row"].to_numpy()]]
            src_top = df["is_tier"].to_numpy()[dn["top_row"].to_numpy()]
            ref_top = df["has_ref"].to_numpy()[dn["top_row"].to_numpy()]
            ref_tru = df["has_ref"].to_numpy()[dn["truth_row"].to_numpy()]
            summary[t] = dict(base=base, n_down=int(len(dn)),
                              n_same_formula=int(sum(a is not None and a == b for a, b in zip(ft, fu))),
                              n_top_from_Q=int(src_top.sum()), n_top_has_ref=int(ref_top.sum()),
                              n_truth_has_ref=int(ref_tru.sum()),
                              n_truth_out_of_top25=int((dn["rank_exp"] > 25).sum()))
        else:
            summary[t] = dict(base=base, n_down=0)
    return dict(exp_id=exp_dir.name, issue=257, table_sha256=meta["table_sha256"], n_pos=int(y.sum()),
                n_molecules=int(len(per)), features=feats, counts_vs_ctrl=counts["ctrl"], counts_vs_q=counts["q"],
                down_summary=summary, cases=recs,
                selection="型 a・b・d は ctrl、c・e は q との比べで下がった分子、molecule_id の sha1 昇順の先頭 4 件",
                note="dev の値は開発用。採否・ハイパラは決めない")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp003 の誤り分析（#257）")
    ap.add_argument("--exp-dir", default=str(ROOT / "outputs/local_eval/ranker/lambdarank/exp003"))
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Path(a.exp_dir), Path(a.eval_root))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("n_pos", "n_molecules", "counts_vs_ctrl", "counts_vs_q", "down_summary")},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
