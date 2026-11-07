#!/usr/bin/env python3
"""
ランカーの特徴量表（Issue #29）。1 つのホールドアウト（v3_* または学習グループ rk_*）について、
プール候補（hide_rule 適用後の ±10 ppm 窓を f·z で並べた上位 60、0.479 版 v19 の prior 段と同じ集合）と
PubChem 候補（scripts/local_eval/pubchem.py、プールのキーを除いた上位 60）を 1 つの候補集合にし、候補ごとの特徴量を作る。

特徴量（クエリ・候補・参照だけから計算。正解ラベルは label 列にだけ使う）:
  fz            f·z（そのホールドアウトのロジット z。fold k の構造は fold k を学習していないモデル）
  fz_rank_src   出所ごとの窓内での f·z 順位（1 始まり）
  src_pubchem   出所（0 = プール、1 = PubChem）
  lib_sim       候補の参照スペクトルと問い合わせのエントロピー類似度の最大値（参照が無ければ NaN）。参照は hide_rule を当てた後
  lib_max       分子の lib_sim の最大値（参照の無い分子は 0。run.py の lib 段と同じ）
  delta_pc      PubChem 最良 f·z − プール最良 f·z（分子ごと。PubChem 候補が無ければ NaN）
  prior         文献 prior: log1p 物質数 + log1p PubMed（プール候補のみ。PubChem 候補は NaN）
  frag          MetFrag-lite 断片説明率（data/ext/frag/frag_rescore.py、学習なし）
参照は run.py の lib 段と同じキャッシュ（outputs/local_eval/<h>/refs_top60.parquet、プール候補のキー）を使う。
PubChem 候補はプールのキー（学習構造を含む）を除いてあるので、参照は持たない扱い（NaN）。

    python scripts/ranker/ranker_features.py --holdout v3_c1                      # ロジットは <holdout>_fold0
    python scripts/ranker/ranker_features.py --holdout rk_f1_c1 --logits-run rk_f1_c1_fold1
出力: outputs/ranker/features/<holdout>.parquet
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["fz", "fz_rank_src", "src_pubchem", "lib_sim", "lib_max", "delta_pc", "prior", "frag"]


def _run_module():
    spec = importlib.util.spec_from_file_location("local_eval_run", ROOT / "scripts/local_eval/run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["local_eval_run"] = mod
    spec.loader.exec_module(mod)
    return mod


def molecule_rows(mid: str, pool_rows: list, pc_rows: list, sim: dict, pop: dict, frag: dict, truth_key: str | None) -> list[dict]:
    """1 分子分の候補行。pool_rows = [(fz, key, smiles)]（f·z 降順）、pc_rows = [(smiles, fz, key)]（f·z 降順）。"""
    lib_max = max(sim.values(), default=0.0)
    best_pool = pool_rows[0][0] if pool_rows else np.nan
    delta = (pc_rows[0][1] - best_pool) if pc_rows and pool_rows else np.nan
    out = []
    for r, (fz, k, s) in enumerate(pool_rows, 1):
        out.append(dict(molecule_id=mid, key=k, smiles=s, fz=fz, fz_rank_src=r, src_pubchem=0, lib_sim=sim.get(k, np.nan),
                        lib_max=lib_max, delta_pc=delta, prior=pop.get(k, np.nan), frag=frag.get(s, np.nan), label=int(k == truth_key)))
    for r, (s, fz, k) in enumerate(pc_rows, 1):
        out.append(dict(molecule_id=mid, key=k, smiles=s, fz=fz, fz_rank_src=r, src_pubchem=1, lib_sim=np.nan,
                        lib_max=lib_max, delta_pc=delta, prior=np.nan, frag=frag.get(s, np.nan), label=int(k == truth_key)))
    return out


def build(holdout: str, logits_run: str | None) -> pd.DataFrame:
    le = _run_module()
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as F
    import library as L
    import pubchem as PC
    run_name = logits_run or f"{holdout}_fold0"
    zf = np.load(ROOT / f"outputs/holdout_eval/{run_name}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = le.load_pool()
    fpc = le.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = pd.read_csv(ROOT / f"outputs/holdouts/{holdout}/truth.csv").set_index("molecule_id").inchikey14.to_dict()
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    spectra = le.query_spectra(holdout)
    lists = {}
    for mid in truth:
        if mid not in targets:
            continue
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        lists[mid] = le.dedup(le.score_window(idx, pool, zf[mid].astype(np.float32), fpc), pool, le.TOPN)
    keys = {pool.key[i] for rows in lists.values() for _, i in rows}
    refs = L.load_refs(holdout, keys, ROOT / f"outputs/local_eval/{holdout}/refs_top{le.TOPN}.parquet")
    pcres = PC.run_holdout(holdout, {m: targets[m] for m in lists}, zf, fpc.bits, set(pool.key) - hidden)
    F.warmup()
    fcache, rows = {}, []
    for mid, prs in lists.items():
        pool_rows = [(fz, pool.key[i], pool.smiles[i]) for fz, i in prs]
        pc_rows = [tuple(p) for p in (pcres.get(mid) or [])]
        sim = L.lib_scores(spectra.get(mid, []), [k for _, k, _ in pool_rows], refs)
        pop = {pool.key[i]: float(pool.pop[i]) for _, i in prs}
        smis = [s for _, _, s in pool_rows] + [s for s, _, _ in pc_rows]
        sc = F.score_molecule(spectra.get(mid, []), smis, cache=fcache) if smis and spectra.get(mid) else []
        frag = {s: float(v) for s, v in zip(smis, sc) if v is not None}
        rows += molecule_rows(mid, pool_rows, pc_rows, sim, pop, frag, truth.get(mid))
    df = pd.DataFrame(rows)
    df["holdout"] = holdout
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--logits-run", default=None)
    args = ap.parse_args()
    df = build(args.holdout, args.logits_run)
    out = ROOT / f"outputs/ranker/features/{args.holdout}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    g = df.groupby("molecule_id")
    print(f"wrote {out}: {len(df)} rows, {g.ngroups} molecules, truth in candidates {int(g.label.max().sum())}, "
          f"pool truth {int(df[(df.label == 1) & (df.src_pubchem == 0)].molecule_id.nunique())}, "
          f"pubchem truth {int(df[(df.label == 1) & (df.src_pubchem == 1)].molecule_id.nunique())}")


if __name__ == "__main__":
    main()
