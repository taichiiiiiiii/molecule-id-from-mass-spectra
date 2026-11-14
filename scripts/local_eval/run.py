#!/usr/bin/env python3
"""
リーク無しのローカル縮小パイプライン（Issue #20）。改善ループの評価環境。

段:
  1. candidates : 候補プール（pool_meta）の ±PPM 窓
  2. score      : fold-0 FPNet ロジット z で fp[bits] @ z（ホールドアウト分子を学習で見ていないモデル）
  3. prior      : 上位 TOPN を z 標準化 + POOLPOP_MU * (log1p 物質数 + log1p PubMed)（0.479 版 v19 と同じ定数）
  2a. pmi      : （#97、指定時だけ）候補の点を fp @ (z − z̄) にする。z̄ は同じ評価集合の全問い合わせロジット
                  （pc_logits.npz の __targets__ 以外の全キー）の単純平均で、係数は 1。joint・pubchem とは併用しない
  2c. nbz      : （#103、指定時だけ）評価集合の中の問い合わせ同士の相互最近傍 j（#101 の定義、しきい値なし）がある
                  分子 i だけ、z_i' = (z_i + s_ij·z_j) / (1 + s_ij) にしてから fp @ z_i' で採点する（z_j は元のロジット）。
                  joint・pubchem・sanpier・lib・reflast とは併用しない
  2b. sanpier   : （#60、指定時だけ）fold-safe sanpier の点 n と s を分子ごとに z 化し、s' = μ_s + σ_s·(z_s+z_n)/2（等重み）。
                  点は scripts/local_eval/sanpier.py が outputs/local_eval/<h>/sanpier_scores.npz に作る
  4. frag       : 上位 TOPN の同一組成式グループを MetFrag-lite 断片説明率で並べ替え（0.479 版 v7c と同じ:
                  gl_fuse.rerank_multi、λ=0.5、学習なしなのでリーク無し）
  4b. elem      : （#90、指定時だけ）dedup 後の上位 TOPN で、C・H・N・O・P・S・F・Cl・Br・I 以外の元素を含む候補を
                  含まない候補の後ろへ回す（安定な分割、定数なし。scripts/local_eval/elem_rule.py）
  5. output     : InChIKey14 で重複除去した上位 25 → submission 形式 → MRR@25

    python scripts/local_eval/run.py --holdout np_v2
    python scripts/local_eval/run.py --holdout enveda_v1 --stages score      # 文献 prior 無し

ロジットが無ければ scripts/cv/local_fold0_logits.py で作る（CPU で数秒）。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

PPM = 10.0          # 0.479 版の PubChem チャネルと同じ
TOPN = 60           # 0.479 版の base リスト長
POOLPOP_MU = 0.15   # 0.479 版 v19
FRAG_LAM = 0.5      # 0.479 版 v7c（frag_gap の λ）
LIB_TAU = 0.7       # 0.479 版のライブラリ一致の既存しきい値
K = 25


# ---------------------------------------------------------------- 段 1: 候補
class Pool:
    def __init__(self, meta: pd.DataFrame, pop: np.ndarray | None):
        order = np.argsort(meta["mass"].values, kind="stable")
        self.key = meta["key"].values[order]
        self.smiles = meta["smiles"].values[order]
        self.mass = meta["mass"].values[order].astype(np.float64)
        self.pop = pop[meta["pid"].values[order]] if pop is not None else np.zeros(len(order))

    def window(self, target: float, ppm: float = PPM) -> np.ndarray:
        tol = target * ppm * 1e-6
        a = np.searchsorted(self.mass, target - tol, "left")
        b = np.searchsorted(self.mass, target + tol, "right")
        return np.arange(a, b)


def load_pool(root: Path = ROOT) -> Pool:
    meta = pd.read_parquet(root / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet",
                           columns=["key", "smiles", "mass", "pid"])
    pp = pd.read_csv(root / "data/ext/pop/pool_popularity.csv", usecols=["pool_row", "log1p_substances", "log1p_pubmed"])
    pop = np.zeros(len(meta))
    pop[pp.pool_row.values] = pp.log1p_substances.fillna(0).values + pp.log1p_pubmed.fillna(0).values
    return Pool(meta, pop)


# ---------------------------------------------------------------- 段 2: 採点
class FPCache:
    """SMILES -> 選択ビットの指紋。同じ構造を何度も計算しない。"""
    bits: np.ndarray

    def __init__(self, bits: np.ndarray):
        from casmi import chem
        self.chem, self.bits, self.c = chem, bits, {}

    def __call__(self, smi: str):
        if smi not in self.c:
            fp = self.chem.raw_fingerprint(smi)
            self.c[smi] = None if fp is None else fp[self.bits].astype(np.float32)
        return self.c[smi]


def score_window(idx: np.ndarray, pool: Pool, z: np.ndarray, fpc: FPCache):
    rows = []
    for i in idx:
        fp = fpc(pool.smiles[i])
        if fp is not None:
            rows.append((float(fp @ z), int(i)))
    rows.sort(key=lambda r: -r[0])
    return rows


# ---------------------------------------------------------------- 段 2a: pmi（#97、候補ごとの hubness 補正）
PMI_EXCLUSIVE = ("joint", "pubchem")   # 生の z で score_window を呼び直す経路。pmi と併用しない


def query_mean(zf) -> np.ndarray:
    """z̄: npz の全問い合わせベクトル（__targets__ 以外の全キー）の float64 の単純平均（#97）。"""
    keys = [k for k in zf.files if k != "__targets__"]
    assert keys, "問い合わせのロジットが無い"
    return np.mean([np.asarray(zf[k], np.float64) for k in keys], axis=0)


def stage_z(z: np.ndarray, zbar: np.ndarray | None) -> np.ndarray:
    """score_window に渡す z。zbar が None（pmi 無し）なら今までどおり float32 の z。"""
    if zbar is None:
        return z.astype(np.float32)
    return (np.asarray(z, np.float64) - zbar).astype(np.float32)


# ---------------------------------------------------------------- 段 2a: nbz（#103、相互最近傍の問い合わせのロジットを s で混ぜる）
NBZ_EXCLUSIVE = ("joint", "pubchem", "sanpier", "lib", "reflast")   # 生の z を使う経路・測らない組み合わせ


def nbz_mix(zs: dict, pairs: list) -> dict:
    """pairs = [(a, b, s_ab)] の相互最近傍の組。両端のロジットがある組だけ、各端 i に
    z_i' = (z_i + s·z_j) / (1 + s) を float64 で計算して float32 で返す（z_j は元のロジット。反復しない）。
    返すのはつながりのある分子だけ。その他の分子は呼び出し側で元の z を使う。"""
    out = {}
    for a, b, s in pairs:
        if a not in zs or b not in zs:
            continue
        s = float(s)
        za, zb = np.asarray(zs[a], np.float64), np.asarray(zs[b], np.float64)
        out[a] = ((za + s * zb) / (1.0 + s)).astype(np.float32)
        out[b] = ((zb + s * za) / (1.0 + s)).astype(np.float32)
    return out


def nbz_pairs(spectra: dict) -> list:
    """#101 と同じ定義（query_network.sim_matrix / mutual_pairs、library._prep と同じ前処理、entropy_sim）で、
    spectra（query_spectra の形）の中の相互最近傍の組 [(a, b, s)] を返す。spectra は書き換えない。"""
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import query_network as QN
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    c = SearchCfg()
    specs = {}
    for m, v in spectra.items():
        specs[m] = []
        for q in v:
            pm, pp = sp.prep_query(q["mz"], q["it"], c.floor, c.topk, c.power, c.ent_weight)
            specs[m].append(dict(mode=q["mode"], pm=pm, pp=pp))

    def sim(x, y):
        return sp.entropy_sim(x["pm"], x["pp"], y["pm"], y["pp"], c.tol, c.ppm_tol)

    ids = list(specs)
    S = QN.sim_matrix(ids, specs, sim)
    return [(ids[i], ids[j], float(S[i, j])) for i, j in QN.mutual_pairs(ids, S)]


def nbz_crosscheck(pairs: list, path: Path) -> str:
    """#101 の出力 json の pairs と照合する。json が無ければ 'no_json'。組か s が違えば止める。"""
    if not path.exists():
        return "no_json"
    ref = {(p["a"], p["b"]): p["s"] for p in json.loads(path.read_text())["pairs"]}
    got = {(a, b): s for a, b, s in pairs}
    assert set(ref) == set(got), f"#101 と相互最近傍の組が違う: {sorted(set(ref) ^ set(got))[:5]}"
    assert all(abs(ref[k] - got[k]) <= 1e-9 for k in ref), "#101 と s が違う"
    return "match"


def check_stages(stages: list[str]) -> None:
    """pmi（#97）・nbz（#103）と、それぞれ測っていない段の同時指定を止める。pmi と nbz の併用も止める。"""
    bad = [s for s in PMI_EXCLUSIVE if s in stages]
    assert not ("pmi" in stages and bad), f"pmi は {bad} と併用できない（#97 の範囲外）"
    bad = [s for s in NBZ_EXCLUSIVE if s in stages]
    assert not ("nbz" in stages and bad), f"nbz は {bad} と併用できない（#103 の範囲外）"
    assert not ("pmi" in stages and "nbz" in stages), "pmi と nbz は併用できない（どちらの事前登録の範囲でもない）"


# ---------------------------------------------------------------- 段 2b: sanpier（#60、等重み）
def sanpier_mix(rows, nmap: dict):
    """rows=(s, pool 行) の全候補について s と n を z 化し、s' = μ_s + σ_s·(z_s+z_n)/2 で並べ直す。
    n が無い（NaN・欠け）候補は z_n = 0。点のある候補が 2 未満、または σ_s / σ_n = 0 の分子は rows をそのまま返す。"""
    if len(rows) < 2:
        return rows
    s = np.array([r[0] for r in rows], np.float64)
    n = np.array([nmap.get(i, np.nan) for _, i in rows], np.float64)
    ok = np.isfinite(n)
    mu, sd = s.mean(), s.std()
    if ok.sum() < 2 or sd == 0:
        return rows
    nm, ns = n[ok].mean(), n[ok].std()
    if ns == 0:
        return rows
    zn = np.zeros(len(s)); zn[ok] = (n[ok] - nm) / ns
    s2 = mu + sd * ((s - mu) / sd + zn) / 2
    order = np.argsort(-s2, kind="stable")
    return [(float(s2[o]), rows[o][1]) for o in order]


# ---------------------------------------------------------------- 段 3: 文献 prior（v19 と同じ形）
def rerank_with_prior(rows, pool: Pool, mu: float = POOLPOP_MU, topn: int = TOPN):
    head = dedup(rows, pool, topn)
    if len(head) <= 1 or mu == 0:
        return head
    s = np.array([r[0] for r in head]); zs = (s - s.mean()) / (s.std() + 1e-9)
    f = zs + mu * np.array([pool.pop[r[1]] for r in head])
    order = np.argsort(-f, kind="stable")
    return [(float(f[o]), head[o][1]) for o in order]


def dedup(rows, pool: Pool, n: int):
    out, seen = [], set()
    for sc, i in rows:
        k = pool.key[i]
        if k in seen:
            continue
        seen.add(k); out.append((sc, i))
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------- 段 4: 断片説明率（v7c と同じ並べ替え）
def _load(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def query_spectra(holdout: str) -> dict:
    te = pd.read_parquet(ROOT / f"data/holdouts/{holdout}/test.parquet")
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        out[str(mid)] = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                              adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1)
                         for r in g.itertuples()][:16]
    return out


def frag_rerank(rows, pool: Pool, spectra, frag, fuse, cache, formula_of, topn: int = TOPN, lam: float = FRAG_LAM):
    head = dedup(rows, pool, topn)
    if len(head) < 2 or not spectra:
        return head
    smis = [pool.smiles[i] for _, i in head]
    sc = frag.score_molecule(spectra, smis, cache=cache)
    order = fuse.rerank_multi(smis, [pool.key[i] for _, i in head], [r[0] for r in head],
                              [formula_of[i] for _, i in head], [{s: v for s, v in zip(smis, sc) if v is not None}],
                              [lam], top_n=len(head))
    return [head[o] for o in order]


# ---------------------------------------------------------------- 段: PubChem 提案の差し込み（0.479 版 cell 33 の merge と同じ）
PC_LIB_TAU, REL_TH, PROMOTE_REL = 0.9, 600.0, 150.0
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]


def merge_slots(base, base_keys, pc, pc_keys, slots, n=K):
    """PubChem 提案を指定の順位（1 始まり）に差し込む。base と同じキーの提案は飛ばす。"""
    seen = set(base_keys)
    props = [(s, k) for s, k in zip(pc, pc_keys) if k not in seen]
    out, bi, pi = [], 0, 0
    while len(out) < n and (bi < len(base) or pi < len(props)):
        if (len(out) + 1) in slots and pi < len(props):
            out.append(props[pi][0]); pi += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        else:
            out.append(props[pi][0]); pi += 1
    return out


# ---------------------------------------------------------------- 実行
def run(holdout: str, stages: list[str], logits_run: str | None = None) -> dict:
    from mrr25 import calculate_mrr25, read_truth
    from elem_rule import demote_unusual
    check_stages(stages)
    t0 = time.time()
    run_name = logits_run or f"{holdout}_fold0"
    zf = np.load(ROOT / f"outputs/holdout_eval/{run_name}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    zbar = query_mean(zf) if "pmi" in stages else None   # #97: 集合ごとに 1 回だけ
    pool = load_pool()
    fpc = FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(ROOT / f"outputs/holdouts/{holdout}/truth.csv")
    if "frag" in stages:
        sys.path.insert(0, str(ROOT / "data/ext/frag"))   # numba のキャッシュは通常の import でないと読めない
        import frag_rescore as frag
        frag.warmup()
        fuse = _load("gl_fuse", ROOT / "data/ext/v4g/casmi26-glacier/gl_fuse.py")
        meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["mass", "formula"])
        formula_of = meta["formula"].values[np.argsort(meta["mass"].values, kind="stable")]
        spectra, fcache = query_spectra(holdout), {}
    import library as L
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()   # クラス 2 型: 正解はプールに居ない
    need_lib = "lib" in stages or "reflast" in stages
    if need_lib and "frag" not in stages:
        spectra = query_spectra(holdout)
    if "sanpier" in stages:
        import sanpier as SP
        nscores = SP.load_scores(ROOT / f"outputs/local_eval/{holdout}/sanpier_scores.npz")
    nbz_z, nbz_diag = {}, {}
    if "nbz" in stages:   # #103: 集合ごとに 1 回だけ。z_j は同じ npz（zf）から取る
        pairs = nbz_pairs(query_spectra(holdout))
        nbz_z = nbz_mix({m: zf[m] for m in zf.files if m != "__targets__"}, pairs)
        nbz_diag = dict(nbz_n_pairs=len(pairs), nbz_n_connected=len(nbz_z),
                        nbz_s_median=round(float(np.median([p[2] for p in pairs])), 4) if pairs else None,
                        nbz_crosscheck_101=nbz_crosscheck(pairs, ROOT / f"outputs/local_eval/query_network_{holdout}.json"),
                        nbz_connected=sorted(nbz_z))
    lists = {}
    for mid in truth:
        if mid not in targets:
            continue
        idx = pool.window(targets[mid])
        if hidden:
            idx = idx[~np.isin(pool.key[idx], list(hidden))]
        z = nbz_z[mid] if mid in nbz_z else zf[mid]
        rows = score_window(idx, pool, stage_z(z, zbar), fpc)
        if "sanpier" in stages:
            rows = sanpier_mix(rows, nscores[mid])
        if "prior" in stages:
            rows = rerank_with_prior(rows, pool)
        if "frag" in stages:
            rows = frag_rerank(rows, pool, spectra.get(mid, []), frag, fuse, fcache, formula_of)
        lists[mid] = dedup(rows, pool, TOPN)
        if "elem" in stages:   # #90: 天然物らしくない元素を含む候補を後ろへ（安定な分割、定数なし）
            lists[mid] = demote_unusual(lists[mid], lambda r: pool.smiles[r[1]])
    diag = {}
    if need_lib:
        keys = {pool.key[i] for rows in lists.values() for _, i in rows}
        refs = L.load_refs(holdout, keys, ROOT / f"outputs/local_eval/{holdout}/refs_top{TOPN}.parquet")
        for mid, rows in lists.items():
            ks = [pool.key[i] for _, i in rows]
            sim = L.lib_scores(spectra.get(mid, []), ks, refs)
            lib_max = max(sim.values(), default=0.0)
            diag[mid] = dict(lib_max=lib_max, n_ref=len(sim))
            if "lib" in stages and lib_max >= LIB_TAU:   # クラス 1: 一致の強い候補を類似度順に先頭へ
                hit = sorted([r for r, k in zip(rows, ks) if sim.get(k, 0) >= LIB_TAU], key=lambda r: -sim[pool.key[r[1]]])
                rows = hit + [r for r, k in zip(rows, ks) if sim.get(k, 0) < LIB_TAU]
            if "reflast" in stages and lib_max < LIB_TAU:   # #22 の ref_last と同じ規則
                rows = [r for r in rows if pool.key[r[1]] not in refs] + [r for r in rows if pool.key[r[1]] in refs]
            lists[mid] = rows
    preds = {mid: [pool.smiles[i] for _, i in lists[mid][:K]] if mid in lists else [] for mid in truth}
    if "joint" in stages:   # 構造案: プール候補と PubChem 候補を同じ f·z の尺度で一列に並べる（差し込み枠を使わない）
        import pubchem as PC
        pool_keys = set(pool.key) - hidden
        pcres = PC.run_holdout(holdout, {m: targets[m] for m in lists}, zf, fpc.bits, pool_keys)
        for mid in lists:
            idx = pool.window(targets[mid])
            if hidden:
                idx = idx[~np.isin(pool.key[idx], list(hidden))]
            prow = [(fz, pool.smiles[i], pool.key[i]) for fz, i in dedup(score_window(idx, pool, zf[mid].astype(np.float32), fpc), pool, TOPN)]
            allr = sorted(prow + [(fz, s_, k) for s_, fz, k in (pcres.get(mid) or [])], key=lambda r: -r[0])
            out, seen = [], set()
            for fz, s_, k in allr:
                if k in seen:
                    continue
                seen.add(k); out.append(s_)
                if len(out) >= K:
                    break
            preds[mid] = out
    if "pubchem" in stages:   # 0.479 版の merge / promote（cell 33）と同じ規則で PubChem 提案を差し込む
        import pubchem as PC
        pool_keys = set(pool.key) - hidden
        pcres = PC.run_holdout(holdout, {m: targets[m] for m in lists}, zf, fpc.bits, pool_keys)
        n_ins = n_prom = 0
        for mid, rows in lists.items():
            prop = pcres.get(mid) or []
            lib_max = diag.get(mid, {}).get("lib_max", 0.0)
            if not prop or lib_max >= PC_LIB_TAU:
                continue
            # ゲート量は 0.479 版の生ロジット尺度（1.5 × モデル平均）に合わせて 1.5 倍する（近似: fold-0 単独）
            best_pool = max((r[0] for r in score_window(pool.window(targets[mid]), pool, zf[mid].astype(np.float32), fpc)
                             if pool.key[r[1]] not in hidden), default=None) if "prior" in stages else (rows[0][0] if rows else None)
            rel = 1e9 if best_pool is None else 1.5 * (prop[0][1] - best_pool)
            slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
            base = preds[mid]; bk = [pool.key[i] for _, i in rows[:K]]
            merged = merge_slots(base, bk, [p[0] for p in prop], [p[2] for p in prop], slots)
            if rel > PROMOTE_REL:   # 文献 prior の昇格条件（S, pop）はローカルに無いので Δ だけ
                merged = ([prop[0][0]] + [x for x in merged if x != prop[0][0]])[:K]; n_prom += 1
            preds[mid] = merged; n_ins += 1
        diag_pc = dict(pubchem_inserted=n_ins, pubchem_promoted=n_prom)
    else:
        diag_pc = {}
    tag = "+".join(stages)
    out_dir = ROOT / f"outputs/local_eval/{holdout}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"submission_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["molecule_id", "smiles"])
        for mid, c in preds.items():
            w.writerow([mid, ";".join(c)])
    res = dict(holdout=holdout, stages=stages, logits=run_name, n=len(truth),
               mrr25=round(calculate_mrr25(truth, preds), 4), secs=round(time.time() - t0, 1))
    if diag:
        lm = np.array([d["lib_max"] for d in diag.values()])
        res["lib_max_ge_tau"] = round(float((lm >= LIB_TAU).mean()), 3)
        res["lib_max_median"] = round(float(np.median(lm)), 3)
    res.update(diag_pc)
    if zbar is not None:   # #97: z̄ を作った問い合わせ数と z̄ の要約（記録用）
        res.update(pmi_n_queries=len(zf.files) - 1, pmi_zbar_mean=round(float(zbar.mean()), 4), pmi_zbar_std=round(float(zbar.std()), 4))
    res.update(nbz_diag)   # #103: nbz の組の数・つながりのある分子（記録用）
    (out_dir / f"result_{tag}.json").write_text(json.dumps(res, indent=2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--stages", default="score,prior", help="カンマ区切り: score[,pmi|,nbz][,sanpier][,prior][,frag][,elem][,lib][,reflast][,pubchem]")
    ap.add_argument("--logits-run", default=None, help="outputs/holdout_eval/<run>/pc_logits.npz（既定 <holdout>_fold0）")
    args = ap.parse_args()
    print(json.dumps(run(args.holdout, args.stages.split(","), args.logits_run), indent=2))


if __name__ == "__main__":
    main()
