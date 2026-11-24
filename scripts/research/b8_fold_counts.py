#!/usr/bin/env python3
"""
B8 振り分け器の段取り（Issue #216、2026-11-24 昼 項目 4）: fold 1〜4 の hold で 4 型の eligible を数える（集合は作らない）。

規則は数える前に #216 にコメントで固定した（issuecomment-7121356702）。要点:
  - fold k の hold = data/ext/foldsafe/make_split.py の規則（train_structs × structures.parquet の fold_identity、libmask の
    bit 9 を持つ構造の identity_group は np）。k = 0 は split.parquet の hold と鍵の集合で一致すること、k = 1〜4 は
    check_inputs_fold{k}.json の excluded_fold_structure_rows・excluded_np_structure_rows と一致することを関門にする
  - 4 型: (a) build_holdout_v3.eligible の c1、(b) select_c2p の条件（質量 ∩ hold ∩ COCONUT）、(c) eligible の c2、
    (d) commit 072ec66 の build_holdout.py の条件（holdout_expand.enveda_eligible）を fold k に当てたもの（単位は identity_group）
  - 除外: v3 の 4 集合と A3a の 3 集合の全正解（holdout_expand.Excl: 生・正規化・scorer_key・identity_group）
  - N_bal(k) = min(|A|, |B∖G(A)|, |C∖G(A∪B)|, |D∖G(A∪B)|, ⌊|(C∪D)∖G(A∪B)|/2⌋)。学習は各回 3 fold の和 ≥ 80、評価は
    Σ_{k=1..4} ≥ 200（1 型あたり）
  - 部品ごとの fold k との重なり（生 inchikey14・正規化 key・scorer_key）

    .venv/bin/python scripts/research/b8_fold_counts.py --check-dir <check_inputs_fold*.json のある dir> \
        --out outputs/local_eval/b8_fold_counts.json
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
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/cv", "scripts/validate", "scripts/ranker", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
SPLIT = ROOT / "data/ext/foldsafe/split.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
POOL = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
POP = ROOT / "data/ext/pop/pool_popularity.csv"
TRAIN = ROOT / "data/train.parquet"
ORIG = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")          # truth は outputs/holdouts/<h>/truth.csv
A3A = ("v3_c1b", "v3_c2nb", "enveda_v1b")                 # truth は outputs/local_eval/<h>/truth.csv
NP_BIT = 1 << 9                                           # make_split.py の is_np
MIN_TRAIN, MIN_EVAL = 80, 200                             # 規則 §3（EPV 10 × 8 特徴、v3 の集合の大きさ）
FOLDS_OOF = (1, 2, 3, 4)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 小さな道具（単体テストの対象）

def split_labels(ts: pd.DataFrame, st: pd.DataFrame, fold: int) -> pd.DataFrame:
    """make_split.py と同じ区分（np / hold / train）を train_structs の行ごとに返す（列 inchikey14・key・sid・fold）。"""
    s = st[["inchikey14", "identity_group", "fold_identity"]].drop_duplicates("inchikey14")
    j = ts.merge(s, on="inchikey14", how="left")
    assert j.fold_identity.notna().all(), "structures.parquet に無い train 構造がある"
    is_np = (j.libmask.astype("int64") & NP_BIT) != 0
    np_groups = set(j.identity_group[is_np])
    lab = np.where(j.identity_group.isin(np_groups), "np", np.where(j.fold_identity == fold, "hold", "train"))
    return pd.DataFrame(dict(inchikey14=j.inchikey14.values, key=j.key.values, sid=j.sid.values, fold=lab))


def n_bal(A: set, B: set, C: set, D: set, grp) -> int:
    """規則 §3 の控えめな 1 型あたりの数。A〜D は構造キー（D は identity_group）の集合、grp(x) は x の identity_group。
    D の要素は群そのもの（grp は恒等）として扱えるよう、grp は D に対しても呼ばれる。"""
    gA = {grp(x) for x in A}
    gAB = gA | {grp(x) for x in B}
    b = {x for x in B if grp(x) not in gA}
    c = {x for x in C if grp(x) not in gAB}
    d = {x for x in D if grp(x) not in gAB}
    cd_groups = {grp(x) for x in c} | {grp(x) for x in d}
    return int(min(len(A), len(b), len(c), len(d), len(cd_groups) // 2))


def feasible(nb: dict, folds=FOLDS_OOF, min_train: int = MIN_TRAIN, min_eval: int = MIN_EVAL) -> dict:
    """規則 §3 の判定。nb = {fold: N_bal}。学習 = 各回（1 fold を抜く）の残り 3 fold の和、評価 = 全 fold の和。"""
    train_sums = {k: int(sum(nb[j] for j in folds if j != k)) for k in folds}
    eval_sum = int(sum(nb[k] for k in folds))
    ok_train = all(v >= min_train for v in train_sums.values())
    ok_eval = eval_sum >= min_eval
    return dict(train_sums_leave_one_out=train_sums, eval_sum=eval_sum, min_train=min_train, min_eval=min_eval,
                ok_train=ok_train, ok_eval=ok_eval, feasible=bool(ok_train and ok_eval))


def overlap_counts(side_a: pd.DataFrame, side_b: pd.DataFrame, sc_key: dict) -> dict:
    """2 つの構造の表（列 inchikey14・key）の重なり: 生（inchikey14）・正規化（train_structs.key）・scorer_key。"""
    ra, rb = set(side_a.inchikey14), set(side_b.inchikey14)
    ka, kb = set(side_a.key), set(side_b.key)
    sa = {sc_key.get(k) for k in ra} - {None}
    sb = {sc_key.get(k) for k in rb} - {None}
    return dict(raw_inchikey14=len(ra & rb), canonical_key=len(ka & kb), scorer_key=len(sa & sb))


# ---------------------------------------------------------------- 読み込み

def read_meta() -> pd.DataFrame:
    """train.parquet の 4 列を 1 MiB バッファ・pre_buffer なしで 8192 行ずつ読む。文字列は category（メモリ節約）。"""
    from casmi import chem
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    parts = []
    for b in pf.iter_batches(batch_size=8192, columns=["inchikey14", "ingest_lib", "precursor_mz", "adduct"]):
        d = b.to_pandas()
        d["neutral_mass"] = chem.neutral_mass(d.precursor_mz.values.astype(np.float64), d.adduct.astype(str).values)
        parts.append(pd.DataFrame(dict(inchikey14=d.inchikey14.astype(str).values, ingest_lib=d.ingest_lib.astype(str).values,
                                       neutral_mass=d.neutral_mass.values)))
    m = pd.concat(parts, ignore_index=True)
    m["inchikey14"] = m.inchikey14.astype("category")
    m["ingest_lib"] = m.ingest_lib.astype("category")
    return m


def build_excl(st: pd.DataFrame, ts_key: dict):
    import holdout_expand as HE
    import mrr25
    ex = HE.Excl(st, ts_key, mrr25.to_key)
    for h in ORIG:
        ex.add(pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv"),
               set((ROOT / f"data/holdouts/{h}/drop_keys.txt").read_text().split()))
    for h in A3A:
        ex.add(pd.read_csv(ROOT / f"outputs/local_eval/{h}/truth.csv"),
               set((ROOT / f"data/holdouts/{h}/drop_keys.txt").read_text().split()))
    return ex


def enveda_eligible_fold(meta, st, hold, fold, bad_keys_fn=None):
    """holdout_expand.enveda_eligible（commit 072ec66 の条件）を fold k のプールで。返り値: (群の行, 群 → キーの集合)。"""
    import holdout_expand as HE
    pool = HE.enveda_pool(st, fold)
    grp = st[st.identity_group.isin(set(pool.identity_group))]
    gkeys = {}
    for r in grp.itertuples():
        gkeys.setdefault(r.identity_group, set()).update((r.inchikey14, r.scorer_key))
    train_keys = set(meta.inchikey14.cat.categories.astype(str))
    lib = meta.ingest_lib.astype(str)
    env_keys = set(meta.inchikey14[lib.str.startswith("enveda").values].astype(str).unique())
    ok = []
    for g in pool.identity_group:
        ks = gkeys[g]
        good = (ks & train_keys) <= hold and bool(ks & env_keys)
        if good and bad_keys_fn is not None and bad_keys_fn(ks):
            good = False
        ok.append(good)
    return pool[np.array(ok, dtype=bool)], gkeys


# ---------------------------------------------------------------- 数える

def main() -> None:
    import build_groups as BG
    import build_holdout_v3 as V3
    import holdout_expand as HE
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-dir", required=True, help="check_inputs_fold{1..4}.json のある dir（Kaggle の fold-safe FPNet）")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/b8_fold_counts.json"))
    args = ap.parse_args()
    t0 = time.time()
    meta = read_meta()
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity", "mono_mass",
                                         "n_enveda_spectra"])
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "key", "libmask", "sid"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    sc_key = dict(zip(st.inchikey14, st.scorer_key))
    grp_of = dict(zip(st.inchikey14, st.identity_group))
    coco = BG.coconut_keys()
    pool_keys = set(pd.read_parquet(POOL, columns=["key"]).key.astype(str))
    pop_keys = set(pd.read_csv(POP, usecols=["inchikey14"]).inchikey14.astype(str))
    lib_counts = meta.groupby("inchikey14", observed=True).size()
    env_mask = meta.ingest_lib.astype(str).str.startswith("enveda").values
    lib_env = meta[env_mask].groupby("inchikey14", observed=True).size()
    ex = build_excl(st, ts_key)
    ex_orig = HE.orig_excl(st, ts_key)   # a3a-counts.json の再現（k = 0）の点検用

    res = dict(rule_comment="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/216#issuecomment-7121356702",
               exclusion=dict(sets=list(ORIG + A3A), e_raw=len(ex.e_raw), e_key=len(ex.e_key), e_grp=len(ex.e_grp)),
               gates={}, folds={}, components_note={})
    # 関門 1: fold 0 の作り直し = split.parquet
    sp = pd.read_parquet(SPLIT)
    lab0 = split_labels(ts, st, 0)
    g0 = dict(hold_equal=set(lab0.inchikey14[lab0.fold == "hold"]) == set(sp.inchikey14[sp.fold == "hold"]),
              np_equal=set(lab0.inchikey14[lab0.fold == "np"]) == set(sp.inchikey14[sp.fold == "np"]),
              n_hold=int((lab0.fold == "hold").sum()), n_np=int((lab0.fold == "np").sum()))
    res["gates"]["fold0_split_reproduced"] = g0
    assert g0["hold_equal"] and g0["np_equal"], f"fold 0 の split が再現しない {g0}"

    for k in (0, 1, 2, 3, 4):
        lab = split_labels(ts, st, k)
        hold_df, train_df = lab[lab.fold == "hold"], lab[lab.fold == "train"]
        hold = set(hold_df.inchikey14)
        fk = dict(n_hold=len(hold), n_np=int((lab.fold == "np").sum()), n_train_side=len(train_df))
        if k >= 1:
            ck = json.loads((Path(args.check_dir) / f"check_inputs_fold{k}.json").read_text())
            fk["check_inputs"] = dict(excluded_fold_structure_rows=ck["excluded_fold_structure_rows"],
                                      excluded_np_structure_rows=ck["excluded_np_structure_rows"],
                                      training_structure_rows=ck["training_structure_rows"],
                                      fold_training_overlap=ck["fold_training_overlap"], passed=ck["passed"])
            gate = (ck["excluded_fold_structure_rows"] == fk["n_hold"] and ck["excluded_np_structure_rows"] == fk["n_np"]
                    and ck["training_structure_rows"] == fk["n_train_side"])
            fk["check_inputs_match"] = bool(gate)
            assert gate, f"fold {k}: 作り直しの行数が check_inputs と違う {fk}"
        # v3・A3a の正解との重なり（fold 1〜4 は 0 の見込み）
        fk["hold_hit_by_v3_a3a_exclusion"] = len(ex.hit(hold))
        # 部品ごとの重なり
        comp = {}
        comp["fpnet_fold_k_train_side_vs_hold"] = overlap_counts(train_df, hold_df, sc_key)
        hk = set(hold_df.key)
        comp["pool_meta_contains_hold"] = dict(by_canonical_key=len(hk & pool_keys),
                                               by_raw_inchikey14=len(hold & pool_keys), n_hold=len(hold))
        comp["prior_pool_popularity_rows_of_hold"] = dict(by_raw_inchikey14=len(hold & pop_keys),
                                                          by_canonical_key=len(hk & pop_keys))
        hl = lib_counts[lib_counts.index.astype(str).isin(hold)]
        he = lib_env[lib_env.index.astype(str).isin(hold)]
        comp["library_ref_rows_of_hold_before_hiding"] = dict(rows=int(hl.sum()), structures=int((hl > 0).sum()),
                                                             enveda_rows=int(he.sum()), public_rows=int(hl.sum() - he.sum()))
        fk["components"] = comp
        assert comp["fpnet_fold_k_train_side_vs_hold"] == dict(raw_inchikey14=0, canonical_key=0, scorer_key=0), \
            f"fold {k}: FPNet の学習側と hold が重なる"

        # 4 型
        # eligible・c2n_eligible はキーごとの集計なので、hold のキーの行だけに絞っても結果は同じ（速さのため）
        meta_k = meta[meta.inchikey14.astype(str).isin(hold).values]
        meta_k = meta_k.assign(inchikey14=meta_k.inchikey14.astype(str), ingest_lib=meta_k.ingest_lib.astype(str))
        c1_all, c2_all = V3.eligible(meta_k, hold)
        c2n_all = HE.c2n_eligible(meta_k, hold, coco)
        ev_all, _ = enveda_eligible_fold(meta, st, hold, k)
        ev_new, gkeys = enveda_eligible_fold(meta, st, hold, k, lambda ks: bool(ex.hit(ks)))
        A = set(c1_all) - ex.hit(c1_all)
        B = set(c2n_all) - ex.hit(c2n_all)
        C = set(c2_all) - ex.hit(c2_all)
        D = set(ev_new.identity_group)
        types = {"a_v3_c1u": dict(eligible_all=len(c1_all), excluded=len(c1_all) - len(A), eligible=len(A)),
                 "b_v3_c2nu": dict(eligible_all=len(c2n_all), excluded=len(c2n_all) - len(B), eligible=len(B)),
                 "c_v3_c2": dict(eligible_all=len(c2_all), excluded=len(c2_all) - len(C), eligible=len(C)),
                 "d_enveda_v1u_groups": dict(pool_groups=int(len(HE.enveda_pool(st, k))), eligible_all=len(ev_all),
                                             excluded=len(ev_all) - len(D), eligible=len(D))}
        # (a) の参照: 公開ライブラリの参照が残る構造（定義上すべてのはず）
        types["a_v3_c1u"]["with_public_refs"] = int(sum(1 for x in A if lib_counts.get(x, 0) > lib_env.get(x, 0)))
        gA, gB, gC = {grp_of.get(x) for x in A}, {grp_of.get(x) for x in B}, {grp_of.get(x) for x in C}
        fk["type_overlap_groups"] = dict(a_b=len(gA & gB), a_c=len(gA & gC), a_d=len(gA & D), b_c=len(gB & gC),
                                         b_d=len(gB & D), c_d=len(gC & D))
        grp = lambda x: x if x in D else grp_of.get(x)   # D の要素は群そのもの
        fk["types"] = types
        fk["n_bal"] = n_bal(A, B, C, D, grp)
        if k == 0:   # a3a-counts.json の再現（元の 4 集合だけで除いた数）
            c1o = len(set(c1_all) - ex_orig.hit(c1_all))
            c2no = len(set(c2n_all) - ex_orig.hit(c2n_all))
            evo, _ = enveda_eligible_fold(meta, st, hold, 0, lambda ks: bool(ex_orig.hit(ks)))
            a3a = json.loads((ROOT / "docs/research/a3a-holdout-expand/a3a-counts.json").read_text())["types"]
            rep = dict(c1=(len(c1_all), c1o), c2n=(len(c2n_all), c2no), enveda=(len(ev_all), len(evo)))
            exp = dict(c1=(a3a["v3_c1"]["eligible_all"], a3a["v3_c1"]["eligible_new"]),
                       c2n=(a3a["v3_c2n"]["eligible_all"], a3a["v3_c2n"]["eligible_new"]),
                       enveda=(a3a["enveda_v1"]["eligible_all_groups"], a3a["enveda_v1"]["eligible_new_groups"]))
            res["gates"]["fold0_a3a_counts_reproduced"] = dict(got=rep, expected=exp, equal=rep == exp)
            assert rep == exp, f"a3a-counts.json が再現しない {rep} {exp}"
        res["folds"][k] = fk
        print(json.dumps({k: dict(types={t: v["eligible"] for t, v in types.items()}, n_bal=fk["n_bal"])}), flush=True)

    nb = {k: res["folds"][k]["n_bal"] for k in FOLDS_OOF}
    res["feasibility"] = feasible(nb)
    res["table_eligible"] = {k: {t: v["eligible"] for t, v in res["folds"][k]["types"].items()} for k in (0,) + FOLDS_OOF}
    res["components_note"] = dict(
        frag="data/ext/frag/frag_rescore.py（学習なしの MetFrag-lite）・gl_fuse.rerank_multi はファイルを読まない（学習した値・"
             "train 由来の表なし）。対象なし",
        pubchem_tier="学習しない外部の表。数えない（(c) の正解の有無は集合を作る段で記録）",
        fixed_constants="fp_bits・PPM 10・TOPN 60・μ 0.15・λ 0.5 は公開の部品の固定値。fold k を見ていないとは確かめられない")
    res["inputs_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in (SPLIT, FOLDS, TSTRUCT, POOL, POP)}
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(dict(table=res["table_eligible"], n_bal=nb, feasibility=res["feasibility"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
