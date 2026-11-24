#!/usr/bin/env python3
"""
B8 振り分け器（Issue #216）: fold 1〜4 × 5 型の集合 b8_f{k}_{a,e,b,c,d} を作り、`_t10` に絞る。

事前登録: 下書き issuecomment-7121558819 §1 ＋ 確定 issuecomment-7134581003（(e) の追加、引く順番 (a)→(e)→(b)→(c)→(d)）。
  - seed S = 20261124、型 t の番号 a=0, b=1, c=2, d=3, e=4、s_kt = S + 100·k + t
  - (a)(b)(c)(e): 候補を sorted にして default_rng(s_kt).choice(200, replace=False)（select と同じ形）。
    write_holdout の rng は default_rng(s_kt + 1000)、seed 引数は s_kt、id_prefix は f{k}{t}
  - (d): enveda の群の表から pool.sample(n=200, random_state=s_kt)、write_enveda の rng は default_rng(s_kt)
  - 作り方: (a) rule=enveda_libs, pool=keep, query=enveda / (e) rule=enveda_libs, pool=remove, query=enveda /
    (b) rule=all, pool=keep, query=prefer_enveda, test_adducts / (c) rule=all, pool=remove, query=enveda / (d) write_enveda
  - 除外: v3 の 4 集合・A3a の 3 集合の正解と、それより前に作った b8 の集合の正解（holdout_expand.Excl の hit:
    生・正規化・scorer_key・identity_group）。集合ごとに holdout_expand.finish（#116 の rebuild_set と点検）
  - 作った後に holdout_expand.t10 で `_t10`（減った分は補充しない）

    .venv/bin/python scripts/research/b8_build_sets.py --count-only   # 候補数だけ（集合を作る前に記録する）
    .venv/bin/python scripts/research/b8_build_sets.py                # 作る
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/cv", "scripts/validate", "scripts/ranker", "scripts/research", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

S = 20261124
N = 200
FOLDS = (1, 2, 3, 4)
TYPE_NO = dict(a=0, b=1, c=2, d=3, e=4)
ORDER = ("a", "e", "b", "c", "d")          # 確定の事前登録: (a) → (e) → (b) → (c) → (d)
SPEC = {   # (rule, pool, query)。d は write_enveda
    "a": ("enveda_libs", "keep", "enveda"),
    "e": ("enveda_libs", "remove", "enveda"),
    "b": ("all", "keep", "prefer_enveda"),
    "c": ("all", "remove", "enveda"),
}
TRAIN_TYPES = ("a", "b", "c", "d")         # 当てはめに使う型（(e) は評価だけ）
ALL_TYPES = ("a", "b", "c", "d", "e")


def seed_of(k: int, t: str) -> int:
    return S + 100 * k + TYPE_NO[t]


def set_name(k: int, t: str, t10: bool = False) -> str:
    return f"b8_f{k}_{t}" + ("_t10" if t10 else "")


def draw(cands, n: int, seed: int) -> list:
    """sorted の候補から seed で最大 n 個（select・select_c2p と同じ形）。sorted で返す。"""
    c = sorted(cands)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(c, size=min(n, len(c)), replace=False).tolist()) if c else []


def candidates(t: str, c1_all, c2_all, c2n_all, ex, pending_groups: set = frozenset()) -> list:
    """型 t の候補（除外 ex の hit を除いた sorted）。(a)(e) は c1、(b) は c2n、(c) は c2。
    pending_groups は --count-only のとき、書かずに引いた前の型の identity_group（数えるだけの近似）。"""
    base = {"a": c1_all, "e": c1_all, "b": c2n_all, "c": c2_all}[t]
    return sorted(k for k in set(base) - ex.hit(base) if ex.grp_of.get(k) not in pending_groups)


def overlap_by_key(keys_raw: set, smiles, ex_ref, ts_key: dict) -> dict:
    """集合の正解（生 inchikey14 と SMILES）と、ex_ref（v3・A3a の正解の Excl）の重なりを鍵の種類ごとに数える。"""
    import mrr25
    canon = {k for k in (mrr25.to_key(s) for s in smiles) if k} | {ts_key[k] for k in keys_raw if k in ts_key}
    sc = {ex_ref.sc_key.get(k) for k in keys_raw} - {None}
    grp = {ex_ref.grp_of.get(k) for k in keys_raw} - {None}
    return dict(raw=len(keys_raw & ex_ref.e_raw), canonical=len(canon & ex_ref.e_key), scorer_key=len(sc & ex_ref.e_key),
                group=len(grp & ex_ref.e_grp))


def main() -> None:
    import build_groups as BG
    import build_holdout_v3 as V3
    import holdout_expand as HE
    import mrr25
    import b8_fold_counts as FC
    ap = argparse.ArgumentParser()
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/b8_build_sets.json"))
    args = ap.parse_args()
    t0 = time.time()
    meta = FC.read_meta()
    st = pd.read_parquet(FC.FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity", "mono_mass",
                                            "n_enveda_spectra"])
    ts = pd.read_parquet(FC.TSTRUCT, columns=["inchikey14", "key", "libmask", "sid"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    coco = BG.coconut_keys()
    test_adducts = set(pd.read_parquet(ROOT / "data/test.parquet", columns=["adduct"]).adduct.astype(str))
    ex = FC.build_excl(st, ts_key)            # v3 の 4 集合・A3a の 3 集合の正解（作った集合を順に足す）
    ex_ref = FC.build_excl(st, ts_key)        # 重なりの記録用（v3・A3a だけ、足さない）
    train_keys = set(meta.inchikey14.cat.categories.astype(str))
    rep = dict(prereg="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/216#issuecomment-7134581003",
               seed_S=S, n=N, order=list(ORDER), count_only=bool(args.count_only), folds={}, sets={})
    for k in FOLDS:
        lab = FC.split_labels(ts, st, k)
        hold = set(lab.inchikey14[lab.fold == "hold"])
        meta_k = meta[meta.inchikey14.astype(str).isin(hold).values]
        meta_k = meta_k.assign(inchikey14=meta_k.inchikey14.astype(str), ingest_lib=meta_k.ingest_lib.astype(str))
        c1_all, c2_all = V3.eligible(meta_k, hold)
        c2n_all = HE.c2n_eligible(meta_k, hold, coco)
        ctx = dict(hold=hold, ex=ex, st=st, rep=rep, ts_pairs=list(ts_key.items()), to_key=mrr25.to_key)
        fk = dict(n_hold=len(hold), candidates={})
        rep["folds"][k] = fk
        pending = set()
        for t in ORDER:
            s = seed_of(k, t)
            name = set_name(k, t)
            if t == "d":
                ev, gkeys = FC.enveda_eligible_fold(meta, st, hold, k, lambda ks: bool(ex.hit(ks)))
                if args.count_only:
                    ev = ev[~ev.identity_group.isin(pending)]
                fk["candidates"][t] = len(ev)
                print(json.dumps(dict(fold=k, type=t, candidates=len(ev))), flush=True)
                if args.count_only:
                    continue
                pick = ev.sample(n=min(N, len(ev)), random_state=s)
                picked = set().union(*[gkeys[g] & train_keys for g in pick.identity_group])
                info = HE.write_enveda(pick, st, meta, FC.TRAIN, HE.SETS_DIR / name, HE.OUT_TRUTH / name, s,
                                       np.random.default_rng(s), f"f{k}{t}")
                info.update(name=name, fold=k, type=t, eligible_new_groups=len(ev), picked=len(pick))
                HE.finish(name, info, picked, ctx)
                (HE.SETS_DIR / name / "HOLDOUT.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str))
                continue
            cands = candidates(t, c1_all, c2_all, c2n_all, ex, pending)
            fk["candidates"][t] = len(cands)
            print(json.dumps(dict(fold=k, type=t, candidates=len(cands))), flush=True)
            keys = draw(cands, N, s)
            if args.count_only:
                pending |= {ex.grp_of.get(x) for x in keys} - {None}
                continue
            rule, pool, query = SPEC[t]
            info = HE.write_holdout(name, keys, rule, meta, np.random.default_rng(s + 1000), s, FC.TRAIN, pool=pool,
                                    query=query, test_adducts=test_adducts if t == "b" else frozenset(),
                                    id_prefix=f"f{k}{t}", truth_root=HE.OUT_TRUTH)
            info.update(fold=k, type=t, eligible_new=len(cands), picked=len(keys))
            HE.finish(name, info, set(keys), ctx)
            hj = HE.SETS_DIR / name / "HOLDOUT.json"
            hj.write_text(json.dumps({**json.loads(hj.read_text()), **{kk: info[kk] for kk in ("fold", "type", "eligible_new", "picked")}},
                                     ensure_ascii=False, indent=2, default=str))
        if args.count_only:
            continue
        for t in ALL_TYPES:
            name = set_name(k, t)
            info = HE.t10(name, set_name(k, t, True), HE.OUT_TRUTH / name / "truth.csv")
            tr = pd.read_csv(HE.OUT_TRUTH / set_name(k, t, True) / "truth.csv", dtype=str)
            info["overlap_v3_a3a"] = overlap_by_key(set(tr.inchikey14), tr.smiles, ex_ref, ts_key)
            assert not any(info["overlap_v3_a3a"].values()), f"{name}: v3/A3a の正解と重なる {info['overlap_v3_a3a']}"
            rk = {}
            for c in ("c1", "c2", "c2p"):
                p = ROOT / f"outputs/holdouts/rk_f{k}_{c}/truth.csv"
                if p.is_file():
                    rt = pd.read_csv(p, dtype=str)
                    rk[c] = dict(raw=len(set(tr.inchikey14) & set(rt.inchikey14)),
                                 group=len({ex.grp_of.get(x) for x in tr.inchikey14} - {None}
                                           & {ex.grp_of.get(x) for x in rt.inchikey14} - {None}))
                else:
                    rk[c] = "truth.csv なし"
            info["overlap_rk_f_k"] = rk
            rep["sets"][set_name(k, t, True)] = info
    rep["seconds"] = round(time.time() - t0, 1)
    HE._dump(rep, args.out)


if __name__ == "__main__":
    main()
