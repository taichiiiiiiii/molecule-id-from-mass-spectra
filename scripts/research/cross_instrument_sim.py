#!/usr/bin/env python3
"""
装置をまたぐ参照の照合（Issue #280、2026-11-27 昼 項目 3）: train の中だけで、同じ構造（inchikey14）・同じ付加イオンの
スペクトルの組のエントロピー類似度（今の lib_max_sim と同じ casmi.spectra.entropy_sim・SearchCfg の既定・
scripts/local_eval/library.py の _prep）を、同じ instrument_type / 別の instrument_type に分けて数える。

数え方は数える前に #280 の本文で固定した。要点:
  - 装置の値は instrument_type の生の文字列。欠け（NaN・空・nan・N/A）は装置とみなさず、その行は組に入れない
  - 対象: 2 つ以上の（生の）装置にスペクトルがある構造を inchikey14 の辞書順に並べ、先に除外（12/6 の判定用 4 集合と
    dev B8 の 20 集合とその _t10 版の正解・drop_keys、holdout_expand.Excl）を除いてから最初の N（既定 2,000）構造
  - 組: 同じ構造・同じ付加イオン（生の文字列）の中のすべてのスペクトルの組（ファイルの順で前の行を問い合わせ側にする。
    entropy_sim の許容幅は問い合わせ側の m/z で決まるのでほぼ対称）
  - 数: (1) スペクトルごとの最大（同じ装置の相手・別の装置の相手）、(2) 群ごとの最大（構造×付加イオン×問い合わせの装置、
    問い合わせ全部 × 参照全部の最大 = lib_max_sim と同じ形）、(3) すべての組。中央値・10/90% 分位・数
  - 副: 装置の系統（family、下の規則）と「問い合わせが timsTOF、参照が timsTOF 以外」
  - 補助: dev と判定用の集合の正解に、隠す規則を当てた後に残る参照の装置（timsTOF あり / timsTOF 以外だけ / 無し）

    /usr/bin/time -l .venv/bin/python scripts/research/cross_instrument_sim.py \\
        --out outputs/local_eval/cross_instrument_sim.json
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/cv", "scripts/validate", "scripts/local_eval", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

TRAIN = ROOT / "data/train.parquet"
FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
JUDGE = ("v3_c1u_t10", "enveda_v1u_t10", "v3_c2_t10", "v3_c2nu_t10")
DEV = tuple(f"b8_f{k}_{t}" for k in (1, 2, 3, 4) for t in "abcde")
EXCL_SETS = JUDGE + DEV + tuple(f"{h}_t10" for h in DEV)
ENVEDA = {"enveda-180", "enveda-np-examples"}
META_COLS = ["inchikey14", "instrument_type", "adduct", "ingest_lib"]
SPEC_COLS = META_COLS + ["ms2_mzs", "ms2_normalized_intensities"]
QS = (0.1, 0.25, 0.5, 0.75, 0.9)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 小さな道具（単体テストの対象）

# inst_value・family は #287 で scripts/local_eval/instrument.py に移した（中身は同じ）。ここでは名前をそのまま使う
from instrument import family, inst_value  # noqa: E402,F401


def pick_structures(inst_sets: dict, excluded: set, n: int) -> tuple[list, dict]:
    """2 つ以上の装置にスペクトルがある構造を辞書順に並べ、除外を先に除いて最初の n 構造。返り値: (構造, 数)。"""
    elig = sorted(k for k, s in inst_sets.items() if len(s) >= 2)
    kept = [k for k in elig if k not in excluded]
    pick = kept[:n]
    last = pick[-1] if pick else None
    n_ex_prefix = sum(1 for k in elig if k in excluded and (last is None or k <= last))
    return pick, dict(n_eligible=len(elig), n_eligible_excluded=len(elig) - len(kept),
                      n_excluded_before_cutoff=n_ex_prefix, n_picked=len(pick), first=pick[0] if pick else None,
                      last=last)


def pair_table(groups: dict, sim) -> pd.DataFrame:
    """groups: (ik14, adduct) -> [(row_id, inst, spec), ...]（ファイルの順）。同じ群の中のすべての組（i < j、i を問い合わせ側）。
    列: ik14・adduct・i・j・inst_i・inst_j・sim。"""
    rows = []
    for (k, a), lst in groups.items():
        for x in range(len(lst)):
            ri, ii, si = lst[x]
            for y in range(x + 1, len(lst)):
                rj, ij, sj = lst[y]
                rows.append((k, a, ri, rj, ii, ij, float(sim(si, sj))))
    return pd.DataFrame(rows, columns=["ik14", "adduct", "i", "j", "inst_i", "inst_j", "sim"])


def directed(pairs: pd.DataFrame) -> pd.DataFrame:
    """組を両向きにした表（q = 問い合わせの行、r = 相手の行）。列: ik14・adduct・q・r・inst_q・inst_r・sim。"""
    a = pairs.rename(columns=dict(i="q", j="r", inst_i="inst_q", inst_j="inst_r"))
    b = pairs.rename(columns=dict(j="q", i="r", inst_j="inst_q", inst_i="inst_r"))
    return pd.concat([a, b], ignore_index=True)[["ik14", "adduct", "q", "r", "inst_q", "inst_r", "sim"]]


def summary(x) -> dict:
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return dict(n=0)
    q = np.quantile(x, QS)
    return dict(n=int(len(x)), mean=round(float(x.mean()), 4), **{f"q{int(p * 100):02d}": round(float(v), 4)
                                                                     for p, v in zip(QS, q)})


def max_tables(d: pd.DataFrame, fam: dict) -> dict:
    """d（directed）から、相手の区分ごとの最大を、スペクトルごと・群ごとに。fam: 生の装置 → 系統。"""
    d = d.copy()
    d["same_raw"] = d.inst_q == d.inst_r
    d["fam_q"] = d.inst_q.map(fam)
    d["fam_r"] = d.inst_r.map(fam)
    d["same_fam"] = d.fam_q == d.fam_r
    out = {}
    kinds = dict(
        same_raw=d.same_raw,
        diff_raw=~d.same_raw,
        diff_raw_same_family=(~d.same_raw) & d.same_fam,
        diff_family=~d.same_fam,
        tims_q__non_tims_r=(d.fam_q == "timsTOF") & (d.fam_r != "timsTOF"),
        tims_q__tims_r=(d.fam_q == "timsTOF") & (d.fam_r == "timsTOF"),
        non_tims_q__tims_r=(d.fam_q != "timsTOF") & (d.fam_r == "timsTOF"),
    )
    spec_max, grp_max = {}, {}
    for name, m in kinds.items():
        s = d[m].groupby("q").sim.max()
        g = d[m].groupby(["ik14", "adduct", "inst_q"]).sim.max()
        spec_max[name], grp_max[name] = s, g
        out[name] = dict(per_spectrum_max=summary(s.values), per_group_max=summary(g.values),
                         per_group_max_median_ci=boot_median_ci(g.values, g.index.get_level_values(0)),
                         all_pairs_directed=summary(d.sim[m].values),
                         n_structures=int(d.ik14[m].nunique()))
    # 同じスペクトル（群）が両方の相手を持つときの差（同じ − 別）
    paired = {}
    for a, b in (("same_raw", "diff_raw"), ("tims_q__tims_r", "tims_q__non_tims_r")):
        js = pd.concat([spec_max[a].rename("a"), spec_max[b].rename("b")], axis=1, join="inner")
        jg = pd.concat([grp_max[a].rename("a"), grp_max[b].rename("b")], axis=1, join="inner")
        paired[f"{a}_minus_{b}"] = dict(
            per_spectrum=dict(a=summary(js.a), b=summary(js.b), diff=summary(js.a - js.b),
                              share_b_ge_a=round(float((js.b >= js.a).mean()), 4) if len(js) else None),
            per_group=dict(a=summary(jg.a), b=summary(jg.b), diff=summary(jg.a - jg.b),
                           share_b_ge_a=round(float((jg.b >= jg.a).mean()), 4) if len(jg) else None))
    out["paired"] = paired
    return out


def boot_median_ci(values, clusters, n_boot: int = 1000, seed: int = 0) -> dict:
    """構造（clusters）ごとに復元抽出したブートストラップの中央値の 95% 区間（不確かさの記述だけ、何も選ばない）。"""
    v = np.asarray(values, np.float64)
    c = np.asarray(clusters)
    if len(v) == 0:
        return dict(n=0)
    uc, inv = np.unique(c, return_inverse=True)
    idx = [np.nonzero(inv == u)[0] for u in range(len(uc))]
    rng = np.random.default_rng(seed)
    meds = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        meds.append(np.median(np.concatenate([v[idx[u]] for u in pick])))
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return dict(n=int(len(v)), n_clusters=int(len(uc)), median=round(float(np.median(v)), 4),
                ci95=[round(float(lo), 4), round(float(hi), 4)], n_boot=n_boot, seed=seed)


def ref_instruments_after_hide(truth_keys: set, drop: set, rule: str, meta_agg: pd.DataFrame,
                               alias: dict) -> dict:
    """正解の構造に、隠す規則を当てた後に残る参照の装置。meta_agg: 列 inchikey14・inst・ingest_lib・n。
    alias: 正解の鍵 → train の inchikey14 の集合（生・scorer_key の別名）。返り値: 区分ごとの正解の数。"""
    m = meta_agg
    hid = m.inchikey14.isin(drop)
    if rule == "enveda_libs":
        hid &= m.ingest_lib.isin(ENVEDA)
    rem = m[~hid]
    fam_of = [family(v) for v in rem.inst]
    by_key = {}
    for k, f in zip(rem.inchikey14, fam_of):
        by_key.setdefault(k, set()).add(f)
    cnt = dict(has_timsTOF=0, non_timsTOF_only=0, missing_inst_only=0, none=0)
    for t in truth_keys:
        fs = set()
        for k in alias.get(t, {t}):
            fs |= by_key.get(k, set())
        if not fs:
            cnt["none"] += 1
        elif "timsTOF" in fs:
            cnt["has_timsTOF"] += 1
        elif fs == {None}:
            cnt["missing_inst_only"] += 1
        else:
            cnt["non_timsTOF_only"] += 1
    return cnt


# ---------------------------------------------------------------- 読み込み

def read_meta_agg() -> tuple[dict, pd.DataFrame]:
    """1 回目: 構造ごとの装置の集合と、(inchikey14, inst, ingest_lib) ごとの行の数。"""
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    sets: dict = {}
    aggs = []
    for b in pf.iter_batches(batch_size=8192, columns=META_COLS):
        d = b.to_pandas()
        inst = [inst_value(v) for v in d.instrument_type]
        for k, v in zip(d.inchikey14, inst):
            if v is not None:
                sets.setdefault(k, set()).add(v)
        d["inst"] = pd.Series(inst, index=d.index, dtype=object)
        aggs.append(d.groupby(["inchikey14", "inst", "ingest_lib"], dropna=False).size().rename("n").reset_index())
    assert all(isinstance(v, str) for s_ in sets.values() for v in s_), "装置の集合に文字列でない値"
    agg = pd.concat(aggs, ignore_index=True).groupby(["inchikey14", "inst", "ingest_lib"], dropna=False).n.sum()
    return sets, agg.reset_index()


def read_spectra(keys: set) -> pd.DataFrame:
    """2 回目: keys の行（SPEC_COLS、ファイルの順、row_id = ファイルの行番号）。"""
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    vs = pa.array(sorted(keys), type=pa.string())
    parts, off = [], 0
    for b in pf.iter_batches(batch_size=8192, columns=SPEC_COLS):
        m = pc.is_in(b.column("inchikey14"), value_set=vs)
        idx = np.nonzero(np.asarray(m.to_numpy(zero_copy_only=False), bool))[0]
        if len(idx):
            d = b.take(pa.array(idx)).to_pandas()
            d["row_id"] = idx + off
            parts.append(d)
        off += b.num_rows
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    import holdout_expand as HE
    import mrr25
    from library import _prep
    from casmi import spectra as sp
    from casmi.library import SearchCfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/cross_instrument_sim.json"))
    args = ap.parse_args()
    t0 = time.time()
    cfg = SearchCfg()

    st = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "identity_group"])
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "key"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    ex = HE.Excl(st, ts_key, mrr25.to_key)
    truths, drops = {}, {}
    for h in EXCL_SETS:
        tr = pd.read_csv(ROOT / f"outputs/local_eval/{h}/truth.csv")
        dr = set((ROOT / f"data/holdouts/{h}/drop_keys.txt").read_text().split())
        truths[h], drops[h] = tr, dr
        ex.add(tr, dr)
    t1 = time.time()

    sets, agg = read_meta_agg()
    t2 = time.time()
    excluded = ex.hit(set(sets))
    pick, sel = pick_structures(sets, excluded, args.n)
    sel.update(n_train_structures=len(sets), n_train_structures_excluded=len(excluded),
               exclusion=dict(sets=list(EXCL_SETS), e_raw=len(ex.e_raw), e_key=len(ex.e_key), e_grp=len(ex.e_grp)))

    df = read_spectra(set(pick))
    t3 = time.time()
    df["inst"] = pd.Series([inst_value(v) for v in df.instrument_type], index=df.index, dtype=object)
    n_rows_all = len(df)
    n_no_adduct = int((df.inst.notna() & df.adduct.isna()).sum())
    df = df[df.inst.notna() & df.adduct.notna()].reset_index(drop=True)
    assert all(isinstance(v, str) for v in df.inst) and all(isinstance(v, str) for v in df.adduct)
    groups: dict = {}
    for k, a, rid, ins, mz, it in zip(df.inchikey14, df.adduct, df.row_id, df.inst, df.ms2_mzs,
                                      df.ms2_normalized_intensities):
        groups.setdefault((k, a), []).append((int(rid), ins, _prep(mz, it)))
    pairs = pair_table(groups, lambda s1, s2: sp.entropy_sim(s1[0], s1[1], s2[0], s2[1], cfg.tol, cfg.ppm_tol))
    t4 = time.time()
    assert np.isfinite(pairs.sim).all(), "entropy_sim に有限でない値"
    d = directed(pairs)
    fam = {v: family(v) for v in set(df.inst)}
    res = dict(issue=280, definition_comment="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/280",
               train_sha256_note="data/train.parquet は元フォルダへの symlink（読み取りだけ）",
               search_cfg=dict(tol=cfg.tol, ppm_tol=cfg.ppm_tol, floor=cfg.floor, topk=cfg.topk, power=cfg.power,
                               ent_weight=cfg.ent_weight),
               selection=sel,
               rows=dict(read=int(n_rows_all), with_instrument_and_adduct=int(len(df)), instrument_but_no_adduct=n_no_adduct,
                         groups_struct_adduct=len(groups),
                         groups_with_pairs=int(sum(len(v) >= 2 for v in groups.values())),
                         groups_with_2plus_inst=int(sum(len({x[1] for x in v}) >= 2 for v in groups.values()))),
               n_pairs_undirected=dict(all=int(len(pairs)), same_raw=int((pairs.inst_i == pairs.inst_j).sum()),
                                       diff_raw=int((pairs.inst_i != pairs.inst_j).sum())),
               family_map=dict(sorted(fam.items())),
               inst_rows_in_pick=df.inst.value_counts().to_dict(),
               adduct_rows_in_pick=df.adduct.value_counts().head(10).to_dict(),
               stats=max_tables(d, fam))
    # 補助: 集合の正解に、隠す規則の後に残る参照の装置
    alias: dict = {}
    sc_of = dict(zip(st.inchikey14, st.scorer_key))
    by_sc: dict = {}
    for k, s in sc_of.items():
        by_sc.setdefault(s, set()).add(k)
    aux = {}
    for h in JUDGE + DEV:
        p = ROOT / f"data/holdouts/{h}/hide_rule.json"
        rule = json.loads(p.read_text())["rule"] if p.is_file() else "all"
        tk = set(truths[h].inchikey14.astype(str))
        for t in tk:
            alias[t] = {t} | by_sc.get(t, set()) | by_sc.get(sc_of.get(t), set())
        q = pq.read_table(ROOT / f"data/holdouts/{h}/test.parquet", columns=["instrument_type"]).column(0).to_pylist()
        aux[h] = dict(rule=rule, n_truth=len(tk), query_instruments=pd.Series(q).value_counts().to_dict(),
                      **ref_instruments_after_hide(tk, drops[h], rule, agg, alias))
    res["truth_refs_after_hide"] = aux
    fam_sets = {}
    for k, v in zip(agg.inchikey14, agg.inst):
        fam_sets.setdefault(k, set()).add(family(v))
    fs = [s_ - {None} for s_ in fam_sets.values()]
    res["train_structures_by_timsTOF"] = dict(
        timsTOF_only=sum(1 for s_ in fs if s_ == {"timsTOF"}),
        timsTOF_and_other=sum(1 for s_ in fs if "timsTOF" in s_ and len(s_) >= 2),
        no_timsTOF=sum(1 for s_ in fs if s_ and "timsTOF" not in s_),
        no_instrument=sum(1 for s_ in fs if not s_))
    res["seconds"] = dict(exclusion=round(t1 - t0, 1), meta=round(t2 - t1, 1), spectra=round(t3 - t2, 1),
                          pairs=round(t4 - t3, 1), total=round(time.time() - t0, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
