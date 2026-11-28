#!/usr/bin/env python3
"""
Issue #300（計画 2026-11-28 昼 項目 2）: 天然物の正解での dev の内訳と、「timsTOF の問い合わせ × 天然物の正解」の
集合を作れる分子数を fold ごとに数える（読むだけ。集合は作らない）。

定義は #300 の本文（数える前に固定）のとおり。要点:
  (a) 天然物 = 正解の正準の鍵（mrr25.to_key / train_structs.key）が v2 プール pool_meta の src ∈ {2, 3} の key に入る。
      補助（記録だけ）: 生の inchikey14 で入るか、enveda-np-examples の 250 構造（libmask bit 9）に入るか
  (b) dev の内訳: 台帳の既存の per_molecule.csv（exp003、対照 exp007・014・015、候補 exp016・017・018）を
      型 × 天然物 / それ以外に分け、n・MRR・差・対応のある bootstrap の 95% 区間（evaluate.paired_bootstrap）。
      n < 30 の部分は n だけ
  (c) fold ごと（と np の行）に、train の構造で ① hold ∩ timsTOF ∩ 天然物 → ② 質量 157〜1159 → ③ MSG と重ならない
      （glacier_l60.msg_key_sets の規則）→ ④ 既存の集合の正解と重ならない（holdout_expand の 4 鍵）。④ の後を
      a′（公開ライブラリの参照あり）と d′（全部）に分ける。fold 0 は鍵と所属だけ（スコア・rr・候補は読まない）
  (d) 足す条件: a′ ⇔ n_a′ ≥ 150、d′ ⇔ （a′ を作るなら min(200, n_a′) を除いた残り）≥ 150。fold 0 と fold 1〜4 の和で

    .venv/bin/python scripts/research/np_subset_counts.py \
        --out outputs/local_eval/np_subset/np_subset_counts.json --copy docs/research/np-subset-counts.json
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
for p in ("scripts/cv", "scripts/validate", "scripts/ranker", "scripts/bench", "scripts/research",
          "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
POOL = ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet"
TRAIN = ROOT / "data/train.parquet"
LEDGER = ROOT / "outputs/local_eval/ranker/lambdarank"
TYPES = ("a", "b", "c", "d", "e")
DEV_FOLDS = (1, 2, 3, 4)
EXP_SINGLE = "exp003"
EXP_CTRL = ("exp007", "exp014", "exp015")
EXP_CAND = ("exp016", "exp017", "exp018")
MIN_N = 30                      # (b): これ未満は n だけ
MIN_SET, DRAW = 150, 200        # (d)
MASS_LO, MASS_HI = 157.0, 1159.0
NP_BIT = 1 << 9
# ④ の除外に使う集合（truth.csv の置き場所）
ORIG = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")                       # outputs/holdouts/<h>/truth.csv
A3A = ("v3_c1b", "v3_c2nb", "enveda_v1b")                              # outputs/local_eval/<h>/truth.csv
T10 = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")     # outputs/local_eval/<h>/truth.csv
B8 = tuple(f"b8_f{k}_{t}_t10" for k in DEV_FOLDS for t in TYPES)        # outputs/local_eval/<h>/truth.csv


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 純粋な関数（テストあり）

def is_np(canon_keys, coconut: set) -> np.ndarray:
    """(a) 正準の鍵が COCONUT（pool_meta の src 2/3 の key）に入るか。None・空は偽。"""
    return np.array([bool(k) and str(k) in coconut for k in canon_keys], dtype=bool)


def msg_hit(raw, canon, ms: dict) -> np.ndarray:
    """(c)③ glacier_l60.mark_msg と同じ規則: 生の InChIKey14 が MSG train ∪ val の inchikey[:14]、または
    正準の鍵が MSG train ∪ val の SMILES の to_key に一致すれば重なり。"""
    raw = pd.Series(list(raw), dtype=object).astype(str)
    canon = pd.Series(list(canon), dtype=object)
    return (raw.isin(ms["raw_tv"]) | canon.isin(ms["key_tv"])).to_numpy()


def bootstrap(a, b):
    from evaluate import paired_bootstrap
    return paired_bootstrap(np.asarray(a, float), np.asarray(b, float))


def seed_mean(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """同じ分子の並びの per_molecule を (set, molecule_id) で揃えて rr を平均する。分子の集合が違えば止める。"""
    base = frames[0][["set", "molecule_id", "type"]].copy()
    keys = set(zip(base.set, base.molecule_id))
    rr = []
    for f in frames:
        assert len(f) == len(base) and set(zip(f.set, f.molecule_id)) == keys, "seed の間で分子が違う"
        m = base.merge(f[["set", "molecule_id", "rr"]], on=["set", "molecule_id"], how="left", validate="one_to_one")
        rr.append(m.rr.to_numpy(float))
    base["rr"] = np.mean(rr, axis=0)
    return base


def cell(rr_new, rr_ref, n_min: int = MIN_N) -> dict:
    """(b) の 1 マス。n < n_min なら n だけ。"""
    n = int(len(rr_new))
    if n < n_min:
        return dict(n=n)
    lo, hi = bootstrap(rr_new, rr_ref)
    return dict(n=n, mrr=round(float(np.mean(rr_new)), 4), mrr_ref=round(float(np.mean(rr_ref)), 4),
                delta=round(float(np.mean(rr_new) - np.mean(rr_ref)), 4), lo=round(lo, 4), hi=round(hi, 4))


def breakdown(df: pd.DataFrame, new: str, ref: str, n_min: int = MIN_N) -> dict:
    """df: 列 type・np（bool）・new・ref。型 × 天然物 / それ以外 / 全体。"""
    out = {}
    for t in TYPES:
        d = df[df.type.astype(str) == t]
        out[t] = {g: cell(x[new].to_numpy(), x[ref].to_numpy(), n_min)
                  for g, x in (("np", d[d.np]), ("non_np", d[~d.np]), ("all", d))}
    return out


def stage_counts(t: pd.DataFrame) -> dict:
    """(c) t: 構造ごとの表（列 tims・np・mass_ok・msg・excl・public・group、いずれも bool、group は identity_group）。
    ① tims ∩ np → ② mass_ok → ③ ¬msg → ④ ¬excl。④ の後を a′（public）と d′（全部）に。"""
    s1 = t.tims & t.np
    s2 = s1 & t.mass_ok
    s3 = s2 & ~t.msg
    s4 = s3 & ~t.excl
    return dict(n_structures=int(len(t)), n_tims=int(t.tims.sum()), n1_tims_np=int(s1.sum()), n2_mass=int(s2.sum()),
                n3_no_msg=int(s3.sum()), n4_no_sets=int(s4.sum()), n4_groups=int(t.group[s4].nunique()),
                a_prime=int((s4 & t.public).sum()), d_prime=int(s4.sum()),
                drop_msg=int((s2 & t.msg).sum()), drop_sets=int((s3 & t.excl).sum()),
                n1_tims_nonnp=int((t.tims & ~t.np).sum()))


def decide(n_a: int, n_all: int, min_set: int = MIN_SET, draw: int = DRAW) -> dict:
    """(d) a′ ⇔ n_a ≥ min_set、d′ ⇔ （a′ を作るなら min(draw, n_a) を除いた残り）≥ min_set。"""
    make_a = n_a >= min_set
    rest = n_all - (min(draw, n_a) if make_a else 0)
    make_d = rest >= min_set
    return dict(n_a_prime=int(n_a), n_d_prime_pool=int(n_all), make_a_prime=bool(make_a),
                d_prime_remainder=int(rest), make_d_prime=bool(make_d), add=bool(make_a or make_d))


# ---------------------------------------------------------------- 読み込み

def read_meta() -> pd.DataFrame:
    """train.parquet の 5 列を 8192 行ずつ読み、構造ごとに timsTOF・公開の参照・中性質量の中央値にまとめる。"""
    from casmi import chem
    import build_holdout_v3 as V3
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    parts = []
    for b in pf.iter_batches(batch_size=8192, columns=["inchikey14", "ingest_lib", "instrument_type", "precursor_mz",
                                                       "adduct"]):
        d = b.to_pandas()
        nm = chem.neutral_mass(d.precursor_mz.values.astype(np.float64), d.adduct.astype(str).values)
        parts.append(pd.DataFrame(dict(inchikey14=d.inchikey14.astype(str).values,
                                       tims=(d.instrument_type.astype(str) == "timsTOF").values,
                                       public=d.ingest_lib.astype(str).isin(V3.PUBLIC).values,
                                       neutral_mass=nm)))
    m = pd.concat(parts, ignore_index=True)
    m["inchikey14"] = m.inchikey14.astype("category")
    g = m.groupby("inchikey14", observed=True)
    out = pd.DataFrame(dict(tims=g.tims.any(), public=g.public.any(), mass=g.neutral_mass.median(),
                            n_rows=g.size()))
    out.index = out.index.astype(str)
    return out, len(m)


def truth_path(h: str) -> Path:
    return ROOT / (f"outputs/holdouts/{h}/truth.csv" if h in ORIG else f"outputs/local_eval/{h}/truth.csv")


def set_exclusion(st, ts_key, to_key):
    """④: 判定用 v3・A3a・_t10・B8 の正解と drop_keys（B8 は元の集合の drop_keys も）。正解の鍵だけを使う。"""
    import holdout_expand as HE
    frames, drops, used = [], [], []
    for h in ORIG + A3A + T10 + B8:
        frames.append(pd.read_csv(truth_path(h), dtype=str)[["inchikey14", "smiles"]])
        for name in {h, h[:-4] if h.endswith("_t10") else h}:
            p = ROOT / f"data/holdouts/{name}/drop_keys.txt"
            if p.is_file():
                drops.append(set(p.read_text().split()))
                used.append(name)
    e_raw, e_key, e_grp = HE.exclusion_sets(frames, drops, to_key, ts_key, st)
    sc_key = dict(zip(st.inchikey14, st.scorer_key))
    grp_of = dict(zip(st.inchikey14, st.identity_group))
    hit = lambda keys: HE.excluded(keys, e_raw, e_key, e_grp, ts_key, sc_key, grp_of)
    return hit, dict(sets=list(ORIG + A3A + T10 + B8), drop_keys_from=sorted(used), e_raw=len(e_raw),
                     e_key=len(e_key), e_grp=len(e_grp))


def dev_np_flags(sets, to_key, coconut, np_raw, np_key) -> pd.DataFrame:
    rows = []
    for h in sets:
        tr = pd.read_csv(truth_path(h), dtype=str)
        k = [to_key(s) for s in tr.smiles]
        rows.append(pd.DataFrame(dict(set=h, molecule_id=tr.molecule_id, key=k, inchikey14=tr.inchikey14)))
    d = pd.concat(rows, ignore_index=True)
    d["np"] = is_np(d.key, coconut)
    d["np_raw"] = d.inchikey14.isin(coconut).to_numpy()
    d["np_examples"] = (d.inchikey14.isin(np_raw) | d.key.isin(np_key)).to_numpy()
    d["key_none"] = d.key.isna() | (d.key == "")
    return d


def read_pm(exp: str) -> pd.DataFrame:
    return pd.read_csv(LEDGER / exp / "per_molecule.csv", dtype=dict(set=str, molecule_id=str, type=str))


# ---------------------------------------------------------------- 本体

def main(argv=None) -> dict:
    import b8_fold_counts as B8F
    import glacier_l60 as GL
    import mrr25
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/np_subset/np_subset_counts.json"))
    ap.add_argument("--copy", default=str(ROOT / "docs/research/np-subset-counts.json"))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    t0 = time.time()
    pm = pd.read_parquet(POOL, columns=["key", "src"])
    coconut = set(pm.key[pm.src.isin([2, 3])].astype(str))
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity"])
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "key", "libmask", "sid"])
    ts_key = dict(zip(ts.inchikey14, ts.key))
    is_npex = (ts.libmask.astype("int64") & NP_BIT) != 0
    np_raw, np_key = set(ts.inchikey14[is_npex]), set(ts.key[is_npex])
    res = dict(issue=300, definitions="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/300",
               np_definition="canonical key in pool_meta src in {2,3}", n_coconut_keys=len(coconut),
               n_np_examples=int(is_npex.sum()))

    # ---- (b) dev の内訳
    exps = (EXP_SINGLE,) + EXP_CTRL + EXP_CAND
    pms = {e: read_pm(e) for e in exps}
    dev_sets = sorted(set(pms[EXP_SINGLE].set))
    assert set(dev_sets) == set(B8), "exp003 の集合が B8 の 20 集合と違う"
    flags = dev_np_flags(dev_sets, mrr25.to_key, coconut, np_raw, np_key)
    assert flags.set.astype(str).str.startswith("b8_").all()      # dev だけ（判定用の集合の rr は読まない）
    one = pms[EXP_SINGLE].merge(flags[["set", "molecule_id", "np", "np_raw", "np_examples"]],
                                on=["set", "molecule_id"], how="left", validate="one_to_one")
    assert one.np.notna().all()
    one["np"] = one.np.astype(bool)
    ctrl, cand = seed_mean([pms[e] for e in EXP_CTRL]), seed_mean([pms[e] for e in EXP_CAND])
    assert set(zip(ctrl.set, ctrl.molecule_id)) == set(zip(cand.set, cand.molecule_id))
    three = ctrl.rename(columns={"rr": "rr_ctrl3"}).merge(cand[["set", "molecule_id", "rr"]].rename(columns={"rr": "rr_cand3"}),
                                                       on=["set", "molecule_id"], validate="one_to_one")
    three = three.merge(flags[["set", "molecule_id", "np"]], on=["set", "molecule_id"], how="left", validate="one_to_one")
    three["np"] = three.np.astype(bool)
    res["dev_breakdown"] = dict(
        rule=f"n < {MIN_N} は n だけ。区間 = evaluate.paired_bootstrap（2,000 回、seed 0）",
        n_key_none=int(flags.key_none.sum()),
        np_count_by_type={t: dict(n=int((one.type == t).sum()), np=int(one.np[one.type == t].sum()),
                                  np_raw=int(one.np_raw[one.type == t].sum()),
                                  np_examples=int(one.np_examples[one.type == t].sum())) for t in TYPES},
        exp003_vs_score_prior_frag=dict(n=len(one), cells=breakdown(one, "rr", "rr_ctrl")),
        cand3_vs_ctrl3=dict(n=len(three), ctrl=list(EXP_CTRL), cand=list(EXP_CAND),
                            np_count_by_type={t: int(three.np[three.type == t].sum()) for t in TYPES},
                            cells=breakdown(three, "rr_cand3", "rr_ctrl3")))
    print(json.dumps(dict(np_by_type=res["dev_breakdown"]["np_count_by_type"]), ensure_ascii=False), flush=True)

    # ---- (c) 作れる数
    meta, n_train_rows = read_meta()
    ms = GL.msg_key_sets(ROOT, a.workers)
    hit, ex_info = set_exclusion(st[["inchikey14", "scorer_key", "identity_group"]], ts_key, mrr25.to_key)
    grp_of = dict(zip(st.inchikey14, st.identity_group))
    base = ts[["inchikey14", "key"]].copy()
    assert base.inchikey14.isin(meta.index).all(), "train_structs に train の行が無い構造がある"
    m = meta.reindex(base.inchikey14)
    base["tims"] = m.tims.to_numpy(bool)
    base["public"] = m.public.to_numpy(bool)
    base["mass_ok"] = ((m.mass >= MASS_LO) & (m.mass <= MASS_HI)).to_numpy(bool)
    base["np"] = is_np(base.key, coconut)
    base["msg"] = msg_hit(base.inchikey14, base.key, ms)
    exc = hit(set(base.inchikey14))
    base["excl"] = base.inchikey14.isin(exc).to_numpy()
    base["group"] = base.inchikey14.map(grp_of)
    folds = {}
    for k in (0,) + DEV_FOLDS:
        lab = B8F.split_labels(ts, st, k)
        lab_of = dict(zip(lab.inchikey14, lab.fold))
        lb = base.inchikey14.map(lab_of)
        folds[k] = stage_counts(base[(lb == "hold").to_numpy()])
        if k == 0:
            folds["np"] = stage_counts(base[(lb == "np").to_numpy()])
    sums = {key: int(sum(folds[k][key] for k in DEV_FOLDS)) for key in folds[1]}
    res["fold_counts"] = dict(rule="① hold∩timsTOF∩天然物 → ② 質量 157〜1159 → ③ MSG 1.0 train∪val と重ならない → "
                                   "④ 既存の集合の正解と重ならない（4 鍵）。a′ = ④ ∩ 公開の参照あり、d′ = ④",
                              by_fold={str(k): v for k, v in folds.items()}, dev_folds_sum=sums,
                              msg=dict(rows=ms["n_rows"], smiles=ms["n_smiles"], key_none=ms["n_key_none"],
                                       raw_tv=len(ms["raw_tv"]), key_tv=len(ms["key_tv"])),
                              set_exclusion=ex_info, n_train_rows=n_train_rows, n_train_structs=len(ts))
    res["decision"] = dict(holdout_fold0=decide(folds[0]["a_prime"], folds[0]["d_prime"]),
                           dev_folds_1_4=decide(sums["a_prime"], sums["d_prime"]),
                           note="np の行は (d) の数に入れない（#300 (d)）")
    res["inputs_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in (FOLDS, TSTRUCT, POOL)}
    res["inputs_sha256"].update({f"{e}/per_molecule.csv": sha256(LEDGER / e / "per_molecule.csv") for e in exps})
    res["seconds"] = round(time.time() - t0, 1)
    for p in (a.out, a.copy):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(dict(fold_counts=res["fold_counts"]["by_fold"], sums=sums, decision=res["decision"]),
                     ensure_ascii=False), flush=True)
    return res


if __name__ == "__main__":
    main()
