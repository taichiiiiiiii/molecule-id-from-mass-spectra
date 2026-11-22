#!/usr/bin/env python3
"""
検証 v3 の拡大（Issue #199、#195 A3a）。fold-0 の split の hold から新しい分子で v3 の各型を大きくする。

  型と元のスクリプト:
    v3_c1     : scripts/cv/build_holdout_v3.py の select の c1（hold ∩ 質量 157〜1159 ∩ enveda 系と公開ライブラリの両方）、
                write_holdout(rule=enveda_libs, pool=keep, query=enveda)
    v3_c2n    : build_holdout_v3.select_c2p（質量 ∩ hold ∩ COCONUT〈pool_meta.src ∈ {2,3}〉）、write_holdout(rule=all, pool=keep,
                query=prefer_enveda, test_adducts=data/test.parquet の付加イオン)（scripts/ranker/build_groups.py --c2p と同じ）
    enveda_v1 : commit 072ec66 の scripts/cv/build_holdout.py（sha256 fa795e05…、既定の引数）。fold_identity 0 ∩ 質量 157〜1159 ∩
                n_enveda_spectra > 0 を identity_group ごとに 1 行、pool.sample(n=200, random_state=20261102)、
                問い合わせは群のキーの enveda 系の行すべて（付加イオンでは絞らない）、構造ごとに 1〜6 本

  元の 4 集合の正解の除き方（raw・正準・別名・同じ identity_group）:
    E_raw = 各集合の drop_keys.txt（#116 の書き足し後）∪ truth.inchikey14、E_key = mrr25.to_key(truth.smiles) ∪
    train_structs.key(E_raw) ∪ E_raw、E_grp = structures.parquet で inchikey14 か scorer_key が E_raw ∪ E_key に入る
    identity_group。候補 k は k ∈ E_raw ∪ E_key、train_structs.key(k) ∈ E_key、scorer_key(k) ∈ E_key、identity_group(k) ∈ E_grp
    のどれかなら除く。

  事前登録（#199、確定）: seed 20261123。
    v3_c1b     = select(..., 200, seed, exclude) の c1 だけ（select は c1 を先に同じ rng で引くので c2 を捨てても c1 は変わらない）
    v3_c2nb    = select_c2p(..., 200, seed+10, exclude)（v3_c1b も除外に足す）
    enveda_v1b = enveda_v1 型の群から sample(n=200, random_state=seed+20)（v3_c1b・v3_c2nb も除外に足す）
  作った後に新しい集合にも #116 の rebuild_set（同じ train_structs・同じ to_key）を当てる。`_t10` は 10 種の付加イオンの行だけに
  絞った版（残り 0 本の分子は除く）、`*u_t10` は元＋新の合併。

    .venv/bin/python scripts/cv/holdout_expand.py count --out outputs/local_eval/a3a_1123/counts.json
    .venv/bin/python scripts/cv/holdout_expand.py regen --scratch <scratch> --out outputs/local_eval/a3a_1123/regen.json
    .venv/bin/python scripts/cv/holdout_expand.py build --out outputs/local_eval/a3a_1123/build.json
    .venv/bin/python scripts/cv/holdout_expand.py summary --out outputs/local_eval/a3a_1123/summary.json   # run.py の対照の後

  置き場所: 集合は data/holdouts/<名前>/（クローンの実のディレクトリ）、truth.csv は outputs/local_eval/<名前>/truth.csv。
  outputs/holdouts・outputs/holdout_eval は元のフォルダへのリンクなので書かない。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ORIG = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
# 隠れテストの 10 種の付加イオン（データのページの一覧、文字列の完全一致）。見える test の 7 種はこの部分集合
TEST10 = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
          "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
VISIBLE7 = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M+Cl]-")   # 見える test の 7 種（参考）
FOLDS = ROOT / "data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet"
SPLIT = ROOT / "data/ext/foldsafe/split.parquet"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
TRAIN = ROOT / "data/train.parquet"
SETS_DIR = ROOT / "data/holdouts"
OUT_TRUTH = ROOT / "outputs/local_eval"
SEED = 20261123   # 事前登録 (b)。引き直さない
N_MAX = 200        # 事前登録 (a)
ORIG_FILES = ("test.parquet", "drop_keys.txt", "hide_rule.json", "HOLDOUT.json", "sample_submission.csv",
              "drop_keys_manifest.json")
UNION = {"v3_c1": ("v3_c1b", "v3_c1u"), "enveda_v1": ("enveda_v1b", "enveda_v1u"), "v3_c2n": ("v3_c2nb", "v3_c2nu")}
HIDE_KEYS = ("rule", "pool", "enveda_libs")   # 事前登録 第 2 版の指摘 3: 比べる鍵はこの 3 つだけ


# ---------------------------------------------------------------- 小さな道具（単体テストの対象）

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


META_COLS = ["inchikey14", "ingest_lib", "precursor_mz", "adduct"]
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
             "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_orig", "collision_energy_ev",
             "collision_energy_orig_units"]   # build_holdout.py・build_holdout_v3.py の test.parquet と同じ列


def hid(prefix: str, s: str, n: int) -> str:
    """build_holdout.py と同じ id（sha1 の先頭 n 文字に接頭を付ける）。"""
    return prefix + hashlib.sha1(s.encode()).hexdigest()[:n]


def read_rows_at(train: Path, rows: np.ndarray, batch_size: int = 8192) -> pd.DataFrame:
    """train.parquet の行番号 rows（昇順）の全列を、8192 行ずつ流して取り出す（_row 列 = 全体の行番号、rows の順）。"""
    pf = pq.ParquetFile(train, buffer_size=1 << 20, pre_buffer=False)
    rows = np.asarray(rows, dtype=np.int64)
    parts, start = [], 0
    for b in pf.iter_batches(batch_size=batch_size):
        n = b.num_rows
        lo, hi = np.searchsorted(rows, [start, start + n])
        if hi > lo:
            parts.append(pa.Table.from_batches([b]).take(pa.array(rows[lo:hi] - start)).to_pandas().assign(_row=rows[lo:hi]))
        start += n
    return pd.concat(parts, ignore_index=True).sort_values("_row").reset_index(drop=True)


def t10_mask(adducts: pd.Series) -> np.ndarray:
    """事前登録 (c): 隠れテストの 10 種の付加イオン（データのページの一覧、文字列の完全一致）の行だけ True。"""
    return adducts.astype(str).isin(TEST10).values


def t10_filter(test: pd.DataFrame, truth: pd.DataFrame, sub: pd.DataFrame):
    """test の行を 10 種に絞り、残りの問い合わせが 0 本の分子を truth・sample_submission からも除く（drop_keys はそのまま）。"""
    t = test[t10_mask(test.adduct)].reset_index(drop=True)
    keep = set(t.molecule_id)
    return t, truth[truth.molecule_id.isin(keep)].reset_index(drop=True), sub[sub.molecule_id.isin(keep)].reset_index(drop=True)


def union(parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
    """(test, truth, sub) を縦に足す。molecule_id・spectrum_id がぶつからないことを assert する。"""
    t = pd.concat([p[0] for p in parts], ignore_index=True)
    tr = pd.concat([p[1] for p in parts], ignore_index=True)
    sb = pd.concat([p[2] for p in parts], ignore_index=True)
    assert tr.molecule_id.is_unique and sb.molecule_id.is_unique, "合併で molecule_id がぶつかる"
    assert t.spectrum_id.is_unique, "合併で spectrum_id がぶつかる"
    return t, tr, sb


def hide_rule_of(d: Path):
    p = d / "hide_rule.json"
    return json.loads(p.read_text()) if p.is_file() else None


def same_hide(a, b) -> bool:
    """hide_rule の rule・pool・enveda_libs だけを比べる（元の v3_c1 には query の鍵が無い）。"""
    if a is None or b is None:
        return a is None and b is None
    return all(a.get(k) == b.get(k) for k in HIDE_KEYS)


def _dump(obj, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def read_meta(train: Path, cols=META_COLS, batch_size: int = 8192) -> pd.DataFrame:
    """train.parquet の必要な列だけを 8192 行ずつ読む（buffer 1 MiB・pre_buffer なし）。文字列は str（object）にそろえる
    （行の順はファイルの順）。#116 の read_table(filters) と同じ行・同じ順を出す。"""
    pf = pq.ParquetFile(train, buffer_size=1 << 20, pre_buffer=False)
    out = pd.concat([b.to_pandas() for b in pf.iter_batches(batch_size=batch_size, columns=list(cols))], ignore_index=True)
    for c in out.columns:   # concat で category の種類が違うと object に戻るので、全部 str にそろえる
        if out[c].dtype != np.float64:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
            out[c] = out[c].map(lambda v: v if v is None else str(v))
    return out


def exclusion_sets(truth_frames: list[pd.DataFrame], drop_lists: list[set], to_key, ts_key: dict,
                   st: pd.DataFrame) -> tuple[set, set, set]:
    """(E_raw, E_key, E_grp)（上の定義）。truth_frames は inchikey14・smiles の列を持つ表、st は inchikey14・scorer_key・identity_group。"""
    e_raw = set()
    for d in drop_lists:
        e_raw |= set(d)
    e_key = set()
    for t in truth_frames:
        e_raw |= set(t.inchikey14.astype(str))
        e_key |= {k for k in (to_key(s) for s in t.smiles.astype(str)) if k}
    e_key |= {ts_key[k] for k in e_raw if k in ts_key}
    e_key |= e_raw
    hit = st.inchikey14.isin(e_raw | e_key) | st.scorer_key.isin(e_raw | e_key)
    return e_raw, e_key, set(st.identity_group[hit])


def excluded(keys, e_raw: set, e_key: set, e_grp: set, ts_key: dict, sc_key: dict, grp_of: dict) -> set:
    """keys のうち除外に当たるもの（raw・正準・scorer_key・群のどれか）。"""
    return {k for k in keys if k in e_raw or k in e_key or ts_key.get(k) in e_key or sc_key.get(k) in e_key
            or grp_of.get(k) in e_grp}


class Excl:
    """元の集合（＋作った新しい集合）の正解の除外。add で集合を足す。"""

    def __init__(self, st: pd.DataFrame, ts_key: dict, to_key):
        self.ts_key, self.to_key = ts_key, to_key
        self.sc_key = dict(zip(st.inchikey14, st.scorer_key))
        self.grp_of = dict(zip(st.inchikey14, st.identity_group))
        self.st, self.frames, self.drops = st, [], []
        self.e_raw, self.e_key, self.e_grp = set(), set(), set()

    def add(self, truth: pd.DataFrame, drop: set) -> None:
        self.frames.append(truth)
        self.drops.append(set(drop))
        self.e_raw, self.e_key, self.e_grp = exclusion_sets(self.frames, self.drops, self.to_key, self.ts_key, self.st)

    def hit(self, keys) -> set:
        return excluded(keys, self.e_raw, self.e_key, self.e_grp, self.ts_key, self.sc_key, self.grp_of)


# ---------------------------------------------------------------- 候補（元のスクリプトと同じ条件）

def c1_eligible(meta: pd.DataFrame, hold: set) -> list:
    """build_holdout_v3.select と同じ c1 の条件の全候補（sorted）。"""
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import build_holdout_v3 as V3
    return V3.eligible(meta, hold)[0]


def c2n_eligible(meta: pd.DataFrame, hold: set, coco: set) -> list:
    """build_holdout_v3.select_c2p と同じ条件（質量 157〜1159 ∩ hold ∩ COCONUT）の全候補（sorted）。"""
    mass = meta.groupby("inchikey14", observed=True).neutral_mass.median()
    ok = set(mass[(mass >= 157.0) & (mass <= 1159.0)].index.astype(str))
    return sorted(ok & set(hold) & set(coco))


def enveda_pool(st: pd.DataFrame, fold: int = 0) -> pd.DataFrame:
    """commit 072ec66 の build_holdout.py（enveda_v1 を作った版、sha256 fa795e05…、既定の引数）と同じプール:
    fold_identity == fold ∩ mono_mass 157〜1159 ∩ n_enveda_spectra > 0 を identity_group ごとに最初の 1 行（st の順）。
    付加イオン・ppm・装置では絞らない（その版には無い）。"""
    base = (st.fold_identity == fold) & st.mono_mass.between(157, 1159)
    return st[base & (st.n_enveda_spectra > 0)].drop_duplicates("identity_group")


def enveda_eligible(meta: pd.DataFrame, st: pd.DataFrame, hold: set, bad_keys_fn=None) -> tuple[pd.DataFrame, dict]:
    """enveda_v1 型の候補（enveda_pool の群のうち、(1) 群の train に載るキーがすべて hold、(2) bad_keys_fn(群のキー) が偽、
    (3) 群のキーのどれかに ingest_lib が enveda で始まる train の行がある）。返り値: (群の行, 群 → キーの集合)。"""
    pool = enveda_pool(st)
    grp = st[st.identity_group.isin(set(pool.identity_group))]
    gkeys = {}
    for r in grp.itertuples():
        gkeys.setdefault(r.identity_group, set()).update((r.inchikey14, r.scorer_key))
    train_keys = set(meta.inchikey14.astype(str).unique())
    env_keys = set(meta.inchikey14[meta.ingest_lib.astype(str).str.startswith("enveda")].astype(str).unique())
    ok = []
    for g in pool.identity_group:
        ks = gkeys[g]
        good = (ks & train_keys) <= hold and bool(ks & env_keys)
        if good and bad_keys_fn is not None and bad_keys_fn(ks):
            good = False
        ok.append(good)
    return pool[np.array(ok, dtype=bool)], gkeys


def orig_excl(st: pd.DataFrame, ts_key: dict, names=ORIG) -> Excl:
    """元の 4 集合の正解の除外（#116 の書き足し後の drop_keys を使う）。"""
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import mrr25
    from casmi import chem  # noqa: F401  （chem の import で RDKit を確かめる）
    ex = Excl(st, ts_key, mrr25.to_key)
    for h in names:
        ex.add(pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv"),
               set((ROOT / f"data/holdouts/{h}/drop_keys.txt").read_text().split()))
    return ex


def load_context():
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    import mrr25
    cols = ["inchikey14", "ingest_lib", "instrument_type", "precursor_mz", "adduct", "precursor_error_ppm"]
    pf = pq.ParquetFile(TRAIN, buffer_size=1 << 20, pre_buffer=False)
    meta = pd.concat([b.to_pandas() for b in pf.iter_batches(batch_size=8192, columns=cols)], ignore_index=True)
    meta["neutral_mass"] = chem.neutral_mass(meta.precursor_mz.values.astype(np.float64), meta.adduct.astype(str).values)
    split = pd.read_parquet(SPLIT)
    st = pd.read_parquet(FOLDS, columns=["inchikey14", "scorer_key", "identity_group", "fold_identity", "mono_mass",
                                         "n_enveda_spectra"])
    ts = pd.read_parquet(TSTRUCT, columns=["inchikey14", "key"])
    return meta, set(split.inchikey14[split.fold == "hold"]), st, dict(zip(ts.inchikey14, ts.key)), mrr25


# ---------------------------------------------------------------- 作成（#199 事前登録の手順 6）

def write_enveda(pick: pd.DataFrame, st: pd.DataFrame, meta: pd.DataFrame, train: Path, out: Path, truth_dir: Path,
                 seed: int, rng, id_prefix: str = "") -> dict:
    """commit 072ec66 の build_holdout.py の後半の写し（行の読み方だけ read_rows に替え、id 接頭を足した）。"""
    pick = pick.copy()
    pick["scenario"] = "no_reference"
    groups = set(pick.identity_group)
    grp = st[st.identity_group.isin(groups)][["inchikey14", "scorer_key", "identity_group"]]
    g_scn = dict(zip(pick.identity_group, pick.scenario))
    keys_all = {}
    for r in grp.itertuples():
        for k in (r.inchikey14, r.scorer_key):
            keys_all[k] = g_scn[r.identity_group]
    # str にする（category の groupby は空の群も回り、rng がずれる。072ec66 では object だった）
    scn = meta.inchikey14.astype(str).map(keys_all)
    is_enveda = meta.ingest_lib.astype(str).str.startswith("enveda")
    out.mkdir(parents=True, exist_ok=True)
    (out / "drop_keys.txt").write_text(format_keys(set(keys_all)))
    # 072ec66 と同じ乱数の使い方: 群ごとに integers(1,7) と integers(1<<31) を順に引く（群は inchikey14 の str 順）
    rows = np.flatnonzero((scn.notna() & is_enveda).to_numpy())
    cand = meta.iloc[rows][["inchikey14"]].astype(str).assign(row=rows)
    chosen = []
    for _, g in cand.groupby("inchikey14"):
        m = int(rng.integers(1, 7))
        chosen.extend(g.row.sample(n=min(m, len(g)), random_state=int(rng.integers(1 << 31))).tolist())
    chosen = np.sort(np.array(chosen))
    te = read_rows_at(train, chosen)
    k2g = dict(zip(grp.inchikey14, grp.identity_group))
    te["group"] = te.inchikey14.map(k2g)
    te["molecule_id"] = te.group.map(lambda g: hid(f"m{id_prefix}_", f"{seed}-{g}", 6))
    te["spectrum_id"] = [hid(f"s{id_prefix}_", f"{seed}-{i}", 8) for i in chosen]   # train の全体の行番号
    truth = (te.groupby("molecule_id").agg(inchikey14=("inchikey14", "first"), smiles=("normalized_smiles", "first"),
                                           group=("group", "first"), n_spectra=("spectrum_id", "size")).reset_index())
    truth["scenario"] = truth.group.map(g_scn)
    truth["inchikey14"] = truth.group.map(dict(zip(pick.identity_group, pick.scorer_key)))
    te = te.sample(frac=1, random_state=seed)[TEST_COLS].reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    te.to_parquet(out / "test.parquet", index=False)
    pd.DataFrame({"molecule_id": sorted(truth.molecule_id), "smiles": "CCO"}).to_csv(out / "sample_submission.csv", index=False)
    (out / "drop_keys.txt").write_text(format_keys(set(keys_all)))
    truth_dir.mkdir(parents=True, exist_ok=True)
    truth.drop(columns="group").to_csv(truth_dir / "truth.csv", index=False)
    return dict(seed=seed, fold=0, np=False, n_molecules=len(truth), n_spectra=len(te),
                scenario=truth.scenario.value_counts().to_dict(),
                spectra_per_molecule=truth.n_spectra.describe().round(2).to_dict(),
                train_rows_dropped=int(scn.notna().sum()), n_drop_keys=len(keys_all))


def write_holdout(name: str, keys: list, rule: str, meta: pd.DataFrame, rng, seed: int, train: Path, out_root: Path = ROOT,
                  pool: str | None = None, query: str = "enveda", test_adducts: set = frozenset(),
                  id_prefix: str = "", truth_root: Path | None = None) -> dict:
    """v3_c1b・v3_c2nb を build_holdout_v3.write_holdout で書く（stream=True、id 接頭つき、truth は truth_root）。"""
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import build_holdout_v3 as V3
    return V3.write_holdout(name, keys, rule, meta, rng, seed, train, out_root=out_root, pool=pool, query=query,
                            test_adducts=test_adducts, id_prefix=id_prefix, truth_root=truth_root, stream=True)


def format_keys(keys: set) -> str:
    return "\n".join(sorted(keys)) + "\n"


def finish(name: str, info: dict, picked_keys: set, ctx: dict) -> None:
    """新しい集合の後始末と点検: #116 の書き足し、hold・除外・id の assert、次の集合のために除外へ足す。"""
    import rebuild_drop_keys as RD
    hold, ex, st, rep = ctx["hold"], ctx["ex"], ctx["st"], ctx["rep"]
    tr = pd.read_csv(OUT_TRUTH / name / "truth.csv", dtype=str)
    # #116 と同じ書き足し（正準キー・別名）を新しい集合にも当てる（truth は outputs/local_eval から読む）
    RD.rebuild_set(SETS_DIR / name, list(zip(tr.molecule_id, tr.inchikey14, tr.smiles)), ctx["ts_pairs"], ctx["to_key"])
    drop = set((SETS_DIR / name / "drop_keys.txt").read_text().split())
    assert picked_keys <= hold, f"{name}: hold に無いキー"
    info["truth_in_hold"] = f"{len(picked_keys & hold)}/{len(picked_keys)}"
    hit = ex.hit(set(tr.inchikey14) | picked_keys | drop)
    assert not hit, f"{name}: 元／前の集合と重なる {sorted(hit)[:5]}"
    info["overlap_with_previous"] = 0
    assert tr.molecule_id.is_unique, f"{name}: molecule_id がぶつかる"
    if "eligible_new_groups" in info:   # enveda 型は 1 群 = 1 分子（群の数と分子数が同じ）
        assert info["n_molecules"] == info["picked"], f"{name}: 群の数 {info['picked']} と分子数 {info['n_molecules']} が違う"
    info["drop_keys_after_rebuild"] = len(drop)
    ex.add(tr, drop)
    rep["sets"][name] = info


# ---------------------------------------------------------------- 実行

def count(args) -> None:
    import build_groups as BG
    meta, hold, st, ts_key, mrr25 = load_context()
    ex = orig_excl(st, ts_key)
    res = dict(n_hold=len(hold), exclusion=dict(e_raw=len(ex.e_raw), e_key=len(ex.e_key), e_grp=len(ex.e_grp)), types={})
    c1_all = c1_eligible(meta, hold)
    c1_new = sorted(set(c1_all) - ex.hit(c1_all))
    res["types"]["v3_c1"] = dict(eligible_all=len(c1_all), excluded_by_orig=len(c1_all) - len(c1_new), eligible_new=len(c1_new))
    c2_all = c2n_eligible(meta, hold, BG.coconut_keys())
    c2_new = sorted(set(c2_all) - ex.hit(c2_all))
    res["types"]["v3_c2n"] = dict(eligible_all=len(c2_all), excluded_by_orig=len(c2_all) - len(c2_new), eligible_new=len(c2_new),
                                  overlap_with_c1_new=len(set(c2_new) & set(c1_new)))
    pool = enveda_pool(st)
    ev_all, _ = enveda_eligible(meta, st, hold)
    ev_new, _ = enveda_eligible(meta, st, hold, lambda ks: bool(ex.hit(ks)))
    c1g, c2g = {ex.grp_of.get(k) for k in c1_new}, {ex.grp_of.get(k) for k in c2_new}
    res["types"]["enveda_v1"] = dict(pool_groups=len(pool), eligible_all_groups=len(ev_all),
                                     excluded_by_orig=len(ev_all) - len(ev_new), eligible_new_groups=len(ev_new),
                                     overlap_groups_with_c1_new=len(c1g & set(ev_new.identity_group)),
                                     overlap_groups_with_c2n_new=len(c2g & set(ev_new.identity_group)))
    res["types"]["v3_c2"] = dict(note="増やさない（事前登録 (f)）")
    orig = {}
    for h in ORIG:
        te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet", columns=["molecule_id", "adduct"])
        m = t10_mask(te.adduct)
        orig[h] = dict(n_mol=int(te.molecule_id.nunique()), n_spec=len(te), n_mol_t10=int(te.molecule_id[m].nunique()),
                       n_spec_t10=int(m.sum()), adducts_outside_t10=te.adduct[~m].astype(str).value_counts().to_dict())
    res["orig_t10"] = orig
    res["inputs_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in (SPLIT, FOLDS, TSTRUCT)}
    _dump(res, args.out)


def build(args) -> None:
    import build_groups as BG
    import build_holdout_v3 as V3
    import mrr25
    meta, hold, st, ts_key, _ = load_context()
    ex = orig_excl(st, ts_key)
    rep = dict(seed=SEED, before_sha256=orig_shas(), sets={})
    ctx = dict(hold=hold, ex=ex, st=st, rep=rep, ts_pairs=list(ts_key.items()), to_key=mrr25.to_key)

    # v3_c1b（c1 だけ使い、c2 は捨てる）
    c1_x = ex.hit(c1_eligible(meta, hold))
    c1b, _c2, n1, _ = V3.select(meta, hold, N_MAX, SEED, exclude=c1_x)
    info = write_holdout("v3_c1b", c1b, "enveda_libs", meta, np.random.default_rng(SEED + 1), SEED, TRAIN, id_prefix="b",
                         truth_root=OUT_TRUTH)
    info.update(eligible_new=n1, picked=len(c1b))
    finish("v3_c1b", info, set(c1b), ctx)
    # v3_c2nb
    coco = BG.coconut_keys()
    test_adducts = set(pd.read_parquet(ROOT / "data/test.parquet", columns=["adduct"]).adduct.astype(str))
    c2nb, n2 = V3.select_c2p(meta, hold, coco, N_MAX, SEED + 10, exclude=ex.hit(c2n_eligible(meta, hold, coco)))
    info = write_holdout("v3_c2nb", c2nb, "all", meta, np.random.default_rng(SEED + 11), SEED, TRAIN, pool="keep",
                         query="prefer_enveda", test_adducts=test_adducts, id_prefix="b", truth_root=OUT_TRUTH)
    info.update(eligible_new=n2, picked=len(c2nb))
    finish("v3_c2nb", info, set(c2nb), ctx)
    # enveda_v1b
    ev, gkeys = enveda_eligible(meta, st, hold, lambda ks: bool(ex.hit(ks)))
    pick = ev.sample(n=min(N_MAX, len(ev)), random_state=SEED + 20)
    train_keys = set(meta.inchikey14.astype(str).unique())
    picked = set().union(*[gkeys[g] & train_keys for g in pick.identity_group])
    info = write_enveda(pick, st, meta, TRAIN, SETS_DIR / "enveda_v1b", OUT_TRUTH / "enveda_v1b", SEED + 20,
                        np.random.default_rng(SEED + 21), "b")
    info.update(eligible_new_groups=len(ev), picked=len(pick))
    finish("enveda_v1b", info, picked, ctx)
    # _t10 と合併
    for h in ORIG:
        rep["sets"][f"{h}_t10"] = t10(h, f"{h}_t10", ROOT / f"outputs/holdouts/{h}/truth.csv")
    for h in ("v3_c1", "v3_c2n", "enveda_v1"):
        rep["sets"][f"{h}b_t10"] = t10(f"{h}b", f"{h}b_t10", OUT_TRUTH / f"{h}b" / "truth.csv")
        rep["sets"][f"{h}u_t10"] = union_sets(f"{h}_t10", f"{h}b_t10", f"{h}u_t10")
    after = orig_shas()
    rep["after_sha256_equal"] = rep["before_sha256"] == after
    assert rep["after_sha256_equal"], "元の集合の sha256 が変わった"
    _dump(rep, args.out)


def t10(src: str, dst: str, truth_src: Path) -> dict:
    """<src> の問い合わせを 10 種の付加イオンの行に限った <dst> を作る。残り 0 本の分子は truth・sample_submission から除く。
    drop_keys.txt・hide_rule.json はそのまま写す。truth は outputs/local_eval/<dst>/truth.csv。"""
    sd, dd = SETS_DIR / src, SETS_DIR / dst
    te = pd.read_parquet(sd / "test.parquet")
    truth = pd.read_csv(truth_src)
    sub = pd.read_csv(sd / "sample_submission.csv")
    t, tr, sb = t10_filter(te, truth, sub)
    dd.mkdir(parents=True, exist_ok=True)
    t.to_parquet(dd / "test.parquet", index=False)
    sb.to_csv(dd / "sample_submission.csv", index=False)
    for f in ("drop_keys.txt", "hide_rule.json"):
        if (sd / f).is_file():
            shutil.copyfile(sd / f, dd / f)
    (OUT_TRUTH / dst).mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT_TRUTH / dst / "truth.csv", index=False)
    info = dict(source=src, rule="t10", adducts=list(TEST10), n_molecules=len(tr), n_spectra=len(t),
                n_molecules_removed=len(truth) - len(tr), n_spectra_removed=len(te) - len(t),
                adducts_outside_t10=te.adduct[~t10_mask(te.adduct)].astype(str).value_counts().to_dict())
    (dd / "HOLDOUT.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def union_sets(a: str, b: str, dst: str) -> dict:
    """<a> と <b>（t10 版）を縦に合わせた <dst>。drop_keys は和、hide_rule は 3 つの鍵が同じことを確かめて写す。"""
    da, db, dd = SETS_DIR / a, SETS_DIR / b, SETS_DIR / dst
    ha, hb = hide_rule_of(da), hide_rule_of(db)
    assert same_hide(ha, hb), f"hide_rule が違う: {ha} / {hb}"
    dd.mkdir(parents=True, exist_ok=True)
    parts = [(pd.read_parquet(d / "test.parquet"), pd.read_csv(OUT_TRUTH / n / "truth.csv"),
              pd.read_csv(d / "sample_submission.csv")) for d, n in ((da, a), (db, b))]
    t, tr, sb = union(parts)
    t.to_parquet(dd / "test.parquet", index=False)
    sb.to_csv(dd / "sample_submission.csv", index=False)
    keys = set((da / "drop_keys.txt").read_text().split()) | set((db / "drop_keys.txt").read_text().split())
    (dd / "drop_keys.txt").write_text(format_keys(keys))
    if (da / "hide_rule.json").is_file():
        shutil.copyfile(da / "hide_rule.json", dd / "hide_rule.json")
    (OUT_TRUTH / dst).mkdir(parents=True, exist_ok=True)
    tr.to_csv(OUT_TRUTH / dst / "truth.csv", index=False)
    info = dict(parts=[a, b], n_molecules=len(tr), n_spectra=len(t), n_drop_keys=len(keys))
    (dd / "HOLDOUT.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def orig_shas(names=ORIG) -> dict:
    out = {}
    for h in names:
        for p in sorted((SETS_DIR / h).iterdir()):
            out[str(p.relative_to(ROOT))] = sha256(p)
        out[f"outputs/holdouts/{h}/truth.csv"] = sha256(ROOT / f"outputs/holdouts/{h}/truth.csv")
    return out


def regen(args) -> None:
    """元の v3_c1（build_holdout_v3 の既定）と enveda_v1（commit 072ec66 の手順の写し）を scratch に作り直し、元と比べる。"""
    import build_holdout_v3 as V3
    import mrr25
    import rebuild_drop_keys as RD
    scratch = Path(args.scratch)
    meta, hold, st, ts_key, _ = load_context()
    res = {}
    c1, _c2, _, _ = V3.select(meta, hold, 200, 20261107)
    V3.write_holdout("v3_c1", c1, "enveda_libs", meta, np.random.default_rng(20261108), 20261107, TRAIN, out_root=scratch,
                     stream=True)
    res["v3_c1"] = compare(scratch / "data/holdouts/v3_c1", scratch / "outputs/holdouts/v3_c1/truth.csv", "v3_c1", RD,
                           ts_key, mrr25.to_key)
    pick = enveda_pool(st).sample(n=200, random_state=20261102)
    write_enveda(pick, st, meta, TRAIN, scratch / "enveda_v1", scratch / "enveda_v1_truth", 20261102,
                 np.random.default_rng(20261102))
    res["enveda_v1"] = compare(scratch / "enveda_v1", scratch / "enveda_v1_truth/truth.csv", "enveda_v1", RD, ts_key,
                               mrr25.to_key)
    _dump(res, args.out)


def compare(new_dir: Path, new_truth: Path, name: str, RD, ts_key, to_key) -> dict:
    od = SETS_DIR / name
    a, b = pd.read_parquet(od / "test.parquet"), pd.read_parquet(new_dir / "test.parquet")
    try:
        pd.testing.assert_frame_equal(a, b)
        frame_eq = True
    except AssertionError:
        frame_eq = False
    man = json.loads((od / "drop_keys_manifest.json").read_text())
    raw = (new_dir / "drop_keys.txt").read_bytes()
    tr = pd.read_csv(new_truth, dtype=str)
    reb, _ = RD.rebuild_keys(list(zip(tr.molecule_id, tr.inchikey14, tr.smiles)), set(raw.decode().split()),
                             list(ts_key.items()), to_key)
    return dict(test_frame_equal=frame_eq, test_bytes_equal=sha256(od / "test.parquet") == sha256(new_dir / "test.parquet"),
                drop_keys_eq_manifest_before=RD.sha256_bytes(raw) == man["before"]["sha256"],
                drop_keys_rebuilt_eq_manifest_after=RD.sha256_bytes(RD.format_keys(reb).encode()) == man["after"]["sha256"],
                drop_keys_current_eq_manifest_after=sha256(od / "drop_keys.txt") == man["after"]["sha256"],
                truth_csv_bytes_equal=sha256(ROOT / f"outputs/holdouts/{name}/truth.csv") == sha256(new_truth),
                sample_submission_bytes_equal=sha256(od / "sample_submission.csv") == sha256(new_dir / "sample_submission.csv"))


def summary(args) -> None:
    """手順 7 の記録: 各集合の分子数・スペクトル数・付加イオンの内訳・対照（score+prior+frag）の MRR、合併と部分の一致、
    判定に使う 4 集合の MDE（net_mde.json の sd_delta から、事前登録 (e) の式）。採点はしない（run.py の出力を読むだけ）。"""
    import math
    import statistics
    import mrr25
    tag = "score+prior+frag"
    names = ["v3_c1b", "v3_c2nb", "enveda_v1b"] + [f"{h}_t10" for h in ORIG] + \
            [f"{b}_t10" for b, _ in UNION.values()] + [f"{u}_t10" for _, u in UNION.values()]
    res = dict(stages=tag, sets={}, unions={}, mde={})
    rr = {}
    for n in names:
        te = pd.read_parquet(SETS_DIR / n / "test.parquet", columns=["molecule_id", "adduct"])
        truth = mrr25.read_truth(OUT_TRUTH / n / "truth.csv")
        sub = OUT_TRUTH / n / f"submission_{tag}.csv"
        pred = mrr25.read_submission(sub)
        rr[n] = {m: mrr25.reciprocal_rank(mrr25.to_key(s), pred.get(m, [])) for m, s in truth.items()}
        r = json.loads((OUT_TRUTH / n / f"result_{tag}.json").read_text())
        res["sets"][n] = dict(n_molecules=len(truth), n_spectra=len(te), adducts=te.adduct.astype(str).value_counts().to_dict(),
                              mrr25=r["mrr25"], mrr25_from_rr=round(sum(rr[n].values()) / len(rr[n]), 4),
                              submission_sha256=sha256(sub))
    for h, (b, u) in UNION.items():
        a_, b_, u_ = rr[f"{h}_t10"], rr[f"{b}_t10"], rr[f"{u}_t10"]
        same = all(u_[m] == v for m, v in {**a_, **b_}.items()) and set(u_) == set(a_) | set(b_)
        wmean = (sum(a_.values()) + sum(b_.values())) / (len(a_) + len(b_))
        umean = sum(u_.values()) / len(u_)
        res["unions"][u] = dict(per_molecule_rr_equal=same, weighted_mean_parts=round(wmean, 6), union_mean=round(umean, 6),
                                abs_diff=abs(wmean - umean), within_1e4=abs(wmean - umean) <= 1e-4)
    nm = json.loads((OUT_TRUTH / "net_mde.json").read_text())
    judge = {"v3_c1": "v3_c1u_t10", "v3_c2": "v3_c2_t10", "enveda_v1": "enveda_v1u_t10", "v3_c2n": "v3_c2nu_t10"}
    for h, j in judge.items():
        n = res["sets"][j]["n_molecules"]
        sds = [v["sd_delta"] for v in nm[h]["variants"].values() if v["n_changed"] > 0]
        out = dict(set=j, n=n, n_variants=len(sds))
        if sds:
            med = statistics.median(sds)
            out.update(sd_delta_median=round(med, 4), sd_delta_min=round(min(sds), 4), sd_delta_max=round(max(sds), 4),
                       mde_median=round(2.8 * med / math.sqrt(n), 4), mde_min=round(2.8 * min(sds) / math.sqrt(n), 4),
                       mde_max=round(2.8 * max(sds) / math.sqrt(n), 4))
            fv = nm[h]["variants"].get("score+fragnn+prior+frag")
            if fv:
                out["mde_fragnn_v1_sd"] = round(2.8 * fv["sd_delta"] / math.sqrt(n), 4)
            out["direction_only"] = out["mde_median"] >= 0.02   # 事前登録 (e): 主の MDE が 0.02 を下回らない型は向きの報告だけ
            out["mde_median_orig_n"] = nm[h]["mde_median"]
        else:
            out["note"] = "変種がすべて差 0（基準 0）。MDE なし"
        res["mde"][h] = out
    _dump(res, args.out)


def main() -> None:
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/ranker"))
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, default in (("count", "counts.json"), ("regen", "regen.json"), ("build", "build.json"),
                          ("summary", "summary.json")):
        a = sub.add_parser(name)
        a.add_argument("--out", default=str(ROOT / "outputs/local_eval/a3a_1123" / default))
        if name == "regen":
            a.add_argument("--scratch", required=True)
    args = ap.parse_args()
    {"count": count, "regen": regen, "build": build, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    main()
