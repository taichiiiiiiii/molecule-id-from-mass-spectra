#!/usr/bin/env python3
"""
Issue #162: v2 プール（pool_meta.parquet）の `src` の意味を、`src` を作ったコードと独立の所属とのクロス表で確かめ、
検証 v3 の 4 集合（v3_c1・v3_c2・v3_c2n・enveda_v1）の正解・おとりの `src` 内訳を数える、読むだけの監査（採点しない）。

定義（数える前に固定、Issue #162 の本文と同じ）:
  キー: canon = casmi.chem.score_key（互変異性体を正準化した InChIKey14、pool_meta.key と同じ作り方）、
        raw = 正準化しない InChIKey14 の先頭 14 文字
  表 1（src の値 × 独立の所属、プールの行数）:
    train_raw   key ∈ data/train.parquet の inchikey14（競技データ、row group ごとに 1 列だけ読む）
    train_canon key ∈ train_structs.key（canon。v2 プールと同じ作者の表。無作為 2,000 行で score_key(smiles) == key の率も記録）
    train_any   train_raw か train_canon か key ∈ train_structs.rawkey
    coco_raw    key ∈ COCONUT 2026-09 全件 CSV の standard_inchi_key[:14]（raw。互変異性体の別名の分は取りこぼす）
    coco_np     pool_popularity.csv の np_coconut == 1（PubChem 側の COCONUT 注記。別のデータセット）
    coco_any    coco_raw か coco_np
    pubchem     pool_popularity.csv の pubchem_cid がある（正の値。「無し」は −1 で書かれている）
    内部の列（独立ではない）: train_sid_ge0（train_sid >= 0）、np_likeness_notna
    PubChem 層（data/ext/pubchem_tier、キーの列なし）との積は数えない（未測定）
  判定（judge）: 「確かめた」= (i) 作ったコードに src の定義があり行番号で引用できる、
    (ii) src ∈ {1,3} の train_any と train_sid_ge0 が 99% 以上・src 2 の train_any が 1% 以下、
    (iii) src ∈ {2,3} の coco_any が 99% 以上・src 1 の coco_any が 1% 以下。1 つでも外れれば「未確認」（外れたセルを書く）
  表 2（集合ごと）: 正解 = canon(truth.smiles) のプールの行の src（無ければ「プール外」）。おとり = run.visible_window
    （±10 ppm、pool: remove なら drop_keys を除く）の中の key != 正解の canon の全行の src。
    分子ごとの「窓の中で正解と同じ src のおとりの行数」の分位点。記録だけ（採否に使わない）

  追加の確認（事前登録の外。判定は変えない、読み方の補足だけ）: --coco-canon のとき、COCONUT CSV の行のうち raw キーが
    プールの key に無いものについて score_key(standardize_smiles(canonical_smiles))（02_build_coconut.py と同じ
    「最大の断片・電荷を外す・立体を落とす」→ 互変異性体の正準化）を自前で計算し、src ∈ {2,3} で coco_raw に当たらない
    行の key がそれで何行当たるかを数える（coco_canon）。pool_popularity の np_coconut は上流のプールの COCONUT の印を
    写したもの（TECHNICAL_NOTES.md「COCONUT membership follows the upstream pool」）で独立ではないと分かったため

    python scripts/cv/src_audit.py [--out outputs/local_eval/src_audit/src_audit.json] [--coco-canon --workers 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1", "v3_c2", "v3_c2n", "enveda_v1")
SRC_VALUES = (1, 2, 3)
FLAGS = ("train_raw", "train_canon", "train_any", "coco_raw", "coco_np", "coco_any", "pubchem",
         "train_sid_ge0", "np_likeness_notna")
HI, LO = 0.99, 0.01
N_KEYCHECK = 2000
COCONUT_CSV = ROOT / "outputs/engine2_fp_retrain/work/ext/coconut_csv-09-2026.csv"
# ahmedberatozer/casmi26-training-code の scripts/05_build_pool.py（Kaggle の版 2026-10-19、取得して sha256 を照合）
TC_BUILD_POOL_SHA256 = "faa560b5427c825b0895802d35f31dfd55396feb28491bab6346937cf89c888f"
TC_LINES = {3: "src (1=train, 2=coconut, 3=both)",
            29: "both = set(co.key) & tr_keys",
            30: "src_tr = np.where(trk.key.isin(both), 3, 1)",
            37: "src=np.concatenate([src_tr, np.full(len(cok), 2, np.int8)])"}


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）
def crosstab(src: np.ndarray, flags: dict[str, np.ndarray]) -> dict:
    """src の値ごとに行数 n と各フラグが真の行数。"""
    out = {}
    for v in np.unique(src):
        m = src == v
        out[str(int(v))] = dict(n=int(m.sum()), **{f: int(np.asarray(a)[m].sum()) for f, a in flags.items()})
    return out


def cid_present(cid) -> np.ndarray:
    """pool_popularity.csv の pubchem_cid は「無し」を −1 で書く（NaN ではない）。正の CID だけを「ある」とする。"""
    c = np.asarray(cid, dtype=np.float64)
    return np.isfinite(c) & (c > 0)


def rate(table: dict, src_vals, flag: str) -> float | None:
    n = sum(table[str(v)]["n"] for v in src_vals if str(v) in table)
    k = sum(table[str(v)][flag] for v in src_vals if str(v) in table)
    return k / n if n else None


def judge(table: dict, code_ok: bool, hi: float = HI, lo: float = LO) -> dict:
    """Issue #162 の事前登録の判定。checks の各項目と、外れた項目の一覧を返す。"""
    checks = {
        "i_code_definition": bool(code_ok),
        "ii_src13_train_any_ge": (rate(table, (1, 3), "train_any"), ">=", hi),
        "ii_src13_train_sid_ge0_ge": (rate(table, (1, 3), "train_sid_ge0"), ">=", hi),
        "ii_src2_train_any_le": (rate(table, (2,), "train_any"), "<=", lo),
        "iii_src23_coco_any_ge": (rate(table, (2, 3), "coco_any"), ">=", hi),
        "iii_src1_coco_any_le": (rate(table, (1,), "coco_any"), "<=", lo),
    }
    failed = []
    for k, c in checks.items():
        if isinstance(c, bool):
            ok = c
        else:
            r, op, th = c
            ok = r is not None and (r >= th if op == ">=" else r <= th)
        if not ok:
            failed.append(k)
    view = {k: (c if isinstance(c, bool) else dict(rate=None if c[0] is None else round(c[0], 6), op=c[1], th=c[2]))
            for k, c in checks.items()}
    return dict(verdict="確かめた" if not failed else "未確認", checks=view, failed=failed)


def window_src(keys: np.ndarray, src: np.ndarray, idx: np.ndarray, truth_key: str):
    """窓 idx の中の正解の行の src（無ければ None）、おとり（key != truth_key）の src の配列。"""
    k = keys[idx]
    t = idx[k == truth_key]
    d = idx[k != truth_key]
    return (int(src[t[0]]) if len(t) else None), src[d]


def counts(a) -> dict:
    a = np.asarray(a)
    return {str(int(v)): int(c) for v, c in zip(*np.unique(a, return_counts=True))} if len(a) else {}


def check_code(path: Path) -> dict:
    """05_build_pool.py の sha256 と、src の定義の行（行番号は 1 始まり）を照合する。"""
    if not path.is_file():
        return dict(ok=False, reason=f"not found: {path}")
    b = path.read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    lines = b.decode("utf-8").splitlines()
    got = {n: lines[n - 1].strip() if n <= len(lines) else "" for n in TC_LINES}
    ok_lines = all(s in got[n] for n, s in TC_LINES.items())
    return dict(ok=bool(sha == TC_BUILD_POOL_SHA256 and ok_lines), sha256=sha, sha_match=sha == TC_BUILD_POOL_SHA256,
                lines={str(n): got[n] for n in TC_LINES}, lines_match=ok_lines)


# ---------------------------------------------------------------- 読み込み
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def train_raw_keys(path: Path) -> set:
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    out = set()
    for b in f.iter_batches(batch_size=8192, columns=["inchikey14"]):
        out.update(str(x)[:14] for x in b.column(0).to_pylist() if x is not None)
    return out


def coconut_raw_keys(path: Path) -> tuple[set, int]:
    import pandas as pd
    out, n = set(), 0
    for ch in pd.read_csv(path, usecols=["standard_inchi_key"], chunksize=200_000, dtype=str):
        s = ch.standard_inchi_key.dropna()
        n += len(ch)
        out.update(s.str[:14].tolist())
    return out, n


def _std_key(smi):
    if str(ROOT / "data/ext/v4g/casmi26-v3-models/code") not in sys.path:   # spawn の子プロセス用
        sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    if not isinstance(smi, str):
        return None
    x = chem.standardize_smiles(smi)
    return chem.score_key(x) if x else None


def coco_canon_keys(path: Path, pool_keys: set, workers: int) -> tuple[set, int]:
    """COCONUT の行のうち raw キーがプールに無い行の、自前の標準化 + score_key の集合と、計算した行数。"""
    import pandas as pd
    from multiprocessing import Pool as MP
    smis = []
    for ch in pd.read_csv(path, usecols=["canonical_smiles", "standard_inchi_key"], chunksize=200_000, dtype=str):
        raw = ch.standard_inchi_key.fillna("").str[:14]
        smis += ch.canonical_smiles[~raw.isin(pool_keys)].tolist()
    with MP(workers) as mp:
        ks = mp.map(_std_key, smis, chunksize=200)
    return {k for k in ks if k}, len(smis)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/src_audit/src_audit.json"))
    ap.add_argument("--tc-build-pool", default=str(Path.home() / ".hermes/profiles/massspec/cache/scratch/tc_src/scripts/scripts/05_build_pool.py"),
                    help="Kaggle の ahmedberatozer/casmi26-training-code から取得した scripts/05_build_pool.py")
    ap.add_argument("--coco-canon", action="store_true", help="追加の確認（事前登録の外）: COCONUT の自前の正準キー")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import pandas as pd
    from casmi import chem
    import library as L
    from run import load_pool, visible_window
    t0 = time.time()
    out = dict(issue=162, definition=__doc__.split("定義（数える前に固定、Issue #162 の本文と同じ）:")[1].split("    python")[0].strip())

    pm = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet",
                         columns=["key", "mass", "src", "train_sid", "np_likeness", "pid"])
    assert pm.key.is_unique, "pool_meta.key が一意でない"
    assert (pm.pid.values == np.arange(len(pm))).all(), "pid が行番号でない"
    ts = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet",
                         columns=["inchikey14", "key", "rawkey", "smiles", "sid"])
    tr_raw = train_raw_keys(ROOT / "data/train.parquet")
    coco, n_coco_rows = coconut_raw_keys(COCONUT_CSV)
    pp = pd.read_csv(ROOT / "data/ext/pop/pool_popularity.csv", usecols=["pool_row", "pubchem_cid", "np_coconut"])
    print(f"loaded {time.time()-t0:.0f}s: pool {len(pm)}, train_raw {len(tr_raw)}, coconut rows {n_coco_rows} keys {len(coco)}, "
          f"popularity {len(pp)}", flush=True)

    keys = pm.key.astype(str).values
    np_coco = np.zeros(len(pm), bool)
    np_coco[pp.pool_row.values] = pp.np_coconut.fillna(0).values == 1
    has_cid = np.zeros(len(pm), bool)
    has_cid[pp.pool_row.values] = cid_present(pp.pubchem_cid.values)
    ts_key, ts_raw = set(ts.key.dropna().astype(str)), set(ts.rawkey.dropna().astype(str)) | set(ts.inchikey14.dropna().astype(str))
    ks = pd.Series(keys)
    isin = lambda st: ks.isin(st).values
    f = dict(train_raw=isin(tr_raw), train_canon=isin(ts_key))
    f["train_any"] = f["train_raw"] | f["train_canon"] | isin(ts_raw)
    f["coco_raw"] = isin(coco)
    f["coco_np"] = np_coco
    f["coco_any"] = f["coco_raw"] | f["coco_np"]
    f["pubchem"] = has_cid
    f["train_sid_ge0"] = pm.train_sid.values >= 0
    f["np_likeness_notna"] = pm.np_likeness.notna().values
    src = pm.src.values.astype(int)
    table = crosstab(src, f)
    # train_structs.key が自前の score_key と同じ作り方か（無作為 N_KEYCHECK 行、seed 0）
    rng = np.random.default_rng(0)
    samp = ts.iloc[np.sort(rng.choice(len(ts), size=min(N_KEYCHECK, len(ts)), replace=False))]
    eq = [chem.score_key(s) == k for s, k in zip(samp.smiles, samp.key)]
    code = check_code(Path(a.tc_build_pool))
    out.update(pool_rows=len(pm), train_raw_keys=len(tr_raw), train_structs_rows=len(ts), coconut_csv_rows=n_coco_rows,
               coconut_raw_keys=len(coco), popularity_rows=len(pp),
               train_structs_key_eq_score_key=dict(n=len(eq), eq=int(sum(eq))),
               code=code, table1=table, judge=judge(table, code["ok"]),
               coconut_csv_sha256=sha256_file(COCONUT_CSV))
    print("table1", json.dumps(table), flush=True)
    if a.coco_canon:
        t1 = time.time()
        cc, n_calc = coco_canon_keys(COCONUT_CSV, set(keys), a.workers)
        miss = (src >= 2) & ~f["coco_raw"]
        hit = miss & isin(cc)
        out["extra_coco_canon"] = dict(
            note="事前登録の外。判定（judge）は変えない。np_coconut は上流プールの印の写しで独立でない",
            coconut_rows_computed=n_calc, canon_keys=len(cc), secs=round(time.time() - t1, 1),
            src23_rows=int((src >= 2).sum()), src23_coco_raw=int(((src >= 2) & f["coco_raw"]).sum()),
            src23_miss_raw=int(miss.sum()), src23_miss_raw_hit_canon=int(hit.sum()),
            src1_coco_canon=int(((src == 1) & isin(cc)).sum()),
            per_src={str(v): dict(coco_raw_or_canon=int(((src == v) & (f["coco_raw"] | isin(cc))).sum()), n=int((src == v).sum()))
                     for v in SRC_VALUES})
        print("extra_coco_canon", json.dumps(out["extra_coco_canon"], ensure_ascii=False), flush=True)
    print("judge", json.dumps(out["judge"], ensure_ascii=False), flush=True)

    # 表 2: run.py の Pool と同じ並び（mass の stable argsort）
    pool = load_pool()
    order = np.argsort(pm.mass.values, kind="stable")
    assert (pool.key == pm.key.values[order]).all(), "Pool の並びが pool_meta の mass の stable argsort と違う"
    psrc = src[order]
    key2src = dict(zip(keys, src))
    sets = {}
    for h in HOLDOUTS:
        tr = pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv")
        zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
        targets = {m: float(t) for m, t in zf["__targets__"]}
        rule = L.hide_rule(h)
        hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
        t_src_pool, t_src_win, d_src, same, n_t = [], [], [], [], 0
        for mid, smi in zip(tr.molecule_id, tr.smiles):
            ck = chem.score_key(smi)
            t_src_pool.append(key2src.get(ck, 0))      # 0 = プール外
            if mid not in targets:
                continue
            n_t += 1
            idx = visible_window(pool, targets[mid], hidden)
            ts_, ds = window_src(pool.key, psrc, idx, ck)
            t_src_win.append(0 if ts_ is None else ts_)
            d_src.append(ds)
            if ts_ is not None:
                same.append(int((ds == ts_).sum()))
        dall = np.concatenate(d_src) if d_src else np.array([], int)
        sets[h] = dict(hide_rule=rule, n_truth=len(tr), n_target=n_t,
                       truth_src_full_pool=counts(t_src_pool), truth_src_in_window=counts(t_src_win),
                       decoy_src_window_rows=counts(dall),
                       decoy_src_window_share={k: round(v / len(dall), 4) for k, v in counts(dall).items()} if len(dall) else {},
                       decoy_rows=int(len(dall)),
                       same_src_decoys_per_molecule=dict(n=len(same), **({q: float(np.percentile(same, p)) for q, p in
                                                                          (("min", 0), ("p25", 25), ("p50", 50), ("p75", 75), ("max", 100))}
                                                                         if same else {})))
        print(h, json.dumps({k: v for k, v in sets[h].items() if k != "hide_rule"}), flush=True)
    out["table2"] = sets
    out["secs"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", a.out, out["secs"], "s")


if __name__ == "__main__":
    main()
