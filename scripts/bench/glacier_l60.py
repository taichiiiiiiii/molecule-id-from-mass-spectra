#!/usr/bin/env python3
"""Issue #242: GLACIER の予測スペクトルを手元の CPU で書き出す（B8 の dev 3,901 分子の L60 版、(r) 区分）。

    # 1) 入力の一覧（軽い。予測の前）: jobs.parquet・molecules.parquet・manifest.json
    .venv/bin/python scripts/bench/glacier_l60.py list --out-dir outputs/local_eval/glacier_l60
    # 2) 予測（重い。shard ごとに tmp → os.replace、再開できる）
    .venv/bin/python scripts/bench/glacier_l60.py run --out-dir outputs/local_eval/glacier_l60 \
        --pkg outputs/local_eval/glacier_l60/env/pkg --site outputs/local_eval/glacier_l60/env/site

定義（#242 の定義のコメントと同じ。数え直さない）:
  - 候補 L = 対照 score,prior,frag の重複除去した上位 60（#244 の dev 表 `outputs/local_eval/features/dev_b8.parquet`
    の is_tier = 0 の行、ctrl_rank の順）。分子ごとに |L| = min(60, n_i)（#236 の n_i）であることを確かめる
  - 区分 (r) = glacier_counts.spectrum_jobs（gl_runner と同じ規則、[M+H]+・[M+Na]+、CE が NaN なら 20・40・60 の 3 つ）。
    問い合わせは各集合の data/holdouts/<set>/test.parquet
  - 1 行 = (分子, 候補, 区分) = 予測 1 本。合計は #242 の L60 版 (r) 544,886 本と一致しなければ止める
  - 予測 = gl_runner.Predictor（canonical 形ごとに 1 回、post 済み: top 100・m/z 順・最大 1 の float32）。
    **一致の点（問い合わせとの類似度）は計算しない**。値の要約（平均など）も出さない
  - 漏れの印 msg_overlap（#242 の漏れの規則のコメント）を分子の表に付ける: 正解が MassSpecGym の train ∪ val と
    生の InChIKey14 か正規化キー（mrr25.to_key）で一致 → 1。msg_overlap_all は全体（test を含む）、参考

出力（--out-dir、クローンの実ディレクトリ）:
  jobs.parquet       row, set, molecule_id, cand_rank, cand_key, smiles, adduct, instr, ce
  molecules.parquet  set, molecule_id, type, inchikey14, key, n_L, n_cat_r, n_jobs, msg_overlap, msg_overlap_all
  manifest.json      本数・入力の sha256・shard の数
  shards/shard_NNNNN.npz   row, ok, fail（0 成功 / 1 予測失敗〈featurize 失敗を含む。理由別の数は json〉/ 2 SMILES 不可）,
                           offsets, mz, it（ok の行の peaks を連結）
  shards/shard_NNNNN.json  本数・失敗の数（理由別）・秒・peak RSS・npz の sha256（これがあれば完了）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/bench", "scripts/validate"):
    sys.path.insert(0, str(ROOT / _p))

EXPECTED_N_JOBS = 544_886          # #242 の L60 版 (r)
EXPECTED_N_MOLS = 3_901
EXPECTED_N_L = 197_170
SHARD_SIZE = 20_000
DEV_B8 = "outputs/local_eval/features/dev_b8.parquet"
PER_MOL = "outputs/local_eval/i236_recall/per_molecule.parquet"
DEV_LIST = "docs/bench/dev_list.csv.gz"
MSG = "data/ext/msg/MassSpecGym.tsv"
TOPN = 60
FAIL_PRED, FAIL_SMILES = 1, 2


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- shard（純粋な関数、テストあり）
def shard_bounds(n: int, size: int) -> list[tuple[int, int]]:
    """[0, n) を size ずつの半開区間に切る（最後は短い）。n = 0 なら空。"""
    if size <= 0:
        raise ValueError("size > 0")
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def shard_paths(out_dir: Path, i: int) -> tuple[Path, Path]:
    d = out_dir / "shards"
    return d / f"shard_{i:05d}.npz", d / f"shard_{i:05d}.json"


def shard_done(out_dir: Path, i: int, a: int, b: int, jobs_sha256: str | None = None) -> bool:
    """json があり、npz の sha256・区間（と jobs.parquet の sha256）が合えば完了。合わなければ未完了（やり直す）。"""
    npz, js = shard_paths(out_dir, i)
    if not (npz.exists() and js.exists()):
        return False
    try:
        m = json.loads(js.read_text())
    except Exception:
        return False
    if jobs_sha256 is not None and m.get("jobs_sha256") != jobs_sha256:
        return False
    return m.get("row_start") == a and m.get("row_end") == b and m.get("npz_sha256") == sha256(npz)


def pack(rows: np.ndarray, preds: list, fail: np.ndarray) -> dict:
    """予測の list（(mz, it) か None）を連結した配列に。"""
    ok = np.array([p is not None and len(p[0]) > 0 for p in preds], bool)
    lens = np.array([len(p[0]) if o else 0 for p, o in zip(preds, ok)], np.int64)
    offsets = np.zeros(len(preds) + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])
    mz = np.concatenate([np.asarray(p[0], np.float32) for p, o in zip(preds, ok) if o]) if ok.any() \
        else np.zeros(0, np.float32)
    it = np.concatenate([np.asarray(p[1], np.float32) for p, o in zip(preds, ok) if o]) if ok.any() \
        else np.zeros(0, np.float32)
    fail = fail.copy()
    fail[(~ok) & (fail == 0)] = FAIL_PRED
    return dict(row=rows.astype(np.int64), ok=ok, fail=fail.astype(np.int8), offsets=offsets, mz=mz, it=it)


def write_shard(out_dir: Path, i: int, a: int, b: int, arrays: dict, stats: dict) -> dict:
    npz, js = shard_paths(out_dir, i)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    fail = arrays["fail"]
    meta = dict(shard=i, row_start=a, row_end=b, n_rows=int(b - a), n_ok=int(arrays["ok"].sum()),
                n_fail=int((~arrays["ok"]).sum()), n_fail_pred=int((fail == FAIL_PRED).sum()),
                n_fail_smiles=int((fail == FAIL_SMILES).sum()), n_peaks=int(arrays["offsets"][-1]),
                npz_sha256=sha256(npz), **stats)
    tmpj = js.with_name(js.name + ".tmp")
    tmpj.write_text(json.dumps(meta, indent=1))
    os.replace(tmpj, js)
    return meta


def run_shards(out_dir: Path, jobs, predict_fn, shard_size: int = SHARD_SIZE, max_shards: int | None = None,
               log=print, jobs_sha256: str | None = None) -> dict:
    """jobs（row 順の DataFrame）を shard に切り、完了していない shard だけ predict_fn で予測して書く。
    predict_fn(list[(smiles, adduct, instr, ce)]) -> (list[(mz, it) | None], fail ndarray[int8], stats dict)"""
    n = len(jobs)
    assert (jobs["row"].values == np.arange(n)).all(), "row が 0..n-1 の順でない"
    bounds = shard_bounds(n, shard_size)
    done_n, ran = 0, []
    for i, (a, b) in enumerate(bounds):
        if shard_done(out_dir, i, a, b, jobs_sha256):
            done_n += 1
            continue
        if max_shards is not None and len(ran) >= max_shards:
            break
        sub = jobs.iloc[a:b]
        keys = list(zip(sub.smiles.astype(str), sub.adduct.astype(str), sub.instr.astype(str),
                        sub.ce.astype(float)))
        t0 = time.time()
        preds, fail, stats = predict_fn(keys)
        assert len(preds) == len(keys) == len(fail)
        stats = dict(stats, seconds=round(time.time() - t0, 2),
                     maxrss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                     finished_at=time.strftime("%Y-%m-%d %H:%M:%S %z"), jobs_sha256=jobs_sha256)
        meta = write_shard(out_dir, i, a, b, pack(sub["row"].values, preds, np.asarray(fail, np.int8)), stats)
        ran.append(i)
        log("SHARD " + json.dumps({k: meta[k] for k in ("shard", "n_rows", "n_ok", "n_fail_pred", "n_fail_smiles",
                                                         "seconds", "maxrss_bytes")}))
    n_done = sum(shard_done(out_dir, i, a, b, jobs_sha256) for i, (a, b) in enumerate(bounds))
    return dict(n_shards=len(bounds), n_done=n_done, ran=ran, skipped=done_n)


# ---------------------------------------------------------------- 段: list
def msg_key_sets(root: Path, workers: int = 6) -> dict:
    import pandas as pd
    from multiprocessing import Pool
    d = pd.read_csv(root / MSG, sep="\t", usecols=["smiles", "inchikey", "fold"], dtype=str)
    d["ik14"] = d.inchikey.str[:14]
    tv = d.fold.isin(["train", "val"])
    smis = sorted(set(d.smiles))
    with Pool(workers) as p:
        keys = p.map(_to_key, smis, chunksize=500)
    k_of = dict(zip(smis, keys))
    d["k"] = d.smiles.map(k_of)
    return dict(raw_tv=set(d.ik14[tv]), raw_all=set(d.ik14), key_tv=set(d.k[tv].dropna()),
                key_all=set(d.k.dropna()), n_rows=len(d), n_smiles=len(smis), n_key_none=sum(k is None for k in keys))


def _to_key(s):
    import mrr25
    try:
        return mrr25.to_key(s)
    except Exception:
        return None


def stage_list(root: Path, out_dir: Path, workers: int) -> dict:
    import pandas as pd
    import glacier_counts as G
    t0 = time.time()
    dev = pd.read_parquet(root / DEV_B8, columns=["set", "type", "molecule_id", "cand_key", "cand_smiles", "is_tier",
                                                  "ctrl_rank"])
    mols = dev[["set", "type", "molecule_id"]].drop_duplicates().copy()
    mols["molecule_id"] = mols.molecule_id.astype(str)
    assert len(mols) == EXPECTED_N_MOLS, len(mols)
    L = dev[dev.is_tier == 0].copy()
    L["molecule_id"] = L.molecule_id.astype(str)
    L = L.sort_values(["set", "molecule_id", "ctrl_rank"], kind="stable")
    assert not L.duplicated(["set", "molecule_id", "cand_smiles"]).any()
    assert len(L) == EXPECTED_N_L, len(L)
    L["cand_rank"] = L.groupby(["set", "molecule_id"]).cumcount()
    # |L| = min(60, n_i)（#236）
    pm = pd.read_parquet(root / PER_MOL, columns=["set", "molecule_id", "type", "n_i"])
    pm = pm[pm.type != "none"].copy()
    pm["molecule_id"] = pm.molecule_id.astype(str)
    nL = L.groupby(["set", "molecule_id"]).size().rename("n_L").reset_index()
    mols = mols.merge(pm[["set", "molecule_id", "n_i"]], on=["set", "molecule_id"], how="left", validate="one_to_one")
    mols = mols.merge(nL, on=["set", "molecule_id"], how="left").fillna({"n_L": 0})
    mols["n_L"] = mols.n_L.astype(np.int64)
    assert mols.n_i.notna().all()
    assert (np.minimum(TOPN, mols.n_i.astype(np.int64)) == mols.n_L).all(), "|L| が min(60, n_i) と違う"
    # 区分 (r)
    cols = ["molecule_id", "adduct", "ionization_mode", "instrument_type", "collision_energy_orig",
            "collision_energy_ev", "collision_energy_orig_units"]
    cats = {}
    n_spec = 0
    for s in sorted(mols.set.unique()):
        q = pd.read_parquet(root / f"data/holdouts/{s}/test.parquet", columns=cols)
        for r in q.itertuples(index=False):
            n_spec += 1
            cats.setdefault((s, str(r.molecule_id)), set()).update(
                G.spectrum_jobs(r.adduct, r.ionization_mode, r.instrument_type, r.collision_energy_orig_units,
                                r.collision_energy_orig, r.collision_energy_ev, G.COV_R))
    mols["n_cat_r"] = [len(cats.get((s, m), ())) for s, m in zip(mols.set, mols.molecule_id)]
    # 行 = 分子 × 候補 × 区分（区分は (adduct, instr, ce) の順）
    rec = []
    for (s, m), g in L.groupby(["set", "molecule_id"], sort=True):
        cs = sorted(cats.get((s, m), ()), key=lambda t: (t[0], t[1], t[2]))
        for cr, ck, smi in zip(g.cand_rank, g.cand_key, g.cand_smiles):
            for ad, ins, ce in cs:
                rec.append((s, m, int(cr), str(ck), str(smi), ad, ins, float(ce)))
    jobs = pd.DataFrame(rec, columns=["set", "molecule_id", "cand_rank", "cand_key", "smiles", "adduct", "instr", "ce"])
    jobs.insert(0, "row", np.arange(len(jobs), dtype=np.int64))
    nj = jobs.groupby(["set", "molecule_id"]).size().rename("n_jobs").reset_index()
    mols = mols.merge(nj, on=["set", "molecule_id"], how="left").fillna({"n_jobs": 0})
    mols["n_jobs"] = mols.n_jobs.astype(np.int64)
    assert int((mols.n_L * mols.n_cat_r).sum()) == len(jobs)
    assert len(jobs) == EXPECTED_N_JOBS, f"本数 {len(jobs)} が #242 の {EXPECTED_N_JOBS} と違う"
    assert np.isfinite(jobs.ce.values).all()
    # 漏れの印（#242 の規則）
    dl = pd.read_csv(root / DEV_LIST, dtype={"identity_group": str})
    dl = dl[dl.used_b8 == 1].rename(columns={"b8_set": "set", "b8_molecule_id": "molecule_id"})
    dl["molecule_id"] = dl.molecule_id.astype(str)
    mols = mols.merge(dl[["set", "molecule_id", "inchikey14", "key"]], on=["set", "molecule_id"], how="left",
                      validate="one_to_one")
    assert mols.inchikey14.notna().all() and mols.key.notna().all()
    ms = msg_key_sets(root, workers)
    mols["msg_overlap"] = (mols.inchikey14.isin(ms["raw_tv"]) | mols.key.isin(ms["key_tv"])).astype(np.int8)
    mols["msg_overlap_all"] = (mols.inchikey14.isin(ms["raw_all"]) | mols.key.isin(ms["key_all"])).astype(np.int8)
    mols = mols[["set", "molecule_id", "type", "inchikey14", "key", "n_i", "n_L", "n_cat_r", "n_jobs",
                 "msg_overlap", "msg_overlap_all"]].sort_values(["set", "molecule_id"]).reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("jobs", jobs), ("molecules", mols)):
        tmp = out_dir / f"{name}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out_dir / f"{name}.parquet")
    n_uniq = int(jobs[["smiles", "adduct", "instr", "ce"]].drop_duplicates().shape[0])
    man = dict(
        n_mols=len(mols), n_L=int(len(L)), n_jobs=int(len(jobs)), n_jobs_unique_smiles_cat=n_uniq,
        n_query_spectra=n_spec, n_mols_no_cov=int((mols.n_cat_r == 0).sum()),
        n_mols_no_L=int((mols.n_L == 0).sum()), shard_size=SHARD_SIZE,
        n_shards=len(shard_bounds(len(jobs), SHARD_SIZE)),
        msg_overlap=dict(n1=int(mols.msg_overlap.sum()), n1_all=int(mols.msg_overlap_all.sum()),
                         by_type={t: dict(n=int((mols.type == t).sum()), n1=int(mols.msg_overlap[mols.type == t].sum()))
                                  for t in sorted(mols.type.unique())},
                         msg_rows=ms["n_rows"], msg_smiles=ms["n_smiles"], msg_key_none=ms["n_key_none"]),
        inputs={p: sha256(root / p) for p in (DEV_B8, PER_MOL, DEV_LIST, MSG)},
        outputs={f"{n}.parquet": sha256(out_dir / f"{n}.parquet") for n in ("jobs", "molecules")},
        seconds=round(time.time() - t0, 1))
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps(man, ensure_ascii=False))
    return man


# ---------------------------------------------------------------- 段: run
def make_predict_fn(pkg: Path, site: Path, threads: int, log):
    sys.path.insert(0, str(pkg))
    import gl_runner as R
    a = types.SimpleNamespace(pkg=str(pkg), site="none", any_rdkit=True, wheels=None)
    sys.path.insert(0, str(site))
    rd = R.setup_env(a, log)
    import torch
    if threads > 0:
        torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = R.load_model(str(pkg), "cpu", log)
    env = dict(rdkit=rd, torch=torch.__version__, threads=torch.get_num_threads(),
               gl_runner_sha256=sha256(pkg / "gl_runner.py"), ckpt_sha256=sha256(pkg / "ckpt" / "glacier.pt"))
    log("ENV " + json.dumps(env))

    def predict(keys):
        pred = R.Predictor(model, "cpu", 32, log)        # shard ごとに新しいキャッシュ（メモリを抑える）
        fail = np.zeros(len(keys), np.int8)
        good = []
        for n, (smi, ad, ins, ce) in enumerate(keys):
            ok, _ = R.smiles_ok(smi)
            if ok:
                good.append(n)
            else:
                fail[n] = FAIL_SMILES
        jobs = {keys[n] for n in good}
        finished = pred.predict(jobs, float("inf"))
        assert finished
        preds = [None] * len(keys)
        for n in good:
            v = pred.cache.get(keys[n])
            preds[n] = None if v is None else (v[0].astype(np.float32), v[1].astype(np.float32))
        # n_model_* と n_feat_fail は canonical 形 × 区分の単位（行の単位の n_fail_pred とは違う）
        stats = dict(count_unit_model="canonical (smiles, adduct, instr, ce)",
                     n_model_calls=int(pred.n_pred), n_model_fail=int(pred.n_fail),
                     n_feat_fail=int(pred.n_feat_fail), t_pred=round(pred.t_pred, 2), bs_final=int(pred.bs), **env)
        if len(R._CANON) > 200_000:
            R._CANON.clear()
        return preds, fail, stats
    return predict


def stage_run(out_dir: Path, pkg: Path, site: Path, threads: int, max_shards: int | None) -> dict:
    import pandas as pd
    man = json.loads((out_dir / "manifest.json").read_text())
    jp = out_dir / "jobs.parquet"
    assert sha256(jp) == man["outputs"]["jobs.parquet"], "jobs.parquet が manifest と違う"
    jobs = pd.read_parquet(jp, columns=["row", "smiles", "adduct", "instr", "ce"])
    assert len(jobs) == man["n_jobs"] == EXPECTED_N_JOBS

    def log(*x):
        print(time.strftime("%H:%M:%S"), *x, flush=True)
    fn = make_predict_fn(pkg, site, threads, log)
    res = run_shards(out_dir, jobs, fn, man["shard_size"], max_shards, log, jobs_sha256=man["outputs"]["jobs.parquet"])
    log("RUN " + json.dumps(res))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("list", "run"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=ROOT, help="data/・outputs/・docs/ を読む根（既定はこのファイルの repo）")
    ap.add_argument("--pkg", type=Path)
    ap.add_argument("--site", type=Path)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-shards", type=int, default=None, help="この回に予測する shard の上限（完了済みは数えない）")
    a = ap.parse_args(argv)
    out = a.out_dir.resolve()
    if a.stage == "list":
        stage_list(a.root.resolve(), out, a.workers)
    else:
        assert a.pkg and a.site
        stage_run(out, a.pkg.resolve(), a.site.resolve(), a.threads, a.max_shards)


if __name__ == "__main__":
    main()
