#!/usr/bin/env python3
"""
dev 表に GLACIER L60 の予測と観測の一致の列（Issue #259）を足して別名で書く。run.py は変えない。
正解のラベル・MRR は使わない（記録は行数・sha256・NaN/inf・件数だけ）。

    PY=".../.venv/bin/python"
    $PY scripts/features/add_glacier_match.py \
        --in  /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc.parquet \
        --out /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc_gl.parquet \
        --gl-dir /Users/taichi/work/molecule-id-dev/outputs/local_eval/glacier_l60

定義（#259 の定義のコメント）:
- 観測 = data/holdouts/<set>/test.parquet、分子ごとに run.py の query_spectra と同じ並び・先頭 16 本
- 区分 = glacier_counts.spectrum_jobs(..., COV_R)、予測 = jobs.parquet の行の shard（ok = 0 は予測なし）
- m/z 許容 = frag_rescore.resolve_cfg(None)["tol"]（MetFrag-lite と同じ）
- 列 = gl_cos_mean・gl_cos_max・gl_ent_mean・gl_ent_max・has_gl ＋ gl_cos_mean・gl_ent_mean の式 (8) の相対化 10 列

配線の点検: 観測の m/z・強度が run.py の query_spectra と一致、先頭 16 本の区分 ⊆ jobs の区分、jobs の候補 = 表の
プールの行（被覆のある分子）、has_gl = 1 の行に NaN/inf 無し、元の列は値も順序も同じ。
集合ごとに <out の親>/parts_gl_dev/<set>.parquet に書くので途中から再開できる。

Issue #268: `--gl-dir-tier <glacier_tier>`（glacier_l60.py list --part tier の出力）を渡すと、PubChem 層の行
（is_tier = 1）にも同じ関数・同じ定義で列を埋める（parts は parts_gl_dev_tier/）。渡さなければ #259 と同じ動き。

Issue #290（項目 6）: `--which holdout` で判定用 4 集合（holdout_v3_cc_n1、1,358 分子。行数は dump_dev の記録の
n_rows の和 = add_neighbour_ref.expect_for と同じ）に同じ定義・同じ点検で足す。parts は parts_gl_holdout[_tier]/。
holdout は値を開かない: 記録は行数・分子数・has_gl の数・NaN/inf・sha256 だけ（正解のラベルの列は読まない）。

    $PY scripts/features/add_glacier_match.py --which holdout \
        --in  /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/holdout_v3_cc_n1.parquet \
        --out /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/holdout_v3_cc_n1_gl.parquet \
        --gl-dir      /Users/taichi/work/molecule-id-dev/outputs/local_eval/glacier_holdout/pool \
        --gl-dir-tier /Users/taichi/work/molecule-id-dev/outputs/local_eval/glacier_holdout/tier
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/features", "scripts/local_eval", "scripts/bench"):
    sys.path.insert(0, str(ROOT / _p))

import glacier_match as GM  # noqa: E402

KEY_COLS = ["set", "molecule_id", "cand_smiles", "is_tier"]
N_SPEC = 16                      # run.py の query_spectra の上限
EXPECT = dict(n_mol=3901, n_rows=430979)
SHARD_SIZE = 20_000              # glacier_l60.SHARD_SIZE
NEW_COLS = GM.SIM_COLS + ("has_gl",)


def expect_for(which: str, parts_dir: Path | None = None) -> dict:
    """期待の集合と数。dev は固定の値（#259）、holdout は add_neighbour_ref.expect_for（dump_dev の記録の和、#290）。"""
    if which == "dev":
        return dict(sets=None, **EXPECT)
    import add_neighbour_ref as NR
    return NR.expect_for(which) if parts_dir is None else NR.expect_for(which, parts_dir)


def parts_name(which: str, tier: bool) -> str:
    """集合ごとの parts の置き場の名前（dev は #259/#268 と同じ名前のまま）。"""
    return ("parts_gl_dev" if which == "dev" else "parts_gl_holdout") + ("_tier" if tier else "")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mz_tol() -> float:
    sys.path.insert(0, str(ROOT / "data/ext/frag"))
    import frag_rescore as frag
    return float(frag.resolve_cfg(None)["tol"])


def observed(holdout: str) -> dict:
    """分子 → [(prep 済みの (mz, it), 区分の集合)]（run.py の query_spectra と同じ並び・先頭 16 本）。"""
    import glacier_counts as G
    te = pd.read_parquet(ROOT / f"data/holdouts/{holdout}/test.parquet")
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        lst = []
        for r in list(g.itertuples())[:N_SPEC]:
            cats = G.spectrum_jobs(r.adduct, r.ionization_mode, r.instrument_type, r.collision_energy_orig_units,
                                   r.collision_energy_orig, r.collision_energy_ev, G.COV_R)
            lst.append((np.asarray(r.ms2_mzs, np.float64), np.asarray(r.ms2_normalized_intensities, np.float64),
                        cats))
        out[str(mid)] = lst
    return out


def load_preds(gl_dir: Path, jobs: pd.DataFrame) -> dict:
    """jobs（1 集合の行、row 順）→ {(molecule_id, smiles, (adduct, instr, ce)): prep 済みの (mz, it)}。ok = 0 は入れない。"""
    rows = jobs["row"].to_numpy()
    lo, hi = int(rows.min()), int(rows.max()) + 1
    need = range(lo // SHARD_SIZE, (hi - 1) // SHARD_SIZE + 1)
    pos = {}
    for i in need:
        z = np.load(gl_dir / "shards" / f"shard_{i:05d}.npz")
        r, ok, off, mz, it = z["row"], z["ok"], z["offsets"], z["mz"], z["it"]
        sel = np.flatnonzero((r >= lo) & (r < hi) & ok)
        for k in sel:
            pos[int(r[k])] = (mz[off[k]:off[k + 1]], it[off[k]:off[k + 1]])
    out = {}
    for t in jobs.itertuples(index=False):
        p = pos.get(int(t.row))
        if p is not None:
            out[(t.molecule_id, t.smiles, (t.adduct, t.instr, float(t.ce)))] = GM.prep(*p)
    return out


def check_jobs(holdout, mid, jc, job_cands, cats_first, cand_smis) -> bool:
    """1 分子の配線の点検（pool・tier 共通）。被覆のある分子なら True。
    候補の行が無い分子（n_L = 0）は jobs に行が無い。それ以外は先頭 16 本の区分 ⊆ jobs の区分。"""
    assert cats_first <= jc or not cand_smis, f"{holdout}/{mid}: 先頭 16 本の区分が jobs に無い"
    if jc:
        assert job_cands.get(mid, set()) == cand_smis, f"{holdout}/{mid}: jobs の候補と表の行が違う"
        return True
    assert mid not in job_cands
    return False


def fill_rows(pos, smis, mid, qprep, preds, tol, cols, has) -> tuple[int, int]:
    """行 pos（同じ分子）に一致の列を埋める。preds = {(molecule_id, smiles, 区分): prep 済み予測}。
    返り値 = (使った (観測, 予測) の対の数, 値のあるスペクトルの数)。"""
    n_pairs = n_spec_used = 0
    for p in pos:
        vals = []
        for q, cs in qprep:
            ps = [preds[k] for k in ((mid, smis[p], c) for c in sorted(cs)) if k in preds]
            n_pairs += len(ps)
            vals.append(GM.spectrum_value(q, ps, tol))
        a = GM.aggregate(vals)
        n_spec_used += sum(v is not None for v in vals)
        for c in GM.SIM_COLS:
            cols[c][p] = a[c]
        has[p] = a["has_gl"]
    return n_pairs, n_spec_used


def _set_jobs(gl_dir: Path, holdout: str):
    jobs = pd.read_parquet(gl_dir / "jobs.parquet", filters=[("set", "==", holdout)])
    jobs = jobs.sort_values("row").reset_index(drop=True)
    if len(jobs) == 0:
        return jobs, {}, {}, {}
    preds = load_preds(gl_dir, jobs)
    job_cats = jobs.groupby("molecule_id")[["adduct", "instr", "ce"]].apply(
        lambda g: set(zip(g.adduct, g.instr, g.ce.astype(float)))).to_dict()
    job_cands = jobs.groupby("molecule_id")["smiles"].apply(set).to_dict()
    return jobs, preds, job_cats, job_cands


def run_set(args) -> str:
    holdout, sub_path, parts, gl_dir, tol = args[:5]
    gl_dir_tier = args[5] if len(args) > 5 else None
    out, rec_p = parts / f"{holdout}.parquet", parts / f"{holdout}.json"
    if out.is_file() and rec_p.is_file():
        return str(rec_p)
    t0 = time.time()
    from run import query_spectra
    sub = pd.read_parquet(sub_path)
    jobs, preds, job_cats, job_cands = _set_jobs(gl_dir, holdout)
    if gl_dir_tier is not None:
        jobs_t, preds_t, job_cats_t, job_cands_t = _set_jobs(gl_dir_tier, holdout)
    obs = observed(holdout)
    qs = query_spectra(holdout)
    assert set(qs) == set(obs)
    for mid, lst in obs.items():   # 配線: run.py の query_spectra と同じ観測
        assert len(lst) == len(qs[mid])
        for (mz, it, _), q in zip(lst, qs[mid]):
            assert np.array_equal(mz, q["mz"]) and np.array_equal(it, q["it"]), f"{holdout}/{mid}: 観測が違う"
    n = len(sub)
    cols = {c: np.full(n, np.nan) for c in GM.SIM_COLS}
    has = np.zeros(n, np.int8)
    mids = sub.molecule_id.astype(str).to_numpy()
    smis = sub.cand_smiles.astype(str).to_numpy()
    tier = sub.is_tier.to_numpy()
    n_pairs = n_spec_used = 0
    n_mol_cov = n_mol_cov_t = n_pairs_t = n_spec_used_t = 0
    for mid in pd.unique(mids):
        pos = np.flatnonzero(mids == mid)
        lst = obs.get(mid, [])
        cats_first = set().union(*[c for _, _, c in lst]) if lst else set()
        qprep = [(GM.prep(mz, it), c) for mz, it, c in lst]
        pos_pool = pos[tier[pos] == 0]
        n_mol_cov += check_jobs(holdout, mid, job_cats.get(mid, set()), job_cands, cats_first, set(smis[pos_pool]))
        a, b = fill_rows(pos_pool, smis, mid, qprep, preds, tol, cols, has)
        n_pairs += a
        n_spec_used += b
        if gl_dir_tier is not None:
            pos_t = pos[tier[pos] != 0]
            n_mol_cov_t += check_jobs(holdout, mid, job_cats_t.get(mid, set()), job_cands_t, cats_first,
                                      set(smis[pos_t]))
            a, b = fill_rows(pos_t, smis, mid, qprep, preds_t, tol, cols, has)
            n_pairs_t += a
            n_spec_used_t += b
    h1 = has == 1
    for c in GM.SIM_COLS:
        assert np.isfinite(cols[c][h1]).all() and np.isnan(cols[c][~h1]).all(), f"{holdout}: {c} の欠け"
    if gl_dir_tier is None:
        assert (has[tier != 0] == 0).all()
    df = pd.DataFrame({**cols, "has_gl": has})
    tmp = out.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    rec = dict(set=holdout, n_rows=n, n_molecules=int(len(pd.unique(mids))), n_molecules_covered=n_mol_cov,
               n_has_gl=int(h1.sum()), n_has_gl0=int((~h1).sum()), n_has_gl0_pool=int(((~h1) & (tier == 0)).sum()),
               n_jobs=int(len(jobs)), n_preds_ok=len(preds), n_pairs=int(n_pairs), n_spectrum_values=int(n_spec_used),
               **({} if gl_dir_tier is None else dict(
                   n_molecules_covered_tier=n_mol_cov_t, n_has_gl_tier=int((h1 & (tier != 0)).sum()),
                   n_has_gl0_tier=int(((~h1) & (tier != 0)).sum()), n_jobs_tier=int(len(jobs_t)),
                   n_preds_ok_tier=len(preds_t), n_pairs_tier=int(n_pairs_t),
                   n_spectrum_values_tier=int(n_spec_used_t))),
               secs=round(time.time() - t0, 1),
               maxrss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    rec_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps({k: rec[k] for k in ("set", "n_rows", "secs")}), flush=True)
    return str(rec_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gl-dir", type=Path, required=True)
    ap.add_argument("--gl-dir-tier", type=Path, default=None,
                    help="#268: PubChem 層の予測（glacier_l60.py list --part tier の out-dir）。渡すと層の行も埋める")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sets", default="", help="試しに一部の集合だけ（parts だけ書いて表は書かない）")
    ap.add_argument("--which", choices=["dev", "holdout"], default="dev",
                    help="#290: holdout = 判定用 4 集合（期待の数は dump_dev の記録）")
    a = ap.parse_args()
    exp = expect_for(a.which)
    assert a.inp.is_absolute() and a.out.is_absolute() and a.gl_dir.is_absolute(), "絶対パスで渡す"
    assert a.gl_dir_tier is None or a.gl_dir_tier.is_absolute(), "絶対パスで渡す"
    assert a.inp.resolve() != a.out.resolve(), "入力と同じ名前には書かない"
    t0 = time.time()
    tol = mz_tol()
    parts = a.out.parent / parts_name(a.which, a.gl_dir_tier is not None)
    parts.mkdir(parents=True, exist_ok=True)
    key = pd.read_parquet(a.inp, columns=KEY_COLS)
    key["molecule_id"] = key.molecule_id.astype(str)
    sets = list(dict.fromkeys(key.set.tolist()))
    if exp["sets"] is not None and sorted(sets) != sorted(exp["sets"]):
        raise ValueError(f"表の集合が {a.which} の {len(exp['sets'])} 集合と違う: {sets}")
    run_sets = [s for s in a.sets.split(",") if s] or sets
    jobs = []
    for s in run_sets:
        sp = parts / f"{s}.in.parquet"
        if not sp.is_file():
            key[key.set == s].reset_index(drop=True).to_parquet(sp, index=False)
        jobs.append((s, sp, parts, a.gl_dir, tol, a.gl_dir_tier))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = [json.loads(Path(p).read_text()) for p in ex.map(run_set, jobs)]
    if a.sets:
        print(json.dumps(recs, ensure_ascii=False, indent=1))
        return
    tab = pd.read_parquet(a.inp)
    added = list(NEW_COLS) + [f"{c}_{s}" for c in GM.REL_SRC for s in ("dmean", "z", "dmax", "softmax", "sd")]
    assert not set(added) & set(tab.columns), "gl の列が既にある"
    sv = tab.set.to_numpy()
    assert list(dict.fromkeys(sv)) == sets and (np.r_[True, sv[1:] != sv[:-1]].sum() == len(sets)), "集合の行が連続していない"
    new = []
    for s in sets:
        p = pd.read_parquet(parts / f"{s}.parquet")
        assert len(p) == int((tab.set == s).sum())
        new.append(p)
    new = pd.concat(new, ignore_index=True)
    out = pd.concat([tab, new], axis=1)
    out = GM.relativize_masked(out, ["set", "molecule_id"])
    assert list(out.columns) == list(tab.columns) + added
    h1 = out.has_gl.to_numpy() == 1
    g = out[added].drop(columns=["has_gl"]).to_numpy(np.float64)
    n_bad_h1 = int((~np.isfinite(g[h1])).sum())
    assert n_bad_h1 == 0, "has_gl = 1 の行に NaN/inf"
    assert np.isnan(g[~h1]).all(), "has_gl = 0 の行に値がある"
    tmp = a.out.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(a.out)
    back = pd.read_parquet(a.out)
    pd.testing.assert_frame_equal(back[tab.columns.tolist()], tab)   # 元の列は値も順序も同じ
    old = tab.select_dtypes(include=[np.number]).to_numpy(np.float64)
    tier = out.is_tier.to_numpy()
    rep = dict(issue=(259 if a.gl_dir_tier is None else 268) if a.which == "dev" else 290, which=a.which,
               expect=dict(n_mol=exp["n_mol"], n_rows=exp["n_rows"]), input=str(a.inp), input_sha256=sha256_file(a.inp), output=str(a.out),
               output_sha256=sha256_file(a.out), gl_dir=str(a.gl_dir),
               jobs_sha256=sha256_file(a.gl_dir / "jobs.parquet"), mz_tol=tol,
               **({} if a.gl_dir_tier is None else dict(
                   gl_dir_tier=str(a.gl_dir_tier), jobs_tier_sha256=sha256_file(a.gl_dir_tier / "jobs.parquet"))),
               n_rows=int(len(out)), n_molecules=int(out[["set", "molecule_id"]].drop_duplicates().shape[0]),
               n_cols_in=int(tab.shape[1]), n_cols_out=int(out.shape[1]), added=added,
               n_has_gl1=int(h1.sum()), n_has_gl0=int((~h1).sum()), n_has_gl0_pool=int(((~h1) & (tier == 0)).sum()),
               n_has_gl0_tier=int(((~h1) & (tier != 0)).sum()), n_has_gl1_tier=int((h1 & (tier != 0)).sum()),
               n_nonfinite_has_gl1=n_bad_h1,
               n_nan_old_cols=int(np.isnan(old).sum()), n_inf_old_cols=int(np.isinf(old).sum()),
               n_molecules_has_gl1=int(out.loc[h1, ["set", "molecule_id"]].drop_duplicates().shape[0]),
               secs=round(time.time() - t0, 1), per_set=recs)
    assert rep["n_molecules"] == exp["n_mol"] and rep["n_rows"] == exp["n_rows"], rep
    assert rep["n_rows"] == int(len(tab)), rep   # 行数不変
    rp = a.out.with_name(a.out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_set"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
