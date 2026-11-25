#!/usr/bin/env python3
"""
項目 1（#244・#245）の特徴表に、式 (6) の候補の中で対比した断片の被覆（Issue #253）の列 `cc`・`cc_z` を足して別名で書く。
run.py は変えない。正解のラベル・MRR は使わない。holdout は値を開かない（記録は行数・sha256・NaN/inf・件数だけ）。

    .venv/bin/python scripts/features/add_contrastive_coverage.py --which dev \
        --in /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8.parquet \
        --out /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc.parquet
    .venv/bin/python scripts/features/add_contrastive_coverage.py --which holdout \
        --in /Users/taichi/.hermes/profiles/massspec/cache/scratch/features_holdout/holdout_v3.parquet \
        --out /Users/taichi/.hermes/profiles/massspec/cache/scratch/features_holdout/holdout_v3_cc.parquet

配線の点検: 同じ断片と同じスペクトル（run.py の query_spectra）から metfrag_lite を作り直し、表の値と |差| ≤ 1e−9 で一致しなければ止める。
集合ごとに <out の親>/parts_cc_<which>/<set>.parquet に書くので途中から再開できる。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

KEY_COLS = ["set", "molecule_id", "cand_smiles", "metfrag_lite", "has_frag"]
MFL_TOL = 1e-9
EXPECT = dict(dev=dict(n_mol=3901, n_rows=430979), holdout=dict(n_mol=1358, n_rows=None))


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def molecule_groups(df: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """(molecule_id, 行の位置) を表の順に。分子の行が連続していなければ止める。"""
    mid = df.molecule_id.astype(str).to_numpy()
    start = np.r_[0, np.flatnonzero(mid[1:] != mid[:-1]) + 1]
    ends = np.r_[start[1:], len(mid)]
    ids = mid[start]
    assert len(set(ids)) == len(ids), "分子の行が連続していない"
    return [(ids[k], np.arange(start[k], ends[k])) for k in range(len(start))]


def run_set(args) -> str:
    """1 集合: 列を作って parts に書く（再開できる）。戻り値は記録の json のパス。"""
    holdout, sub_path, parts = args
    import contrastive_coverage as CC
    out, rec_p = parts / f"{holdout}.parquet", parts / f"{holdout}.json"
    if out.is_file() and rec_p.is_file():
        return str(rec_p)
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "data/ext/frag"))   # numba のキャッシュは通常の import でないと読めない
    import frag_rescore as frag
    frag.warmup()
    from run import query_spectra
    sub = pd.read_parquet(sub_path)
    spectra = query_spectra(holdout)
    cache: dict = {}
    n = len(sub)
    cc, ccz = np.zeros(n), np.zeros(n)
    n_single = n_nopeak = n_nospec = n_unparsed = 0
    max_diff = 0.0
    for mid, pos in molecule_groups(sub):
        smis = sub.cand_smiles.to_numpy()[pos].tolist()
        sp = spectra.get(mid, [])
        r = CC.molecule_features(sp, smis, frag, cache)
        cc[pos], ccz[pos] = r["cc"], r["cc_z"]
        # 配線の点検: 表の metfrag_lite（score_molecule、None は 0）と作り直しの一致
        reb = np.array([0.0 if (v is None or not sp) else float(v) for v in r["metfrag_lite_rebuilt"]])
        d = float(np.max(np.abs(reb - sub.metfrag_lite.to_numpy()[pos]))) if len(pos) else 0.0
        assert d <= MFL_TOL, f"{holdout}/{mid}: metfrag_lite の作り直しが表と違う"
        hf = sub.has_frag.to_numpy()[pos]
        parsed = np.array([v is not None for v in r["metfrag_lite_rebuilt"]])
        assert np.array_equal(hf == 1, parsed & bool(sp)), f"{holdout}/{mid}: has_frag と読めた候補が違う"
        max_diff = max(max_diff, d)
        n_single += len(pos) == 1
        n_nospec += not sp
        n_nopeak += r["n_spectra_with_peaks"] == 0
        n_unparsed += int((~parsed).sum())
    assert np.all(np.isfinite(cc)) and np.all(np.isfinite(ccz)), f"{holdout}: NaN/inf"
    tmp = out.with_suffix(".tmp.parquet")
    pd.DataFrame(dict(cc=cc, cc_z=ccz)).to_parquet(tmp, index=False)
    tmp.replace(out)
    rec = dict(set=holdout, n_rows=n, n_molecules=int(sub.molecule_id.nunique()), n_single_candidate=int(n_single),
               n_no_spectra=int(n_nospec), n_no_spectrum_with_peaks=int(n_nopeak), n_unparsed_candidates=n_unparsed,
               metfrag_rebuild_ok=True, metfrag_rebuild_max_abs_diff=max_diff, secs=round(time.time() - t0, 1))
    rec_p.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(json.dumps({k: rec[k] for k in ("set", "n_rows", "secs")}), flush=True)
    return str(rec_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dev", "holdout"], required=True)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    assert a.inp.is_absolute() and a.out.is_absolute(), "絶対パスで渡す"
    assert a.inp.resolve() != a.out.resolve(), "入力と同じ名前には書かない"
    t0 = time.time()
    parts = a.out.parent / f"parts_cc_{a.which}"
    parts.mkdir(parents=True, exist_ok=True)
    key = pd.read_parquet(a.inp, columns=KEY_COLS)
    sets = list(dict.fromkeys(key.set.tolist()))
    jobs = []
    for s in sets:
        sp = parts / f"{s}.in.parquet"
        if not sp.is_file():
            key[key.set == s].reset_index(drop=True).to_parquet(sp, index=False)
        jobs.append((s, sp, parts))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        recs = [json.loads(Path(p).read_text()) for p in ex.map(run_set, jobs)]
    tab = pd.read_parquet(a.inp)
    assert not set(("cc", "cc_z")) & set(tab.columns), "cc の列が既にある"
    new = []
    for s in sets:
        p = pd.read_parquet(parts / f"{s}.parquet")
        assert len(p) == int((tab.set == s).sum())
        new.append(p)
    new = pd.concat(new, ignore_index=True)
    # 集合の行が表の中で連続していること（concat の順 = 表の順）
    sv = tab.set.to_numpy()
    assert list(dict.fromkeys(sv)) == sets and (np.r_[True, sv[1:] != sv[:-1]].sum() == len(sets)), "集合の行が連続していない"
    out = pd.concat([tab, new], axis=1)
    assert len(out) == len(tab)
    num = out.select_dtypes(include=[np.number]).to_numpy(np.float64)
    tmp = a.out.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(a.out)
    back = pd.read_parquet(a.out)
    pd.testing.assert_frame_equal(back[tab.columns.tolist()], tab)   # 元の列は値も順序も同じ
    rep = dict(which=a.which, issue=253, input=str(a.inp), input_sha256=sha256_file(a.inp),
               output=str(a.out), output_sha256=sha256_file(a.out), output_is_symlink=a.out.is_symlink(),
               output_realpath=str(a.out.resolve()),
               n_rows=int(len(out)), n_molecules=int(out[["set", "molecule_id"]].drop_duplicates().shape[0]),
               n_cols_in=int(tab.shape[1]), n_cols_out=int(out.shape[1]), added=["cc", "cc_z"],
               n_nan=int(np.isnan(num).sum()), n_inf=int(np.isinf(num).sum()),
               n_single_candidate=sum(r["n_single_candidate"] for r in recs),
               n_no_spectra=sum(r["n_no_spectra"] for r in recs),
               n_no_spectrum_with_peaks=sum(r["n_no_spectrum_with_peaks"] for r in recs),
               n_unparsed_candidates=sum(r["n_unparsed_candidates"] for r in recs),
               metfrag_rebuild_ok=all(r["metfrag_rebuild_ok"] for r in recs), metfrag_tol=MFL_TOL,
               secs=round(time.time() - t0, 1), per_set=recs)
    if a.which == "dev":
        rep["metfrag_rebuild_max_abs_diff"] = max(r["metfrag_rebuild_max_abs_diff"] for r in recs)
    else:   # holdout は値に関わる数を書かない
        for r in rep["per_set"]:
            r.pop("metfrag_rebuild_max_abs_diff", None)
    exp = EXPECT[a.which]
    assert rep["n_molecules"] == exp["n_mol"], rep["n_molecules"]
    if exp["n_rows"] is not None:
        assert rep["n_rows"] == exp["n_rows"], rep["n_rows"]
    rp = a.out.with_name(a.out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_set"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
