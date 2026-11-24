#!/usr/bin/env python3
"""#230（#195 A5 の残り）: 見える test 400 分子が、train 以外の外部ライブラリ（MassSpecGym）の参照と一致する割合（読むだけ）。

定義（#230 の本文 §3、数える前に固定）: headroom.md §1.2 と同じ方法を外部ライブラリに当てる。
- 比べ方: `scripts/eda/library_coverage.py` の `cosine`（平方根強度・L2 正規化・許容 0.01 Da）。test のスペクトルごとに、
  同じ付加イオン（文字列が同じ）・前駆体 ±10 ppm の参照と比べ、分子ごとに最大値（best_cos_ext）
- 外部ライブラリ: MassSpecGym の tsv（1.0 と 1.5 の両方の行）。付加イオンは [M+H]+ と [M+Na]+ だけなので、
  負イオンのスペクトルは比べる相手が無い（0 として数える）
- (a) best_cos_ext ≥ 0.99 の分子、(b) そのうち最良一致の参照の構造（inchikey 先頭 14 文字）が train の inchikey14 に無い分子。
  補助: best_cos_ext ≥ 0.7（run.py の LIB_TAU）、比べる相手が 1 件以上ある分子。割合には Wilson 95%
- 見える test は training dataset の例なので、この値は隠れテストの型の比率の推定に使えない

    .venv/bin/python scripts/research/a5_extlib_match.py --msg data/ext/msg/MassSpecGym.tsv data/ext/msg/MassSpecGym1.5.tsv \\
        --out-dir outputs/local_eval/a5_extlib_1124
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
sys.path.insert(0, str(ROOT / "scripts/eda"))
sys.path.insert(0, str(ROOT / "scripts/research"))
from a13_slot_dup import wilson  # noqa: E402
from library_coverage import PPM, cosine, load_flat  # noqa: E402

COPY, TAU = 0.99, 0.7


def prep_peaks(mzs: str, its: str) -> tuple[np.ndarray, np.ndarray]:
    """tsv の 1 行のピーク（カンマ区切り）→ m/z の昇順、平方根強度を L2 正規化（load_flat と同じ処理）。"""
    mz = np.array([float(x) for x in str(mzs).split(",") if x], np.float64)
    it = np.sqrt(np.clip(np.array([float(x) for x in str(its).split(",") if x], np.float64), 0, None))
    o = np.argsort(mz, kind="stable"); mz, it = mz[o], it[o]
    n = np.sqrt((it * it).sum())
    return mz, (it / n if n > 0 else it)


def read_refs(paths, need: dict[str, np.ndarray], chunksize: int = 50_000) -> dict[str, list]:
    """付加イオンごとに、test の前駆体のどれかの ±PPM に入る参照だけを読む。{adduct: [(prec, key14, mz, it), ...]}。"""
    out: dict[str, list] = {}
    cols = ["mzs", "intensities", "inchikey", "precursor_mz", "adduct"]
    for p in paths:
        for ch in pd.read_csv(p, sep="\t", usecols=cols, chunksize=chunksize, dtype={"inchikey": str}):
            for ad, g in ch.groupby("adduct", sort=False):
                q = need.get(ad)
                if q is None or not len(q):
                    continue
                pm = g.precursor_mz.to_numpy(float)
                j = np.searchsorted(q, pm)
                near = np.minimum(np.abs(q[np.clip(j - 1, 0, len(q) - 1)] - pm), np.abs(q[np.clip(j, 0, len(q) - 1)] - pm))
                keep = near <= pm * PPM * 1e-6 * 1.01
                for r in g[keep].itertuples(index=False):
                    mz, it = prep_peaks(r.mzs, r.intensities)
                    out.setdefault(ad, []).append((float(r.precursor_mz), str(r.inchikey)[:14], mz, it))
    return out


def match(test: pd.DataFrame, off, mz, it, refs: dict[str, list]) -> pd.DataFrame:
    """test のスペクトルごとに best_cos・best_key・n_ref（比べた参照の数）。"""
    idx = {ad: (np.array([r[0] for r in rs]), rs) for ad, rs in refs.items()}
    for ad, (pm, rs) in idx.items():
        o = np.argsort(pm, kind="stable"); idx[ad] = (pm[o], [rs[i] for i in o])
    rows = []
    for k, r in enumerate(test.itertuples(index=False)):
        qm, qv = mz[off[k]:off[k + 1]], it[off[k]:off[k + 1]]
        best, bkey, n = 0.0, None, 0
        if r.adduct in idx:
            pm, rs = idx[r.adduct]
            d = r.precursor_mz * PPM * 1e-6
            lo, hi = np.searchsorted(pm, r.precursor_mz - d), np.searchsorted(pm, r.precursor_mz + d, side="right")
            n = hi - lo
            for _, key, rm, ri in rs[lo:hi]:
                s = cosine(qm, qv, rm, ri)
                if s > best:
                    best, bkey = s, key
        rows.append(dict(molecule_id=r.molecule_id, adduct=r.adduct, n_ref=int(n), best_cos=best, best_key=bkey))
    return pd.DataFrame(rows)


def per_molecule(spec: pd.DataFrame, train_keys: set) -> pd.DataFrame:
    s = spec.sort_values("best_cos", ascending=False, kind="stable")
    m = s.groupby("molecule_id", sort=True).agg(n_spectra=("best_cos", "size"), n_ref=("n_ref", "sum"),
                                                 best_cos_ext=("best_cos", "first"), best_ext_key=("best_key", "first"))
    m["best_ext_key_in_train"] = m.best_ext_key.map(lambda k: bool(k) and k in train_keys)
    m["copy_like_ext"] = m.best_cos_ext >= COPY
    m["copy_like_ext_not_in_train"] = m.copy_like_ext & ~m.best_ext_key_in_train
    m["ge_tau_ext"] = m.best_cos_ext >= TAU
    return m.reset_index()


def summarize(m: pd.DataFrame) -> dict:
    n = len(m)

    def share(col):
        k = int(m[col].sum()); lo, hi = wilson(k, n)
        return dict(k=k, n=n, share=round(k / n, 4), wilson95=[round(lo, 4), round(hi, 4)])
    return dict(n=n, copy_like_ext=share("copy_like_ext"), copy_like_ext_not_in_train=share("copy_like_ext_not_in_train"),
                ge_tau_ext=share("ge_tau_ext"), has_ref=dict(k=int((m.n_ref > 0).sum()), n=n),
                best_cos_ext_median=round(float(m.best_cos_ext.median()), 4))


def struct_overlap(best_keys, tables: dict[str, set]) -> dict:
    """補助（#230 の定義の外、結果の後に足した説明）: 見える test の train の最良一致の構造（headroom の A5 表の best_key）が、
    各外部ライブラリの構造表（inchikey 先頭 14 文字）にいくつあるか。"""
    bk = [str(k) for k in best_keys]
    return {name: dict(k=sum(k in keys for k in bk), n=len(bk), n_struct=len(keys)) for name, keys in tables.items()}


def struct_tables(msg_paths, extlib_meta=None, mbh_struct=None) -> dict[str, set]:
    t = {"massspecgym": set()}
    for p in msg_paths:
        for ch in pd.read_csv(p, sep="\t", usecols=["inchikey"], chunksize=100_000, dtype=str):
            t["massspecgym"] |= set(ch.inchikey.astype(str).str[:14])
    if extlib_meta:
        t["extlib_meta"] = set(pd.read_parquet(extlib_meta, columns=["inchikey14"]).inchikey14.astype(str))
    if mbh_struct:
        t["massbank_harmonized"] = set(pd.read_parquet(mbh_struct, columns=["inchikey"]).inchikey.astype(str).str[:14])
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", nargs="+", required=True)
    ap.add_argument("--test", default=str(ROOT / "data/test.parquet"))
    ap.add_argument("--train", default=str(ROOT / "data/train.parquet"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a5-table", default=str(ROOT / "docs/research/headroom-1123/a5_visible_test_per_molecule.csv"))
    ap.add_argument("--extlib-meta", default=None, help="補助: ExtLib の ext_meta.parquet（構造の重なりだけ）")
    ap.add_argument("--mbh-struct", default=None, help="補助: MassBank harmonized の structures_with_fingerprints.parquet")
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    te, off, mz, it = load_flat(a.test, ["molecule_id", "spectrum_id"])
    need = {ad: np.sort(g.precursor_mz.to_numpy(float)) for ad, g in te.groupby("adduct")}
    refs = read_refs(a.msg, need)
    spec = match(te, off, mz, it, refs)
    tk = pq.read_table(a.train, columns=["inchikey14"], read_dictionary=["inchikey14"]).to_pandas().inchikey14
    train_keys = set(tk.astype(str).unique())
    m = per_molecule(spec, train_keys)
    summ = dict(summarize(m), n_test_spectra=int(len(te)), n_ref_loaded={k: len(v) for k, v in refs.items()},
                spectra_without_ref=int((spec.n_ref == 0).sum()), secs=round(time.time() - t0, 1),
                inputs={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [a.test, *a.msg]})
    a5 = pd.read_csv(a.a5_table)
    summ["aux_visible_best_key_in_ext_struct"] = struct_overlap(
        a5.best_key, struct_tables(a.msg, a.extlib_meta, a.mbh_struct))
    m.to_csv(out / "a5_extlib_per_molecule.csv", index=False)
    (out / "a5_extlib.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
