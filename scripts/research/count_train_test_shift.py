#!/usr/bin/env python3
"""#196: train と見える test の入力の違いを全行で数え直す（#193 / PR #194 §2 の独立の数え直し、B2）。学習・推論はしない。

数えるもの（train.parquet 全行・test.parquet〈見える test〉全行、加えて fragnn v1 が学習・内部検証に使った行）:
- adduct・instrument_type・ionization_mode の内訳。timsTOF の中の adduct の内訳。多量体（[2M…] など、`is_multimer`）の割合
- collision_energy_ev: 欠測 = null または NaN を除いて 0 個（`ce_state`。null / 空リスト / 全 NaN を別々にも数える）、
  複数値 = NaN を除いて 2 個以上。#193 は定義を書いていないので、ここでは上の定義を明記して数える
- ピーク数: (raw) ms2_mzs のリストの長さ、(valid) fragnn の `top_peaks` と同じく m/z・強度が有限で強度 > 0 の本数。
  中央値・p90 を全行で出し、#193 と同じ「先頭 2 万行」の値も並べる
- 同じ分子のスペクトルの本数: 見える test は molecule_id ごと・(molecule_id, mode) ごと、train（v1 の学習の行）は
  (inchikey14, mode) ごと

train.parquet は `pq.ParquetFile(p, buffer_size=1<<20, pre_buffer=False)` で必要な列だけ 8192 行ずつ読む。
v1 の学習の行は outputs/fragnn_train/prep.npz の spec_row・is_val（#120 の学習が書いたもの）から読む。

純粋な関数（CI でテストする）は numpy・pyarrow だけを使う。
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ISSUE = 196
FIRST_N = 20_000                    # #193 の「先頭 2 万行」
COLS_TRAIN = ["precursor_mz", "base_peak_intensity", "adduct", "instrument_type", "ionization_mode", "collision_energy_ev", "ms2_mzs",
              "ms2_normalized_intensities", "num_peaks", "inchikey14", "ingest_lib"]
COLS_TEST = ["precursor_mz", "base_peak_intensity", "molecule_id", "adduct", "instrument_type", "ionization_mode", "collision_energy_ev", "ms2_mzs",
             "ms2_normalized_intensities"]
_MULTIMER = re.compile(r"^\[\s*([2-9]|\d{2,})\s*M")

ABOVE_PREC_DA = 2.0                 # データのページ「Peaks above precursor + 2 Da were removed」
BASE_PEAK_MIN = 1_000.0             # データのページ「base peak was below 1,000 raw counts were dropped」
CE_STATES = ("ok_single", "ok_multi", "missing_null", "missing_empty", "missing_allnan")


# ---------------------------------------------------------------- 純粋な関数（CI）

def is_multimer(adduct) -> bool:
    """'[2M+Na]+'・'[2M-H]-'・'[3M+H]+' などの多量体の付加イオンなら True（'[M+H]+'・'[M+2H]2+' は False）。"""
    return isinstance(adduct, str) and bool(_MULTIMER.match(adduct))


def ce_state(ce) -> str:
    """collision_energy_ev の 1 行の状態。欠測 = null / 空 / NaN を除いて 0 個。複数値 = NaN を除いて 2 個以上。"""
    if ce is None:
        return "missing_null"
    a = np.asarray(ce, np.float64).ravel()
    if a.size == 0:
        return "missing_empty"
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "missing_allnan"
    return "ok_multi" if a.size >= 2 else "ok_single"


def valid_peak_counts(mz_list, it_list, prec=None):
    """pyarrow の ListArray 2 本（同じ行数）→ (raw の本数, 有効な本数)。有効 = m/z・強度が有限で強度 > 0
    （fragnn_bench.top_peaks と同じ条件。m/z と強度の長さが違う行は短い方まで、top_peaks と同じ）。
    prec（行ごとの precursor_mz）を渡すと、3 つ目に「有効なピークのうち m/z > prec + 2 Da の本数」も返す
    （データのページ: 隠れテストは precursor + 2 Da より上のピークを除いてある）。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    n = len(mz_list)
    lm = np.asarray(pc.fill_null(pc.list_value_length(mz_list), 0), np.int64)
    li = np.asarray(pc.fill_null(pc.list_value_length(it_list), 0), np.int64)
    k = np.minimum(lm, li)
    out = np.zeros(n, np.int64)
    above = np.zeros(n, np.int64)
    if k.sum() == 0:
        return (lm, out) if prec is None else (lm, out, above)
    mzv = np.asarray(pc.list_flatten(mz_list).to_numpy(zero_copy_only=False), np.float64)
    itv = np.asarray(pc.list_flatten(it_list).to_numpy(zero_copy_only=False), np.float64)
    pm = np.asarray(pc.list_parent_indices(mz_list), np.int64)
    pi = np.asarray(pc.list_parent_indices(it_list), np.int64)
    # 行ごとの先頭からの位置（j < k[row] だけ使う）
    sm = np.concatenate([[0], np.cumsum(lm)[:-1]])
    si = np.concatenate([[0], np.cumsum(li)[:-1]])
    jm = np.arange(len(mzv)) - sm[pm]
    ji = np.arange(len(itv)) - si[pi]
    mzv, pm = mzv[jm < k[pm]], pm[jm < k[pm]]
    itv = itv[ji < k[pi]]
    ok = np.isfinite(mzv) & np.isfinite(itv) & (itv > 0)
    out = np.bincount(pm[ok], minlength=n).astype(np.int64)
    del pa
    if prec is None:
        return lm, out
    pr = np.asarray(prec, np.float64)
    hi = ok & (mzv > pr[pm] + ABOVE_PREC_DA)
    above = np.bincount(pm[hi], minlength=n).astype(np.int64)
    return lm, out, above


def quantiles(x) -> dict:
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), p10=float(np.percentile(x, 10)), median=float(np.median(x)),
                p90=float(np.percentile(x, 90)), max=float(x.max()), zero=int((x == 0).sum()))


def shares(counter: Counter, total: int, top: int | None = None) -> list:
    items = counter.most_common(top)
    return [[str(k), int(v), round(v / total, 6) if total else None] for k, v in items]


def summarize(rows: dict, idx: np.ndarray | None = None, timstof: str = "timsTOF") -> dict:
    """rows: 列名 → 行ごとの配列（adduct, instrument_type, ionization_mode, ce_state, peaks_raw, peaks_valid）。
    idx: 数える行（None なら全行）。"""
    sel = np.arange(len(rows["adduct"])) if idx is None else np.asarray(idx, np.int64)
    ad = rows["adduct"][sel]; ins = rows["instrument_type"][sel]; mo = rows["ionization_mode"][sel]
    ce = rows["ce_state"][sel]
    n = len(sel)
    multi = np.array([is_multimer(a) for a in ad], bool) if n else np.zeros(0, bool)
    tims = ins == timstof
    ce_c = Counter(ce.tolist())
    miss = sum(ce_c.get(s, 0) for s in CE_STATES if s.startswith("missing"))
    return dict(
        n_rows=int(n),
        instrument=shares(Counter(ins.tolist()), n),
        mode=shares(Counter(mo.tolist()), n),
        adduct=shares(Counter(ad.tolist()), n),
        multimer=dict(n=int(multi.sum()), share=float(multi.mean()) if n else None,
                      adducts=shares(Counter(ad[multi].tolist()), n)),
        timstof=dict(n=int(tims.sum()), share=float(tims.mean()) if n else None,
                     adduct=shares(Counter(ad[tims].tolist()), int(tims.sum()), top=15),
                     multimer_n=int((multi & tims).sum()),
                     multimer_share_in_timstof=float((multi & tims).sum() / max(1, tims.sum()))),
        ce=dict(states={s: int(ce_c.get(s, 0)) for s in CE_STATES}, missing=int(miss),
                missing_share=float(miss / n) if n else None, multi=int(ce_c.get("ok_multi", 0)),
                multi_share=float(ce_c.get("ok_multi", 0) / n) if n else None),
        peaks_raw=quantiles(rows["peaks_raw"][sel]),
        peaks_valid=quantiles(rows["peaks_valid"][sel]),
        **({} if "peaks_above_prec2" not in rows else dict(above_prec2=_above(rows, sel))),
        **({} if "base_peak" not in rows else dict(base_peak=_base(rows, sel))),
    )


def _above(rows: dict, sel) -> dict:
    a = rows["peaks_above_prec2"][sel]; v = rows["peaks_valid"][sel]
    return dict(rows_with_any=int((a > 0).sum()), rows_share=float((a > 0).mean()) if len(a) else None,
                peaks=int(a.sum()), peak_share=float(a.sum() / max(1, v.sum())))


def _base(rows: dict, sel) -> dict:
    b = rows["base_peak"][sel]
    return dict(null=int(np.isnan(b).sum()), below_1000=int((b < BASE_PEAK_MIN).sum()),
                below_1000_share_of_nonnull=float((b < BASE_PEAK_MIN).sum() / max(1, (~np.isnan(b)).sum())))


def group_sizes(keys) -> dict:
    """同じキーの行の数（1 キーあたりのスペクトルの本数）の分布。"""
    _, c = np.unique(np.asarray(keys), return_counts=True)
    q = quantiles(c)
    q["n_groups"] = int(len(c))
    q["share_groups_ge2"] = float((c >= 2).mean()) if len(c) else None
    q["hist_1_to_16plus"] = [int((c == i).sum()) for i in range(1, 16)] + [int((c >= 16).sum())]
    return q


# ---------------------------------------------------------------- 読み込み（ローカル）

def read_rows(path: Path, cols: list, batch_size: int = 8192) -> dict:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    acc = {k: [] for k in ("adduct", "instrument_type", "ionization_mode", "ce_state", "peaks_raw", "peaks_valid",
                           "num_peaks", "key", "ingest_lib", "peaks_above_prec2", "base_peak")}
    for b in pf.iter_batches(batch_size=batch_size, columns=cols):
        d = {c: b.column(c) for c in cols}
        acc["adduct"].append(np.array(d["adduct"].to_pylist(), object))
        acc["instrument_type"].append(np.array(d["instrument_type"].to_pylist(), object))
        acc["ionization_mode"].append(np.array(d["ionization_mode"].to_pylist(), object))
        acc["ce_state"].append(np.array([ce_state(x) for x in d["collision_energy_ev"].to_pylist()], object))
        prec = np.asarray(d["precursor_mz"].fill_null(np.nan).to_numpy(zero_copy_only=False), np.float64)
        raw, val, above = valid_peak_counts(d["ms2_mzs"], d["ms2_normalized_intensities"], prec)
        acc["peaks_raw"].append(raw.astype(np.int32)); acc["peaks_valid"].append(val.astype(np.int32))
        acc["peaks_above_prec2"].append(above.astype(np.int32))
        acc["base_peak"].append(np.asarray(d["base_peak_intensity"].fill_null(np.nan).to_numpy(zero_copy_only=False),
                                           np.float64))
        if "num_peaks" in d:
            acc["num_peaks"].append(np.asarray(d["num_peaks"].fill_null(-1).to_numpy(), np.int64).astype(np.int32))
        kcol = "inchikey14" if "inchikey14" in d else "molecule_id"
        acc["key"].append(np.array(d[kcol].to_pylist(), object))
        if "ingest_lib" in d:
            acc["ingest_lib"].append(np.array(d["ingest_lib"].to_pylist(), object))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in acc.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="#196 train と見える test の入力の違いの数え直し")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--prep", default=None, help="fragnn v1 の prep.npz（spec_row・is_val）。無ければ v1 の行は数えない")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    te = read_rows(Path(a.test), COLS_TEST)
    t_test = time.time() - t0
    tr = read_rows(Path(a.train), COLS_TRAIN)
    t_train = time.time() - t0 - t_test
    res = dict(issue=ISSUE, definitions=dict(
        multimer="adduct が正規表現 ^\\[\\s*([2-9]|\\d{2,})\\s*M に合う（[2M…]・[3M…] など）",
        ce_missing="collision_energy_ev が null、空リスト、または NaN を除いて 0 個",
        ce_multi="NaN を除いて 2 個以上",
        peaks_raw="ms2_mzs のリストの長さ",
        peaks_valid="m/z・強度が有限で強度 > 0 の本数（fragnn_bench.top_peaks と同じ条件）",
        above_prec2="有効なピークのうち m/z > precursor_mz + 2 Da（データのページで隠れテストは除去済み）",
        base_peak="base_peak_intensity < 1,000（データのページで隠れテストは除去済み）。null は別に数える",
        first_n=f"ファイルの先頭 {FIRST_N} 行（#193 と同じ切り方）"))
    res["visible_test"] = summarize(te)
    res["visible_test"]["ce_values"] = shares(Counter(
        round(float(v), 3) for x in pq_ce_values(Path(a.test)) for v in x), int(len(te["adduct"])), top=20)
    res["visible_test"]["spectra_per_molecule"] = group_sizes(te["key"])
    res["visible_test"]["spectra_per_molecule_mode"] = group_sizes(
        np.array([f"{k}|{m}" for k, m in zip(te["key"], te["ionization_mode"])], object))
    res["visible_test"]["adduct_list"] = sorted({str(x) for x in te["adduct"]})
    res["train_all"] = summarize(tr)
    res["train_first20k"] = summarize(tr, np.arange(min(FIRST_N, len(tr["adduct"]))))
    tims = tr["instrument_type"] == "timsTOF"
    res["train_all"]["instrument_contains_tims"] = shares(Counter(
        x for x in tr["instrument_type"].tolist() if isinstance(x, str) and "tims" in x.lower()), len(tr["adduct"]))
    res["train_all"]["timstof_ingest_lib"] = shares(Counter(tr["ingest_lib"][tims].tolist()), int(tims.sum()), top=5)
    # 隠れテストと同じ装置の部分（timsTOF）と、データのページが「test と同じ装置・同じ処理」と書く enveda-np-examples
    res["train_timstof"] = summarize(tr, np.flatnonzero(tims))
    res["train_enveda_np_examples"] = summarize(tr, np.flatnonzero(tr["ingest_lib"] == "enveda-np-examples"))
    npk = tr["num_peaks"]
    res["train_all"]["num_peaks_col_vs_list"] = dict(equal=int((npk == tr["peaks_raw"]).sum()),
                                                     differ=int((npk != tr["peaks_raw"]).sum()))
    if a.prep:
        z = np.load(a.prep)
        row, is_val = z["spec_row"].astype(np.int64), z["is_val"].astype(bool)
        ik = np.array([s.decode() for s in z["spec_ik"]], object)
        assert np.all(tr["key"][row] == ik), "prep.npz の spec_ik と train.parquet の inchikey14 が合わない"
        res["v1_train"] = summarize(tr, row[~is_val])
        res["v1_val"] = summarize(tr, row[is_val])
        trr = row[~is_val]
        gk = np.array([f"{k}|{m}" for k, m in zip(tr["key"][trr], tr["ionization_mode"][trr])], object)
        res["v1_train"]["spectra_per_struct_mode"] = group_sizes(gk)
        keep = np.array([not is_multimer(x) for x in tr["adduct"][trr]], bool)
        res["v1_train"]["after_p1"] = dict(n_rows=int(keep.sum()), n_dropped=int((~keep).sum()),
                                           n_struct_before=int(len(np.unique(tr["key"][trr]))),
                                           n_struct_after=int(len(np.unique(tr["key"][trr][keep]))),
                                           spectra_per_struct_mode=group_sizes(gk[keep]))
        vr = row[is_val]
        vkeep = np.array([not is_multimer(x) for x in tr["adduct"][vr]], bool)
        res["v1_val"]["after_p1"] = dict(n_rows=int(vkeep.sum()), n_dropped=int((~vkeep).sum()),
                                         n_struct_before=int(len(np.unique(tr["key"][vr]))),
                                         n_struct_after=int(len(np.unique(tr["key"][vr][vkeep]))))
    res["seconds"] = dict(test=round(t_test, 1), train=round(t_train, 1), total=round(time.time() - t0, 1))
    res["ru_maxrss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 3)   # macOS はバイト
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(dict(out=str(out), seconds=res["seconds"], ru_maxrss_gb=res["ru_maxrss_gb"]), ensure_ascii=False))


def pq_ce_values(path: Path):
    import pyarrow.parquet as pq
    for x in pq.read_table(path, columns=["collision_energy_ev"]).column(0).to_pylist():
        yield [v for v in (x or []) if v is not None and np.isfinite(v)]


if __name__ == "__main__":
    sys.exit(main())
