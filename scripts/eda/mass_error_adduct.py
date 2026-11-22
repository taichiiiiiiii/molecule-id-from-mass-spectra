#!/usr/bin/env python3
"""#203: 前駆体の質量誤差（A11）・付加イオンの誤りと in-source 断片（A12）の数え上げ。学習なし・採点なし。

事前登録: Issue #203（標本の数・seed・既知の差の表・窓の定義はそこで固定。ここでは変えない）。
段（それぞれ別のプロセスで動かし、`/usr/bin/time -l` で peak RSS を測る）:

  train    : train.parquet を row group ごとに読み、`molecular_formula` の単同位体質量 M と
             `casmi.chem.neutral_mass(precursor_mz, adduct)` の ppm 誤差を、装置 × 付加イオンごとに集計する。
             あわせて断片の標本（row group ごとに default_rng(SEED + rg) で 25 行）を選び、ピークと SMILES を書く。
             見える test の推定の正解（test_library_match の best_key）の組成式も引く
  frag     : 標本の各スペクトルの強度の上位 32 本のピークを、frag_rescore.fragment_levels の断片 + fragnn_bench の
             ずらし（担い手 × H −2..+3）と照らし、最も近いものとの符号つき誤差（Da）の分布を出す
  holdouts : v3 の 4 集合と見える test で、スペクトルごと・分子ごと（窓の中心 = 中性質量の中央値）の ppm 誤差、
             ±10 ppm の窓の外の正解、既知の差（KNOWN_DELTAS）に当たる数、付加イオン違いの中性質量の一致

    python scripts/eda/mass_error_adduct.py train
    python scripts/eda/mass_error_adduct.py frag
    python scripts/eda/mass_error_adduct.py holdouts
    python scripts/eda/mass_error_adduct.py width      # §5-3 の案の窓の半幅（train.json と holdouts.json から）

純粋な関数（CI でテストする）は rdkit・casmi を import しない。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ISSUE = 203
SEED = 20261123
N_PER_RG = 25
MIN_PEAKS = 3
PPM = 10.0                 # scripts/local_eval/run.py の PPM（窓の半幅）
TOL = 0.005                # scripts/train/fragnn_bench.py の TOL（frag_rescore.DEFAULT_CFG の tol）
FRAG_MAXERR = 0.05         # 断片の誤差の分布を取る範囲（Da）
FRAG_BIN = 0.001           # 刻み（Da）
BG_LO, BG_HI = 0.02, 0.05  # 偶然の一致の背景の密度を取る区間（Da）
MIN_GROUP = 100            # これより少ない装置 × 付加イオンの組は「その他」
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
import os  # noqa: E402
OUT = Path(os.environ.get("MASS_ERR_OUT", ROOT / "outputs/local_eval/mass_error_1123"))
MAX_RG = int(os.environ.get("MASS_ERR_MAX_RG", "0"))   # 0 = 全部。煙の試しのときだけ小さくする

# 既知の差（Da）。Issue #203 の表（結果の前に固定）
KNOWN_DELTAS = {
    "H2O": 18.010565, "2H2O": 36.021129, "NH3": 17.026549, "Na-H": 21.981942, "K-H": 37.955882,
    "CO2": 43.989829, "HCOOH": 46.005479, "CH3OH": 32.026215, "CH3CN": 41.026549, "H2": 2.015650,
    "hexose": 162.052824, "deoxyhexose": 146.057909, "pentose": 132.042259, "glucuronic": 176.032088,
    "13C": 1.003355,
}
MULTIMER_RATIOS = {"2M": 2.0, "M/2": 0.5}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


# ---------------------------------------------------------------- 純粋な関数
def parse_formula(formula, masses: dict) -> float:
    """組成式 → 単同位体質量。末尾の電荷（+ / - / 数字つき）は無視する。読めなければ nan。
    masses は元素 → 最多同位体の質量。表に無い元素・余分な文字があれば nan。"""
    if not isinstance(formula, str) or not formula:
        return float("nan")
    f = re.sub(r"[+-]+$", "", formula.strip())   # train の電荷の書き方は末尾の + / - だけ（数字なし）
    pos, tot = 0, 0.0
    for m in _FORMULA_TOKEN.finditer(f):
        if m.start() != pos:
            return float("nan")
        el, n = m.group(1), m.group(2)
        if el not in masses:
            return float("nan")
        tot += masses[el] * (int(n) if n else 1)
        pos = m.end()
    if pos != len(f) or pos == 0:
        return float("nan")
    return tot


def ppm_error(observed, exact) -> np.ndarray:
    o = np.asarray(observed, np.float64); e = np.asarray(exact, np.float64)
    return (o - e) / e * 1e6


def group_stats(ppm: np.ndarray, ppm_lim: float = PPM) -> dict:
    """件数・符号つき中央値・|ppm| の 99% 点・|ppm| > ppm_lim の割合（有限の値だけ）。"""
    v = np.asarray(ppm, np.float64); v = v[np.isfinite(v)]
    if not len(v):
        return dict(n=0, median=None, abs_p99=None, frac_out=None, n_out=0)
    a = np.abs(v)
    return dict(n=int(len(v)), median=round(float(np.median(v)), 3), abs_p99=round(float(np.quantile(a, 0.99)), 3),
                frac_out=round(float((a > ppm_lim).mean()), 6), n_out=int((a > ppm_lim).sum()))


def grouped_table(instr, adduct, ppm, min_group: int = MIN_GROUP) -> list:
    """装置 × 付加イオンごとの group_stats。件数 < min_group の組は装置ごとに「その他」にまとめる。"""
    instr = np.asarray([str(x) for x in instr], dtype=object)   # None / nan も文字列の組にする
    adduct = np.asarray([str(x) for x in adduct], dtype=object); ppm = np.asarray(ppm, np.float64)
    ok = np.isfinite(ppm)
    keys = {}
    for i, a, f in zip(instr, adduct, ok):
        if f:
            keys[(i, a)] = keys.get((i, a), 0) + 1
    rows, other = [], {}
    for (i, a), n in sorted(keys.items(), key=lambda kv: (str(kv[0][0]), -kv[1])):
        m = (instr == i) & (adduct == a) & ok
        if n >= min_group:
            rows.append(dict(instrument=str(i), adduct=str(a), **group_stats(ppm[m])))
        else:
            other.setdefault(i, np.zeros(len(ppm), bool)); other[i] |= m
    for i, m in other.items():
        rows.append(dict(instrument=str(i), adduct="その他", **group_stats(ppm[m])))
    return rows


def match_known_delta(delta_da: float, exact: float, ppm_lim: float = PPM, deltas: dict = KNOWN_DELTAS) -> list:
    """Δ = 観測の中性質量 − M が ±d（d は既知の差）に ±ppm_lim × M で当たる名前（符号つき）の一覧。"""
    tol = abs(exact) * ppm_lim * 1e-6
    out = []
    for name, d in deltas.items():
        if abs(delta_da - d) <= tol:
            out.append("+" + name)
        if abs(delta_da + d) <= tol:
            out.append("-" + name)
    return out


def match_multimer(observed: float, exact: float, ppm_lim: float = PPM, ratios: dict = MULTIMER_RATIOS) -> list:
    out = []
    for name, r in ratios.items():
        if abs(observed - r * exact) <= abs(r * exact) * ppm_lim * 1e-6:
            out.append(name)
    return out


def nearest_errors(peaks_mz, frag_masses, shifts) -> np.ndarray:
    """各ピークについて、frag_masses（昇順）+ shifts のうち最も近い理論値との符号つき誤差（観測 − 理論、Da）。
    断片が無ければ nan。"""
    pk = np.asarray(peaks_mz, np.float64); fm = np.asarray(frag_masses, np.float64); sv = np.asarray(shifts, np.float64)
    out = np.full(len(pk), np.nan)
    if not len(fm) or not len(sv):
        return out
    for p, mz in enumerate(pk):
        q = mz - sv
        j = np.searchsorted(fm, q)
        best = np.inf; bestv = np.nan
        for jj in (j - 1, j):
            ok = (jj >= 0) & (jj < len(fm))
            if not ok.any():
                continue
            e = q[ok] - fm[jj[ok]]
            k = int(np.argmin(np.abs(e)))
            if abs(e[k]) < best:
                best, bestv = abs(e[k]), float(e[k])
        out[p] = bestv
    return out


def background_corrected(abs_err: np.ndarray, tol: float = TOL, bin_w: float = FRAG_BIN, lo: float = BG_LO,
                         hi: float = BG_HI) -> dict:
    """|誤差| の分布から、[lo, hi) の平均の密度（件/Da）を偶然の一致の背景として引いた一致の数。
    in_tol = |e| ≤ tol、mid = tol < |e| ≤ lo。true_frac_outside = mid_corr / (in_tol_corr + mid_corr)。"""
    a = np.asarray(abs_err, np.float64); a = a[np.isfinite(a)]
    dens = float(((a >= lo) & (a < hi)).sum()) / (hi - lo)
    n_in = int((a <= tol).sum()); n_mid = int(((a > tol) & (a <= lo)).sum())
    c_in = n_in - dens * tol; c_mid = n_mid - dens * (lo - tol)
    tot = c_in + c_mid
    nb = int(round(hi / bin_w))
    hist = np.histogram(a, bins=np.linspace(0, hi, nb + 1))[0].tolist()
    return dict(n_peaks_within_hi=int((a < hi).sum()), bg_density_per_da=round(dens, 2), n_in_tol=n_in, n_mid=n_mid,
                in_tol_corr=round(c_in, 1), mid_corr=round(c_mid, 1),
                true_frac_outside_tol=round(c_mid / tot, 4) if tot > 0 else None,
                abs_err_median_within_tol=round(float(np.median(a[a <= tol])), 5) if n_in else None,
                hist_bin_da=bin_w, hist=hist)


def multi_adduct_consistency(mol, adduct, nm, ppm_lim: float = PPM) -> dict:
    """2 種以上の付加イオンを持つ分子で、付加イオンごとの中性質量の中央値が分子の中央値（全スペクトル）から
    全部 ±ppm_lim 以内の割合。"""
    import pandas as pd
    d = pd.DataFrame(dict(m=np.asarray(mol, dtype=object), a=np.asarray(adduct, dtype=object), nm=np.asarray(nm, float)))
    d = d[np.isfinite(d.nm)]
    tgt = d.groupby("m").nm.median()
    per = d.groupby(["m", "a"]).nm.median().reset_index()
    per["t"] = per.m.map(tgt)
    per["dev"] = (per.nm - per.t) / per.t * 1e6
    na = per.groupby("m").a.nunique()
    multi = na[na >= 2].index
    sub = per[per.m.isin(multi)]
    ok = sub.groupby("m").dev.apply(lambda v: bool((v.abs() <= ppm_lim).all()))
    bad = sub[sub.dev.abs() > ppm_lim]
    return dict(n_molecules=int(len(tgt)), n_multi_adduct=int(len(multi)), n_consistent=int(ok.sum()),
                frac_consistent=round(float(ok.mean()), 4) if len(ok) else None,
                bad_adducts=bad.a.value_counts().to_dict(),
                bad_examples=[dict(molecule_id=str(r.m), adduct=str(r.a), dev_ppm=round(float(r.dev), 2))
                              for r in bad.head(10).itertuples()])


# ---------------------------------------------------------------- 重いところ（ローカルだけ）
def _elem_masses() -> dict:
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    out = {}
    for z in range(1, 119):
        s = pt.GetElementSymbol(z)
        m = pt.GetMostCommonIsotopeMass(z)
        if m > 0:
            out[s] = float(m)
    return out


def _chem():
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    return chem


def phase_train() -> dict:
    import pyarrow.parquet as pq
    chem = _chem(); masses = _elem_masses()
    pf = pq.ParquetFile(ROOT / "data/train.parquet", buffer_size=1 << 20, pre_buffer=False)
    light = ["inchikey14", "molecular_formula", "instrument_type", "adduct", "precursor_mz", "precursor_error_ppm",
             "ionization_mode", "ingest_lib", "normalized_smiles"]
    import pandas as pd
    vis = pd.read_parquet(ROOT / "outputs/test_library_match.parquet", columns=["molecule_id", "best_key"])
    want_vis = set(vis.best_key.astype(str))
    vis_formula, vis_smiles = {}, {}
    fcache = {}
    all_instr, all_add, all_ppm, all_org = [], [], [], []
    n_unk_adduct, n_bad_formula, unk_adducts = 0, 0, {}
    sample = []
    a12_train = {}
    t0 = time.time()
    for rg in range(pf.num_row_groups if not MAX_RG else MAX_RG):
        L = {c: [] for c in light}; npos = []
        for b in pf.iter_batches(batch_size=8192, row_groups=[rg], columns=light + ["ms2_normalized_intensities"]):
            d = b.to_pydict()
            for c in light:
                L[c].extend(d[c])
            npos.extend(sum(1 for x in (it or []) if x is not None and x > 0) for it in d["ms2_normalized_intensities"])
        n = len(L["precursor_mz"])
        add = np.asarray(L["adduct"], dtype=object)
        nm = chem.neutral_mass(np.asarray(L["precursor_mz"], np.float64), add)
        M = np.empty(n)
        for i, f in enumerate(L["molecular_formula"]):
            if f not in fcache:
                fcache[f] = parse_formula(f, masses)
            M[i] = fcache[f]
        ppm = ppm_error(nm, M)
        unk = ~np.isfinite(nm)
        n_unk_adduct += int(unk.sum())
        for a in add[unk]:
            unk_adducts[str(a)] = unk_adducts.get(str(a), 0) + 1
        n_bad_formula += int((~np.isfinite(M)).sum())
        all_instr.extend(L["instrument_type"]); all_add.extend(L["adduct"]); all_ppm.append(ppm.astype(np.float32))
        for i in np.flatnonzero(np.isfinite(ppm) & (np.abs(ppm) > PPM)):   # 参考（事前登録外）: 窓の外の既知の差
            hits = match_known_delta(float(nm[i] - M[i]), float(M[i])) + match_multimer(float(nm[i]), float(M[i]))
            key = (str(L["instrument_type"][i]), str(L["adduct"][i]))
            t = a12_train.setdefault(key, {})
            for h in (hits or ["(none)"]):
                t[h] = t.get(h, 0) + 1
        all_org.append(np.asarray([np.nan if x is None else x for x in L["precursor_error_ppm"]], np.float32))
        for i, ik in enumerate(L["inchikey14"]):
            if ik in want_vis and ik not in vis_formula:
                vis_formula[ik] = L["molecular_formula"][i]; vis_smiles[ik] = L["normalized_smiles"][i]
        elig = np.flatnonzero(np.isfinite(nm) & (np.asarray(npos) >= MIN_PEAKS))
        rng = np.random.default_rng(SEED + rg)
        pick = np.sort(rng.choice(elig, size=min(N_PER_RG, len(elig)), replace=False))
        pk = set(int(x) for x in pick)
        got, base = {}, 0
        for b in pf.iter_batches(batch_size=8192, row_groups=[rg], columns=["ms2_mzs", "ms2_normalized_intensities"]):
            for j in [x - base for x in pk if base <= x < base + b.num_rows]:
                got[base + j] = (b.column(0)[j].as_py(), b.column(1)[j].as_py())
            base += b.num_rows
        for i in pick:
            sample.append(dict(rg=rg, row=int(i), ik=L["inchikey14"][i], smiles=L["normalized_smiles"][i],
                               adduct=L["adduct"][i], mode=1 if L["ionization_mode"][i] == "positive" else -1,
                               instrument=L["instrument_type"][i], prec=float(L["precursor_mz"][i]),
                               mz=got[int(i)][0], it=got[int(i)][1]))
        print(f"rg {rg} rows {n} t={time.time() - t0:.0f}s", flush=True)
    ppm = np.concatenate(all_ppm).astype(np.float64); org = np.concatenate(all_org).astype(np.float64)
    both = np.isfinite(ppm) & np.isfinite(org)
    diff = np.abs(ppm[both] - org[both])
    res = dict(issue=ISSUE, n_rows=int(len(ppm)), n_unknown_adduct=n_unk_adduct, unknown_adducts_top=dict(
        sorted(unk_adducts.items(), key=lambda kv: -kv[1])[:15]), n_bad_formula=n_bad_formula,
        overall=group_stats(ppm), by_instrument_adduct=grouped_table(all_instr, all_add, ppm),
        organiser_column=dict(n_both=int(both.sum()), abs_diff_median=round(float(np.median(diff)), 4),
                              abs_diff_p99=round(float(np.quantile(diff, 0.99)), 4),
                              frac_abs_diff_gt_1ppm=round(float((diff > 1).mean()), 6),
                              organiser_stats=group_stats(org)),
        n_sample=len(sample), seconds=round(time.time() - t0, 1))
    top = sorted(a12_train.items(), key=lambda kv: -sum(kv[1].values()))[:25]
    res["a12_train_reference"] = dict(
        note="事前登録外の参考: train の |ppm| > 10 のスペクトルで既知の差に当たる数（装置 × 付加イオンの上位 25 組）",
        total_out=int(sum(sum(v.values()) for v in a12_train.values())),
        total_any_hit=int(sum(sum(c for h, c in v.items() if h != "(none)") for v in a12_train.values())),
        groups=[dict(instrument=k[0], adduct=k[1], n_out=int(sum(v.values())),
                     hits=dict(sorted(v.items(), key=lambda kv: -kv[1])[:6])) for k, v in top])
    by_instr = {}
    for i in sorted(set(map(str, all_instr))):
        m = np.asarray([str(x) for x in all_instr], dtype=object) == i
        by_instr[i] = group_stats(ppm[m])
    res["by_instrument"] = by_instr
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frag_sample.json").write_text(json.dumps(sample))
    (OUT / "visible_truth.json").write_text(json.dumps(dict(formula=vis_formula, smiles=vis_smiles)))
    (OUT / "train.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def phase_frag() -> dict:
    sys.path.insert(0, str(ROOT / "data/ext/frag")); sys.path.insert(0, str(ROOT / "scripts/train"))
    import frag_rescore as fr
    import fragnn_bench as fb
    fr.warmup(); c = fr.resolve_cfg(None)
    sample = json.loads((OUT / "frag_sample.json").read_text())
    errs, instr_of, n_none, nfrag, mzs = [], [], 0, [], []
    t0 = time.time()
    for s in sample:
        lv = fr.fragment_levels(s["smiles"], c["max_breaks"], c["max_bonds_2"], c["max_bonds_3"])
        if lv is None:
            n_none += 1; continue
        fm = np.sort(np.concatenate([np.asarray(x, np.float64) for x in lv]))
        nfrag.append(len(fm))
        sv, _, _ = fb.shift_table(s["adduct"], s["mode"])
        mz, _ = fb.top_peaks(s["mz"], s["it"])
        e = nearest_errors(mz, fm, sv)
        mzs.append(np.asarray(mz, np.float64))
        errs.append(e); instr_of.extend([str(s["instrument"])] * len(e))
    e = np.concatenate(errs); a = np.abs(e); pm = a / np.concatenate(mzs) * 1e6
    w = a <= TOL
    instr_of = np.asarray(instr_of, dtype=object)
    res = dict(issue=ISSUE, n_spectra=len(sample), n_unparsed=n_none, n_peaks=int(len(e)), tol=TOL,
               all=background_corrected(a), seconds=round(time.time() - t0, 1),
               signed_err_median_within_tol=round(float(np.median(e[a <= TOL])), 5),
               frag_masses_per_struct_median=float(np.median(nfrag)), frag_masses_per_struct_min=int(np.min(nfrag)),
               abs_ppm_within_tol=dict(median=round(float(np.median(pm[w])), 2), p90=round(float(np.quantile(pm[w], 0.9)), 2),
                                       p99=round(float(np.quantile(pm[w], 0.99)), 2)))
    res["by_instrument"] = {str(i): dict(n_spectra=sum(1 for s in sample if str(s["instrument"]) == i),
                                         **background_corrected(a[instr_of == i]))
                            for i in sorted(set(map(str, instr_of)))}
    for k in list(res["by_instrument"]):
        if k != "timsTOF":          # 隠れテストの装置だけ刻みの分布を残す
            res["by_instrument"][k].pop("hist", None)
    (OUT / "frag.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def _truth_mass(smiles: str, masses: dict) -> float:
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    m = Chem.MolFromSmiles(smiles)
    return parse_formula(CalcMolFormula(m), masses) if m is not None else float("nan")


def _analyse_set(name, te, truth_M, pool=None, truth_key=None, hidden=None) -> dict:
    """te: test.parquet、truth_M: molecule_id → M。"""
    chem = _chem()
    te = te.copy()
    te["nm"] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    te["M"] = te.molecule_id.map(truth_M).astype(float)
    te["ppm"] = ppm_error(te.nm.values, te.M.values)
    spec = dict(n_spectra=int(len(te)), n_unknown_adduct=int((~np.isfinite(te.nm)).sum()),
                n_no_truth=int((~np.isfinite(te.M)).sum()), overall=group_stats(te.ppm.values),
                by_instrument_adduct=grouped_table(te.instrument_type.values, te.adduct.values, te.ppm.values, 1))
    mols, a12_mol, a12_spec = [], [], []
    for mid, sub in te.groupby("molecule_id", sort=False):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        M = float(sub.M.iloc[0])
        if not len(nms) or not np.isfinite(M):
            continue
        tgt = float(np.median(nms)); p = (tgt - M) / M * 1e6
        row = dict(molecule_id=str(mid), target=tgt, M=M, ppm=p, out=bool(abs(p) > PPM))
        if pool is not None and truth_key is not None:
            k = truth_key.get(mid)
            idx = pool.window(tgt, PPM)
            row["truth_in_pool"] = bool(k in pool.keyset)
            row["truth_in_window"] = bool(k in set(pool.key[idx]))
            row["truth_hidden"] = bool(hidden is not None and k in hidden)   # hidden は pool: remove の集合だけ渡す
        mols.append(row)
        if row["out"]:
            a12_mol.append(dict(molecule_id=str(mid), ppm=round(p, 2), delta_da=round(tgt - M, 5),
                                adducts=sorted({str(x) for x in sub.adduct}),
                                instruments=sorted({str(x) for x in sub.instrument_type}),
                                hits=match_known_delta(tgt - M, M) + match_multimer(tgt, M)))
        for r in sub.itertuples():
            if np.isfinite(r.ppm) and abs(r.ppm) > PPM:
                a12_spec.append(dict(molecule_id=str(mid), adduct=str(r.adduct), instrument=str(r.instrument_type),
                                     ppm=round(float(r.ppm), 2),
                                     delta_da=round(float(r.nm - M), 5),
                                     hits=match_known_delta(float(r.nm - M), M) + match_multimer(float(r.nm), M)))
    mp = np.array([m["ppm"] for m in mols])
    res = dict(name=name, spectra=spec, molecules=dict(n=len(mols), stats=group_stats(mp),
                                                        n_out_window=int(sum(m["out"] for m in mols))))
    if mols and "truth_in_pool" in mols[0]:
        res["molecules"].update(
            n_truth_in_pool=int(sum(m["truth_in_pool"] for m in mols)),
            n_truth_in_window=int(sum(m["truth_in_window"] for m in mols)),
            n_in_pool_not_window=int(sum(m["truth_in_pool"] and not m["truth_in_window"] for m in mols)),
            n_hidden=int(sum(m["truth_hidden"] for m in mols)))
    def tally(lst):
        t = {}
        for x in lst:
            for h in (x["hits"] or ["(none)"]):
                t[h] = t.get(h, 0) + 1
        return t
    res["a12_molecules"] = dict(n=len(a12_mol), hit_counts=tally(a12_mol), n_any_hit=sum(bool(x["hits"]) for x in a12_mol),
                                rows=a12_mol[:50])
    res["a12_spectra"] = dict(n=len(a12_spec), hit_counts=tally(a12_spec), n_any_hit=sum(bool(x["hits"]) for x in a12_spec),
                              by_adduct={a: sum(1 for x in a12_spec if x["adduct"] == a) for a in sorted({x["adduct"] for x in a12_spec})},
                              by_instrument={i: sum(1 for x in a12_spec if x["instrument"] == i)
                                             for i in sorted({x["instrument"] for x in a12_spec})},
                              rows=a12_spec[:50])
    res["multi_adduct"] = multi_adduct_consistency(te.molecule_id.values, te.adduct.values, te.nm.values)
    return res


def phase_holdouts() -> dict:
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import run as R
    import library as L
    chem = _chem()
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem as cchem
    masses = _elem_masses()
    pool = R.load_pool(ROOT); pool.keyset = set(pool.key)
    out = dict(issue=ISSUE, ppm=PPM, sets={})
    for h in HOLDOUTS:
        te = pd.read_parquet(ROOT / f"data/holdouts/{h}/test.parquet")
        tr = pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv")
        tM = {r.molecule_id: _truth_mass(r.smiles, masses) for r in tr.itertuples()}
        tkey = {r.molecule_id: cchem.score_key(r.smiles) for r in tr.itertuples()}
        hid = L.drop_keys(h) if L.hide_rule(h).get("pool") == "remove" else set()   # run.py と同じ規則
        out["sets"][h] = _analyse_set(h, te, tM, pool, tkey, hid)
        print(h, out["sets"][h]["molecules"], flush=True)
    vt = json.loads((OUT / "visible_truth.json").read_text())
    vis = pd.read_parquet(ROOT / "outputs/test_library_match.parquet", columns=["molecule_id", "best_key", "best_cos"])
    te = pd.read_parquet(ROOT / "data/test.parquet")
    vM = {r.molecule_id: parse_formula(vt["formula"].get(str(r.best_key)), masses) for r in vis.itertuples()}
    out["sets"]["visible_test"] = _analyse_set("見える test（cos 1.0 の train 一致で推定した正解）", te, vM)
    out["sets"]["visible_test"]["best_cos_min"] = float(vis.best_cos.min())
    (OUT / "holdouts.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def molecule_half_width(instr, adduct, table: dict) -> float:
    """§5-3 の案（#195 A11 の 99% 点の幅）: 分子のスペクトルの（装置, 付加イオン）の組ごとの |ppm| 99% 点の最大値。
    組が表に無ければその装置の「その他」、それも無ければ nan。"""
    vals = []
    for i, a in zip(instr, adduct):
        v = table.get((str(i), str(a)), table.get((str(i), "その他")))
        if v is not None:
            vals.append(float(v))
    return max(vals) if vals else float("nan")


def _molecule_ppm(name: str) -> dict:
    import pandas as pd
    chem = _chem(); masses = _elem_masses()
    te = pd.read_parquet(ROOT / f"data/holdouts/{name}/test.parquet", columns=["molecule_id", "precursor_mz", "adduct"])
    tr = pd.read_csv(ROOT / f"outputs/holdouts/{name}/truth.csv")
    tM = {r.molecule_id: _truth_mass(r.smiles, masses) for r in tr.itertuples()}
    te["nm"] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    out = {}
    for m, sub in te.groupby("molecule_id"):
        v = sub.nm.values[np.isfinite(sub.nm.values)]
        if len(v) and np.isfinite(tM.get(m, np.nan)):
            out[m] = (float(np.median(v)) - tM[m]) / tM[m] * 1e6
    return out


def _control_rr(name: str) -> dict:
    """対照（score+prior+frag、#178 と同じファイル）の分子ごとの rr（mrr25）。"""
    import pandas as pd
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import mrr25
    sub = mrr25.read_submission(ROOT / f"outputs/local_eval/{name}/submission_score+prior+frag.csv")
    tr = pd.read_csv(ROOT / f"outputs/holdouts/{name}/truth.csv")
    return {r.molecule_id: mrr25.reciprocal_rank(mrr25.truth_keys(r.smiles), sub.get(r.molecule_id, [])) for r in tr.itertuples()}


def phase_width() -> dict:
    """§5-3 の案の窓の半幅を、v3 の 4 集合の分子ごとに出す（採点はしない）。"""
    import pandas as pd
    t = json.loads((OUT / "train.json").read_text())
    table = {(r["instrument"], r["adduct"]): r["abs_p99"] for r in t["by_instrument_adduct"]}
    h = json.loads((OUT / "holdouts.json").read_text())
    res = dict(issue=ISSUE, rule="分子の（装置, 付加イオン）の組の train |ppm| 99% 点の最大値", sets={})
    for name in HOLDOUTS:
        te = pd.read_parquet(ROOT / f"data/holdouts/{name}/test.parquet", columns=["molecule_id", "instrument_type", "adduct"])
        hw = {m: molecule_half_width(sub.instrument_type, sub.adduct, table) for m, sub in te.groupby("molecule_id")}
        v = np.array(list(hw.values()), np.float64)
        outm = {r["molecule_id"]: r["ppm"] for r in h["sets"][name]["a12_molecules"]["rows"]}
        # 分子の窓の中心と正解の M の ppm（holdouts 段と同じ定義）を test.parquet と truth.csv から作り直す
        mol_ppm = _molecule_ppm(name)
        rr = _control_rr(name)
        narrow = [m for m in hw if hw[m] < PPM]
        widen = [m for m in hw if hw[m] > PPM]
        enter = [m for m, p in outm.items() if abs(p) <= hw.get(m, 0)]
        drop = [m for m in hw if m in mol_ppm and abs(mol_ppm[m]) > hw[m] and abs(mol_ppm[m]) <= PPM]
        res["sets"][name] = dict(n=int(len(v)), n_nan=int(np.isnan(v).sum()), median=round(float(np.nanmedian(v)), 3),
                                 min=round(float(np.nanmin(v)), 3), max=round(float(np.nanmax(v)), 3),
                                 n_lt10=int((v < PPM).sum()), n_gt10=int((v > PPM).sum()), n_gt1000=int((v > 1000).sum()),
                                 n_out_window_now=len(outm),
                                 n_out_window_enter=len(enter), n_narrow=len(narrow), n_widen=len(widen),
                                 n_truth_drops_out=len(drop),
                                 bound_enter=round(len(enter) / len(hw), 4),
                                 bound_narrow_all_to_rank1=round(sum(1.0 - rr.get(m, 0.0) for m in narrow) / len(hw), 4),
                                 note="広がる分子は候補が増えるだけで上がらない（点は窓に依らない）。bound_* は全部 1 位に来る粗い上限")
    (OUT / "width.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main() -> None:
    ph = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = dict(train=phase_train, frag=phase_frag, holdouts=phase_holdouts, width=phase_width).get(ph)
    if fn is None:
        raise SystemExit("usage: mass_error_adduct.py train|frag|holdouts|width")
    r = fn()
    print(json.dumps({k: v for k, v in r.items() if k not in ("sets", "by_instrument_adduct")}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
