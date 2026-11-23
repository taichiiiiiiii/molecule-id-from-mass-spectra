#!/usr/bin/env python3
"""#196: fragnn v2 の入力の作り方（候補）の費用を、v1 と同じ切片・同じプロセスの形で測る。学習はしない。v3 は読まない。

v1（#120 の `fragnn_bench.py`、PR #121）の 1 ステップ（batch 128 × 候補 128、CPU の特徴作り＋MPS の順・逆伝播、
組の数は 2,048 の倍数に詰める、空回し 5 回のあと 50 回、順伝播だけ 20 回）を、次の 2 つの構成で測る:

  v1 : 上位 32 本、スペクトルの特徴 3（mode・log1p(本数)/log 33・CE/100）、全付加イオン、merge なし（v1 と同じ）
  v2 : P1 = 学習の行を隠れテストの 10 種の付加イオンに限る（データのページの一覧。[2M…] などを外す）
       P5' = m/z > precursor + 2 Da のピークを除く（データのページ: 隠れテストは除去済み）
       P2 = 上位 64 本（#193 の値 1 つ）、スペクトルの特徴の本数の項は log1p(本数)/log 65
       P8(a) = スペクトルの特徴に「instrument_type == 'timsTOF'」の 1 ビット（F_SPEC 4）
       P4 = 確率 0.3 で、同じ (構造, 付加イオン) の 2〜4 本（FPNet の peers と同じ引き方、kmax 4）を merge
            （FPNet の merge_spectra と同じ: 各本を基準ピーク 1 に正規化 → 0.005 Da 以内は強い方 → 上位 64 本）

切片: v1 の prep.npz（outputs/fragnn_train/prep.npz、#120 の本番の学習が書いた前計算）の学習の正解構造から
seed 0 で 1,000 構造（置き換えなし）、そのスペクトル全部。負例は prep.npz の design の CSR（v1 と同じ）。
断片・特徴の関数は fragnn_bench のものをそのまま使う（numba、tol 0.005、h_shifts −2..+3）。

段（別のプロセスで動かし、`/usr/bin/time -l` で peak RSS を測る）:
  spectra : train.parquet を row group ごとに 1 回読み、切片のスペクトルを v1・v2 の 2 通りに詰める（秒/本も測る）
  step_v1 / step_v2 : 1 ステップの秒・順伝播の秒、終わりに `footprint -f bytes -p <自分の pid>` と
            torch.mps.driver_allocated_memory を記録
  report  : 外挿（v1 の本番の実績 39,408 s / 106,000 ステップに v2/v1 の秒の比を掛ける、および #120 の式で 28,000 ステップ）

純粋な関数（CI でテストする）は torch・rdkit を import しない。
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import fragnn_bench as B  # noqa: E402

ISSUE = 196
# データのページ（2026-11-23 に `KaggleApi.competition_list_pages` で読んだ data-description）:
# 「Test spectra use one of ten adducts」
TEST_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
                "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
ABOVE_PREC_DA = 2.0            # 「Peaks above precursor + 2 Da were removed」
TOP_PEAKS_V2 = 64              # #193 P2 の値（1 つだけ）
F_SPEC_V2 = 4
MERGE_P, MERGE_KMAX, MERGE_TOL = 0.3, 4, 0.005   # FPNet traindata.item / peers / merge_spectra の値を借りる
TIMSTOF = "timsTOF"
N_SLICE, SEED_SLICE = 1_000, 0
V1_TRAIN_S, V1_TRAIN_STEPS = 39_408.4, 106_000    # outputs/fragnn_train/train_report.json（#120 の本番の実績）


# ---------------------------------------------------------------- 純粋な関数（CI）

def keep_adduct(adduct) -> bool:
    """P1: 隠れテストの 10 種の付加イオンなら True。"""
    return adduct in TEST_ADDUCTS


def drop_above_precursor(mz, it, prec, da: float = ABOVE_PREC_DA):
    """P5': m/z > prec + da のピークを除く（prec が有限でなければそのまま）。"""
    mz = np.asarray(mz, np.float64).ravel(); it = np.asarray(it, np.float64).ravel()
    n = min(len(mz), len(it)); mz, it = mz[:n], it[:n]
    if prec is None or not np.isfinite(prec):
        return mz, it
    k = mz <= float(prec) + da
    return mz[k], it[k]


def merge_peak_lists(pairs, tol: float = MERGE_TOL, k: int = TOP_PEAKS_V2):
    """FPNet の merge_spectra と同じ規則（各本を最大 1 に正規化、m/z の昇順で 0.005 Da 未満の隣は強い方を残す）の後に
    fragnn_bench.top_peaks で上位 k 本。返り値は m/z の昇順。"""
    mzs, its = [], []
    for mz, it in pairs:
        mz = np.asarray(mz, np.float64).ravel(); it = np.asarray(it, np.float64).ravel()
        ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz, it = mz[ok], it[ok]
        if len(mz) == 0:
            continue
        mzs.append(mz); its.append(it / it.max())
    if not mzs:
        return np.zeros(0), np.zeros(0)
    mz = np.concatenate(mzs); it = np.concatenate(its)
    o = np.argsort(mz, kind="stable"); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    last = 0
    for j in range(1, len(mz)):
        if mz[j] - mz[last] < tol:
            if it[j] >= it[last]:
                keep[last] = False; last = j
            else:
                keep[j] = False
        else:
            last = j
    return B.top_peaks(mz[keep], it[keep], k=k)


def peers(rng, members: np.ndarray, i: int, kmax: int = MERGE_KMAX) -> list:
    """FPNet traindata.peers と同じ引き方: 群の大きさ 1 なら [i]、そうでなければ n ~ U{2..min(kmax, |群|)} 本を
    置き換えなしで引き、i が入っていなければ先頭を i にする。"""
    members = np.asarray(members, np.int64)
    if len(members) <= 1:
        return [int(i)]
    n = int(rng.integers(2, min(kmax, len(members)) + 1))
    pick = rng.choice(members, size=min(n, len(members)), replace=False)
    if i not in pick:
        pick[0] = i
    return [int(x) for x in pick]


def spec_feat_v2(mode, n_peaks: int, ce, instrument) -> tuple:
    """(mode ±1, log1p(本数)/log(65), CE/100（無ければ 0、v1 と同じ）, timsTOF なら 1)。"""
    return (1.0 if (mode or 1) > 0 else -1.0, math.log1p(n_peaks) / math.log(TOP_PEAKS_V2 + 1),
            float(ce or 0.0) / 100.0, 1.0 if instrument == TIMSTOF else 0.0)


def pack_v2(rows: list, k: int = TOP_PEAKS_V2) -> dict:
    """rows: dict(mz, it, prec, adduct, mode, ce, instrument)。mz・it は詰める前に P5' を当てたもの（merge 済みでもよい）。
    fragnn_bench.pack_spectra と同じ形（上位 k 本、shift 最大 12）で、spec_feat だけ F_SPEC_V2 列。"""
    n = len(rows)
    out = dict(mz=np.zeros((n, k)), it=np.zeros((n, k)), n=np.zeros(n, np.int64), prec=np.zeros(n),
               sh_v=np.zeros((n, 12)), sh_c=np.zeros((n, 12), np.int8), sh_d=np.zeros((n, 12), np.int8),
               sh_n=np.zeros(n, np.int64), spec_feat=np.zeros((n, F_SPEC_V2), np.float32))
    for i, r in enumerate(rows):
        mz, it = B.top_peaks(r["mz"], r["it"], k=k)
        m = len(mz)
        out["mz"][i, :m], out["it"][i, :m], out["n"][i], out["prec"][i] = mz, it, m, float(r["prec"])
        v, c, d = B.shift_table(r.get("adduct"), r.get("mode"))
        out["sh_v"][i, :len(v)], out["sh_c"][i, :len(v)], out["sh_d"][i, :len(v)], out["sh_n"][i] = v, c, d, len(v)
        out["spec_feat"][i] = spec_feat_v2(r.get("mode"), m, r.get("ce"), r.get("instrument"))
    return out


def group_index(struct: np.ndarray, adduct_code: np.ndarray) -> dict:
    """(構造, 付加イオン) → その行の番号の配列。"""
    key = np.asarray(struct, np.int64) * 64 + np.asarray(adduct_code, np.int64)
    o = np.argsort(key, kind="stable"); ks = key[o]
    b = np.flatnonzero(np.diff(ks)) + 1
    st = np.concatenate([[0], b]); en = np.concatenate([b, [len(ks)]])
    return {int(ks[s]): o[s:e] for s, e in zip(st, en)}


def extrapolate_ratio(v1_bench_s: float, v2_bench_s: float, v1_real_s: float = V1_TRAIN_S,
                      v1_steps: int = V1_TRAIN_STEPS) -> dict:
    """v1 の本番の学習ループの秒（評価を含む）に、同じ切片の 1 ステップの秒の比 v2/v1 を掛ける（同じステップ数と仮定）。"""
    r = v2_bench_s / v1_bench_s
    return dict(ratio=r, v1_real_h=v1_real_s / 3600, v2_expected_h=v1_real_s * r / 3600, steps=v1_steps)


# ---------------------------------------------------------------- ローカルの段（CI では動かさない）

def _slice(prep: Path) -> dict:
    z = np.load(prep)
    is_val = z["is_val"]
    tr_i = np.flatnonzero(~is_val)
    up = np.unique(z["pos_struct"][tr_i])
    sl = np.sort(np.random.default_rng(SEED_SLICE).choice(up, size=N_SLICE, replace=False))
    sel = tr_i[np.isin(z["pos_struct"][tr_i], sl)]
    return dict(sel=sel, spec_row=z["spec_row"][sel], spec_ik=z["spec_ik"][sel], pos_struct=z["pos_struct"][sel],
                cid=z["cid"][sel], z=z)


def phase_spectra(root: Path, prep: Path, out: Path) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    S = _slice(prep)
    want = S["spec_row"]; order = np.argsort(want, kind="stable"); srow = want[order]
    cols = ["inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
            "collision_energy_ev", "instrument_type"]
    t0 = time.time()
    pf = pq.ParquetFile(root / "data/train.parquet", buffer_size=1 << 20, pre_buffer=False)
    rows = [None] * len(want); base = 0
    for b in pf.iter_batches(batch_size=8192, columns=cols):
        lo, hi = np.searchsorted(srow, base), np.searchsorted(srow, base + b.num_rows)
        if hi > lo:
            sub = b.take(pa.array(srow[lo:hi] - base)).to_pydict()
            for j in range(hi - lo):
                ce = sub["collision_energy_ev"][j]
                assert sub["inchikey14"][j].encode() == S["spec_ik"][order[lo + j]]
                rows[order[lo + j]] = dict(adduct=sub["adduct"][j], prec=sub["precursor_mz"][j],
                                           mode=1 if sub["ionization_mode"][j] == "positive" else -1,
                                           mz=sub["ms2_mzs"][j], it=sub["ms2_normalized_intensities"][j],
                                           ce=float(np.mean(ce)) if ce else 0.0, instrument=sub["instrument_type"][j])
        base += b.num_rows
    scan = time.time() - t0
    assert all(r is not None for r in rows)
    t1 = time.time(); p1 = B.pack_spectra(rows); s1 = time.time() - t1
    keep = np.array([keep_adduct(r["adduct"]) for r in rows], bool)
    t2 = time.time()
    rows2 = []
    for r, kp in zip(rows, keep):
        if kp:
            mz, it = drop_above_precursor(r["mz"], r["it"], r["prec"])
            mz, it = B.top_peaks(mz, it, k=TOP_PEAKS_V2)     # 学習で持つのは上位 64 本（merge もこれを使う）
            rows2.append(dict(r, mz=mz, it=it))
    p2 = pack_v2(rows2); s2 = time.time() - t2
    np.savez(out / "spectra_v1.npz", sel=S["sel"], **p1)
    adc = np.array([TEST_ADDUCTS.index(r["adduct"]) for r in rows2], np.int64)
    np.savez(out / "spectra_v2.npz", sel=S["sel"][keep], adduct_code=adc,
             raw_mz=np.stack([np.pad(r["mz"], (0, TOP_PEAKS_V2 - len(r["mz"]))) for r in rows2]),
             raw_it=np.stack([np.pad(r["it"], (0, TOP_PEAKS_V2 - len(r["it"]))) for r in rows2]),
             raw_meta=np.array(json.dumps([dict(prec=r["prec"], adduct=r["adduct"], mode=r["mode"], ce=r["ce"],
                                                instrument=r["instrument"]) for r in rows2])), **p2)
    return dict(n_spectra_v1=len(rows), n_spectra_v2=len(rows2), n_dropped_p1=int((~keep).sum()), scan_s=round(scan, 1),
                pack_v1_s=round(s1, 3), pack_v2_s=round(s2, 3), sec_pack_per_spectrum_v1=s1 / len(rows),
                sec_pack_per_spectrum_v2=s2 / max(1, len(rows2)), peaks_median_v1=float(np.median(p1["n"])),
                peaks_median_v2=float(np.median(p2["n"])),
                n_rows_scanned=base)


def build_model(f_spec: int):
    import torch
    from torch import nn

    class FragNN(nn.Module):   # fragnn_bench.build_model と同じ形（rho の入力だけ F_SPEC に合わせる）
        def __init__(self):
            super().__init__()
            self.phi = nn.Sequential(nn.Linear(B.F_PEAK, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.rho = nn.Sequential(nn.Linear(256 + B.F_PAIR + f_spec, 256), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

        def forward(self, feats, valid, pair, spec):
            h = self.phi(feats)
            w = feats[..., 0:1] * valid[..., None]
            s = (h * w).sum(1)
            m = torch.where(valid[..., None], h, torch.zeros_like(h)).amax(1)
            return self.rho(torch.cat([s, m, pair, spec], -1)).squeeze(-1)

    return FragNN()


def footprint_bytes(pid: int) -> dict:
    try:
        txt = subprocess.run(["footprint", "-f", "bytes", "-p", str(pid)], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e))
    return dict(**parse_footprint(txt), head=txt[:1500])


def parse_footprint(txt: str) -> dict:
    """`footprint -f bytes` の出力 → 全体の Footprint と「IOAccelerator (graphics)」の行の Dirty のバイト数。"""
    m = re.search(r"Footprint:\s*([\d,]+)\s*B", txt)
    g = re.search(r"^\s*([\d,]+)\s*B\s+[\d,]+\s*B\s+[\d,]+\s*B\s+\d+\s+IOAccelerator \(graphics\)\s*$", txt, re.M)
    return dict(footprint_b=int(m.group(1).replace(",", "")) if m else None,
                graphics_b=int(g.group(1).replace(",", "")) if g else None)


def phase_step(root: Path, prep: Path, out: Path, which: str, steps: int = 50, warm: int = 5, fwd_batches: int = 20,
               device: str = "mps") -> dict:
    import torch
    torch.manual_seed(B.SEED)
    S = _slice(prep); z = S["z"]
    masses, off, off_tr, flat_tr = z["masses"], z["off"], z["off_tr"], z["flat_tr"]
    sp = dict(np.load(out / f"spectra_{which}.npz"))
    sel = sp["sel"]
    pos = dict(zip(S["sel"].tolist(), range(len(S["sel"]))))
    li = np.array([pos[int(s)] for s in sel], np.int64)
    pos_struct, cid = S["pos_struct"][li], S["cid"][li]
    P = TOP_PEAKS_V2 if which == "v2" else B.TOP_PEAKS
    f_spec = F_SPEC_V2 if which == "v2" else B.F_SPEC
    if which == "v2":
        groups = group_index(pos_struct, sp["adduct_code"])
        gkey = pos_struct * 64 + sp["adduct_code"]
        meta = json.loads(str(sp["raw_meta"]))
    dev = torch.device(device)
    model = build_model(f_spec).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=B.LR, weight_decay=B.WD)
    n_spec = len(sel)

    def batch_spec(step, bi):
        """v1: 詰めた配列をそのまま。v2: 確率 0.3 で peers を merge して詰め直す（学習の on-the-fly と同じ）。"""
        if which == "v1":
            return {k: sp[k][bi] for k in ("mz", "it", "n", "prec", "sh_v", "sh_c", "sh_d", "sh_n", "spec_feat")}, 0
        rng = np.random.default_rng([B.SEED, 2, step])
        rows, n_merged = [], 0
        for i in bi:
            idxs = [int(i)]
            if rng.random() < MERGE_P:
                idxs = peers(rng, groups[int(gkey[i])], int(i))
            m = meta[int(i)]
            if len(idxs) == 1:
                ok = sp["raw_it"][i] > 0
                mz, it = sp["raw_mz"][i][ok], sp["raw_it"][i][ok]
                ce = m["ce"]
            else:
                n_merged += 1
                mz, it = merge_peak_lists([(sp["raw_mz"][j][sp["raw_it"][j] > 0], sp["raw_it"][j][sp["raw_it"][j] > 0])
                                           for j in idxs])
                ces = [meta[j]["ce"] for j in idxs if meta[j]["ce"]]
                ce = float(np.mean(ces)) if ces else 0.0
            rows.append(dict(m, mz=mz, it=it, ce=ce))
        pk = pack_v2(rows)
        return pk, n_merged

    def run_step(step, train=True):
        t0 = time.time()
        rng = np.random.default_rng([B.SEED, 1, step])
        bi = np.random.default_rng([B.SEED, 0, step]).choice(n_spec, size=B.BATCH, replace=False)
        cand, mask = B.sample_candidates(rng, pos_struct[bi], cid[bi], off_tr, flat_tr)
        tm = time.time()
        bsp, n_merged = batch_spec(step, bi)
        tmerge = time.time() - tm
        vi, vk = np.nonzero(mask); n = len(vi)
        tf0 = time.time()
        f0, fs0 = B.batch_features(masses, off, cand[vi, vk], vi, bsp)
        tf = time.time() - tf0
        p = B.padded_pairs(n)
        feats = np.zeros((p, P, B.F_PEAK), np.float32); feats[:n] = f0
        fs = np.zeros(p, np.float32); fs[:n] = fs0
        vi_p = np.zeros(p, np.int64); vi_p[:n] = vi
        slot_np = np.full(p, B.BATCH * B.K_CAND, np.int64); slot_np[:n] = vi * B.K_CAND + vk
        vmask = np.arange(P)[None, :] < bsp["n"][vi_p][:, None]; vmask[n:] = False
        valid = torch.from_numpy(vmask).to(dev); x = torch.from_numpy(feats).to(dev)
        pair = torch.from_numpy(fs[:, None]).to(dev); sf = torch.from_numpy(bsp["spec_feat"][vi_p]).to(dev)
        slot = torch.from_numpy(slot_np).to(dev)
        t1 = time.time()
        if train:
            model.train()
            logit = model(x, valid, pair, sf)
            flat = torch.full((B.BATCH * B.K_CAND + 1,), float("-inf"), device=dev).index_put((slot,), logit)
            loss = torch.nn.functional.cross_entropy(flat[:B.BATCH * B.K_CAND].view(B.BATCH, B.K_CAND),
                                                     torch.zeros(B.BATCH, dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), B.CLIP); opt.step()
            lv = float(loss.detach().cpu())
        else:
            model.eval()
            with torch.no_grad():
                logit = model(x, valid, pair, sf)
            lv = float(logit.mean().cpu())
        if dev.type == "mps":
            torch.mps.synchronize()
        t2 = time.time()
        return dict(total=t2 - t0, merge=tmerge, feat=tf, gpu=t2 - t1, n_pairs=n, n_merged=n_merged, loss=lv)

    for i in range(warm):
        run_step(10_000 + i)
    rec = [run_step(i) for i in range(steps)]
    fwd = [run_step(20_000 + i, train=False) for i in range(fwd_batches)]
    fp = footprint_bytes(os.getpid())
    mean = lambda xs, k: float(np.mean([x[k] for x in xs]))  # noqa: E731
    return dict(which=which, device=str(dev), steps=steps, warm=warm, n_spectra=n_spec, top_peaks=P, f_spec=f_spec,
                sec_per_step=mean(rec, "total"), sec_per_step_sd=float(np.std([x["total"] for x in rec])),
                sec_merge=mean(rec, "merge"), sec_feat=mean(rec, "feat"), sec_gpu=mean(rec, "gpu"),
                pairs_per_step=mean(rec, "n_pairs"), merged_per_step=mean(rec, "n_merged"),
                sec_fwd_batch=mean(fwd, "total"), loss_first=rec[0]["loss"], loss_last=rec[-1]["loss"],
                mps_driver_allocated_gb=(round(torch.mps.driver_allocated_memory() / 1e9, 3) if dev.type == "mps" else None),
                footprint=fp)


def phase_report(out: Path) -> dict:
    sp = json.loads((out / "spectra.json").read_text())
    s1 = json.loads((out / "step_v1.json").read_text()); s2 = json.loads((out / "step_v2.json").read_text())
    ratio = extrapolate_ratio(s1["sec_per_step"], s2["sec_per_step"])
    # #120 の式（fragnn_bench.extrapolate）で 28,000 ステップ: v2 の上限の計算に使う形（前計算は v1 の実績 58.8 s を使う）
    n_spec_v2 = int(round((B.EXPECTED_TRAIN_SPECTRA + B.N_VAL_SPECTRA) * sp["n_spectra_v2"] / sp["n_spectra_v1"]))
    ex = {w: B.extrapolate(219_155, 0.00023370790481567382, sp["scan_s"], s["sec_per_step"], s["sec_fwd_batch"],
                           sec_pack_per_spectrum=sp[f"sec_pack_per_spectrum_{w}"],
                           n_spectra=(B.EXPECTED_TRAIN_SPECTRA + B.N_VAL_SPECTRA) if w == "v1" else n_spec_v2)
          for w, s in (("v1", s1), ("v2", s2))}
    cap = {w: min(1.5 * ex[w]["total_h"], 96.0) for w in ex}
    return dict(issue=ISSUE, ratio=ratio, extrapolate_28k=ex, wallclock_cap_h_by_120_rule=cap,
                n_spectra_v2_full_estimate=n_spec_v2,
                footprint_gb={w: (parse_footprint(s["footprint"]["head"])["footprint_b"] or 0) / 1e9
                              for w, s in (("v1", s1), ("v2", s2))},
                graphics_gb={w: (parse_footprint(s["footprint"]["head"])["graphics_b"] or 0) / 1e9
                             for w, s in (("v1", s1), ("v2", s2))},
                per_pair_ratio=(s2["sec_per_step"] / s2["pairs_per_step"]) / (s1["sec_per_step"] / s1["pairs_per_step"]),
                mps_driver_allocated_gb={w: s["mps_driver_allocated_gb"] for w, s in (("v1", s1), ("v2", s2))})


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="#196 fragnn v2 の入力の費用の測定（学習しない）")
    ap.add_argument("phase", choices=["spectra", "step_v1", "step_v2", "report"])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.phase.startswith("step"):
        for k, v in (("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")):
            if os.environ.get(k) != v:
                raise SystemExit(f"set {k}={v}")
    fn = {"spectra": lambda: phase_spectra(root, Path(a.prep), out),
          "step_v1": lambda: phase_step(root, Path(a.prep), out, "v1", device=a.device),
          "step_v2": lambda: phase_step(root, Path(a.prep), out, "v2", device=a.device),
          "report": lambda: phase_report(out)}[a.phase]
    r = fn()
    (out / f"{a.phase}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in r.items() if k != "footprint"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
