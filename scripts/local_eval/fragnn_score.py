#!/usr/bin/env python3
"""
Issue #120: fragnn の段。学習した重み（model_best.pt）で、縮小版の候補窓の各候補に点 n と blind の点を付けてキャッシュする。

    .venv/bin/python scripts/local_eval/fragnn_score.py --holdout v3_c1 [--weights <model_best.pt>] [--out-dir <dir>]

定義は #120 本文（25 行の blind・52 行の段）と、2026-11-17 の「fragnn 段の実装の細目の事前登録」のコメントのとおり:
- 集合の分子 = truth.csv の分子のうち pc_logits の __targets__ にあるもの（sanpier.py と同じ）
- 候補 = run.py の visible_window の全行（±10 ppm、pool: remove の集合は drop_keys を除く）。npz の idx は run.py の Pool の添字
- スペクトル = run.py の query_spectra と同じ行（分子ごとの先頭 16 本）。学習の load_spectra と同じ dict（ce は
  collision_energy_ev の平均、空なら 0）にして fragnn_bench.pack_spectra で詰める。使えるスペクトル = 有効なピークが 1 本以上
- 断片 = frag_rescore.fragment_levels（resolve_cfg(None) の値、学習の prepare と同じ呼び方）。None・例外の候補は NaN
- 点 n = 使えるスペクトルごとの logit の単純平均（model.eval()、inference_mode、fp32、CPU）。使えるスペクトルが無ければ NaN
- blind = molecule_id の文字列の昇順で次の、1 本目のスペクトルの正負が同じ分子のスペクトルに入れ替えた点（最後は先頭に戻る。
  同じ正負がほかに無ければ正負を問わず次、1 分子なら blind = full）
- 出力: <out-dir>/fragnn_scores.npz・fragnnblind_scores.npz（mids, offsets, idx, n。sanpier と同じ形）と fragnn_scores.json
  （既定の out-dir は outputs/local_eval/<holdout>。run.py の fragnn・fragnnblind 段が読む）。MRR・judge は計算しない

純粋な関数（CI でテストする）は torch・rdkit を import しない。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts/train", ROOT / "scripts/local_eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import fragnn_train as T  # noqa: E402  モデル・特徴は学習と同じ関数（T.B = fragnn_bench）
import sanpier as SP  # noqa: E402  npz の形（pack / load_scores）は sanpier と同じ

B = T.B
MAX_SPECTRA = 16             # run.py の query_spectra と同じ
DEFAULT_WEIGHTS = ROOT / "outputs/fragnn_train/model_best.pt"
CHUNK_PAIRS = 8192           # 推論で一度に流す組の数（実装の定数。点の値は変わらない）


# ---------------------------------------------------------------- 純粋な関数（CI）

def spectrum_row(r) -> dict:
    """test.parquet の 1 行 → 学習の load_spectra と同じ dict（mode は ±1、ce は collision_energy_ev の平均、空なら 0）。"""
    ce = r.collision_energy_ev
    ce = float(np.mean(ce)) if ce is not None and len(ce) else 0.0
    return dict(adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == "positive" else -1,
                mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64), ce=ce)


def molecule_rows(te) -> dict[str, list[dict]]:
    """run.py の query_spectra と同じ行（groupby sort=False の各分子の先頭 16 本）を学習の形の dict で返す。"""
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        out[str(mid)] = [spectrum_row(r) for r in g.itertuples()][:MAX_SPECTRA]
    return out


def usable(rows: list[dict]) -> list[dict]:
    """使えるスペクトル = top_peaks の後に有効なピーク（有限・強度 > 0）が 1 本以上あるもの。"""
    return [r for r in rows if len(B.top_peaks(r["mz"], r["it"])[0]) > 0]


def first_mode(rows: list[dict]):
    """その分子の 1 本目のスペクトルの正負（使えるかを問わない）。スペクトルが無ければ None。"""
    return rows[0]["mode"] if rows else None


def blind_targets(mids, modes: dict) -> dict:
    """#120 本文 25 行: molecule_id の文字列の昇順で次の分子のうち、正負が同じ最初の分子（最後は先頭に戻る）。
    同じ正負の分子がほかに無ければ正負を問わず次の分子、集合が 1 分子なら自分（blind = full）。
    返り値 = {mid: (入れ替え先, 種類)}、種類は same_mode / any_mode / self。"""
    order = sorted(set(map(str, mids)))
    n = len(order)
    out = {}
    for i, m in enumerate(order):
        if n == 1:
            out[m] = (m, "self")
            continue
        nxt = [order[(i + d) % n] for d in range(1, n)]
        same = [x for x in nxt if modes.get(x) == modes.get(m)]
        out[m] = (same[0], "same_mode") if same else (nxt[0], "any_mode")
    return out


def mean_logits(logit: np.ndarray, n_cand: int, n_spec: int) -> np.ndarray:
    """組の logit（候補が外側、スペクトルが内側の並び: 組 c*n_spec + s）→ 候補ごとのスペクトルの単純平均（float64）。"""
    return np.asarray(logit, np.float64).reshape(n_cand, n_spec).mean(axis=1)


def pair_layout(n_cand: int, n_spec: int) -> tuple[np.ndarray, np.ndarray]:
    """(候補の番号, スペクトルの番号) の組の並び（候補が外側）。"""
    return np.repeat(np.arange(n_cand), n_spec), np.tile(np.arange(n_spec), n_cand)


# ---------------------------------------------------------------- 推論（torch はここで初めて読む）

def load_model(weights: Path, device: str = "cpu"):
    import torch
    model = B.build_model()
    sd = torch.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def score_candidates(model, levels: list, spectra: list[dict], device: str = "cpu", chunk: int = CHUNK_PAIRS) -> np.ndarray:
    """候補の断片（fragment_levels の結果、None は読めない）と分子のスペクトル（学習の形の dict）→ 候補ごとの n。
    使えるスペクトルの logit の単純平均。読めない候補・使えるスペクトルが無い分子は NaN。"""
    import torch
    out = np.full(len(levels), np.nan, np.float64)
    sp_rows = usable(spectra)
    ok = np.array([lv is not None for lv in levels], bool)
    if not sp_rows or not ok.any():
        return out
    spec = B.pack_spectra(sp_rows)
    masses, off = B.levels_to_csr([lv for lv in levels if lv is not None])
    n_c, n_s = int(ok.sum()), len(sp_rows)
    cs, sp = pair_layout(n_c, n_s)
    spec_feat = torch.from_numpy(np.ascontiguousarray(spec["spec_feat"], np.float32))
    logit = np.zeros(len(cs), np.float64)
    with torch.inference_mode():
        for a in range(0, len(cs), chunk):
            c, s = cs[a:a + chunk], sp[a:a + chunk]
            feats, fs = B.batch_features(masses, off, c, s, spec)
            valid = np.arange(B.TOP_PEAKS)[None, :] < np.asarray(spec["n"])[s][:, None]
            x = torch.from_numpy(np.ascontiguousarray(feats, np.float32)).to(device)
            vt = torch.from_numpy(valid).to(device)
            pair = torch.from_numpy(np.asarray(fs, np.float32)[:, None]).to(device)
            sf = spec_feat[torch.from_numpy(s)].to(device)
            logit[a:a + chunk] = model(x, vt, pair, sf).double().cpu().numpy()
    out[ok] = mean_logits(logit, n_c, n_s)
    return out


class LevelCache:
    """pool の行 → fragment_levels（学習の prepare と同じ呼び方）。None・例外は None（= 読めない）。"""

    def __init__(self, fr, smiles):
        self.fr, self.smiles, self.c = fr, smiles, {}
        cfg = fr.resolve_cfg(None)
        self.args = (cfg["max_breaks"], cfg["max_bonds_2"], cfg["max_bonds_3"])

    def __call__(self, i: int):
        i = int(i)
        if i not in self.c:
            try:
                self.c[i] = self.fr.fragment_levels(str(self.smiles[i]), *self.args)
            except Exception:  # noqa: BLE001
                self.c[i] = None
        return self.c[i]


def score_holdout(holdout: str, weights: Path, out_dir: Path, device: str = "cpu", limit: int = 0,
                  root: Path = ROOT, logits: Path | None = None, truth: Path | None = None) -> dict:
    """集合の全分子の fragnn・fragnnblind の点を作って npz と json を書く。limit > 0 なら先頭 N 分子だけ
    （blind の入れ替え先は集合の全分子で決める。費用の測定用）。"""
    import pandas as pd
    import torch
    import numba
    from rdkit import rdBase
    import run as R
    import library as L
    from mrr25 import read_truth
    t0 = time.time()
    weights = Path(weights)
    w_sha = T.sha256_file(weights)
    model = load_model(weights, device)
    fr = T.import_frag_rescore(root)
    fr.warmup()
    # #146: 読み先を指定できる（既定は今までどおり）
    logits = Path(logits) if logits else root / f"outputs/holdout_eval/{holdout}_fold0/pc_logits.npz"
    truth = Path(truth) if truth else root / f"outputs/holdouts/{holdout}/truth.csv"
    zf = np.load(logits, allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    te = pd.read_parquet(root / f"data/holdouts/{holdout}/test.parquet")
    spec = molecule_rows(te)
    truth_path = truth
    truth = read_truth(truth_path)
    mids = [m for m in truth if m in targets]
    bt = blind_targets(mids, {m: first_mode(spec.get(m, [])) for m in mids})
    run_mids = mids[:limit] if limit else mids
    lc = LevelCache(fr, pool.smiles)
    full, blind = {}, {}
    n_rows = n_unparsable = n_nospec_full = n_nospec_blind = 0
    t_mol = time.time()
    for mid in run_mids:
        idx = R.visible_window(pool, targets[mid], hidden)
        lv = [lc(i) for i in idx]
        n_rows += len(idx)
        n_unparsable += sum(x is None for x in lv)
        sp_full = spec.get(mid, [])
        tgt, kind = bt[mid]
        sp_blind = spec.get(tgt, [])
        n_nospec_full += not usable(sp_full)
        n_nospec_blind += not usable(sp_blind)
        nf = score_candidates(model, lv, sp_full, device)
        nb = nf.copy() if kind == "self" else score_candidates(model, lv, sp_blind, device)
        full[mid] = (idx, nf.astype(np.float32))
        blind[mid] = (idx, nb.astype(np.float32))
    secs_mol = time.time() - t_mol
    kinds = [bt[m][1] for m in run_mids]
    rep = dict(issue=120, holdout=holdout, molecules=len(run_mids), molecules_in_set=len(mids), limit=limit,
               rows=n_rows, unparsable_rows=n_unparsable, unique_pool_rows=len(lc.c),
               molecules_without_usable_spectra=n_nospec_full, molecules_blind_target_without_usable_spectra=n_nospec_blind,
               nan_full=int(sum(np.isnan(v[1]).sum() for v in full.values())),
               nan_blind=int(sum(np.isnan(v[1]).sum() for v in blind.values())),
               blind_same_mode=kinds.count("same_mode"), blind_any_mode=kinds.count("any_mode"), blind_self=kinds.count("self"),
               hidden_keys=len(hidden), logits=str(logits), truth=str(truth_path), weights=str(weights), weights_sha256=w_sha, device=device,
               torch=torch.__version__, rdkit=rdBase.rdkitVersion, numba=numba.__version__,
               torch_threads=torch.get_num_threads(), secs_molecules=round(secs_mol, 1),
               secs_per_molecule=round(secs_mol / max(len(run_mids), 1), 3), secs=round(time.time() - t0, 1),
               ram_peak_gb=T.ram_peak_gb())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, per in (("fragnn_scores.npz", full), ("fragnnblind_scores.npz", blind)):
        tmp = out_dir / (name + ".tmp.npz")
        np.savez_compressed(tmp, **SP.pack(per))
        os.replace(tmp, out_dir / name)
        rep[name] = dict(sha256=T.sha256_file(out_dir / name), rows=int(sum(len(v[0]) for v in per.values())))
    T.atomic_write_text(out_dir / "fragnn_scores.json", json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="#120 fragnn の点（full と blind）を候補窓に付けて npz に書く")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out-dir", default=None, help="既定 outputs/local_eval/<holdout>")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 分子だけ（費用の測定用）")
    ap.add_argument("--logits", default=None, help="#146: pc_logits.npz（既定 outputs/holdout_eval/<holdout>_fold0/pc_logits.npz）")
    ap.add_argument("--truth", default=None, help="#146: truth.csv（既定 outputs/holdouts/<holdout>/truth.csv）")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    out = Path(a.out_dir) if a.out_dir else ROOT / f"outputs/local_eval/{a.holdout}"
    rep = score_holdout(a.holdout, Path(a.weights), out, a.device, a.limit, logits=a.logits, truth=a.truth)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
