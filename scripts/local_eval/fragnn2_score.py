#!/usr/bin/env python3
"""
Issue #196 (h): fragnn v2 の段。v2 の重みで、縮小版の候補窓の各候補に点 n と blind の点を付けてキャッシュする。

    .venv/bin/python scripts/local_eval/fragnn2_score.py --holdout <h> --weights <v2 の model_best.pt> \
        --logits outputs/local_eval/<h>/pc_logits.npz --truth outputs/local_eval/<h>/truth.csv [--out-dir <dir>]
    .venv/bin/python scripts/local_eval/fragnn2_score.py --holdout sub_test --random-init 0 --no-truth --limit 10 \
        --logits outputs/holdout_eval/sub_test_fold0/pc_logits.npz --out-dir <smoke の dir>     # smoke（ラベル無し）

v1 の `fragnn_score.py`（#120）は変えない。v1 と同じもの（そのまま import して使う）: 集合の分子の決め方、候補 = run.py の
visible_window、断片 = frag_rescore.fragment_levels（LevelCache）、blind の入れ替え先（blind_targets・first_mode）、
組の並び（pair_layout）、npz の形（sanpier.pack）、NaN の扱い。

v2 で違うもの（#196 (h)。式は学習のコード〈PR #202 の head b6bc90f の fragnn_bench〉の関数をそのまま使う）:
- 各本の dict に instrument（instrument_type）を足す。CE は v1 と同じ（collision_energy_ev の平均、空なら 0）
- 各本に P5'（precursor + 2 Da より上を除く）→ 上位 64 本 → timsTOF の印: B.pack_spectra_v2
- P4 の推論の形: 分子ごとに (付加イオン, timsTOF の印) が同じ本を出てくる順に群にし、各群を順に 4 本ずつの塊に切る
  （B.infer_chunks）。塊が 2 本以上なら学習の merge と同じ B.merged_row（precursor・shift・mode・印は塊の先頭の本、
  CE は値のある本の平均〈無ければ 0〉）、1 本ならその本のまま。学習の B.merge_batch が merge した本を書き戻すのと同じ形
- 点 n = 使える塊（有効なピークが 1 本以上）ごとの logit の単純平均。使える塊が無ければ NaN
- モデルは B.build_model(B.F_SPEC_V2)、CPU 固定（torch.device("cpu")）、model.eval()、inference_mode、fp32
- 出力: <out-dir>/fragnn2_scores.npz・fragnn2blind_scores.npz と fragnn2_scores.json（既定の out-dir は
  outputs/local_eval/<holdout>。run.py の fragnn2・fragnn2blind 段が読む）。MRR・judge は計算しない

守り（#196 の (h) の時期の変更 7089857166・レビュー 7089895090）:
- --random-init <seed> は乱数で初期化した重み（学習の重みを読まない）。smoke 用で、--holdout sub_test（見える test、
  ラベル無し）と --no-truth のときだけ使える
- --weights を読むときは、その隣の train_report.json の status が "ok"（学習が止まった）でなければ止める
  （学習の途中の model_best.pt で採点しない）

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
import fragnn_score as F1  # noqa: E402  v1 の段（変えない）。同じ定義の関数はここから使う
import fragnn_train as T  # noqa: E402  学習のコード（T.B = fragnn_bench、v2 の前処理の関数）
import sanpier as SP  # noqa: E402  npz の形（pack / load_scores）

B = T.B
MAX_SPECTRA = F1.MAX_SPECTRA           # run.py の query_spectra と同じ（分子ごとの先頭 16 本）
CHUNK_PAIRS = F1.CHUNK_PAIRS           # 推論で一度に流す組の数（実装の定数。点の値は変わらない）
SMOKE_HOLDOUT = "sub_test"             # 見える test（ラベル無し、何も隠さない）
NPZ_FULL, NPZ_BLIND, REPORT = "fragnn2_scores.npz", "fragnn2blind_scores.npz", "fragnn2_scores.json"

# v1 と同じ定義（そのまま使う）
blind_targets = F1.blind_targets
first_mode = F1.first_mode
pair_layout = F1.pair_layout
mean_logits = F1.mean_logits
LevelCache = F1.LevelCache


# ---------------------------------------------------------------- 純粋な関数（CI）

def spectrum_row(r) -> dict:
    """test.parquet の 1 行 → v1 の spectrum_row（mode ±1、CE は平均・空なら 0）に instrument を足した dict。
    学習の load_spectra_v2 が pack_spectra_v2 に渡す dict と同じ鍵。"""
    d = F1.spectrum_row(r)
    d["instrument"] = r.instrument_type
    return d


def molecule_rows(te) -> dict[str, list[dict]]:
    """run.py の query_spectra と同じ行（groupby sort=False の各分子の先頭 16 本）を v2 の dict で返す。"""
    out = {}
    for mid, g in te.groupby("molecule_id", sort=False):
        out[str(mid)] = [spectrum_row(r) for r in g.itertuples()][:MAX_SPECTRA]
    return out


def chunk_spectra(rows: list[dict]) -> tuple[dict, list[list[int]]]:
    """分子の本（rows の順）→ (塊の詰めた値〈B.SPEC_KEYS、行 = 塊〉, 塊ごとの本の番号)。
    各本は B.pack_spectra_v2（P5' → 上位 64 本 → 印）、塊は B.infer_chunks、2 本以上の塊は B.merged_row で merge し、
    学習の B.merge_batch と同じく mz・it・n・spec_feat だけを書き戻す（prec・shift は塊の先頭の本のもの）。"""
    if not rows:
        return {k: np.zeros((0,) + v.shape[1:], v.dtype) for k, v in B.pack_spectra_v2([]).items()
                if k in B.SPEC_KEYS}, []
    pk = B.pack_spectra_v2(rows)
    chunks = B.infer_chunks([r.get("adduct") for r in rows], pk["tims"].tolist())
    first = np.array([c[0] for c in chunks], np.int64)
    out = {k: pk[k][first].copy() for k in B.SPEC_KEYS}
    for r, c in enumerate(chunks):
        if len(c) > 1:
            m = B.merged_row(pk, c, c[0])
            out["mz"][r] = 0.0; out["it"][r] = 0.0
            out["mz"][r, :m["n"]], out["it"][r, :m["n"]], out["n"][r] = m["mz"], m["it"], m["n"]
            out["spec_feat"][r] = m["spec_feat"]
    return out, chunks


def usable_chunks(spec: dict) -> np.ndarray:
    """使える塊 = 有効なピークが 1 本以上（n > 0）の塊の番号。"""
    return np.flatnonzero(np.asarray(spec["n"]) > 0)


def take_rows(spec: dict, idx) -> dict:
    idx = np.asarray(idx, np.int64)
    return {k: np.asarray(v)[idx] for k, v in spec.items()}


def check_weights_report(weights: Path) -> dict:
    """学習が止まったかの確かめ: weights の隣の train_report.json の status が "ok" でなければ止める。
    status と stop_reason だけを読む（損失・内部検証の値は読まない）。"""
    rp = Path(weights).parent / "train_report.json"
    if not rp.exists():
        raise SystemExit(f"{rp} が無い（学習が止まったことを確かめられないので採点しない）")
    rep = json.loads(rp.read_text())
    tr = rep.get("train") if isinstance(rep.get("train"), dict) else {}
    st = dict(status=rep.get("status"), stop_reason=tr.get("stop_reason"))
    if st["status"] != "ok":
        raise SystemExit(f"学習が止まっていない（train_report.json の status = {st['status']!r}）。途中の重みで採点しない")
    return st


def check_random_init(holdout: str, no_truth: bool) -> None:
    if holdout != SMOKE_HOLDOUT or not no_truth:
        raise SystemExit(f"--random-init は smoke（--holdout {SMOKE_HOLDOUT} --no-truth）だけで使える")


# ---------------------------------------------------------------- 推論（torch はここで初めて読む）

def build_v2_model(weights: Path | None = None, random_init: int | None = None):
    """v2 の形のモデル（CPU）。weights があれば読む。無ければ torch.manual_seed(random_init) の乱数の重み。"""
    import torch
    dev = torch.device("cpu")
    if weights is None:
        if random_init is None:
            raise ValueError("weights か random_init のどちらかが要る")
        torch.manual_seed(int(random_init))
        model = B.build_model(B.F_SPEC_V2)
    else:
        model = B.build_model(B.F_SPEC_V2)
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(dev).eval()


def score_candidates(model, levels: list, rows: list[dict], chunk: int = CHUNK_PAIRS) -> np.ndarray:
    """候補の断片（None は読めない）と分子の本（v2 の dict）→ 候補ごとの n = 使える塊の logit の単純平均。
    読めない候補・使える塊が無い分子は NaN。"""
    import torch
    dev = torch.device("cpu")
    out = np.full(len(levels), np.nan, np.float64)
    spec, _ = chunk_spectra(rows)
    use = usable_chunks(spec)
    ok = np.array([lv is not None for lv in levels], bool)
    if len(use) == 0 or not ok.any():
        return out
    spec = take_rows(spec, use)
    top = int(spec["mz"].shape[1])
    masses, off = B.levels_to_csr([lv for lv in levels if lv is not None])
    n_c, n_s = int(ok.sum()), len(use)
    cs, sp = pair_layout(n_c, n_s)
    spec_feat = torch.from_numpy(np.ascontiguousarray(spec["spec_feat"], np.float32))
    logit = np.zeros(len(cs), np.float64)
    with torch.inference_mode():
        for a in range(0, len(cs), chunk):
            c, s = cs[a:a + chunk], sp[a:a + chunk]
            feats, fs = B.batch_features(masses, off, c, s, spec)
            valid = np.arange(top)[None, :] < np.asarray(spec["n"])[s][:, None]
            x = torch.from_numpy(np.ascontiguousarray(feats, np.float32)).to(dev)
            vt = torch.from_numpy(valid).to(dev)
            pair = torch.from_numpy(np.asarray(fs, np.float32)[:, None]).to(dev)
            sf = spec_feat[torch.from_numpy(s)].to(dev)
            logit[a:a + chunk] = model(x, vt, pair, sf).double().cpu().numpy()
    out[ok] = mean_logits(logit, n_c, n_s)
    return out


def score_holdout(holdout: str, out_dir: Path, weights: Path | None = None, random_init: int | None = None,
                  limit: int = 0, root: Path = ROOT, logits: Path | None = None, truth: Path | None = None,
                  no_truth: bool = False) -> dict:
    """集合の全分子の fragnn2・fragnn2blind の点を作って npz と json を書く。limit > 0 なら先頭 N 分子だけ
    （blind の入れ替え先は集合の全分子で決める）。no_truth: 分子 = test.parquet の分子の順のうち logits にあるもの
    （ラベルを読まない。smoke 用）。"""
    import pandas as pd
    import torch
    import numba
    from rdkit import rdBase
    import run as R
    import library as L
    t0 = time.time()
    if random_init is not None:
        check_random_init(holdout, no_truth)
        w_sha, w_desc, train_status = None, f"random-init seed {int(random_init)}", None
        model = build_v2_model(random_init=random_init)
    else:
        weights = Path(weights)
        train_status = check_weights_report(weights)
        w_sha, w_desc = T.sha256_file(weights), str(weights)
        model = build_v2_model(weights=weights)
    fr = T.import_frag_rescore(root)
    fr.warmup()
    logits = Path(logits) if logits else root / f"outputs/holdout_eval/{holdout}_fold0/pc_logits.npz"
    zf = np.load(logits, allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool()
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    te = pd.read_parquet(root / f"data/holdouts/{holdout}/test.parquet")
    spec = molecule_rows(te)
    if no_truth:
        truth_path = None
        mids = [m for m in spec if m in targets]
    else:
        from mrr25 import read_truth
        truth_path = Path(truth) if truth else root / f"outputs/holdouts/{holdout}/truth.csv"
        mids = [m for m in read_truth(truth_path) if m in targets]
    bt = blind_targets(mids, {m: first_mode(spec.get(m, [])) for m in mids})
    run_mids = mids[:limit] if limit else mids
    lc = LevelCache(fr, pool.smiles)
    full, blind = {}, {}
    n_rows = n_unparsable = n_nospec_full = n_nospec_blind = 0
    n_chunks, n_merged_chunks, n_spectra = 0, 0, 0
    t_mol = time.time()
    for mid in run_mids:
        idx = R.visible_window(pool, targets[mid], hidden)
        lv = [lc(i) for i in idx]
        n_rows += len(idx)
        n_unparsable += sum(x is None for x in lv)
        sp_full = spec.get(mid, [])
        tgt, kind = bt[mid]
        sp_blind = spec.get(tgt, [])
        ch_full, chunks = chunk_spectra(sp_full)
        n_spectra += len(sp_full); n_chunks += len(chunks); n_merged_chunks += sum(len(c) > 1 for c in chunks)
        n_nospec_full += len(usable_chunks(ch_full)) == 0
        n_nospec_blind += len(usable_chunks(chunk_spectra(sp_blind)[0])) == 0
        nf = score_candidates(model, lv, sp_full)
        nb = nf.copy() if kind == "self" else score_candidates(model, lv, sp_blind)
        full[mid] = (idx, nf.astype(np.float32))
        blind[mid] = (idx, nb.astype(np.float32))
    secs_mol = time.time() - t_mol
    kinds = [bt[m][1] for m in run_mids]
    rep = dict(issue=196, variant="v2", holdout=holdout, molecules=len(run_mids), molecules_in_set=len(mids), limit=limit,
               rows=n_rows, unparsable_rows=n_unparsable, unique_pool_rows=len(lc.c), spectra=n_spectra,
               chunks=n_chunks, merged_chunks=n_merged_chunks,
               molecules_without_usable_chunks=n_nospec_full, molecules_blind_target_without_usable_chunks=n_nospec_blind,
               nan_full=int(sum(np.isnan(v[1]).sum() for v in full.values())),
               nan_blind=int(sum(np.isnan(v[1]).sum() for v in blind.values())),
               nonfinite_nonnan=int(sum((~np.isfinite(v[1]) & ~np.isnan(v[1])).sum() for d in (full, blind) for v in d.values())),
               blind_same_mode=kinds.count("same_mode"), blind_any_mode=kinds.count("any_mode"), blind_self=kinds.count("self"),
               hidden_keys=len(hidden), logits=str(logits), truth=None if truth_path is None else str(truth_path),
               no_truth=bool(no_truth), weights=w_desc, weights_sha256=w_sha, random_init=random_init,
               train_status=train_status, device="cpu", top_peaks=B.TOP_PEAKS_V2, f_spec=B.F_SPEC_V2,
               merge_kmax=B.MERGE_KMAX, torch=torch.__version__, rdkit=rdBase.rdkitVersion, numba=numba.__version__,
               torch_threads=torch.get_num_threads(), secs_molecules=round(secs_mol, 1),
               secs_per_molecule=round(secs_mol / max(len(run_mids), 1), 3), secs=round(time.time() - t0, 1),
               ram_peak_gb=T.ram_peak_gb())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, per in ((NPZ_FULL, full), (NPZ_BLIND, blind)):
        tmp = out_dir / (name + ".tmp.npz")
        np.savez_compressed(tmp, **SP.pack(per))
        os.replace(tmp, out_dir / name)
        rep[name] = dict(sha256=T.sha256_file(out_dir / name), rows=int(sum(len(v[0]) for v in per.values())))
    T.atomic_write_text(out_dir / REPORT, json.dumps(rep, indent=1, ensure_ascii=False))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="#196 (h) fragnn v2 の点（full と blind）を候補窓に付けて npz に書く")
    ap.add_argument("--holdout", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--weights", default=None, help="v2 の model_best.pt（隣の train_report.json の status が ok のときだけ）")
    g.add_argument("--random-init", type=int, default=None, help="乱数の重み（smoke 用、--holdout sub_test --no-truth だけ）")
    ap.add_argument("--out-dir", default=None, help="既定 outputs/local_eval/<holdout>")
    ap.add_argument("--limit", type=int, default=0, help="先頭 N 分子だけ（smoke・費用の測定用）")
    ap.add_argument("--logits", default=None, help="pc_logits.npz（既定 outputs/holdout_eval/<holdout>_fold0/pc_logits.npz）")
    ap.add_argument("--truth", default=None, help="truth.csv（既定 outputs/holdouts/<holdout>/truth.csv）")
    ap.add_argument("--no-truth", action="store_true", help="truth を読まない（分子 = test.parquet の順のうち logits にあるもの）")
    a = ap.parse_args()
    if a.random_init is not None:
        check_random_init(a.holdout, a.no_truth)
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    out = Path(a.out_dir) if a.out_dir else ROOT / f"outputs/local_eval/{a.holdout}"
    rep = score_holdout(a.holdout, out, weights=a.weights, random_init=a.random_init, limit=a.limit,
                        logits=a.logits, truth=a.truth, no_truth=a.no_truth)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
