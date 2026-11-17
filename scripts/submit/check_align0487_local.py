#!/usr/bin/env python3
"""#127 手順 5（ローカル CPU、学習なし、v3 を見ない）: 提出のノートに足す段（align0487_runner.py）の a が、
`align.py`（PR #110）の式（molecule_vectors・window_scores）と学習が書いた `h_pool.npy` で作った a と一致するかを、
見えているテスト（data/test.parquet）の数分子で確かめる。足す段の秒も測る（時間の関門の参考値）。

    PYTHONPYCACHEPREFIX=<scratch>/pyc .venv/bin/python scripts/submit/check_align0487_local.py \
        --root <元のフォルダ> --code-dir <runner・align_train.py・align_speed_trial.py・base_fs.py・align.py を置いた所> \
        --dreams-state <scratch>/dreams_ssl_state.pt --work <scratch>/n1117n_check --n-mol 6

- 候補: 各分子の precursor_mz −1.007276（[M+H]+ の仮定。候補を作るためだけ）の ±10 ppm にある pool の行を先頭から最大 30、
  ＋ pool に無い SMILES 2 つ（pool の SMILES に "C" を足したもので RDKit が読めてキーが pool に無いもの）。最初の分子だけは
  わざと読めない SMILES を 1 つ足し、(b) の「その分子ごと 2 項に戻る」が働くことを見る
- 参照の a（pool の候補）: runner の分子ベクトル v（--save-g）に対して `align.window_scores`（PR #110 の式）を使う。h は
  (1) ここで `pool_fp.npy` の行（runner のキーの引き方を通さず、候補を作ったときの行番号）から分子側の MLP を別に通した fp32 の h、
  (2) 学習が書いた `h_pool.npy`（fp16 で保存）の行、の 2 通り
- 合格（事前登録の rtol 1e-4）: (1) で |a − a_ref| ≤ 1e-4·|a_ref| がすべての pool の候補で成り立つこと。(2) は fp16 の丸めの
  分だけ違うので記録（最大の絶対・相対の差）
- (3) g の側の独立の確かめ（--embed-dir を渡したとき。2026-11-17 昼、結果を見る前に決めた）: 見えているテストは train の写し
  なので、各スペクトルを train の行（precursor_mz と ms2_mzs の完全一致、1 行だけに当たること）に引き、学習が使った DreaMS の
  埋め込みのシャード（`outputs/align_embed/emb_*.npy`、fp16、`idx_*.parquet` の row で引く）から、学習と同じ
  `encode_spectrum` → 行ごとに正規化 → `align.molecule_vectors` で v_ref を作る。runner の v との cos の最小と、v_ref・fp32 の h で
  `align.window_scores` の a との差（最大の絶対・相対、rtol 1e-4 に入る数）を記録する。ローカルの DreaMS の前向き（CPU）と
  シャード（MPS）の差・fp16 の丸めがあるので rtol 1e-4 は合否にしない。cos の最小 < 0.999 の分子があれば「g の作り方が
  学習と違う」疑いとして Issue に書き、提出の手順に進まない
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def train_rows_for(train_path, sub):
    """見えているテストの各スペクトルを train の行に引く（precursor_mz と ms2_mzs の完全一致）。row group ごとに読む。"""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    want = {}
    for sid, p, mz in zip(sub.spectrum_id.astype(str), sub.precursor_mz.values, sub.ms2_mzs.values):
        want.setdefault((float(p), tuple(float(x) for x in mz)), []).append(sid)
    pf = pq.ParquetFile(train_path, buffer_size=1 << 20, pre_buffer=False)
    vs = pa.array(sorted({k[0] for k in want}))
    hits, off = {}, 0
    for b in pf.iter_batches(batch_size=8192, columns=["precursor_mz", "ms2_mzs"]):
        sel = np.nonzero(pc.is_in(b.column("precursor_mz"), value_set=vs).to_numpy(zero_copy_only=False))[0]
        for i in sel:
            k = (float(b.column("precursor_mz")[i].as_py()), tuple(b.column("ms2_mzs")[i].as_py()))
            for sid in want.get(k, []):
                hits.setdefault(sid, []).append(off + int(i))
        off += b.num_rows
    return hits


def shard_g_check(args, R, sub, a, pm, prow, vec, h32, cfg, mdl, AL, AT, torch, pd):
    """(3): 学習の埋め込みのシャードから v_ref を作り、runner の v と a を比べる（記録。合否は docstring の cos の条件）。"""
    t0 = time.time()
    hits = train_rows_for(args.train or str(R / "data/train.parquet"), sub)
    uniq = {sid: rows[0] for sid, rows in hits.items() if len(rows) == 1}
    E = Path(args.embed_dir)
    loc = {}
    for ip in sorted(E.glob("idx_*.parquet")):
        k = ip.stem.split("_")[1]
        ix = pd.read_parquet(ip, columns=["row", "adduct", "ce"])
        pos = {int(r): j for j, r in enumerate(ix.row.values)}
        for sid, r in uniq.items():
            if r in pos and sid not in loc:
                loc[sid] = (k, pos[r], ix.adduct.values[pos[r]], float(ix.ce.values[pos[r]]))
    sids = [s for s in sub.spectrum_id.astype(str) if s in loc]
    mids = sub.set_index(sub.spectrum_id.astype(str)).loc[sids].molecule_id.astype(str).values
    embs = np.stack([np.asarray(np.load(E / f"emb_{loc[s][0]}.npy", mmap_mode="r")[loc[s][1]], np.float32) for s in sids]) \
        if sids else np.zeros((0, cfg["d_ms"]), np.float32)
    with torch.inference_mode():
        g = torch.nn.functional.normalize(mdl.encode_spectrum(
            torch.from_numpy(embs), torch.from_numpy(np.array([loc[s][3] for s in sids], np.float32)),
            torch.from_numpy(AT.adduct_ids([loc[s][2] for s in sids], cfg["adduct_vocab"]))).float(), dim=-1).numpy()
    vref = AL.molecule_vectors(g, ["t"] * len(sids), mids, "t")
    n_spec_mol = sub.groupby(sub.molecule_id.astype(str)).size().to_dict()
    n_ref_mol = pd.Series(mids).value_counts().to_dict() if len(mids) else {}
    full = [m for m in vref if n_ref_mol.get(m, 0) == n_spec_mol.get(m, -1) and vec.get(m) is not None]
    cos = {m: float(np.dot(vref[m], vec[m])) for m in full}
    ref = np.full(len(a), np.nan)
    for m, idx in a.groupby("molecule_id", sort=False).indices.items():
        sel = idx[pm[idx]]
        if m in cos and len(sel):
            ref[sel] = AL.window_scores(sel, vref[m], h32)
    got = a.a_align.values.astype(np.float64)
    ok = pm & np.isfinite(got) & np.isfinite(ref)
    d = np.abs(got[ok] - ref[ok])
    rel = d / np.maximum(np.abs(ref[ok]), 1e-12)
    return {"spectra": int(len(sub)), "matched_one_train_row": len(uniq),
            "matched_many_rows": int(sum(len(v) > 1 for v in hits.values())), "found_in_shards": len(sids),
            "molecules_all_spectra_in_shards": len(full), "cos_min": min(cos.values()) if cos else None,
            "cos_by_molecule": cos, "n_compared": int(ok.sum()),
            "max_abs_diff": float(d.max()) if len(d) else None, "max_rel_diff": float(rel.max()) if len(rel) else None,
            "n_within_rtol_1e4": int((d <= 1e-4 * np.abs(ref[ok])).sum()),
            "suspect_g_mismatch": bool(cos and min(cos.values()) < 0.999), "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--code-dir", required=True)
    ap.add_argument("--dreams-state", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--n-mol", type=int, default=6)
    ap.add_argument("--align-model", default="", help="既定: <root>/outputs/align_train/model_best.pt")
    ap.add_argument("--h-pool", default="", help="既定: <root>/outputs/align_train/h_pool.npy")
    ap.add_argument("--embed-dir", default="", help="(3) 用: 学習の埋め込みのシャード（outputs/align_embed）")
    ap.add_argument("--train", default="", help="(3) 用: 既定 <root>/data/train.parquet")
    args = ap.parse_args()
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    R, W = Path(args.root), Path(args.work)
    W.mkdir(parents=True, exist_ok=True)
    pool = R / "data/ext/v4g/casmi26-v2-pool"
    te = pd.read_parquet(R / "data/test.parquet")
    mids = list(dict.fromkeys(te.molecule_id.astype(str)))[: args.n_mol]
    sub = te[te.molecule_id.astype(str).isin(mids)]
    sub.to_parquet(W / "test_sub.parquet", index=False)
    meta = pd.read_parquet(pool / "pool_meta.parquet", columns=["key", "smiles", "mass"])
    keys = set(meta.key.values)
    mass = meta.mass.values
    rows = []
    for m in mids:
        p = float(sub[sub.molecule_id.astype(str) == m].precursor_mz.iloc[0]) - 1.007276
        lo, hi = np.searchsorted(mass, p * (1 - 1e-5)), np.searchsorted(mass, p * (1 + 1e-5))
        for r in range(lo, min(hi, lo + 30)):
            rows.append((m, meta.smiles.values[r], int(r)))
        added = 0
        for r in range(lo, min(hi + 500, len(meta))):
            s = meta.smiles.values[r] + "C"
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            if Chem.MolToInchiKey(mol)[:14] in keys:
                continue
            rows.append((m, s, -1)); added += 1
            if added >= 2:
                break
        if m == mids[0]:
            rows.append((m, "C1CC(", -1))
    cands = pd.DataFrame(rows, columns=["molecule_id", "smiles", "pool_row"])
    cands[["molecule_id", "smiles"]].to_parquet(W / "cands.parquet", index=False)
    D = R / "data/ext/dreams/DreaMS-dbec3a0b514a99e5056cfccde4559fda8cfe8129"
    model_path = args.align_model or str(R / "outputs/align_train/model_best.pt")
    h_pool_path = args.h_pool or str(R / "outputs/align_train/h_pool.npy")
    cmd = [sys.executable, str(Path(args.code_dir) / "align0487_runner.py"), "--test", str(W / "test_sub.parquet"),
           "--cands", str(W / "cands.parquet"), "--out", str(W / "a_align.parquet"), "--report", str(W / "runner.json"),
           "--pool-dir", str(pool), "--casmi-code", str(R / "data/ext/v4g/casmi26-v3-models/code"),
           "--code-dir", args.code_dir, "--dreams-root", str(D), "--dreams-state", args.dreams_state,
           "--align-model", model_path,
           "--align-config", str(Path(args.code_dir) / "align_config.json"), "--save-g", str(W / "v.npz")]
    t0 = time.time()
    rc = subprocess.run(cmd).returncode
    wall = time.time() - t0
    rep = json.loads((W / "runner.json").read_text())
    out = {"runner_rc": rc, "runner_status": rep.get("status"), "wall_s": round(wall, 1)}
    if rc != 0:
        print(json.dumps(out)); (W / "check.json").write_text(json.dumps(out, indent=1)); return
    sys.path.insert(0, args.code_dir)
    import align as AL   # PR #110 の scripts/local_eval/align.py（git show で code-dir に出したもの）
    a = pd.read_parquet(W / "a_align.parquet")
    z = np.load(W / "v.npz", allow_pickle=True)
    vec = dict(zip([str(m) for m in z["mids"]], z["v"]))
    h_pool = np.load(h_pool_path, mmap_mode="r")
    import torch
    import align_train as AT
    cfg = json.loads((Path(args.code_dir) / "align_config.json").read_text())
    mdl = AT.build_model(len(cfg["adduct_vocab"]) + 1, cfg["adduct_signs"], d_ms=cfg["d_ms"], d_mol=cfg["d_mol"])
    mdl.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    mdl.eval()
    pool_fp = np.load(pool / "pool_fp.npy", mmap_mode="r")
    a = a.assign(pool_row=cands.pool_row.values)
    prow = a.pool_row.values
    pm = prow >= 0
    with torch.inference_mode():
        x = AT.unpack_bits_torch(torch.from_numpy(np.asarray(pool_fp[prow[pm]])), AT.FP_BITS, torch.float32)
        h32 = np.zeros((len(a), 512), np.float32)
        h32[pm] = torch.nn.functional.normalize(mdl.encode_molecule(x).float(), dim=-1).numpy()
    ref1, ref2 = np.full(len(a), np.nan), np.full(len(a), np.nan)
    for m, idx in a.groupby("molecule_id", sort=False).indices.items():
        sel = idx[pm[idx]]
        if len(sel) and vec.get(m) is not None:
            ref1[sel] = AL.window_scores(sel, vec.get(m), h32)
            ref2[sel] = AL.window_scores(prow[sel], vec.get(m), h_pool)

    def cmp(ref):
        got, want = a.a_align.values[pm].astype(np.float64), ref[pm]
        ok = np.isfinite(got) & np.isfinite(want)
        d = np.abs(got[ok] - want[ok])
        rel = d / np.maximum(np.abs(want[ok]), 1e-12)
        return {"n_compared": int(ok.sum()), "n_nan_got": int((~np.isfinite(got)).sum()),
                "max_abs_diff": float(d.max()) if len(d) else None, "max_rel_diff": float(rel.max()) if len(rel) else None,
                "n_within_rtol_1e4": int((d <= 1e-4 * np.abs(want[ok])).sum()),
                "pass_rtol_1e4": bool(ok.sum() and (d <= 1e-4 * np.abs(want[ok])).all())}

    g_check = None
    if args.embed_dir:
        g_check = shard_g_check(args, R, sub, a, pm, prow, vec, h32, cfg, mdl, AL, AT, torch, pd)
    n_spec = rep["spectra"]["embedded"]
    n_uniq = rep["candidates"]["unique_smiles"]
    spec_s = rep["t_preproc_s"] + rep["t_embed_s"] + rep["t_g_s"]
    cand_s = rep["t_keys_s"] + rep["check_d"]["seconds"] + rep["t_h_s"]   # t_h_s は t_fp_s を含む（同じ t3 から）
    out.update(molecules=rep["molecules"], candidates=rep["candidates"], check_d=rep["check_d"],
               n_pool_candidates=int(pm.sum()), vs_fp32_h_from_pool_fp=cmp(ref1), vs_h_pool_npy_fp16=cmp(ref2),
               a_range=[float(np.nanmin(a.a_align.values)), float(np.nanmax(a.a_align.values))],
               timing={k: rep.get(k) for k in ("t_sha_s", "t_dreams_load_s", "t_keys_s", "t_fp_s", "t_h_s", "t_preproc_s",
                                                "t_embed_s", "t_g_s", "seconds")},
               spectra_embedded=n_spec,
               per_spectrum_s=round((rep["t_preproc_s"] + rep["t_embed_s"] + rep["t_g_s"]) / max(n_spec, 1), 4),
               per_candidate_s=round(cand_s / max(n_uniq, 1), 5), fixed_s=round(rep["seconds"] - spec_s - cand_s, 2),
               timing_note="per_spectrum = preproc+embed+g; per_candidate (unique SMILES) = keys+check_d+fp+h; "
                           "fixed = runner total - both (sha256 of inputs, imports, DreaMS/model load)",
               runner_sha256=rep.get("input_sha256"), versions=rep.get("versions"), g_vs_training_shards=g_check)
    (W / "check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
