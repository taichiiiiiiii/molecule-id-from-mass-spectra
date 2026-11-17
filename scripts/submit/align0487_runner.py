#!/usr/bin/env python3
"""#127: 0.487 写しに足す整列チャネルの点 a_align を、最終リストの候補（molecule_id, smiles）ごとに計算する。

提出のノート（`scripts/submit/make_lb487_align.py` が作る）の中では、このファイルを作業フォルダに書き、外側セル 15 の
`neural_score` の 1 行から別のプロセスとして走らせる（DreaMS の import のための stub が本体のカーネルに残らないように）。
ローカルでは同じファイルを CPU で走らせて、`align.py`（PR #110）の式との一致を確かめる（`scripts/submit/check_align0487_local.py`）。

    python align0487_runner.py --test test.parquet --cands cands.parquet --out a_align.parquet --report align0487_runner.json \
        --pool-dir <casmi26-v2-pool> --casmi-code <casmi26-v3-models/code> --code-dir <align_train.py 等を置いた所> \
        --dreams-root <DreaMS-<commit>> --dreams-state dreams_ssl_state.pt --align-model model_best.pt \
        --align-config align_config.json [--device cuda] [--expect-rdkit 2026.03.3]

#127 の「提出の前に固定すること」:
1. 候補 → h のキー: 立体を外し互変異性体を正準化してからの InChIKey[:14]（0.487 写しのセル 21 の `_to_k14` の RDKit の部分。
   手順 4 で pool の 1,000 行を 100% 再現した方）で `pool_meta.key` を引く
2. (a) pool にある候補は `pool_fp.npy` の行、無い候補は `fs_fp`（`casmi.chem.raw_fingerprint(smiles)[fp_bits]`）から分子側の MLP で h
   (b) (a) ができない候補が 1 つでもある分子、スペクトルから g が作れない分子は、その分子ごと 2 項に戻す（a を NaN にする）
   (d) 候補が当たった pool の行すべてで、その行の SMILES から作った指紋が `pool_fp.npy` の行とバイト一致し、キーが
       `pool_meta.key` を再現することを確かめる。1 行でも違えば status `fp_mismatch` で止まる（終了コード 3、a は書かない）
4. 付加イオンは学習の語彙（align_config.json の adduct_vocab、無いものは未知の枠）、衝突エネルギーは align_embed.ce_value
- g: DreaMS（ピーク 100、fp32 の前向き、最後の層の先頭トークン）→ 学習のシャードと同じ規則で fp16 に丸める（有限かつ
  |x| ≤ 60,000 のとき）→ encode_spectrum → 行ごとに正規化 → 分子ごとの平均を正規化（align.py の molecule_vectors）
- a = cos(v, h) = h_norm · v（align.py の window_scores と同じ。float64 で計算して float32 に）

torch・RDKit はモジュールの先頭で import しない（CI は純粋な関数だけをテストする）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

POOL_META_SHA256 = "82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0"
POOL_FP_SHA256 = "f02d91eeb4e2be5687ce678c199aa09a3acda1e821478372fc23fd1c7edf55e8"
FP_BITS_SHA256 = "c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677"
DREAMS_STATE_SHA256 = "65eb33fa6d6179438af16d0973b2b461e5712288cc7b5c46a938e4926ba55c11"
ALIGN_MODEL_SHA256 = "4e2a7cbe6a570805f47c4359c57de7a8ed7d75de688aa603d27745ca5507cd51"
FP_BITS = 10_226
PACKED_BYTES = (FP_BITS + 7) // 8
PEAKS = 100
EMB_BATCH = 32
FP16_MAX = 60_000.0
EXIT_FP_MISMATCH = 3


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def ce_value(ce) -> float:
    """align_embed.ce_value と同じ: collision_energy_ev の先頭、無ければ NaN。"""
    if ce is None:
        return float("nan")
    try:
        if len(ce) == 0:
            return float("nan")
        v = ce[0]
    except TypeError:
        v = ce
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def fp16_like_training(emb: np.ndarray) -> np.ndarray:
    """学習のシャード（align_embed.embed_shard）と同じ規則: 有限かつ max|x| ≤ FP16_MAX なら fp16 に丸めて fp32 に戻す。"""
    emb = np.asarray(emb, np.float32)
    if len(emb) and bool(np.isfinite(emb).all()) and float(np.abs(emb).max()) <= FP16_MAX:
        return emb.astype(np.float16).astype(np.float32)
    return emb


def molecule_vector(g_rows: np.ndarray):
    """align.py の molecule_vectors の 1 分子分: 行（正規化済みの g）の平均を正規化。作れなければ None。"""
    g_rows = np.asarray(g_rows)
    if len(g_rows) == 0:
        return None
    v = g_rows.astype(np.float64).mean(0)
    nv = np.linalg.norm(v)
    if nv > 0 and np.isfinite(nv):
        return v / nv
    return None


def scores(h_rows: np.ndarray, v) -> np.ndarray:
    """align.py の window_scores と同じ式（h は正規化済み）。"""
    if v is None or len(h_rows) == 0:
        return np.full(len(h_rows), np.nan, np.float32)
    return (np.asarray(h_rows, np.float64) @ v).astype(np.float32)


def mix_rows(mids, za, zb, a, zfun):
    """外側セル 15 の新しい 1 行の中身（#127 の式）。

    mids: 行ごとの molecule_id、za・zb: 既存の 2 項の z（fillna(0.0) 済み）、a: a_align（NaN = 点なし）。
    a が分子の全行で有限な分子だけ (za + zb + z(a)) / 3、ほかの分子は今と同じ 0.5 * za + 0.5 * zb。
    zfun は分子ごとの z 化（ノートの _z。ここでは配列と mids を受け取る形にした関数を渡す）。
    戻り値: (新しい値の配列, 3 項になった分子の集合)
    """
    mids = np.asarray(mids, dtype=object)
    za, zb, a = (np.asarray(x, np.float64) for x in (za, zb, a))
    base = 0.5 * za + 0.5 * zb
    ok = {}
    for m, v in zip(mids, a):
        ok[m] = ok.get(m, True) and bool(np.isfinite(v))
    ok_m = {m for m, f in ok.items() if f}
    mask = np.array([m in ok_m for m in mids], bool)
    out = base.copy()
    if mask.any():
        out[mask] = (za[mask] + zb[mask] + np.asarray(zfun(a[mask], mids[mask]), np.float64)) / 3
    return out, ok_m


def z_by_molecule(x: np.ndarray, mids) -> np.ndarray:
    """ノートの _z と同じ（分子ごとに平均を引き ddof=0 の std で割る。std は 1e-6 で下を切る）。テストと確認用。"""
    x = np.asarray(x, np.float64)
    mids = np.asarray(mids, dtype=object)
    out = np.empty_like(x)
    for m in dict.fromkeys(mids):
        i = mids == m
        s = max(float(x[i].std(ddof=0)), 1e-6)
        out[i] = (x[i] - x[i].mean()) / s
    return out


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 実行の部分

def canon_key(smiles: str, Chem, taut) -> str:
    """0.487 写しのセル 21 の `_to_k14` の RDKit の部分（失敗は空文字）。"""
    try:
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return ""
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(taut.Canonicalize(m))[:14]
    except Exception:  # noqa: BLE001
        return ""


def embed_test(test_path, dreams_root, dreams_state, device, AST, rep):
    """test.parquet の全スペクトルを DreaMS で埋め込む。戻り値: (emb (n,1024) fp32, 行の情報の list)。"""
    import pyarrow.parquet as pq
    import torch
    t0 = time.time()
    model, SP, DFA, info = AST.load_dreams(dreams_root, dreams_state, device)
    rep["dreams"] = {k: info[k] for k in ("n_layers", "d_model", "params_backbone", "ckpt_global_step", "load_state_dict")}
    rep["t_dreams_load_s"] = round(time.time() - t0, 2)
    pp = SP(dformat=DFA(), n_highest_peaks=PEAKS)
    t = pq.read_table(test_path, columns=["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
                                          "precursor_mz", "collision_energy_ev", "adduct"]).to_pydict()
    rows, buf, stats = [], [], {"spectra": len(t["molecule_id"]), "empty": 0, "preproc_error": 0}
    t1 = time.time()
    for i in range(len(t["molecule_id"])):
        mz, it = t["ms2_mzs"][i], t["ms2_normalized_intensities"][i]
        if not mz:
            stats["empty"] += 1
            continue
        try:
            arr = pp(np.array([mz, it], dtype=np.float64), prec_mz=float(t["precursor_mz"][i]), high_form=False)
        except Exception:  # noqa: BLE001  学習と同じく、前処理で落ちる行は数えて飛ばす
            stats["preproc_error"] += 1
            continue
        buf.append(arr)
        rows.append((str(t["molecule_id"][i]), t["adduct"][i], ce_value(t["collision_energy_ev"][i])))
    rep["t_preproc_s"] = round(time.time() - t1, 2)
    store = {}
    last = model.transformer_encoder.ffs[model.n_layers - 1]
    hook = last.register_forward_hook(lambda mod, inp, o: store.__setitem__("e", o[:, 0, :]))
    chunks = []
    t2 = time.time()
    try:
        x = torch.from_numpy(np.stack(buf).astype(np.float32)) if buf else torch.zeros((0, 1, 2))
        with torch.inference_mode():
            for i in range(0, len(buf), EMB_BATCH):
                model(x[i:i + EMB_BATCH].to(device))
                chunks.append(store["e"].float().cpu().numpy())
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        hook.remove()
    rep["t_embed_s"] = round(time.time() - t2, 2)
    emb = np.concatenate(chunks) if chunks else np.zeros((0, 1024), np.float32)
    stats["embedded"] = int(len(emb))
    stats["finite"] = bool(np.isfinite(emb).all())
    rep["spectra"] = stats
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("test", "cands", "out", "report", "pool-dir", "casmi-code", "code-dir", "dreams-root", "dreams-state",
              "align-model", "align-config"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--expect-rdkit", default="")
    ap.add_argument("--save-g", default="", help="確認用: 分子ごとの v を npz に書く")
    args = ap.parse_args(argv)
    sys.dont_write_bytecode = True
    T0 = time.time()
    rep = {"issue": 127, "status": "started", "argv": vars(args)}
    rp = Path(args.report)

    def dump():
        rp.write_text(json.dumps(rep, indent=1, default=str, ensure_ascii=False))

    dump()
    import pandas as pd
    import rdkit
    import torch
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    rep["versions"] = {"rdkit": rdkit.__version__, "torch": torch.__version__, "numpy": np.__version__}
    if args.expect_rdkit and rdkit.__version__ != args.expect_rdkit:
        rep.update(status="error", error=f"RDKit {rdkit.__version__} != {args.expect_rdkit}")
        dump()
        return 2
    sys.path.insert(0, args.code_dir)
    import align_speed_trial as AST
    import align_train as AT
    import base_fs
    base_fs.set_casmi_code_dir(args.casmi_code)
    assert AT.FP_BITS == FP_BITS
    pool = Path(args.pool_dir)
    sha = {"pool_meta": sha256_file(pool / "pool_meta.parquet"), "pool_fp": sha256_file(pool / "pool_fp.npy"),
           "fp_bits": sha256_file(pool / "fp_bits.npy"), "dreams_state": sha256_file(args.dreams_state),
           "align_model": sha256_file(args.align_model)}
    rep["input_sha256"] = sha
    want = {"pool_meta": POOL_META_SHA256, "pool_fp": POOL_FP_SHA256, "fp_bits": FP_BITS_SHA256,
            "dreams_state": DREAMS_STATE_SHA256, "align_model": ALIGN_MODEL_SHA256}
    if sha != want:
        rep.update(status="error", error="input sha256 mismatch", want_sha256=want)
        dump()
        return 2
    rep["t_sha_s"] = round(time.time() - T0, 2)
    meta = pd.read_parquet(pool / "pool_meta.parquet", columns=["key", "smiles"])
    key_to_row = dict(zip(meta.key.values, range(len(meta))))
    pool_fp = np.load(pool / "pool_fp.npy", mmap_mode="r")
    bits = np.load(pool / "fp_bits.npy")
    assert pool_fp.shape == (len(meta), PACKED_BYTES) and len(bits) == FP_BITS

    cands = pd.read_parquet(args.cands, columns=["molecule_id", "smiles"])
    cands["molecule_id"] = cands.molecule_id.astype(str)
    taut = rdMolStandardize.TautomerEnumerator()
    t1 = time.time()
    uniq = list(dict.fromkeys(cands.smiles.astype(str)))
    key_of = {s: canon_key(s, Chem, taut) for s in uniq}
    rep["t_keys_s"] = round(time.time() - t1, 2)
    row_of = {s: key_to_row.get(k, -1) if k else -1 for s, k in key_of.items()}

    # (d) 当たった pool の行の確認（その行の SMILES → 指紋・キー）
    t2 = time.time()
    hit_rows = sorted({r for r in row_of.values() if r >= 0})
    bad = []
    for r in hit_rows:
        s = meta.smiles.values[r]
        f = base_fs.fs_fp(s, bits)
        k = canon_key(s, Chem, taut)
        fp_ok = f is not None and bool((np.packbits(f, bitorder="big") == np.asarray(pool_fp[r])).all())
        if not fp_ok or k != meta.key.values[r]:
            bad.append({"row": int(r), "smiles": s, "fp_ok": fp_ok, "key": k, "pool_key": str(meta.key.values[r])})
    rep["check_d"] = {"pool_rows_checked": len(hit_rows), "n_bad": len(bad), "bad_examples": bad[:20],
                      "seconds": round(time.time() - t2, 2)}
    if bad:
        rep["status"] = "fp_mismatch"
        dump()
        print("ALIGN0487 fp_mismatch", len(bad), flush=True)
        return EXIT_FP_MISMATCH

    # h
    cfg = json.loads(Path(args.align_config).read_text())
    dev = args.device
    model = AT.build_model(len(cfg["adduct_vocab"]) + 1, cfg["adduct_signs"], d_ms=cfg["d_ms"], d_mol=cfg["d_mol"])
    model.load_state_dict(torch.load(args.align_model, map_location="cpu"), strict=True)
    model = model.to(dev).eval()
    t3 = time.time()
    packed, fs_fail, n_pool, n_fs = {}, set(), 0, 0
    for s in uniq:
        r = row_of[s]
        if r >= 0:
            packed[s] = np.asarray(pool_fp[r]); n_pool += 1
            continue
        f = base_fs.fs_fp(s, bits)
        if f is None:
            fs_fail.add(s)
            continue
        packed[s] = np.packbits(np.asarray(f, np.uint8), bitorder="big"); n_fs += 1
    rep["t_fp_s"] = round(time.time() - t3, 2)
    hs = [s for s in uniq if s in packed]
    H = np.zeros((len(hs), cfg["D_SHARED"] if "D_SHARED" in cfg else 512), np.float32)
    with torch.inference_mode():
        for i in range(0, len(hs), 512):
            x = AT.unpack_bits_torch(torch.from_numpy(np.stack([packed[s] for s in hs[i:i + 512]])).to(dev), FP_BITS,
                                     torch.float32)
            H[i:i + 512] = torch.nn.functional.normalize(model.encode_molecule(x).float(), dim=-1).cpu().numpy()
    h_of = dict(zip(hs, range(len(hs))))
    rep["candidates"] = {"rows": int(len(cands)), "unique_smiles": len(uniq), "in_pool": n_pool, "fs_fp": n_fs,
                         "fs_fp_failed": len(fs_fail)}
    rep["t_h_s"] = round(time.time() - t3, 2)

    # g
    emb, rows = embed_test(args.test, args.dreams_root, args.dreams_state, dev, AST, rep)
    t4 = time.time()
    emb = fp16_like_training(emb)
    vocab = cfg["adduct_vocab"]
    with torch.inference_mode():
        g = torch.nn.functional.normalize(model.encode_spectrum(
            torch.from_numpy(emb).to(dev), torch.from_numpy(np.array([r[2] for r in rows], np.float32)).to(dev),
            torch.from_numpy(AT.adduct_ids([r[1] for r in rows], vocab)).to(dev)).float(), dim=-1).cpu().numpy()
    rep["unknown_adduct_spectra"] = int(sum(AT.normalize_adduct(r[1]) not in set(vocab) for r in rows))
    by_mol = {}
    for j, r in enumerate(rows):
        by_mol.setdefault(r[0], []).append(j)
    vec = {m: molecule_vector(g[js]) for m, js in by_mol.items()}
    vec = {m: v for m, v in vec.items() if v is not None}
    rep["t_g_s"] = round(time.time() - t4, 2)

    # a
    a = np.full(len(cands), np.nan, np.float32)
    reason = {}
    for m, idx in cands.groupby("molecule_id", sort=False).indices.items():
        sm = cands.smiles.values[idx].astype(str)
        v = vec.get(m)
        if v is None:
            reason[m] = "no_spectrum_vector"
            continue
        if any(s in fs_fail for s in sm):
            reason[m] = "candidate_without_h"
            continue
        a[idx] = scores(H[[h_of[s] for s in sm]], v)
    out = cands.assign(a_align=a)
    out.to_parquet(args.out, index=False)
    n_mol = int(cands.molecule_id.nunique())
    rep["molecules"] = {"total": n_mol, "three_term": n_mol - len(reason), "two_term": len(reason),
                        "two_term_frac": (len(reason) / n_mol) if n_mol else None,
                        "two_term_reasons": {k: sum(1 for v in reason.values() if v == k)
                                             for k in sorted(set(reason.values()))},
                        "two_term_examples": dict(list(reason.items())[:20])}
    if args.save_g:
        np.savez(args.save_g, mids=np.array(list(vec), dtype=object), v=np.stack(list(vec.values())) if vec else np.zeros((0, 512)))
    rep["out_sha256"] = sha256_file(args.out)
    rep["status"] = "ok"
    rep["seconds"] = round(time.time() - T0, 2)
    dump()
    print("ALIGN0487 ok", json.dumps(rep["molecules"])[:300], f"{rep['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
