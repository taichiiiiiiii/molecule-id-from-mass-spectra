#!/usr/bin/env python3
"""整列チャネルの分子側 h が、SMILES から作り直せるかを確かめる（Issue #127 の手順 4。ローカル CPU、学習なし、v3 を見ない）。

    PYTHONPYCACHEPREFIX=<scratch>/pyc .venv/bin/python scripts/ship/check_align_h.py \
        --align-src <scratch>/n1116n_alignsrc --pool-dir data/ext/v4g/casmi26-v2-pool \
        --casmi-code data/ext/v4g/casmi26-v3-models/code --align-dir outputs/align_train --out <scratch>/check_align_h.json

- 整列のモデルの定義は未マージの PR #110 の `scripts/train/align_train.py` にしかない。`git show` で scratch に出したものを
  `--align-src` で受け取り、`build_model`・`unpack_bits_torch`・`FP_BITS` だけを使う（`align.py` は呼ばない）
- pool（`pool_meta.parquet` 710,701 行）から seed 0 で 1,000 行を選ぶ（`numpy.random.default_rng(0).choice(n, 1000, replace=False)`
  を昇順に並べたもの）
- 測るもの:
  1. 指紋: SMILES → `scripts/engine2/base_fs.fs_fp`（`casmi.chem.raw_fingerprint(smiles)[fp_bits]`、10,226 ビット）→
     `np.packbits`（上位ビットが先、1,279 バイト）が `pool_fp.npy` の行とバイト単位で完全一致する行の割合
  2. キー: SMILES から (i) 素の `InChIKey[:14]`、(ii) 立体を外し互変異性体を正準化してからの `InChIKey[:14]`（0.487 写しの
     セル 21 の `_to_k14` の RDKit の部分）が `pool_meta.key` と一致する割合
  3. h: `model_best.pt` の分子側（`encode_molecule`、評価モード、fp32）で、(a) `pool_fp.npy` の行から、(b) 指紋が一致した行の
     SMILES から作った指紋から、正規化した h を計算し、`h_pool.npy`（fp16、mmap）の同じ行と 1 − cos を比べる。
     1 − cos ≤ 1e-4 の割合、最大値、中央値
- 出力は --out の json だけ（元のフォルダに書かない）

torch・RDKit はモジュールの先頭で import しない（CI は torch を入れないので、純粋な関数だけをテストする）。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

N_ROWS, SEED = 1_000, 0
FP_BITS = 10_226
PACKED_BYTES = (FP_BITS + 7) // 8      # 1,279
COS_TOL = 1e-4


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def sample_rows(n_total: int, n: int = N_ROWS, seed: int = SEED) -> np.ndarray:
    """pool の行番号の標本（重複なし、昇順）。"""
    if n > n_total:
        raise ValueError(f"cannot sample {n} rows from {n_total}")
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def pack_bits(bits01: np.ndarray) -> np.ndarray:
    """0/1 の (nbits,) → packbits（上位ビットが先）。pool_fp.npy と同じ詰め方（`align_train.unpack_bits_np` の逆）。"""
    b = np.asarray(bits01, dtype=np.uint8)
    if b.ndim != 1 or not np.isin(b, (0, 1)).all():
        raise ValueError("bits must be a 1-D 0/1 vector")
    return np.packbits(b, bitorder="big")


def rows_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行ごとのバイト完全一致（(n, bytes) 同士）。"""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return (a == b).all(axis=1)


def one_minus_cos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """行ごとの 1 − cos（float64 で計算。どちらかが 0 ベクトルなら nan）。"""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (x * y).sum(1) / (nx * ny)
    return 1.0 - c


def summarize_dist(d: np.ndarray, tol: float = COS_TOL) -> dict:
    d = np.asarray(d, np.float64)
    ok = np.isfinite(d)
    if not ok.any():
        return {"n": int(len(d)), "n_finite": 0, "frac_le_tol": 0.0, "tol": tol}
    return {"n": int(len(d)), "n_finite": int(ok.sum()), "n_le_tol": int((d[ok] <= tol).sum()),
            "frac_le_tol": float((d[ok] <= tol).sum() / len(d)), "tol": tol,
            "max": float(d[ok].max()), "median": float(np.median(d[ok])), "mean": float(d[ok].mean())}


def first_diff_bits(a_packed: np.ndarray, b_packed: np.ndarray, nbits: int = FP_BITS) -> list[int]:
    """2 つの詰めた指紋で違うビットの位置（原因の手がかり用）。"""
    a = np.unpackbits(np.asarray(a_packed, np.uint8))[:nbits]
    b = np.unpackbits(np.asarray(b_packed, np.uint8))[:nbits]
    return np.nonzero(a != b)[0].tolist()


# ---------------------------------------------------------------- 実行の部分

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_align_module(src_dir: str):
    """scratch に出した align_train.py を読むだけで import する（このリポジトリには入れない）。"""
    p = Path(src_dir) / "align_train.py"
    if not p.is_file():
        raise FileNotFoundError(p)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    return importlib.import_module("align_train")


def keys_for(smiles: list[str]) -> tuple[list[str], list[str]]:
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    taut = rdMolStandardize.TautomerEnumerator()
    plain, canon = [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            plain.append(""); canon.append(""); continue
        plain.append(Chem.MolToInchiKey(m)[:14])
        try:
            m2 = Chem.MolFromSmiles(s)
            Chem.RemoveStereochemistry(m2)
            canon.append(Chem.MolToInchiKey(taut.Canonicalize(m2))[:14])
        except Exception:  # noqa: BLE001  _to_k14 と同じく失敗は空
            canon.append("")
    return plain, canon


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--align-src", required=True, help="align_train.py（PR #110）を置いた scratch のディレクトリ")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--casmi-code", required=True, help="casmi パッケージ（chem.raw_fingerprint）の親ディレクトリ")
    ap.add_argument("--align-dir", required=True, help="model_best.pt・align_config.json・h_pool.npy のあるディレクトリ")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N_ROWS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    sys.dont_write_bytecode = True
    t0 = time.time()
    import pandas as pd
    import rdkit
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine2"))
    import base_fs
    base_fs.set_casmi_code_dir(args.casmi_code)
    AT = load_align_module(args.align_src)
    assert AT.FP_BITS == FP_BITS
    pool, ad = Path(args.pool_dir), Path(args.align_dir)
    meta = pd.read_parquet(pool / "pool_meta.parquet", columns=["key", "smiles"])
    pool_fp = np.load(pool / "pool_fp.npy", mmap_mode="r")
    bits = np.load(pool / "fp_bits.npy")
    h_pool = np.load(ad / "h_pool.npy", mmap_mode="r")
    assert pool_fp.shape == (len(meta), PACKED_BYTES) and len(bits) == FP_BITS, (pool_fp.shape, len(bits))
    assert h_pool.shape[0] == len(meta), h_pool.shape
    rows = sample_rows(len(meta), args.n, args.seed)
    smiles = meta.smiles.values[rows].tolist()
    ref_fp = np.asarray(pool_fp[rows])

    # 1. 指紋
    t1 = time.time()
    made, n_fail = np.zeros_like(ref_fp), 0
    for j, s in enumerate(smiles):
        f = base_fs.fs_fp(s, bits)
        if f is None:
            n_fail += 1
            continue
        made[j] = pack_bits(f)
    fp_eq = rows_equal(made, ref_fp)
    diffs = [{"row": int(rows[j]), "smiles": smiles[j], "n_bits_diff": len(first_diff_bits(made[j], ref_fp[j])),
              "bits": first_diff_bits(made[j], ref_fp[j])[:10]} for j in np.nonzero(~fp_eq)[0][:20]]
    t_fp = time.time() - t1

    # 2. キー
    plain, canon = keys_for(smiles)
    ref_key = meta.key.values[rows]
    key_plain = np.array(plain) == ref_key
    key_canon = np.array(canon) == ref_key

    # 3. h
    cfg = json.loads((ad / "align_config.json").read_text())
    model = AT.build_model(len(cfg["adduct_vocab"]) + 1, cfg["adduct_signs"], d_ms=cfg["d_ms"], d_mol=cfg["d_mol"])
    model.load_state_dict(torch.load(ad / "model_best.pt", map_location="cpu"), strict=True)
    model.eval()
    t3 = time.time()

    def h_of(packed: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            x = AT.unpack_bits_torch(torch.from_numpy(np.ascontiguousarray(packed)), FP_BITS, torch.float32)
            return torch.nn.functional.normalize(model.encode_molecule(x).float(), dim=-1).numpy()

    ref_h = np.asarray(h_pool[rows], np.float32)
    h_from_pool = np.concatenate([h_of(ref_fp[i:i + 250]) for i in range(0, len(rows), 250)])
    d_pool = one_minus_cos(h_from_pool, ref_h)
    m = np.nonzero(fp_eq)[0]
    h_from_smiles = np.concatenate([h_of(made[m][i:i + 250]) for i in range(0, len(m), 250)]) if len(m) else np.zeros((0, 512))
    d_smiles = one_minus_cos(h_from_smiles, ref_h[m]) if len(m) else np.zeros(0)
    t_h = time.time() - t3

    rep = {
        "issue": 127, "step": 4, "n_rows": int(len(rows)), "seed": args.seed,
        "rows_sha256": hashlib.sha256(rows.astype(np.int64).tobytes()).hexdigest(),
        "rdkit": rdkit.__version__, "torch": torch.__version__, "numpy": np.__version__,
        "inputs": {"pool_meta_sha256": sha256(pool / "pool_meta.parquet"), "fp_bits_sha256": sha256(pool / "fp_bits.npy"),
                   "model_best_sha256": sha256(ad / "model_best.pt"), "align_src_sha256": sha256(Path(args.align_src) / "align_train.py"),
                   "best_step": cfg.get("best_step")},
        "fingerprint": {"n_equal": int(fp_eq.sum()), "frac_equal": float(fp_eq.mean()), "n_rdkit_fail": n_fail,
                        "mismatch_examples": diffs, "seconds": round(t_fp, 1)},
        "key": {"plain_inchikey14": {"n_equal": int(key_plain.sum()), "frac_equal": float(key_plain.mean())},
                "stereo_removed_tautomer_canonical": {"n_equal": int(key_canon.sum()), "frac_equal": float(key_canon.mean())},
                "plain_mismatch_examples": [{"row": int(rows[j]), "smiles": smiles[j], "made": plain[j], "pool": str(ref_key[j])}
                                            for j in np.nonzero(~key_plain)[0][:10]]},
        "h_from_pool_fp": summarize_dist(d_pool),
        "h_from_smiles_fp_on_fp_equal_rows": summarize_dist(d_smiles),
        "h_seconds": round(t_h, 1), "seconds": round(time.time() - t0, 1),
    }
    Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("n_rows", "rdkit", "fingerprint", "key", "h_from_pool_fp",
                                          "h_from_smiles_fp_on_fp_equal_rows", "seconds")}, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
