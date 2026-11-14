"""Engine 2 の 6,930 ビット（`fp_bits.npy`）を公開の選び方で作り直して、配布ファイルと照合する（#84）。

出典とライセンス:
- `select_fingerprints` の選び方（頻度が [low, high] のビットを元の列順で残す、low=.005・high=.995）は
  `prvsiyan/casmi26-generation-code` の `src/fp-models-v2/preparation/prepare_legacy_exports.py` の
  `select_fingerprints(packed, metadata, low=.005, high=.995)` の写し（コードは Apache-2.0、Copyright は原作者）。
  元の関数は packbits の全行を unpack して `full.mean(0)` で頻度を出す。ここでは 27.6 万行 × 10,407 列を一度に
  展開しない（約 2.9 GB）ように、整数のビット数 / 構造数で頻度を出してから同じ比較を当てる（整数の和は float64 で
  厳密なので `mean` と同じ値）。元の関数の入力の形の検査（metadata の型・重複）は、CLI 側の重複除去と長さの
  assert に置き換えた。定数と比較の向き（>= low、<= high）は変えていない。
- `fp10407` の指紋（ECFP4 4096 ＋ ECFP6 4096 ＋ RDKit maxPath6 2048 ＋ MACCS 167 = 10,407 ビット、この順）は
  Engine 2 の推論コード `scripts/engine2/eng/casmi_engine.py` の `_fp_init`・`fp_and_mass` に合わせたもの
  （同じ生成器・同じ引数・同じ連結順、ビット選択の前）。prvsiyan の元の指紋生成スクリプト（`build_fps.py`）は
  手元に無いので、それの写しではない。`_fp_init` は `find("fp_bits.npy")` でファイルを探すので import して呼ばない。
  `fp_and_mass` との一致は tests/test_select_fp_bits.py (b) で確かめる。

使い方:
  .venv/bin/python scripts/engine2/select_fp_bits.py --structs <v2-pool>/train_structs.parquet \
      --ref <coconut-casmi26-candidates>/fp_bits.npy --out outputs/engine2_build/fp_bits_6930_check.json \
      [--save-bits <path>/fp_bits.npy]
入力の構造は inchikey14 で重複を除き（最初の行を残す）、SMILES が読めない・指紋が作れない構造は数えない。
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

NBITS = 10407
BLOCKS = (("ECFP4", 4096), ("ECFP6", 4096), ("RDKit", 2048), ("MACCS", 167))
LOW, HIGH = .005, .995  # prepare_legacy_exports.select_fingerprints の既定値（変えない）

_G = {}


def _gens():
    if not _G:
        from rdkit import RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        _G["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        _G["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        _G["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
    return _G


def fp10407(smi):
    """選択前の 10,407 ビット（uint8 の 0/1）。SMILES が読めない・指紋が作れないときは None（fp_and_mass と同じ扱い）。"""
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    g = _gens()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return np.concatenate([g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                               g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                               g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                               np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
    except Exception:
        return None


def select_fingerprints(freq, low=LOW, high=HIGH):
    """頻度が low 以上 high 以下のビットの添字（昇順）。prepare_legacy_exports.py の比較の写し。"""
    freq = np.asarray(freq, dtype=np.float64)
    return np.where((freq >= low) & (freq <= high))[0]


def _count(smis):
    cnt = np.zeros(NBITS, np.int64)
    n = 0
    for s in smis:
        fp = fp10407(s)
        if fp is None:
            continue
        cnt += fp
        n += 1
    return cnt, n


def bit_counts(smiles, workers=8, chunk=5000):
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    if workers <= 1:
        res = [_count(c) for c in chunks]
    else:
        with Pool(workers) as p:
            res = p.map(_count, chunks)
    cnt = np.zeros(NBITS, np.int64)
    n = 0
    for c, k in res:
        cnt += c
        n += k
    return cnt, n


def breakdown(bits):
    out, start = {}, 0
    for name, size in BLOCKS:
        out[name] = int(((bits >= start) & (bits < start + size)).sum())
        start += size
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--structs", required=True, help="v2-pool の train_structs.parquet")
    ap.add_argument("--ref", required=True, help="照合する配布の fp_bits.npy")
    ap.add_argument("--out", required=True, help="結果の json")
    ap.add_argument("--save-bits", default=None, help="作ったビット列を .npy で保存（任意）")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    import pyarrow.parquet as pq
    import rdkit
    t0 = time.time()
    df = pq.read_table(a.structs, columns=["inchikey14", "smiles"]).to_pandas()
    n_rows = len(df)
    d = df.dropna(subset=["inchikey14", "smiles"]).drop_duplicates("inchikey14")
    smiles = d["smiles"].tolist()
    cnt, n = bit_counts(smiles, workers=a.workers)
    assert n > 0
    freq = cnt / n
    bits = select_fingerprints(freq)
    ref = np.load(a.ref)
    equal = bool(np.array_equal(bits, ref))
    sb, sr = set(bits.tolist()), set(ref.tolist())
    missing, extra = sorted(sr - sb), sorted(sb - sr)
    out_bits = np.setdiff1d(np.arange(NBITS), bits)
    lo_side = out_bits[freq[out_bits] < LOW]
    hi_side = out_bits[freq[out_bits] > HIGH]
    rep = {
        "issue": 84,
        "structs": os.path.abspath(a.structs), "structs_sha256": sha256(a.structs),
        "ref": os.path.abspath(a.ref), "ref_sha256": sha256(a.ref),
        "rows": int(n_rows), "unique_inchikey14": int(len(d)), "structures_fingerprinted": int(n),
        "rdkit_version": rdkit.__version__, "numpy_version": np.__version__,
        "low": LOW, "high": HIGH, "nbits_full": NBITS,
        "n_bits": int(len(bits)), "n_bits_ref": int(len(ref)), "breakdown": breakdown(bits),
        "breakdown_ref": breakdown(np.asarray(ref)),
        "array_equal": equal,
        "missing_vs_ref": [{"bit": int(b), "freq": float(freq[b])} for b in missing],
        "extra_vs_ref": [{"bit": int(b), "freq": float(freq[b])} for b in extra],
        "selected_freq_min": float(freq[bits].min()), "selected_freq_max": float(freq[bits].max()),
        "excluded_freq_max_below_low": float(freq[lo_side].max()) if len(lo_side) else None,
        "excluded_freq_min_above_high": float(freq[hi_side].min()) if len(hi_side) else None,
        "n_excluded_above_high": int(len(hi_side)),
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    if a.save_bits:
        np.save(a.save_bits, bits)
    print(json.dumps({k: rep[k] for k in ("n_bits", "breakdown", "array_equal", "structures_fingerprinted",
                                          "rdkit_version", "selected_freq_min", "excluded_freq_max_below_low",
                                          "seconds")}, ensure_ascii=False))
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
