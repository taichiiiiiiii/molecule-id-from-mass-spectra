"""#152 speed2: 実データの pool_fp.npy で、PackedPFP の行の取り出しが上流の unpack 済みの行とバイト一致するかを数える。

使い方: python check_packed_rows.py <work dir（pool_fp.npy と up_train/fp_meta.pkl がある）> [--n 10000] [--seed 0] [--out x.json]

- 無作為に n 行（numpy default_rng(seed).choice(N, n, replace=False)）を選ぶ
- 比べる側（独立の経路、全体は unpack しない）: 各行 i について
    (1) 上流 20 行の式を 1 行だけに当てたもの np.unpackbits(packed[i:i+1], axis=1)[:, :nbits]
    (2) ビット演算 (packed[i, j // 8] >> (7 - j % 8)) & 1（j = 0..nbits-1、np.unpackbits の既定 bitorder='big'）
- 調べる側: PackedPFP(np.load(pool_fp.npy), nbits).rows_np(idx) の各行（1 次元の idx でまとめて取り出す）
  と、2 次元の idx（n を (n/8, 8) に並べ替え）で取り出したものも同じかを確かめる
- 一致の行数を出す。受け入れ（Issue #152 の事前登録）= n/n
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packed_pfp import PackedPFP  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('work')
    ap.add_argument('--n', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='')
    a = ap.parse_args(argv)
    w = Path(a.work)
    t0 = time.time()
    nbits = pickle.load(open(w / 'up_train/fp_meta.pkl', 'rb'))['nbits']
    mm = np.load(w / 'pool_fp.npy', mmap_mode='r')                  # 比べる側は memmap から 1 行ずつ
    P = PackedPFP(np.load(w / 'pool_fp.npy'), nbits)               # 調べる側（学習と同じ読み方）
    N = mm.shape[0]
    idx = np.random.default_rng(a.seed).choice(N, size=a.n, replace=False)
    got = P.rows_np(idx)
    got2 = P.rows_np(idx.reshape(-1, 8)) if a.n % 8 == 0 else None
    j = np.arange(nbits)
    ok1 = ok2 = ok3 = 0
    for r, i in enumerate(idx):
        row = np.asarray(mm[i:i + 1])
        ref1 = np.unpackbits(row, axis=1)[:, :nbits][0]
        ref2 = ((row[0, j // 8] >> (7 - j % 8)) & 1).astype(np.uint8)
        ok1 += int(got[r].tobytes() == ref1.tobytes())
        ok2 += int(got[r].tobytes() == ref2.tobytes())
        if got2 is not None:
            ok3 += int(got2.reshape(-1, nbits)[r].tobytes() == ref1.tobytes())
    res = {'work': str(w), 'pool_rows': int(N), 'packed_cols': int(mm.shape[1]), 'nbits': int(nbits),
           'n': int(a.n), 'seed': a.seed, 'dtype': str(got.dtype), 'shape': list(got.shape),
           'match_upstream_formula_rowwise': ok1, 'match_bitops': ok2,
           'match_2d_index': ok3 if got2 is not None else None,
           'bits_set_mean_per_row': float(got.sum(1).mean()), 'seconds': round(time.time() - t0, 2)}
    res['pass'] = (ok1 == ok2 == a.n) and (got2 is None or ok3 == a.n)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + '\n')
    return 0 if res['pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
