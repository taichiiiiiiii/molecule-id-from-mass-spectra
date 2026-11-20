"""#152: train.parquet を row group ごとに読んで、元の `prepare_legacy_exports.py flat` と同じ 4 ファイルを書く。

派生元: cubres/enveda-casmi-2026 commit 477897993f006ebd86fbbc4809a732f282781b6b の
reproduction/legacy-v2/public_source/speclib.py（load_flat）と prepare_legacy_exports.py（export_flat）、Apache-2.0
（upstream/LICENSE-SOURCE・upstream/NOTICE）。変更点: 一度に読む read_table を row group ごとの読み込みに替えた
ことと、2 列の表の書き出しを足したこと。変換の式は変えていない。

元（upstream/prepare_legacy_exports.py の export_flat → upstream/speclib.py の load_flat）は `pq.read_table` で
train.parquet を一度に読む。共用の Mac ではメモリが大きすぎるので、`pq.ParquetFile(p, buffer_size=1<<20,
pre_buffer=False)` の row group ごとに同じ変換をする:
- ms2_mzs / ms2_normalized_intensities の値を float32 に丸めて連結、offset は int64 で row group をまたいで足す
- メタ列（load_flat の既定 9 列）は float の列なら `to_numpy`（float64）、それ以外は string に cast して
  `to_pylist` を object 配列に（None は None のまま）
出力は元と同じ名前と形: lib_off.npy / lib_mz.npy / lib_in.npy（allow_pickle=False）、lib_meta.pkl（protocol 4）。
同じ出力になることは tests/test_engine2_fp_retrain.py で元の export_flat と比べて確かめる。

もう 1 つ、upstream/build_dataset.py が `pq.read_table('data/train.parquet', columns=[...])` で読む 2 列
（precursor_error_ppm・collision_energy_ev）だけを、同じく row group ごとに `--err-parquet` に書き出す
（元の build_dataset.py は変えずに、作業場所の data/train.parquet としてこの 2 列の表を置く）。

使い方:
  python prep_flat_stream.py --train <train.parquet> --output <WORK>/data --err-parquet <WORK>/data/train.parquet
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

META_COLS = ['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula',
             'adduct', 'precursor_mz', 'instrument_type', 'ionization_mode', 'collision_energy_orig']
ERR_COLS = ['precursor_error_ppm', 'collision_energy_ev']


def _open(path):
    return pq.ParquetFile(str(path), buffer_size=1 << 20, pre_buffer=False)


def load_flat_stream(path, meta_cols=None):
    """speclib.load_flat と同じ dict を、row group ごとに読んで作る。"""
    meta_cols = meta_cols or META_COLS
    f = _open(path)
    offs, mzs, ins = [], [], []
    meta = {c: [] for c in meta_cols}
    floating = {}
    base = 0
    for g in range(f.num_row_groups):
        t = f.read_row_group(g, columns=meta_cols + ['ms2_mzs', 'ms2_normalized_intensities'])
        mzc = t.column('ms2_mzs').combine_chunks()
        itc = t.column('ms2_normalized_intensities').combine_chunks()
        o = mzc.offsets.to_numpy().astype(np.int64)
        # 元は全体を 1 つの配列にしたときの offsets。row group の先頭が 0 から始まらない場合
        # （スライスされた配列）にも合うよう、値の配列は offsets の範囲で切り出す
        a, b = int(o[0]), int(o[-1])
        vm = mzc.values.slice(a, b - a).to_numpy(zero_copy_only=False).astype(np.float32)
        vi_all = itc.values
        io = itc.offsets.to_numpy().astype(np.int64)
        if not np.array_equal(io - io[0], o - o[0]):
            raise ValueError('ms2_mzs と ms2_normalized_intensities の長さが行ごとに一致しない')
        vi = vi_all.slice(int(io[0]), int(io[-1] - io[0])).to_numpy(zero_copy_only=False).astype(np.float32)
        offs.append((o - a + base)[:-1] if g < f.num_row_groups - 1 else (o - a + base))
        base += b - a
        mzs.append(vm); ins.append(vi)
        for c in meta_cols:
            col = t.column(c)
            if pa.types.is_floating(col.type):
                floating[c] = True
                meta[c].append(col.to_numpy(zero_copy_only=False))
            else:
                floating[c] = False
                meta[c].extend(col.cast(pa.string()).to_pylist())
        del t, mzc, itc
    if f.num_row_groups == 0:
        offs = [np.zeros(1, np.int64)]
    d = {'off': np.concatenate(offs), 'allmz': np.concatenate(mzs) if mzs else np.zeros(0, np.float32),
         'allin': np.concatenate(ins) if ins else np.zeros(0, np.float32)}
    for c in meta_cols:
        d[c] = np.concatenate(meta[c]) if floating.get(c) else np.asarray(meta[c], dtype=object)
    return d


def export_flat_stream(train_path, output_dir):
    """prepare_legacy_exports.export_flat と同じファイルを書く（load_flat の代わりに load_flat_stream）。"""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    library = load_flat_stream(train_path)
    for key, name in [('off', 'lib_off.npy'), ('allmz', 'lib_mz.npy'), ('allin', 'lib_in.npy')]:
        with (destination / name).open('xb') as stream:
            np.save(stream, library[key], allow_pickle=False)
    metadata = {key: value for key, value in library.items() if key not in ('off', 'allmz', 'allin')}
    with (destination / 'lib_meta.pkl').open('xb') as stream:
        pickle.dump(metadata, stream, protocol=4)
    return {'spectra': int(len(library['off']) - 1), 'peaks': int(len(library['allmz']))}


def export_err_columns(train_path, out_path):
    """build_dataset.py が読む 2 列だけを row group ごとに書き出す（行の順は元のまま）。"""
    f = _open(train_path)
    out_path = Path(out_path)
    if out_path.exists():
        raise FileExistsError(out_path)
    w = None
    n = 0
    try:
        for g in range(f.num_row_groups):
            t = f.read_row_group(g, columns=ERR_COLS)
            if w is None:
                w = pq.ParquetWriter(str(out_path), t.schema)
            w.write_table(t)
            n += t.num_rows
    finally:
        if w is not None:
            w.close()
    return {'rows': n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--err-parquet', type=Path, required=True)
    a = ap.parse_args()
    t0 = time.time()
    r = export_flat_stream(a.train, a.output)
    r['err'] = export_err_columns(a.train, a.err_parquet)
    r['seconds'] = round(time.time() - t0, 1)
    print(json.dumps(r, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
