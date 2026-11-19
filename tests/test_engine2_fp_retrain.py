"""Issue #152: Engine 2 の FP モデル（S2）の作り直しのテスト。

- upstream/ の写しが cubres/enveda-casmi-2026 commit 4778979 のバイトのまま（sha256）
- prep_flat_stream.py（row group ごとの読み込み）が元の prepare_legacy_exports.export_flat と同じ 4 ファイルを書く、
  2 列の表が元の read_table の 2 列と同じ（CI でも走る）
- train_s2_resumable.py（torch があるときだけ。CI では飛ぶ）: CPU の合成データで「12 step 続けて」と
  「6 step で止めて再開して 12 step まで」の重み・best のチェックポイントがビット一致、引数が違う state からは再開しない
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / 'scripts/engine2/fp_retrain'
UP = FR / 'upstream'
PINS = {
    'build_coconut.py': '85ee0524d03fbd880e07a8637ea740fbb59c4cc280f9b56c2b633939c745b086',
    'build_dataset.py': 'ec90d3d9ef3ec9b1e45d2dc9e2e766439ef7431cce8f8388ccb96b0be96a517b',
    'build_fps.py': '6cd11e3910dc935ecda4a02f2f406c2bafb4c66e356f15dd88f635d5b993078c',
    'build_pool.py': '3be89299396ea87c25e38c60665c6c165756cba083f98e3d52972d3eb0c627c4',
    'fpmodel.py': '69f2656778a08233e22235bf4853f0b3d2420b431390a32d6c8ed453cde85c76',
    'prepare_legacy_exports.py': '8ccce8764cbf58d0ced0a361d2212b134e5332db653000687bc1853a5c76b544',
    'speclib.py': '27e149dd8f705389a969f1b9232b7b2451c7f3189f5586f5a42160864ff403eb',
    'train_fp3.py': '43bb618dd512def4ccb1119f813aabce2b9f9bcbfa681c9d2db5adfa05e66840',
    'LICENSE-SOURCE': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
}


def test_upstream_pins():
    for name, sha in PINS.items():
        assert hashlib.sha256((UP / name).read_bytes()).hexdigest() == sha, name


def _toy_parquet(path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(23):
        n = int(rng.integers(0, 6))
        rows.append({
            'ingest_lib': ['gnps', 'riken', None][i % 3],
            'inchikey14': f'KEY{i % 7:011d}' if i != 5 else None,
            'normalized_smiles': 'CCO' if i % 2 else 'c1ccccc1',
            'molecular_formula': 'C2H6O',
            'adduct': '[M+H]+' if i % 4 else '[M-H]-',
            'precursor_mz': float(rng.uniform(100, 900)) if i != 3 else None,
            'precursor_error_ppm': float(rng.normal()) if i % 5 else None,
            'instrument_type': 'Orbitrap',
            'ionization_mode': 'positive' if i % 4 else 'negative',
            'collision_energy_orig': None if i % 6 == 0 else '20',
            'collision_energy_ev': [20.0, 40.0] if i % 2 else ([] if i % 4 == 0 else None),
            'ms2_mzs': [float(x) for x in np.sort(rng.uniform(50, 500, n))] if i != 7 else None,
            'ms2_normalized_intensities': [float(x) for x in rng.uniform(0, 1, n)] if i != 7 else None,
        })
    t = pa.Table.from_pylist(rows, schema=pa.schema([
        ('ingest_lib', pa.string()), ('inchikey14', pa.string()), ('normalized_smiles', pa.string()),
        ('molecular_formula', pa.string()), ('adduct', pa.string()), ('precursor_mz', pa.float64()),
        ('precursor_error_ppm', pa.float64()), ('instrument_type', pa.string()), ('ionization_mode', pa.string()),
        ('collision_energy_orig', pa.string()), ('collision_energy_ev', pa.list_(pa.float64())),
        ('ms2_mzs', pa.list_(pa.float64())), ('ms2_normalized_intensities', pa.list_(pa.float64()))]))
    pq.write_table(t, path, row_group_size=5)   # row group 5 つ
    assert pq.ParquetFile(path).num_row_groups == 5


def _obj_equal(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.dtype == object:
        assert list(a) == list(b)
    else:
        np.testing.assert_array_equal(a, b)


def test_prep_flat_stream_matches_export_flat(tmp_path, monkeypatch):
    src = tmp_path / 'train.parquet'
    _toy_parquet(src)
    monkeypatch.syspath_prepend(str(UP))
    monkeypatch.syspath_prepend(str(FR))
    import prepare_legacy_exports as ple
    import prep_flat_stream as pfs
    r1 = ple.export_flat(src, tmp_path / 'orig')
    r2 = pfs.export_flat_stream(src, tmp_path / 'mine')
    assert r1 == r2
    for name in ('lib_off.npy', 'lib_mz.npy', 'lib_in.npy'):
        x = np.load(tmp_path / 'orig' / name); y = np.load(tmp_path / 'mine' / name)
        assert x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)
    m1 = pickle.load(open(tmp_path / 'orig/lib_meta.pkl', 'rb'))
    m2 = pickle.load(open(tmp_path / 'mine/lib_meta.pkl', 'rb'))
    assert list(m1) == list(m2)
    for k in m1:
        _obj_equal(m1[k], m2[k])
    # build_dataset.py が読む 2 列
    pfs.export_err_columns(src, tmp_path / 'err.parquet')
    e1 = pq.read_table(src, columns=['precursor_error_ppm', 'collision_energy_ev'])
    e2 = pq.read_table(tmp_path / 'err.parquet', columns=['precursor_error_ppm', 'collision_energy_ev'])
    assert e1.equals(e2)


# ---------------------------------------------------------------- 学習（torch があるときだけ）

def _toy_train_inputs(d):
    rng = np.random.default_rng(1)
    nb = 24; npool = 60; nstr = 20; nspec = 80
    pool_fp = rng.integers(0, 2, (npool, nb)).astype(np.uint8)
    mass = np.sort(rng.uniform(100, 300, npool))
    mass[10:20] = mass[10]                                   # 同じ質量の候補（異性体の窓）
    keys = [f'P{i:04d}' for i in range(npool)]
    np.save(d / 'pool_fp.npy', np.packbits(pool_fp, axis=1)); np.save(d / 'pool_mass.npy', mass)
    pickle.dump({'keys': np.array(keys, dtype=object)}, open(d / 'pool_meta.pkl', 'wb'))
    (d / 'up').mkdir()
    pickle.dump({'nbits': nb, 'keys': keys[5:5 + nstr]}, open(d / 'up/fp_meta.pkl', 'wb'))
    lens = rng.integers(3, 9, nspec); off = np.zeros(nspec + 1, np.int64); np.cumsum(lens, out=off[1:])
    np.savez(d / 'spec.npz', off=off, mz=rng.uniform(50, 300, off[-1]).astype(np.float32),
             it=rng.uniform(0, 1, off[-1]).astype(np.float32), prec=rng.uniform(150, 320, nspec).astype(np.float32),
             ad=rng.integers(0, 5, nspec).astype(np.int16), ins=rng.integers(0, 5, nspec).astype(np.int8),
             ce=np.where(rng.random(nspec) < .3, -1, 30).astype(np.float32),
             mode=np.ones(nspec, np.float32), six=rng.integers(0, nstr, nspec).astype(np.int32),
             is_val=(np.arange(nspec) % 5 == 0), src=np.arange(nspec, dtype=np.int64))


def _args(d, out, state, steps=12, extra=()):
    return ['--spec', str(d / 'spec.npz'), '--fpdir', str(d / 'up'), '--pooldir', str(d), '--steps', str(steps),
            '--bs', '8', '--lr', '2e-4', '--K', '7', '--lam', '1.0', '--seed', '202', '--d', '32', '--layers', '1',
            '--val_every', '4', '--warm', '3', '--device', 'cpu', '--ckpt_every_min', '1000',
            '--out', str(out), '--state', str(state), *extra]


def _load_train(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(FR))
    import train_s2_resumable as tr
    return tr


def test_resume_bit_identical(tmp_path, monkeypatch):
    tr = _load_train(monkeypatch)
    import torch
    torch.set_num_threads(1)
    _toy_train_inputs(tmp_path)
    a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir()
    # a: 12 step を続けて（best の出力は b と同じパスにならないよう別のディレクトリ）
    assert tr.main(_args(tmp_path, a / 'best.pt', a / 'state.pt')) == 0
    # b: 6 step で止めて、同じ引数で 12 step まで再開
    assert tr.main(_args(tmp_path, b / 'best.pt', b / 'state.pt', extra=('--stop_at_step', '6'))) == 0
    s6 = torch.load(b / 'state.pt', weights_only=False)
    assert s6['step'] == 6
    assert tr.main(_args(tmp_path, b / 'best.pt', b / 'state.pt')) == 0
    sa = torch.load(a / 'state.pt', weights_only=False); sb = torch.load(b / 'state.pt', weights_only=False)
    assert sa['step'] == sb['step'] == 12 and sa['best_val'] == sb['best_val']
    for k in sa['model']:
        assert torch.equal(sa['model'][k], sb['model'][k]), k
    assert sa['np_rng'] == sb['np_rng']
    ba = torch.load(a / 'best.pt', weights_only=False); bb = torch.load(b / 'best.pt', weights_only=False)
    assert ba['step'] == bb['step']
    for k in ba['model']:
        assert torch.equal(ba['model'][k], bb['model'][k]), k
    # 学習で重みが動いている（何もしないで一致、ではない）
    s0 = torch.load(b / 'state.pt', weights_only=False)
    assert any(not torch.equal(s6['model'][k], s0['model'][k]) for k in s0['model'])


def test_resume_refuses_changed_args(tmp_path, monkeypatch):
    tr = _load_train(monkeypatch)
    _toy_train_inputs(tmp_path)
    assert tr.main(_args(tmp_path, tmp_path / 'best.pt', tmp_path / 'state.pt', extra=('--stop_at_step', '2'))) == 0
    bad = _args(tmp_path, tmp_path / 'best.pt', tmp_path / 'state.pt')
    bad[bad.index('--lr') + 1] = '3e-4'
    with pytest.raises(SystemExit, match='引数が違う'):
        tr.main(bad)


def test_lr_schedule_matches_upstream_formula(monkeypatch):
    tr = _load_train(monkeypatch)
    a = tr.parse(['--state', 'x', '--steps', '24000', '--lr', '2e-4'])
    import math
    for s in (0, 1, 1999, 2000, 2001, 12000, 23999):
        if s < 2000:
            exp = 2e-4 * s / 2000
        else:
            p = (s - 2000) / 22000
            exp = 2e-4 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
        assert tr.lr_at(a, s) == exp


# ---------------------------------------------------------------- 再開ループ（macOS だけ）

def _loop_env(tmp_path, guard_body):
    import os
    guard = tmp_path / 'guard.sh'
    guard.write_text('#!/bin/bash\nLOG="$1"; shift\n' + guard_body + '\n')
    run = tmp_path / 'run'; (run / 'work').mkdir(parents=True)
    (run / 'work' / 'x.bin').write_bytes(b'x')
    sha = tmp_path / 'SHA'
    sha.write_text(hashlib.sha256(b'x').hexdigest() + '  x.bin\n')
    out = tmp_path / 'out'; out.mkdir()
    lock = tmp_path / '.heavy_job.d'; lock.mkdir()
    env = dict(os.environ, MEMGUARD=str(guard), MIN_AVAIL_GB='0', WAIT_STEP_S='1', WAIT_MAX_S='2',
               SWAP_CMD="echo 'total = 4096.00M  used = 100.00M  free = 3996.00M'", HEAVY_LOCK=str(lock),
               MAX_RESTARTS='1', SWAP_CHECK_S='1')
    return env, run, sha, out, lock


def _run_loop(env, run, sha, out, lock):
    import subprocess
    # 印の owner にループの pid を書くため、bash -c の中で $$ を使う（exec でループに置き換える）
    cmd = (f'echo "card=test pid=$$ role=loop" > {lock}/owner; '
           f'exec bash {FR / "e2fp_loop.sh"} {out} {run} {sha} -- /bin/true')
    return subprocess.run(['bash', '-c', cmd], env=env, timeout=60)


@pytest.mark.skipif(sys.platform != 'darwin', reason='vm_stat・memory_pressure は macOS だけ')
def test_loop_stops_on_memguard_rc99_and_releases_lock(tmp_path):
    env, run, sha, out, lock = _loop_env(tmp_path, 'echo killed > "$LOG"; exit 99')
    r = _run_loop(env, run, sha, out, lock)
    assert r.returncode == 99
    log = (out / 'loop.log').read_text()
    assert 'decision=stop memguard_rc99' in log and 'round 2' not in log
    assert not lock.exists()


@pytest.mark.skipif(sys.platform != 'darwin', reason='vm_stat・memory_pressure は macOS だけ')
def test_loop_resumes_then_finishes(tmp_path):
    body = ('n=$(ls "$(dirname "$LOG")" | grep -c "^train.log"); '
            'if [ "$n" -le 0 ]; then echo "SIGTERM: state saved" > "$LOG"; exit 143; fi; '
            'echo ALL_DONE > "$LOG"; exit 0')
    env, run, sha, out, lock = _loop_env(tmp_path, body)
    r = _run_loop(env, run, sha, out, lock)
    assert r.returncode == 0
    log = (out / 'loop.log').read_text()
    assert 'round 1 rc=143 decision=resume rc_143' in log and 'round 2 rc=0 decision=done' in log
    assert not lock.exists()


# ---------------------------------------------------------------- speed2: PFP を packbits のまま持つ

def _load_packed(monkeypatch):
    monkeypatch.syspath_prepend(str(FR))
    import packed_pfp
    return packed_pfp


@pytest.mark.parametrize('nbits', [24, 21, 18, 1])   # 8 の倍数と、そうでないもの（18 は 6,930 と同じ余り 2）
def test_packed_rows_match_unpacked_numpy(monkeypatch, nbits):
    """(i) torch 無しでも走る部分: packed からの取り出し = 上流の unpack 済み配列の行（バイト一致）。"""
    pp = _load_packed(monkeypatch)
    rng = np.random.default_rng(nbits)
    bits = rng.integers(0, 2, (50, nbits)).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    ref = np.unpackbits(packed, axis=1)[:, :nbits]          # 上流 20 行の式
    assert np.array_equal(ref, bits)
    P = pp.PackedPFP(packed, nbits)
    assert P.shape == ref.shape and len(P) == 50
    for idx in (np.arange(50), rng.integers(0, 50, 17), rng.integers(0, 50, (6, 5)),
                np.array([3, 3, 0, 49], dtype=np.int32), np.zeros((0,), np.int64)):
        got = P.rows_np(idx)
        exp = ref[idx]
        assert got.dtype == exp.dtype == np.uint8 and got.shape == exp.shape
        assert got.tobytes() == exp.tobytes()
        assert got.flags['C_CONTIGUOUS']


def test_packed_rejects_bad_input(monkeypatch):
    pp = _load_packed(monkeypatch)
    with pytest.raises(ValueError):
        pp.PackedPFP(np.zeros((3, 2), np.int64), 10)
    with pytest.raises(ValueError):
        pp.PackedPFP(np.zeros((3, 2), np.uint8), 17)
    with pytest.raises(TypeError):
        pp.PackedPFP(np.zeros((3, 2), np.uint8), 16).rows_np(np.array([0.0]))


def test_packed_torch_getitem_matches_upstream_tensor(monkeypatch):
    """(i) torch の部分: PFP[torch の idx] = 上流の torch.from_numpy(ascontiguousarray(unpacked))[idx]。"""
    torch = pytest.importorskip('torch')
    pp = _load_packed(monkeypatch)
    rng = np.random.default_rng(7)
    for nbits in (24, 21, 18):
        packed = np.packbits(rng.integers(0, 2, (40, nbits)).astype(np.uint8), axis=1)
        up = torch.from_numpy(np.ascontiguousarray(np.unpackbits(packed, axis=1)[:, :nbits]))   # 上流 20・28 行
        P = pp.PackedPFP(packed, nbits)
        assert P.device == up.device
        for idx in (torch.as_tensor(rng.integers(0, 40, 9)), torch.as_tensor(rng.integers(0, 40, (4, 8)))):
            g = P[idx]; e = up[idx]
            assert g.dtype == e.dtype == torch.uint8 and g.shape == e.shape and torch.equal(g, e)
            assert torch.equal(g.float(), e.float())


def test_packed_data_matches_upstream_data(tmp_path, monkeypatch):
    """PackedData の PFP 以外の属性が上流 Data と同じ、PFP の全行も同じ、batch の出力も同じ。"""
    tr = _load_train(monkeypatch)
    import torch
    _toy_train_inputs(tmp_path)
    A = tr.train_fp3.Data(str(tmp_path / 'spec.npz'), str(tmp_path / 'up'), str(tmp_path), device='cpu')
    B = tr.PackedData(str(tmp_path / 'spec.npz'), str(tmp_path / 'up'), str(tmp_path), device='cpu')
    assert set(vars(A)) == set(vars(B))
    for k, v in vars(A).items():
        w = getattr(B, k)
        if k == 'PFP':
            allrows = torch.arange(len(B.pmass))
            assert torch.equal(v[allrows], w[allrows]) and v.dtype == w[allrows].dtype
        elif isinstance(v, np.ndarray):
            assert v.dtype == w.dtype and v.shape == w.shape and np.array_equal(v, w), k
        else:
            assert v == w if not isinstance(v, (list, tuple)) else list(v) == list(w), k
    ids = B.tr[:6]
    ia, ya, ca = A.batch(ids, K=5, rng=np.random.default_rng(3), aug=True)
    ib, yb, cb = B.batch(ids, K=5, rng=np.random.default_rng(3), aug=True)
    for x, y in zip(ia, ib):
        assert torch.equal(x, y)
    assert torch.equal(ya, yb) and torch.equal(ca, cb)
    assert torch.equal(A.PFP[ca], B.PFP[cb])


def test_packed_training_bit_identical_to_upstream(tmp_path, monkeypatch):
    """(ii) CPU の合成データで --pfp upstream と --pfp packed の 12 step の損失（毎 step）・重み・best・乱数が一致。"""
    tr = _load_train(monkeypatch)
    import torch
    torch.set_num_threads(1)
    _toy_train_inputs(tmp_path)
    realF = tr.F
    rec = {}

    class Shim:
        def __init__(self, key):
            self.key = key

        def __getattr__(self, name):
            return getattr(realF, name)

        def binary_cross_entropy_with_logits(self, *a, **k):
            v = realF.binary_cross_entropy_with_logits(*a, **k); rec[self.key].append(('bce', v.detach().clone())); return v

        def cross_entropy(self, *a, **k):
            v = realF.cross_entropy(*a, **k); rec[self.key].append(('ce', v.detach().clone())); return v

    out = {}
    for mode in ('upstream', 'packed'):
        d = tmp_path / mode; d.mkdir()
        rec[mode] = []
        monkeypatch.setattr(tr, 'F', Shim(mode))
        assert tr.main(_args(tmp_path, d / 'best.pt', d / 'state.pt', extra=('--pfp', mode))) == 0
        out[mode] = (torch.load(d / 'state.pt', weights_only=False), torch.load(d / 'best.pt', weights_only=False))
    monkeypatch.setattr(tr, 'F', realF)
    ru, rp = rec['upstream'], rec['packed']
    assert len(ru) == len(rp) and len(ru) >= 2 * 12            # 学習 12 step × 2 ＋ 評価
    for (na, va), (nb, vb) in zip(ru, rp):
        assert na == nb and torch.equal(va, vb)
    (su, bu), (sp, bp) = out['upstream'], out['packed']
    assert su['step'] == sp['step'] == 12 and su['best_val'] == sp['best_val']
    assert su['args']['pfp'] == 'upstream' and sp['args']['pfp'] == 'packed'
    assert su['np_rng'] == sp['np_rng'] and torch.equal(su['torch_rng'], sp['torch_rng'])
    for k in su['model']:
        assert torch.equal(su['model'][k], sp['model'][k]), k
    assert bu['step'] == bp['step']
    for k in bu['model']:
        assert torch.equal(bu['model'][k], bp['model'][k]), k
    # 違う持ち方の state からは再開しない
    with pytest.raises(SystemExit, match='引数が違う'):
        tr.main(_args(tmp_path, tmp_path / 'packed' / 'best.pt', tmp_path / 'packed' / 'state.pt',
                      steps=12, extra=('--pfp', 'upstream')))
