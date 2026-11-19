"""Issue #167: Engine 2 の FP モデル（merged_m1、train_fp4.py）の作り直しのテスト。

- upstream/train_fp4.py の写しが cubres/enveda-casmi-2026 commit 4778979 のバイトのまま（sha256）
- train_m1_resumable.py（torch があるときだけ。CI では飛ぶ）、CPU の合成データで:
  (i) --pfp upstream（上流 train_fp4.Data のまま）と --pfp packed（PackedData4）で 12 step の損失（毎 step）・重み・
      best・乱数がビット一致。merge_p 0.6 で merge の枝（上流 Data._merged）を実際に通ったことを数える
  (ii) 「12 step 続けて」と「6 step で止めて再開して 12 step まで」で重み・best・乱数（numpy・torch）・step・lr がビット一致
  ほかに: PackedData4 の PFP 以外の属性（構造ごとのまとまりを含む）と batch(merge_p) の出力が上流と同じ、
  merge_p を変えた state からは再開しない、lr の式が上流と同じ
"""
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / 'scripts/engine2/fp_retrain'
UP = FR / 'upstream'
TRAIN_FP4_SHA = 'e2eaef630c5f09f157eea3652095b8b9780cbe7fdffa6f0652ae5845d7b18c8a'


def test_upstream_train_fp4_pin():
    assert hashlib.sha256((UP / 'train_fp4.py').read_bytes()).hexdigest() == TRAIN_FP4_SHA


def _toy_train_inputs(d):
    """#152 のテストと同じ合成データ（構造 20 に対してスペクトル 80 本、同じ構造のスペクトルが複数ある）。"""
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
            '--bs', '8', '--lr', '2e-4', '--K', '7', '--lam', '1.0', '--seed', '404', '--d', '32', '--layers', '1',
            '--merge_p', '0.6', '--val_every', '4', '--warm', '3', '--device', 'cpu', '--ckpt_every_min', '1000',
            '--out', str(out), '--state', str(state), *extra]


def _load_train(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(FR))
    import train_m1_resumable as tr
    return tr


def _count_merged(monkeypatch, tr):
    """上流 train_fp4.Data._merged の呼び出しを数える（中身は上流のまま呼ぶ）。"""
    cnt = {'n': 0}
    orig = tr.train_fp4.Data._merged

    def wrapped(self, idxs, maxlen):
        cnt['n'] += 1
        return orig(self, idxs, maxlen)
    monkeypatch.setattr(tr.train_fp4.Data, '_merged', wrapped)
    return cnt


def test_packed_data4_matches_upstream_data(tmp_path, monkeypatch):
    """PackedData4 の PFP 以外の属性（_grp_order・_grp_bounds を含む）が上流 train_fp4.Data と同じ、PFP の全行も同じ、
    batch(merge_p=0.6, aug=True) の出力も同じ。"""
    tr = _load_train(monkeypatch)
    import torch
    _toy_train_inputs(tmp_path)
    A = tr.train_fp4.Data(str(tmp_path / 'spec.npz'), str(tmp_path / 'up'), str(tmp_path), device='cpu')
    B = tr.PackedData4(str(tmp_path / 'spec.npz'), str(tmp_path / 'up'), str(tmp_path), device='cpu')
    assert set(vars(A)) == set(vars(B))
    assert {'_grp_order', '_grp_bounds'} <= set(vars(B))
    for k, v in vars(A).items():
        w = getattr(B, k)
        if k == 'PFP':
            allrows = torch.arange(len(B.pmass))
            assert torch.equal(v[allrows], w[allrows]) and v.dtype == w[allrows].dtype
        elif isinstance(v, np.ndarray):
            assert v.dtype == w.dtype and v.shape == w.shape and np.array_equal(v, w), k
        else:
            assert v == w if not isinstance(v, (list, tuple)) else list(v) == list(w), k
    cnt = _count_merged(monkeypatch, tr)
    ids = B.tr[:16]
    ia, ya, ca = A.batch(ids, K=5, rng=np.random.default_rng(3), aug=True, merge_p=0.6)
    na = cnt['n']
    ib, yb, cb = B.batch(ids, K=5, rng=np.random.default_rng(3), aug=True, merge_p=0.6)
    assert na > 0 and cnt['n'] == 2 * na                     # merge の枝を両方で同じ回数通った
    for x, y in zip(ia, ib):
        assert torch.equal(x, y)
    assert torch.equal(ya, yb) and torch.equal(ca, cb)
    assert torch.equal(A.PFP[ca], B.PFP[cb])


def test_packed_training_bit_identical_to_upstream_with_merge(tmp_path, monkeypatch):
    """(i) --pfp upstream と --pfp packed の 12 step（merge_p 0.6）の損失（毎 step）・重み・best・乱数が一致。"""
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

    cnt = _count_merged(monkeypatch, tr)
    out = {}; merged = {}
    for mode in ('upstream', 'packed'):
        d = tmp_path / mode; d.mkdir()
        rec[mode] = []
        n0 = cnt['n']
        monkeypatch.setattr(tr, 'F', Shim(mode))
        assert tr.main(_args(tmp_path, d / 'best.pt', d / 'state.pt', extra=('--pfp', mode))) == 0
        merged[mode] = cnt['n'] - n0
        out[mode] = (torch.load(d / 'state.pt', weights_only=False), torch.load(d / 'best.pt', weights_only=False))
    monkeypatch.setattr(tr, 'F', realF)
    # merge の枝を実際に通った（12 step × bs 8 × 0.6 ≈ 58 行の見込み。peers が 1 本なら飛ぶので > 0 だけを求める）
    assert merged['upstream'] > 10 and merged['upstream'] == merged['packed']
    ru, rp = rec['upstream'], rec['packed']
    assert len(ru) == len(rp) and len(ru) >= 2 * 12            # 学習 12 step × 2 ＋ 評価
    for (na, va), (nb, vb) in zip(ru, rp):
        assert na == nb and torch.equal(va, vb)
    (su, bu), (sp, bp) = out['upstream'], out['packed']
    assert su['step'] == sp['step'] == 12 and su['best_val'] == sp['best_val']
    assert su['args']['pfp'] == 'upstream' and sp['args']['pfp'] == 'packed'
    assert su['args']['merge_p'] == sp['args']['merge_p'] == 0.6
    assert su['np_rng'] == sp['np_rng'] and torch.equal(su['torch_rng'], sp['torch_rng'])
    for k in su['model']:
        assert torch.equal(su['model'][k], sp['model'][k]), k
    assert bu['step'] == bp['step']
    for k in bu['model']:
        assert torch.equal(bu['model'][k], bp['model'][k]), k


def test_resume_bit_identical_with_merge(tmp_path, monkeypatch):
    """(ii) 12 step 続けて ＝ 6 step で止めて再開して 12 step（重み・best・乱数・step・lr・opt の状態）。"""
    tr = _load_train(monkeypatch)
    import torch
    torch.set_num_threads(1)
    _toy_train_inputs(tmp_path)
    a = tmp_path / 'a'; b = tmp_path / 'b'; a.mkdir(); b.mkdir()
    assert tr.main(_args(tmp_path, a / 'best.pt', a / 'state.pt')) == 0
    assert tr.main(_args(tmp_path, b / 'best.pt', b / 'state.pt', extra=('--stop_at_step', '6'))) == 0
    s6 = torch.load(b / 'state.pt', weights_only=False)
    assert s6['step'] == 6
    assert tr.main(_args(tmp_path, b / 'best.pt', b / 'state.pt')) == 0
    sa = torch.load(a / 'state.pt', weights_only=False); sb = torch.load(b / 'state.pt', weights_only=False)
    assert sa['step'] == sb['step'] == 12 and sa['best_val'] == sb['best_val']
    for k in sa['model']:
        assert torch.equal(sa['model'][k], sb['model'][k]), k
    assert sa['np_rng'] == sb['np_rng'] and torch.equal(sa['torch_rng'], sb['torch_rng'])
    # lr（opt の param_groups に最後に入れた値）と opt の状態
    la = [g['lr'] for g in sa['opt']['param_groups']]; lb = [g['lr'] for g in sb['opt']['param_groups']]
    assert la == lb and la[0] == tr.lr_at(tr.parse(_args(tmp_path, 'x', 'y')), 11)
    for pid, st in sa['opt']['state'].items():
        for kk, vv in st.items():
            assert torch.equal(vv, sb['opt']['state'][pid][kk]), (pid, kk)
    ba = torch.load(a / 'best.pt', weights_only=False); bb = torch.load(b / 'best.pt', weights_only=False)
    assert ba['step'] == bb['step']
    for k in ba['model']:
        assert torch.equal(ba['model'][k], bb['model'][k]), k
    # 再開の途中の state（6 step）から最後まで重みが動いている（何もしないで一致、ではない）
    assert any(not torch.equal(s6['model'][k], sb['model'][k]) for k in sb['model'])


def test_resume_refuses_changed_merge_p(tmp_path, monkeypatch):
    tr = _load_train(monkeypatch)
    _toy_train_inputs(tmp_path)
    assert tr.main(_args(tmp_path, tmp_path / 'best.pt', tmp_path / 'state.pt', extra=('--stop_at_step', '2'))) == 0
    bad = _args(tmp_path, tmp_path / 'best.pt', tmp_path / 'state.pt')
    bad[bad.index('--merge_p') + 1] = '0.5'
    with pytest.raises(SystemExit, match='引数が違う'):
        tr.main(bad)


def test_defaults_and_lr_match_upstream(monkeypatch):
    """merge_p の既定 0.6・max_minutes の既定 1e9 が上流と同じ、lr の式が上流 train_fp4 と同じ（steps 50000）。"""
    tr = _load_train(monkeypatch)
    import math
    a = tr.parse(['--state', 'x', '--steps', '50000', '--lr', '2e-4'])
    assert a.merge_p == 0.6 and a.max_minutes == 1e9 and a.warm == 2000
    for s in (0, 1, 1999, 2000, 2001, 24000, 49999):
        if s < 2000:
            exp = 2e-4 * s / 2000
        else:
            p = (s - 2000) / 48000
            exp = 2e-4 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))
        assert tr.lr_at(a, s) == exp
    src = (UP / 'train_fp4.py').read_text()
    assert "ap.add_argument('--merge_p',type=float,default=0.6)" in src
    assert "ap.add_argument('--max_minutes',type=float,default=1e9)" in src
