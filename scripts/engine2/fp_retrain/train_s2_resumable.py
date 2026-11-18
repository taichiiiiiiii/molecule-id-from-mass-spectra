"""#152: Engine 2 の FP モデル（S2、train_fp3.py）を途中保存・再開できる形で学習する。

元の upstream/train_fp3.py（cubres/enveda-casmi-2026 commit 4778979、Apache-2.0、sha256 43bb618d…）は変えずに
import し、`Data`（入力の読み込み・負例・バッチ）と `fpmodel.FPNet` をそのまま使う。学習ループは元の `main()` を
同じ式で書き直したもので、違いは次だけ（Issue #152 §4）:
  1. 途中保存と再開: `--state` に model・opt・GradScaler・次の step・best_val・乱数の状態（numpy・torch CPU・MPS）・
     経過時間・引数を、`--ckpt_every_min` ごとと評価の直後に tmp → os.replace で書く。SIGTERM を受けたら今の step を
     終えてから保存して rc 143 で終わる。best のチェックポイント（`--out`）は元と同じ形・同じ条件
  2. seed: モデルを作る前にも torch.manual_seed(seed)（元の作った後の呼び出しも残す）
  3. 精度: cuda は元のまま fp16＋GradScaler。mps は autocast fp16＋torch.amp.GradScaler('mps')（元は scaler 無し）
     cpu は元のまま autocast 無し
  4. 記録だけ: `--speed_every` step ごとに `speed step <n> t <経過秒>`、終わりに ALL_DONE
`--max_minutes` は元と同じ意味（既定 1e9）。ただし経過時間は再開をまたいで足す。
"""
import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'upstream'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import fpmodel  # noqa: E402
import train_fp3  # noqa: E402

# 再開のときに一致を求める引数（学習の中身を決めるもの）
STATE_ARGS = ('steps', 'bs', 'lr', 'd', 'layers', 'K', 'lam', 'warm', 'val_every', 'seed', 'spec', 'fpdir', 'pooldir',
              'device', 'out')


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=80000); ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--layers', type=int, default=6); ap.add_argument('--K', type=int, default=63)
    ap.add_argument('--lam', type=float, default=1.0); ap.add_argument('--warm', type=int, default=2000)
    ap.add_argument('--out', default='fp_model.pt')
    ap.add_argument('--val_every', type=int, default=2000); ap.add_argument('--max_minutes', type=float, default=1e9)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--spec', default='ds/spec.npz'); ap.add_argument('--fpdir', default='up_train')
    ap.add_argument('--pooldir', default='.')
    # 自前の追加
    ap.add_argument('--state', required=True, help='途中保存のパス（あれば再開）')
    ap.add_argument('--ckpt_every_min', type=float, default=10.0)
    ap.add_argument('--speed_every', type=int, default=25)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    ap.add_argument('--run', default='', help='プロセスを探すための印（学習には使わない）')
    ap.add_argument('--stop_at_step', type=int, default=-1, help='テスト用: この step まで進んだら保存して終わる')
    return ap.parse_args(argv)


def lr_at(a, s):
    if s < a.warm:
        return a.lr * s / max(1, a.warm)
    p = (s - a.warm) / max(1, a.steps - a.warm)
    return a.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def make_scaler(dev):
    if dev == 'cuda':
        return torch.amp.GradScaler('cuda')
    if dev == 'mps':
        return torch.amp.GradScaler('mps')
    return None


def autocast_ctx(dev):
    if dev == 'cuda':
        return torch.autocast('cuda', dtype=torch.float16)
    if dev == 'mps':
        return torch.autocast('mps', dtype=torch.float16)
    return torch.autocast('cpu', enabled=False)


def atomic_save(obj, path):
    tmp = f'{path}.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None):
    a = parse(argv)
    dev = train_fp3.dev_of() if a.device == 'auto' else a.device
    print('device', dev, flush=True)
    D = train_fp3.Data(a.spec, a.fpdir, a.pooldir, device=dev)
    torch.manual_seed(a.seed)                                        # 自前の追加（作る前）
    model = fpmodel.FPNet(D.nbits, d=a.d, layers=a.layers).to(dev)
    print('params %.1fM' % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    torch.manual_seed(a.seed)                                        # 元の位置（作った後）
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    scaler = make_scaler(dev)
    rng = np.random.default_rng(a.seed)
    start = 0; best_val = -1.0; elapsed0 = 0.0
    sig = {k: getattr(a, k) for k in STATE_ARGS}
    if os.path.exists(a.state):
        ck = torch.load(a.state, map_location='cpu', weights_only=False)
        if ck['args'] != sig:
            diff = {k: (ck['args'].get(k), sig[k]) for k in sig if ck['args'].get(k) != sig[k]}
            raise SystemExit(f'state の引数が違うので再開しない: {diff}')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        if scaler is not None and ck.get('scaler') is not None:
            scaler.load_state_dict(ck['scaler'])
        start = ck['step']; best_val = ck['best_val']; elapsed0 = ck['elapsed']
        rng.bit_generator.state = ck['np_rng']
        torch.set_rng_state(ck['torch_rng'])
        if dev == 'mps' and ck.get('mps_rng') is not None:
            torch.mps.set_rng_state(ck['mps_rng'])
        if dev == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(ck['cuda_rng'])
        print(f'resumed from {a.state} step {start} best_val {best_val:.4f} elapsed_s {elapsed0:.1f}', flush=True)

    stop = {'term': False}

    def on_term(signum, frame):
        stop['term'] = True
    signal.signal(signal.SIGTERM, on_term)

    t0 = time.time()
    elapsed = lambda: elapsed0 + (time.time() - t0)  # noqa: E731

    def save_best(step):
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step,
                    'nbits': D.nbits, 'd': a.d, 'layers': a.layers}, a.out)

    def save_state(next_step):
        st = {'model': model.state_dict(), 'opt': opt.state_dict(),
              'scaler': scaler.state_dict() if scaler is not None else None,
              'step': next_step, 'best_val': best_val, 'elapsed': elapsed(),
              'np_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
              'mps_rng': torch.mps.get_rng_state() if dev == 'mps' else None,
              'cuda_rng': torch.cuda.get_rng_state() if dev == 'cuda' else None,
              'args': sig}
        atomic_save(st, a.state)
        print(f'state saved step {next_step} elapsed_s {st["elapsed"]:.1f}', flush=True)

    last_ck = time.time()
    rb = rc = racc = 0.0; nr = 0
    step = start
    for step in range(start, a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(a, step)
        ids = rng.choice(D.tr, size=a.bs, replace=False)
        inp, ypos, cand = D.batch(ids, K=a.K, rng=rng, aug=True)
        opt.zero_grad(set_to_none=True)
        with autocast_ctx(dev):
            z = model(*inp)
        z = z.float()
        lb = F.binary_cross_entropy_with_logits(z, ypos)
        FPc = D.PFP[cand].to(z.device).float()
        raw = torch.bmm(FPc, z.unsqueeze(-1)).squeeze(-1)
        sc = (raw - raw.mean(dim=1, keepdim=True)) / math.sqrt(D.nbits)
        tgt = torch.zeros(len(ids), dtype=torch.long, device=z.device)
        lc = F.cross_entropy(sc, tgt)
        loss = lb + a.lam * lc
        if not torch.isfinite(loss):
            print(f'  non-finite loss at step {step}; skipping', flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            if scaler is not None:
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            rb += lb.item(); rc += lc.item(); racc += (sc.argmax(1) == 0).float().mean().item(); nr += 1
            if step % 200 == 0:
                el = elapsed()
                print(f'step {step} bce {rb/max(nr,1):.4f} ctr {rc/max(nr,1):.4f} top1 {racc/max(nr,1):.3f} '
                      f'lr {lr_at(a, step):.2e} {el:.0f}s', flush=True)
                rb = rc = racc = 0.0; nr = 0
            if a.speed_every > 0 and (step + 1) % a.speed_every == 0:
                print(f'speed step {step + 1} t {elapsed():.1f}', flush=True)
            if (step + 1) % a.val_every == 0:
                model.eval(); vb = []; vacc = []
                with torch.no_grad():
                    vids = rng.choice(D.va, size=min(2048, len(D.va)), replace=False)
                    for s in range(0, len(vids), 128):
                        ii = vids[s:s + 128]
                        pos = D.s2p[D.six[ii]]
                        if (pos < 0).any():
                            ii = ii[pos >= 0]
                        if len(ii) == 0:
                            continue
                        inp, ypos, cand = D.batch(ii, K=a.K, rng=rng)
                        z = model(*inp).float()
                        vb.append(F.binary_cross_entropy_with_logits(z, ypos).item())
                        FPc = D.PFP[cand].to(z.device).float()
                        raw = torch.bmm(FPc, z.unsqueeze(-1)).squeeze(-1)
                        sc = (raw - raw.mean(dim=1, keepdim=True)) / math.sqrt(D.nbits)
                        vacc.append((sc.argmax(1) == 0).float().mean().item())
                va = float(np.mean(vacc))
                print(f'  VAL step {step} bce {np.mean(vb):.4f} hardneg-top1 {va:.3f}'
                      + ('  <- best' if va > best_val else ''), flush=True)
                model.train()
                if va > best_val:
                    best_val = va; save_best(step + 1)
                save_state(step + 1); last_ck = time.time()
        # ここで step は終わった。次に進む前に止める・保存するかを決める
        if a.stop_at_step >= 0 and step + 1 >= a.stop_at_step:
            save_state(step + 1)
            print('stopped at stop_at_step', flush=True)
            return 0
        if stop['term']:
            save_state(step + 1)
            print('SIGTERM: state saved, exit 143', flush=True)
            sys.exit(143)
        if (time.time() - last_ck) / 60 >= a.ckpt_every_min:
            save_state(step + 1); last_ck = time.time()
        if elapsed() / 60 > a.max_minutes:
            save_state(step + 1)
            print('time budget reached; keeping best-val checkpoint', flush=True)
            return 0
    save_state(a.steps)
    print('done. best hardneg-top1 = %.3f' % best_val, flush=True)
    print('ALL_DONE', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
