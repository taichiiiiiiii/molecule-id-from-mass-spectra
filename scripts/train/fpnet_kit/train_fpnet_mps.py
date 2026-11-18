"""Fold-safe training of the public FPNet architecture (spectrum -> fingerprint logits, mass-window ranking loss).

The public packages ship the model (casmi/fpnet.py) and the data access (casmi/traindata.py) but no training loop;
this script supplies one so that the encoder can be refitted with chosen identities held out.

usage: python train_fpnet.py --code <casmi code dir> --run <run dir with work/> [--hours 4] [--bs 256] ...
"""
import argparse, json, math, os, sys, time
import queue, signal  # [port:resume]
import multiprocessing as mp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--code', required=True); ap.add_argument('--run', required=True)
ap.add_argument('--hours', type=float, default=4.0); ap.add_argument('--max_steps', type=int, default=1000000000)
ap.add_argument('--bs', type=int, default=256); ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--warmup', type=int, default=2000)
ap.add_argument('--merge_p', type=float, default=0.3); ap.add_argument('--w_bce', type=float, default=1.0)
ap.add_argument('--w_el', type=float, default=0.1); ap.add_argument('--workers', type=int, default=16)
ap.add_argument('--eval_every', type=int, default=5000); ap.add_argument('--eval_groups', type=int, default=3000)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--layers', type=int, default=8); ap.add_argument('--d', type=int, default=512)
ap.add_argument('--holdout', default='hold,np'); ap.add_argument('--tag', default='fpnet')
ap.add_argument('--bias_init', type=int, default=0)
ap.add_argument('--device', default='mps')  # [port:device]
ap.add_argument('--ckpt_every_min', type=float, default=30.0); ap.add_argument('--resume', type=int, default=1)  # [port:resume]
ap.add_argument('--accum', type=int, default=1)  # [port:memory]
args = ap.parse_args()
sys.path.insert(0, args.code)
os.chdir(args.run)
import torch
import torch.nn.functional as F
from casmi import fpnet, traindata
from casmi.spectra import merge_spectra

G = {}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def _qsize(q):  # [port:device]
    try:  # [port:device]
        return q.qsize()  # [port:device]
    except NotImplementedError:  # [port:device]
        return -1  # [port:device]


def worker(wid, q):
    torch.set_num_threads(1)
    D = G['D']
    s0 = G.get('start_step', 0)  # [port:resume]
    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1) if s0 == 0 else np.random.default_rng([args.seed * 1000 + wid + 1, s0])  # [port:resume]
    while True:
        ids = D.sample_ids(args.bs)
        items = [D.item(int(i), args.merge_p, True) for i in ids]
        pos = D.sid2pid[D.sid[ids]]
        item = (tuple(t.numpy() for t in fpnet.collate(items, 'cpu')), pos, D.negatives(pos))  # [port:resume]
        while True:  # [port:resume]
            try:  # [port:resume]
                q.put(item, timeout=30); break  # [port:resume]
            except queue.Full:  # [port:resume]
                if os.getppid() != G['ppid']:  # [port:resume]
                    os._exit(0)  # [port:resume]


def eval_groups(D, fold_names, n, rng):
    """(sid, mode) groups of the given folds that have a pool entry; at most n of them."""
    ok = np.isin(D.fold[D.sid], fold_names) & (D.sid2pid[D.sid] >= 0)
    groups = {}
    for i in np.where(ok)[0]:
        groups.setdefault((int(D.sid[i]), int(D.mode[i])), []).append(int(i))
    keys = sorted(groups)
    if len(keys) > n:
        keys = [keys[i] for i in sorted(rng.choice(len(keys), n, replace=False))]
    return [groups[k][:8] for k in keys]


def accum_backward(model, arrs, pos, negs, PFP, ELEM, nbits, target, accum, dev, dtype, scaler, fdt=torch.float32):  # [port:memory]
    # forward/backward of one optimizer step in `accum` equal contiguous parts; returns the bs-sample means for the log  # [port:memory]
    B = len(pos); mb = B // accum  # [port:memory]
    assert mb * accum == B and len(target) == mb, (B, accum, len(target))  # [port:memory]
    tot = {}  # [port:memory]
    for s in range(0, B, mb):  # [port:memory]
        sl = slice(s, s + mb)  # [port:memory]
        inp = tuple(torch.as_tensor(a[sl], device=dev) for a in arrs)  # [port:memory]
        inp = tuple(t.to(fdt) if t.is_floating_point() else t for t in inp)  # [port:memory]
        pos_t = torch.as_tensor(pos[sl], device=dev)  # [port:memory]
        cand = torch.as_tensor(np.concatenate([pos[sl, None], negs[sl]], 1), device=dev)  # [port:memory]
        with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):  # [port:memory]
            z, el = model(*inp)  # [port:memory]
        z = z.to(fdt); el = el.to(fdt)  # [port:memory]
        fp = fpnet.unpack_gpu(PFP[cand], nbits).to(fdt)  # [port:memory]
        sc = torch.einsum('bkn,bn->bk', fp, z)  # [port:memory]
        l_rank = F.cross_entropy(sc, target)  # [port:memory]
        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])  # [port:memory]
        l_el = F.smooth_l1_loss(el, ELEM[pos_t].to(fdt))  # [port:memory]
        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el  # [port:memory]
        scaler.scale(loss / accum).backward()  # [port:memory]
        for k, v in (('loss', loss), ('rank', l_rank), ('bce', l_bce), ('el', l_el), ('acc', (sc.argmax(1) == 0).to(fdt).mean())):  # [port:memory]
            tot[k] = tot.get(k, 0.0) + v.detach() / accum  # [port:memory]
        del inp, z, el, fp, sc, loss, l_rank, l_bce, l_el  # [port:memory]
    return {k: v.item() for k, v in tot.items()}  # [port:memory]


@torch.no_grad()
def window_eval(model, D, PFP, entries, dtype, bs=128):
    """Rank of the true structure in its +-10 ppm pool window by f.z, all spectra of a (structure, mode) group merged."""
    model.eval(); ranks, sizes = [], []
    for s in range(0, len(entries), bs):
        chunk = entries[s:s + bs]; items = []
        for idxs in chunk:
            i = idxs[0]
            if len(idxs) == 1:
                mz = D.mz[D.off[i]:D.off[i + 1]]; it = D.it[D.off[i]:D.off[i + 1]]; ce = D.ce[i]; nm = max(1, D.ce_n[i])
            else:
                mz, it = merge_spectra([(D.mz[D.off[j]:D.off[j + 1]], D.it[D.off[j]:D.off[j + 1]]) for j in idxs])
                ces = D.ce[idxs]; ces = ces[~np.isnan(ces)]; ce = ces.mean() if len(ces) else np.nan
                nm = int(sum(max(1, D.ce_n[j]) for j in idxs))
            pm, pi = fpnet.prep_peaks(mz, it, D.prec[i])
            items.append(dict(mz=pm, it=pi, prec=float(D.prec[i]), adduct_ix=int(D.adduct[i]), ce=0.0 if np.isnan(ce) else float(ce),
                              ce_known=0.0 if np.isnan(ce) else 1.0, n_merged=min(nm, 8), mode=float(D.mode[i])))
        inp = fpnet.collate(items, PFP.device)  # [port:device]
        with torch.autocast(PFP.device.type, dtype=dtype, enabled=dtype != torch.float32):  # [port:device]
            z = model(*inp)[0].float()
        for r, idxs in enumerate(chunk):
            pid = D.sid2pid[D.sid[idxs[0]]]; m = D.pmass[pid]; tol = m * 10e-6
            lo = np.searchsorted(D.pmass, m - tol, 'left'); hi = np.searchsorted(D.pmass, m + tol, 'right')
            sc = fpnet.unpack_gpu(PFP[lo:hi], D.nbits) @ z[r]
            ranks.append(int((sc > sc[pid - lo]).sum().item()) + 1); sizes.append(int(hi - lo))
    model.train()
    ranks = np.array(ranks)
    return dict(n=len(ranks), mrr25=float(np.where(ranks <= 25, 1.0 / ranks, 0).mean()), top1=float((ranks == 1).mean()),
                top5=float((ranks <= 5).mean()), top25=float((ranks <= 25).mean()), median_window=float(np.median(sizes)))


def save_resume(path, model, opt, scaler, step, elapsed, hist, ema, dev):  # [port:resume]
    d = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, elapsed=elapsed,  # [port:resume]
             hist=hist, ema=dict(ema), rng_cpu=torch.get_rng_state(),  # [port:resume]
             rng_mps=torch.mps.get_rng_state() if dev == 'mps' else None, args=vars(args))  # [port:resume]
    torch.save(d, path + '.tmp'); os.replace(path + '.tmp', path)  # [port:resume]
    log('saved resume', path, 'step', step, 'elapsed_s', round(elapsed, 1))  # [port:resume]


def main():
    hold = tuple(args.holdout.split(','))
    D = traindata.Data('cpu', seed=args.seed, holdout_folds=hold)
    G['D'] = D
    ck_path = f'{args.tag}_resume.pt'  # [port:resume]
    ck = torch.load(ck_path, map_location='cpu', weights_only=False) if args.resume and os.path.exists(ck_path) else None  # [port:resume]
    G['start_step'] = ck['step'] if ck is not None else 0; G['ppid'] = os.getpid()  # [port:resume]
    rng = np.random.default_rng(12345)
    sets = {name: eval_groups(D, names, args.eval_groups, rng) for name, names in
            (('hold', ['hold']), ('np', ['np']), ('train', ['train']))}
    log({k: len(v) for k, v in sets.items()}, 'train spectra', len(D.tr))
    q = mp.get_context('fork').Queue(maxsize=64)
    procs = [mp.get_context('fork').Process(target=worker, args=(w, q), daemon=True) for w in range(args.workers)]
    for p in procs:
        p.start()
    def _term(signum, frame):  # [port:resume]
        for p in procs:  # [port:resume]
            p.terminate()  # [port:resume]
        os._exit(128 + signum)  # [port:resume]
    signal.signal(signal.SIGTERM, _term)  # [port:resume]
    dev = args.device  # [port:device]
    torch.manual_seed(args.seed)
    PFP = D.PFP.to(dev); ELEM = D.elem.to(dev)
    model = fpnet.FPNet(D.nbits, d=args.d, layers=args.layers).to(dev)
    # Optional: start the output bias at the log-odds of each fingerprint bit. With the raw f.z ranking loss this makes
    # the initial scores differ by hundreds between candidates, so it is off by default.
    freq = torch.zeros(D.nbits, device=dev)
    if args.bias_init:
        with torch.no_grad():
            pids = np.unique(D.sid2pid[D.sid[D.tr]]); pids = rng.choice(pids, min(40000, len(pids)), replace=False)
            for s in range(0, len(pids), 8000):
                freq += fpnet.unpack_gpu(PFP[torch.as_tensor(pids[s:s + 8000], device=dev)], D.nbits).sum(0)
            freq = (freq / len(pids)).clamp(1e-4, 1 - 1e-4)
            model.head[-1].bias.copy_(torch.log(freq / (1 - freq)))
    log('parameters', sum(p.numel() for p in model.parameters()), 'mean bit frequency', float(freq.mean()))
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]; no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)], lr=args.lr, betas=(0.9, 0.98))
    dtype = fpnet.autocast_dtype(dev)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    budget = args.hours * 3600.0; t0 = time.time(); step = 0; hist = []
    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}
    if ck is not None:  # [port:resume]
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])  # [port:resume]
        step = ck['step']; t0 = time.time() - ck['elapsed']; hist = ck['hist']; ema = dict(ck['ema'])  # [port:resume]
        torch.set_rng_state(ck['rng_cpu'])  # [port:resume]
        if dev == 'mps' and ck.get('rng_mps') is not None:  # [port:resume]
            torch.mps.set_rng_state(ck['rng_mps'])  # [port:resume]
        log('resumed from', ck_path, 'step', step, 'elapsed_s', round(ck['elapsed'], 1))  # [port:resume]
        del ck  # [port:resume]
    t_ck = time.time()  # [port:resume]
    assert args.bs % args.accum == 0, (args.bs, args.accum)  # [port:memory]
    target = torch.zeros(args.bs // args.accum, dtype=torch.long, device=dev)  # [port:memory]
    while True:
        prog = max(step / args.max_steps, (time.time() - t0) / budget)
        if prog >= 1.0:
            break
        lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = lr
        arrs, pos, negs = q.get()
        opt.zero_grad(set_to_none=True)
        vals = accum_backward(model, arrs, pos, negs, PFP, ELEM, D.nbits, target, args.accum, dev, dtype, scaler)  # [port:memory]
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1
        for k, v in vals.items():
            ema[k] = v if step == 1 else 0.98 * ema[k] + 0.02 * v
        if not math.isfinite(vals['loss']):
            log('non-finite loss at step', step); break
        if step % 200 == 0:
            log(f"step {step} lr {lr:.2e} " + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) + f" {step / (time.time() - t0):.2f} it/s queue {_qsize(q)}")  # [port:device]
        if step % args.eval_every == 0:
            ev = {name: window_eval(model, D, PFP, ent, dtype) for name, ent in sets.items()}
            hist.append(dict(step=step, hours=(time.time() - t0) / 3600, **{f'{n}_{k}': v for n, e in ev.items() for k, v in e.items()}))
            log('EVAL', json.dumps(hist[-1]))
            fpnet.save_ckpt(f'{args.tag}_last.pt', model, step, extra=dict(args=vars(args)))
            json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)
        if time.time() - t_ck >= args.ckpt_every_min * 60:  # [port:resume]
            save_resume(ck_path, model, opt, scaler, step, time.time() - t0, hist, ema, dev)  # [port:resume]
            t_ck = time.time()  # [port:resume]
    if all(math.isfinite(v) for v in ema.values()):  # [port:resume]
        save_resume(ck_path, model, opt, scaler, step, time.time() - t0, hist, ema, dev)  # [port:resume]
    ev = {name: window_eval(model, D, PFP, ent, dtype) for name, ent in sets.items()}
    hist.append(dict(step=step, hours=(time.time() - t0) / 3600, final=True, **{f'{n}_{k}': v for n, e in ev.items() for k, v in e.items()}))
    log('EVAL', json.dumps(hist[-1]))
    fpnet.save_ckpt(f'{args.tag}_final.pt', model, step, extra=dict(args=vars(args)))
    json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)
    log('ALL_DONE')
    for p in procs:
        p.terminate()


if __name__ == '__main__':
    main()
