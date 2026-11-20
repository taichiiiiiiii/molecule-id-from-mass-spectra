"""Derive train_fpnet_mps.py from the kit's train_fpnet.py by exact, asserted replacements (one-off helper, #82)."""
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_bytes().decode().replace('\r\n', '\n')
D = '  # [port:device]'
R = '  # [port:resume]'
M = '  # [port:memory]'


def rep(old, new_lines, tag=None):
    global src
    assert src.count(old) == 1, (src.count(old), old)
    src = src.replace(old, '\n'.join(new_lines))


def m(lines, tag):
    return [l + tag if l else l for l in lines]


# header docstring note + imports
rep('import argparse, json, math, os, sys, time\n',
    ['import argparse, json, math, os, sys, time'] +
    m(['import queue, signal'], R) + [''])
# new arguments
rep("ap.add_argument('--bias_init', type=int, default=0)\n",
    ["ap.add_argument('--bias_init', type=int, default=0)"] +
    m(["ap.add_argument('--device', default='mps')"], D) +
    m(["ap.add_argument('--ckpt_every_min', type=float, default=30.0); ap.add_argument('--resume', type=int, default=1)"], R) +
    m(["ap.add_argument('--accum', type=int, default=1)"], M) + [''])
# worker: fresh RNG stream after a resume; exit when the parent is gone
rep("    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1)\n",
    m(["    s0 = G.get('start_step', 0)",
       "    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1) if s0 == 0 else np.random.default_rng([args.seed * 1000 + wid + 1, s0])"], R) + [''])
rep("        q.put((tuple(t.numpy() for t in fpnet.collate(items, 'cpu')), pos, D.negatives(pos)))\n",
    m(["        item = (tuple(t.numpy() for t in fpnet.collate(items, 'cpu')), pos, D.negatives(pos))",
       "        while True:",
       "            try:",
       "                q.put(item, timeout=30); break",
       "            except queue.Full:",
       "                if os.getppid() != G['ppid']:",
       "                    os._exit(0)"], R) + [''])
# eval on the chosen device, fp32 without autocast
rep("        inp = fpnet.collate(items, 'cuda')\n        with torch.autocast('cuda', dtype=dtype):\n            z = model(*inp)[0].float()\n",
    m(["        inp = fpnet.collate(items, PFP.device)",
       "        with torch.autocast(PFP.device.type, dtype=dtype, enabled=dtype != torch.float32):"], D) +
    ["            z = model(*inp)[0].float()", ''])
# resume-state writer (module level, before main)
rep("def main():\n",
    m(["def save_resume(path, model, opt, scaler, step, elapsed, hist, ema, dev):",
       "    d = dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, elapsed=elapsed,",
       "             hist=hist, ema=dict(ema), rng_cpu=torch.get_rng_state(),",
       "             rng_mps=torch.mps.get_rng_state() if dev == 'mps' else None, args=vars(args))",
       "    torch.save(d, path + '.tmp'); os.replace(path + '.tmp', path)",
       "    log('saved resume', path, 'step', step, 'elapsed_s', round(elapsed, 1))",
       "",
       ""], R) + ["def main():", ''])
# load the resume state before the workers fork (they need the start step)
rep("    G['D'] = D\n",
    ["    G['D'] = D"] +
    m(["    ck_path = f'{args.tag}_resume.pt'",
       "    ck = torch.load(ck_path, map_location='cpu', weights_only=False) if args.resume and os.path.exists(ck_path) else None",
       "    G['start_step'] = ck['step'] if ck is not None else 0; G['ppid'] = os.getpid()"], R) + [''])
# stop the workers on SIGTERM (memguard) so they do not outlive the trainer
rep("    for p in procs:\n        p.start()\n    dev = 'cuda'\n",
    ["    for p in procs:", "        p.start()"] +
    m(["    def _term(signum, frame):",
       "        for p in procs:",
       "            p.terminate()",
       "        os._exit(128 + signum)",
       "    signal.signal(signal.SIGTERM, _term)"], R) +
    m(["    dev = args.device"], D) + [''])
# restore the state after model/opt/scaler exist
rep("    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}\n",
    ["    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}"] +
    m(["    if ck is not None:",
       "        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])",
       "        step = ck['step']; t0 = time.time() - ck['elapsed']; hist = ck['hist']; ema = dict(ck['ema'])",
       "        torch.set_rng_state(ck['rng_cpu'])",
       "        if dev == 'mps' and ck.get('rng_mps') is not None:",
       "            torch.mps.set_rng_state(ck['rng_mps'])",
       "        log('resumed from', ck_path, 'step', step, 'elapsed_s', round(ck['elapsed'], 1))",
       "        del ck",
       "    t_ck = time.time()"], R) + [''])
# gradient accumulation (#82): the worker still sends bs samples; split them into `accum` equal contiguous parts,
# backward each part's loss / accum, then one unscale/clip/step as before. Same gradient as one shot (all losses are means).
rep("    target = torch.zeros(args.bs, dtype=torch.long, device=dev)\n",
    m(["    assert args.bs % args.accum == 0, (args.bs, args.accum)",
       "    target = torch.zeros(args.bs // args.accum, dtype=torch.long, device=dev)"], M) + [''])
rep("""        inp = tuple(torch.as_tensor(a, device=dev) for a in arrs)
        pos_t = torch.as_tensor(pos, device=dev)
        cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=dev)
        with torch.autocast('cuda', dtype=dtype):
            z, el = model(*inp)
        z = z.float(); el = el.float()
        fp = fpnet.unpack_gpu(PFP[cand], D.nbits)
        sc = torch.einsum('bkn,bn->bk', fp, z)
        l_rank = F.cross_entropy(sc, target)
        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])
        l_el = F.smooth_l1_loss(el, ELEM[pos_t])
        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
""",
    ["        opt.zero_grad(set_to_none=True)"] +
    m(["        vals = accum_backward(model, arrs, pos, negs, PFP, ELEM, D.nbits, target, args.accum, dev, dtype, scaler)"], M) + [''])
rep("        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(), acc=(sc.argmax(1) == 0).float().mean().item())\n",
    [''])
rep("@torch.no_grad()\ndef window_eval",
    m(["def accum_backward(model, arrs, pos, negs, PFP, ELEM, nbits, target, accum, dev, dtype, scaler, fdt=torch.float32):",
       "    # forward/backward of one optimizer step in `accum` equal contiguous parts; returns the bs-sample means for the log",
       "    B = len(pos); mb = B // accum",
       "    assert mb * accum == B and len(target) == mb, (B, accum, len(target))",
       "    tot = {}",
       "    for s in range(0, B, mb):",
       "        sl = slice(s, s + mb)",
       "        inp = tuple(torch.as_tensor(a[sl], device=dev) for a in arrs)",
       "        inp = tuple(t.to(fdt) if t.is_floating_point() else t for t in inp)",
       "        pos_t = torch.as_tensor(pos[sl], device=dev)",
       "        cand = torch.as_tensor(np.concatenate([pos[sl, None], negs[sl]], 1), device=dev)",
       "        with torch.autocast(dev, dtype=dtype, enabled=dtype != torch.float32):",
       "            z, el = model(*inp)",
       "        z = z.to(fdt); el = el.to(fdt)",
       "        fp = fpnet.unpack_gpu(PFP[cand], nbits).to(fdt)",
       "        sc = torch.einsum('bkn,bn->bk', fp, z)",
       "        l_rank = F.cross_entropy(sc, target)",
       "        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])",
       "        l_el = F.smooth_l1_loss(el, ELEM[pos_t].to(fdt))",
       "        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el",
       "        scaler.scale(loss / accum).backward()",
       "        for k, v in (('loss', loss), ('rank', l_rank), ('bce', l_bce), ('el', l_el), ('acc', (sc.argmax(1) == 0).to(fdt).mean())):",
       "            tot[k] = tot.get(k, 0.0) + v.detach() / accum",
       "        del inp, z, el, fp, sc, loss, l_rank, l_bce, l_el",
       "    return {k: v.item() for k, v in tot.items()}",
       "",
       ""], M) + ["@torch.no_grad()", "def window_eval"])
# periodic resume save (after the eval/ckpt block of each step)
rep("            json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)\n    ev = ",
    ["            json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)"] +
    m(["        if time.time() - t_ck >= args.ckpt_every_min * 60:",
       "            save_resume(ck_path, model, opt, scaler, step, time.time() - t0, hist, ema, dev)",
       "            t_ck = time.time()",
       "    if all(math.isfinite(v) for v in ema.values()):",
       "        save_resume(ck_path, model, opt, scaler, step, time.time() - t0, hist, ema, dev)"], R) +
    ["    ev = "])
# macOS: multiprocessing.Queue.qsize() raises NotImplementedError (no sem_getvalue); the step log line calls it
rep("def worker(wid, q):\n",
    m(["def _qsize(q):",
       "    try:",
       "        return q.qsize()",
       "    except NotImplementedError:",
       "        return -1",
       "",
       ""], D) + ["def worker(wid, q):", ''])
rep(" it/s queue {q.qsize()}\")\n",
    [" it/s queue {_qsize(q)}\")" + D, ''])
Path(sys.argv[2]).write_text(src)
print('ok')
