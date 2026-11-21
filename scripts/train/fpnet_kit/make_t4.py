"""Derive train_fpnet_t4.py from the kit's train_fpnet.py by exact, asserted replacements (Issue #82, 11/21).

    python scripts/train/fpnet_kit/make_t4.py <kit train_fpnet.py> <out train_fpnet_t4.py>

Adds only session-boundary save/resume (every added line ends with '# [t4:resume]'); device, precision, batch, losses,
hyper-parameters and the schedule stay as in the kit. The kernel runs this file on the dataset's original (sha256 checked)
and requires the result to equal the committed train_fpnet_t4.py.
"""
import sys
from pathlib import Path

TAG = '  # [t4:resume]'


def derive(src: str) -> str:
    src = src.replace('\r\n', '\n')

    def rep(old, new_lines):
        nonlocal src
        assert src.count(old) == 1, (src.count(old), old)
        src = src.replace(old, '\n'.join(new_lines))

    def m(lines):
        return [l + TAG if l else l for l in lines]

    # helpers module next to this file (fpnet_resume.py)
    rep('import numpy as np\n',
        ['import numpy as np'] +
        m(["sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import fpnet_resume as FR"]) + [''])
    # new arguments (session number, session length, resume file); kit hyper-parameters untouched
    rep("ap.add_argument('--bias_init', type=int, default=0)\n",
        ["ap.add_argument('--bias_init', type=int, default=0)"] +
        m(["ap.add_argument('--session', type=int, default=1); ap.add_argument('--session_hours', type=float, default=10.5)",
           "ap.add_argument('--resume_from', default='')"]) + [''])
    # worker RNG: kit formula in a fresh run, seed*1000+wid+1 + 7919*session after a resume (pre-registered on #82)
    rep("    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1)\n",
        m(["    D.rng = np.random.default_rng(FR.worker_seed(args.seed, wid, args.session, G['resumed']))"]) + [''])
    # read the resume file before the workers fork (they need to know whether this is a resumed session)
    rep("    G['D'] = D\n",
        ["    G['D'] = D"] +
        m(["    ck = FR.load_resume(args.resume_from) if args.resume_from else None",
           "    assert (ck is not None) == (args.session > 1), (args.session, args.resume_from)",
           "    if ck is not None:",
           "        same = ('session', 'session_hours', 'resume_from', 'code', 'run')",
           "        bad = {k: (ck['args'].get(k), v) for k, v in vars(args).items() if k not in same and ck['args'].get(k) != v}",
           "        assert not bad and ck['session'] == args.session - 1, (bad, ck['session'], args.session)",
           "    G['resumed'] = ck is not None"]) + [''])
    # restore model / AdamW / GradScaler / step / elapsed / history / EMA / RNG once they exist
    rep("    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}\n",
        ["    ema = {k: 0.0 for k in ('loss', 'rank', 'bce', 'el', 'acc')}"] +
        m(["    lr_check = None",
           "    if ck is not None:",
           "        step, elapsed, hist, ema, lr_check = FR.apply_resume(ck, model, opt, scaler)",
           "        t0 = time.time() - elapsed",
           "        log('resumed from', args.resume_from, 'session', args.session, 'step', step, 'elapsed_s', round(elapsed, 1))",
           "        del ck",
           "    t_sess = time.time()"]) + [''])
    # session boundary: after this step's lr is set, before the batch is taken
    rep("        for g in opt.param_groups:\n            g['lr'] = lr\n        arrs, pos, negs = q.get()\n",
        ["        for g in opt.param_groups:", "            g['lr'] = lr"] +
        m(["        if lr_check is not None:",
           "            assert lr == lr_check, (lr, lr_check)",
           "            log('lr continues', f'{lr:.10e}', 'step', step); lr_check = None",
           "        if time.time() - t_sess >= args.session_hours * 3600:",
           "            assert all(math.isfinite(v) for v in ema.values()), ema",
           "            FR.save_resume(f'{args.tag}_resume.pt', model, opt, scaler, step, time.time() - t0, hist, ema, lr, args.session, vars(args))",
           "            json.dump(hist, open(f'{args.tag}_history.json', 'w'), indent=1)",
           "            log('SESSION_DONE', 'session', args.session, 'step', step, 'lr', f'{lr:.10e}', 'elapsed_s', round(time.time() - t0, 1))",
           "            for p in procs:",
           "                p.terminate()",
           "            return"]) +
        ["        arrs, pos, negs = q.get()", ''])
    return src


if __name__ == '__main__':
    out = derive(Path(sys.argv[1]).read_bytes().decode())
    Path(sys.argv[2]).write_text(out)
    print('ok')
