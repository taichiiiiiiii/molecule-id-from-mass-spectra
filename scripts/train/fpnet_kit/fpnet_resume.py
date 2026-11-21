"""Session-boundary save/resume helpers for the kit's train_fpnet.py on Kaggle T4 (Issue #82, 11/21 pre-registration).

Used by train_fpnet_t4.py (derived from the kit file by make_t4.py) and by the CUDA self-check cell of the kernel.

- save_resume / load_resume / apply_resume: model, AdamW and torch.amp.GradScaler state_dicts, step, training elapsed
  seconds, eval history, loss EMA, lr, session, and the torch (CPU, CUDA) / numpy / python RNG states.
  Written to <path>.tmp and moved with os.replace. The kit has no lr-scheduler object: lr is a pure function of step
  and elapsed time (prog = max(step / max_steps, elapsed / budget)), so restoring step and elapsed continues it.
- worker_seed: the kit's worker seed (seed*1000+wid+1) in a fresh run; seed*1000+wid+1 + 7919*session after a resume.
- state_equal: bitwise comparison of nested state (tensors by dtype, shape and bytes).
- selfcheck: train a separate model copy for N steps, save, load into fresh objects, compare bitwise, run one more
  step; the caller's RNG state is restored on exit.
"""
from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch

RESUME_PRIME = 7919
FORMAT = 'fpnet_t4_resume_v1'


def worker_seed(seed: int, wid: int, session: int, resumed: bool) -> int:
    if not resumed:
        return seed * 1000 + wid + 1                       # kit formula, unchanged
    return seed * 1000 + wid + 1 + RESUME_PRIME * session  # pre-registered on #82 (11/21)


def capture_rng() -> dict:
    st = dict(torch_cpu=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(), cuda=None)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict) -> None:
    torch.set_rng_state(st['torch_cpu'])
    np.random.set_state(st['numpy'])
    random.setstate(st['python'])
    if st.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st['cuda'])


@contextmanager
def preserved_rng():
    st = capture_rng()
    try:
        yield
    finally:
        restore_rng(st)


def save_resume(path, model, opt, scaler, step, elapsed, hist, ema, lr, session, args=None) -> dict:
    d = dict(format=FORMAT, model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=int(step),
             elapsed=float(elapsed), hist=list(hist), ema=dict(ema), lr=float(lr), session=int(session),
             rng=capture_rng(), args=dict(args) if args is not None else None)
    tmp = str(path) + '.tmp'
    torch.save(d, tmp)
    os.replace(tmp, path)
    return d


def load_resume(path) -> dict:
    ck = torch.load(path, map_location='cpu', weights_only=False)
    assert ck.get('format') == FORMAT, ck.get('format')
    return ck


def apply_resume(ck, model, opt, scaler, restore_rng_state=True):
    """Load the state into existing objects; returns (step, elapsed, hist, ema, lr)."""
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    scaler.load_state_dict(ck['scaler'])
    if restore_rng_state:
        restore_rng(ck['rng'])
    return ck['step'], ck['elapsed'], list(ck['hist']), dict(ck['ema']), ck['lr']


def state_equal(a, b, path='') -> list:
    """List of paths where a and b differ (empty = bitwise equal)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return [path + ': type']
        a = a.detach().cpu(); b = b.detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape:
            return [path + f': {a.dtype}{tuple(a.shape)} vs {b.dtype}{tuple(b.shape)}']
        return [] if a.contiguous().reshape(-1).view(torch.uint8).equal(b.contiguous().reshape(-1).view(torch.uint8)) else [path + ': bytes']
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        ok = isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape \
            and a.tobytes() == b.tobytes()
        return [] if ok else [path + ': ndarray']
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [path + f': keys {sorted(map(str, set(a) ^ set(b)))}']
        out = []
        for k in a:
            out += state_equal(a[k], b[k], f'{path}/{k}')
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [path + ': len']
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += state_equal(x, y, f'{path}[{i}]')
        return out
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if (type(a) == type(b) and a == b) else [path + f': {a!r} vs {b!r}']


def kit_lr(step, max_steps, elapsed, budget, lr0, warmup):
    """The kit's inline lr formula (train_fpnet.py, main loop), for tests and the self-check."""
    prog = max(step / max_steps, elapsed / budget)
    return lr0 * min(1.0, (step + 1) / warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))


def make_opt(model, lr=3e-4, wd=0.01):
    """AdamW exactly as the kit builds it (decay on ndim >= 2, betas 0.9/0.98)."""
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=wd), dict(params=no_decay, weight_decay=0.0)],
                             lr=lr, betas=(0.9, 0.98))


def train_step(model, opt, scaler, batch, loss_fn, lr, device, dtype):
    """One optimizer step with the kit's order: set lr, autocast forward, scaled backward, unscale, clip 1.0, step."""
    for g in opt.param_groups:
        g['lr'] = lr
    with torch.autocast(device, dtype=dtype, enabled=dtype != torch.float32):
        out = model(*batch['inp'])
    loss = loss_fn(out, batch)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    return loss.item()


def selfcheck(make_model, make_batch, loss_fn, device, dtype, path, steps=50, seed=20261121, lr0=3e-4, warmup=2000,
              max_steps=82855):
    """Separate model copy and RNG state; N steps -> save -> load into fresh objects -> bitwise compare -> 1 more step.

    make_batch(step, generator) must build the batch from the given torch.Generator only. The caller's RNG state
    (torch CPU/CUDA, numpy, python) is the same before and after the call.
    """
    with preserved_rng():
        torch.manual_seed(seed)
        gen = torch.Generator(device='cpu').manual_seed(seed + 1)
        model = make_model().to(device)
        opt = make_opt(model, lr=lr0)
        scaler = torch.amp.GradScaler(device, enabled=True)
        losses = []
        for step in range(steps):
            lr = kit_lr(step, max_steps, 0.0, 3.6e6, lr0, warmup)
            losses.append(train_step(model, opt, scaler, make_batch(step, gen), loss_fn, lr, device, dtype))
        lr_next = kit_lr(steps, max_steps, 0.0, 3.6e6, lr0, warmup)
        saved = save_resume(path, model, opt, scaler, steps, 0.0, [], dict(loss=losses[-1]), lr_next, 1)
        ck = load_resume(path)
        model2 = make_model().to(device)
        opt2 = make_opt(model2, lr=lr0)
        scaler2 = torch.amp.GradScaler(device, enabled=True)
        step2, _, _, _, lr2 = apply_resume(ck, model2, opt2, scaler2)
        diffs = {k: state_equal(a, b) for k, (a, b) in dict(
            model=(model.state_dict(), model2.state_dict()), opt=(opt.state_dict(), opt2.state_dict()),
            scaler=(scaler.state_dict(), scaler2.state_dict()), file_vs_live_model=(saved['model'], ck['model'])).items()}
        nxt = train_step(model2, opt2, scaler2, make_batch(steps, gen), loss_fn, kit_lr(step2, max_steps, 0.0, 3.6e6, lr0, warmup),
                         device, dtype)
        res = dict(steps=steps, step_loaded=step2, lr_saved=lr_next, lr_loaded=lr2, scaler_state=scaler2.state_dict(),
                   diffs=diffs, next_loss=nxt, losses_first_last=[losses[0], losses[-1]],
                   ok=all(not v for v in diffs.values()) and step2 == steps and lr2 == lr_next and math.isfinite(nxt)
                   and all(math.isfinite(x) for x in losses))
        del model, model2, opt, opt2, ck, saved
        os.remove(path)
    return res
