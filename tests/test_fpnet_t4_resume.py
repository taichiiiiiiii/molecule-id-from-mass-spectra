"""Issue #82 (11/21): session-boundary save/resume for the kit's train_fpnet.py on Kaggle T4.

Pre-registered tests (comment on #82, 2026-11-21):
 (a) save -> load: model / AdamW / GradScaler state_dicts, step and lr are bitwise equal (the kit has no lr-scheduler
     object; lr is a function of step and elapsed time, so step + elapsed + lr are what is saved and compared)
 (b) worker-free, fixed synthetic data on CPU: the loss sequence of "N steps straight" equals "stop at k, resume, go to N"
     (plus b2: the derived train_fpnet_t4.py itself, run end-to-end on CPU against a fake casmi package with constant
     batches, gives bitwise-equal final weights for one session vs two sessions)
 (c) worker RNG formula: kit formula in a fresh run, seed*1000+wid+1 + 7919*session after a resume
 plus: the self-check restores the caller's RNG state; the derived file is generated from the kit file and differs from
 it only by marked lines.
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "scripts/train/fpnet_kit"
ORIG = KIT / "train_fpnet_orig.py"
T4 = KIT / "train_fpnet_t4.py"
ORIG_SHA = "46b4dceebb2badeac78e0d5e55849169f5bef8da70a3adf1a7431fe34e603312"
MARK = "# [t4:resume]"
sys.path.insert(0, str(KIT))
import fpnet_resume as FR  # noqa: E402
import make_t4  # noqa: E402


class Tiny(torch.nn.Module):
    def __init__(self, nin=12, nout=10, drop=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(nin, 32), torch.nn.GELU(), torch.nn.Dropout(drop),
                                       torch.nn.Linear(32, nout))
        self.el = torch.nn.Linear(nin, 3)

    def forward(self, x):
        return self.net(x), self.el(x)


def tiny_batch(step, gen=None):
    g = torch.Generator().manual_seed(1000 + step) if gen is None else gen
    x = torch.randn(16, 12, generator=g)
    return dict(inp=(x,), y=(torch.rand(16, 10, generator=g) > 0.5).float(), e=torch.randn(16, 3, generator=g))


def tiny_loss(out, b):
    z, el = out
    return torch.nn.functional.binary_cross_entropy_with_logits(z, b["y"]) + 0.1 * torch.nn.functional.smooth_l1_loss(el, b["e"])


def _objs(seed=0):
    torch.manual_seed(seed)
    m = Tiny()
    return m, FR.make_opt(m), torch.amp.GradScaler("cpu", enabled=True)


def _run(model, opt, scaler, a, b, N):
    out = []
    for s in range(a, b):
        lr = FR.kit_lr(s, N, 0.0, 3.6e6, 3e-4, 20)
        out.append(FR.train_step(model, opt, scaler, tiny_batch(s), tiny_loss, lr, "cpu", torch.float32))
    return out


# ---------------------------------------------------------------- (a)
def test_a_save_load_bitwise(tmp_path):
    m, o, sc = _objs()
    _run(m, o, sc, 0, 30, 100)
    lr = FR.kit_lr(30, 100, 0.0, 3.6e6, 3e-4, 20)
    p = tmp_path / "r.pt"
    FR.save_resume(p, m, o, sc, 30, 123.5, [dict(step=5)], dict(loss=1.0), lr, 1, dict(seed=0))
    assert not (tmp_path / "r.pt.tmp").exists()
    ck = FR.load_resume(p)
    m2, o2, sc2 = _objs(seed=99)               # different init: everything must come from the file
    assert FR.state_equal(m.state_dict(), m2.state_dict())   # really different before loading
    step, el, hist, ema, lr2 = FR.apply_resume(ck, m2, o2, sc2)
    assert FR.state_equal(m.state_dict(), m2.state_dict()) == []
    assert FR.state_equal(o.state_dict(), o2.state_dict()) == []
    assert sc.state_dict() and FR.state_equal(sc.state_dict(), sc2.state_dict()) == []
    assert (step, el, hist, ema) == (30, 123.5, [dict(step=5)], dict(loss=1.0))
    assert lr2 == lr and lr2 == FR.kit_lr(step, 100, el, 3.6e6, 3e-4, 20)
    assert o2.state_dict()["state"][0]["step"].item() == 30


def test_a_state_equal_detects_one_bit():
    a = {"w": torch.ones(3)}
    b = {"w": torch.ones(3)}
    b["w"][1] = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0))
    assert FR.state_equal(a, b) == ["/w: bytes"]
    assert FR.state_equal({"x": 1}, {"x": 1.0})        # int vs float is a difference


# ---------------------------------------------------------------- (b)
def test_b_loss_sequence_straight_equals_resumed(tmp_path):
    N, k = 60, 23
    m, o, sc = _objs()
    straight = _run(m, o, sc, 0, N, N)

    m1, o1, sc1 = _objs()
    first = _run(m1, o1, sc1, 0, k, N)
    p = tmp_path / "r.pt"
    FR.save_resume(p, m1, o1, sc1, k, 0.0, [], {}, FR.kit_lr(k, N, 0.0, 3.6e6, 3e-4, 20), 1)
    torch.manual_seed(777); np.random.seed(5); random.seed(3)        # disturb every RNG before resuming
    m2, o2, sc2 = _objs(seed=42)
    step, *_ = FR.apply_resume(FR.load_resume(p), m2, o2, sc2)
    second = _run(m2, o2, sc2, step, N, N)
    assert len(first + second) == N
    assert first + second == straight                                 # bitwise float equality
    assert FR.state_equal(m.state_dict(), m2.state_dict()) == []


FAKE_FPNET = '''
import numpy as np, torch
class FPNet(torch.nn.Module):
    def __init__(self, nbits, d=512, layers=8, heads=8, drop=0.1, n_elem=10):
        super().__init__()
        self.nbits = nbits
        self.l1 = torch.nn.Linear(6, 16); self.dr = torch.nn.Dropout(0.2); self.l2 = torch.nn.Linear(16, nbits)
        self.e = torch.nn.Linear(6, n_elem)
    def forward(self, x):
        return self.l2(self.dr(torch.relu(self.l1(x)))), self.e(x)
def collate(items, device):
    return (torch.as_tensor(np.stack([it["x"] for it in items]), device=device),)
def unpack_gpu(packed, nbits):
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    return ((packed.unsqueeze(-1) >> shifts) & 1).reshape(*packed.shape[:-1], -1)[..., :nbits].float()
def autocast_dtype(device="cuda"):
    return torch.float32
def save_ckpt(path, model, step, extra=None):
    d = dict(model=model.state_dict(), step=step); d.update(extra or {}); torch.save(d, path)
def prep_peaks(mz, it, prec):
    return mz, it
'''
FAKE_TRAINDATA = '''
import numpy as np, torch
class Data:
    """Constant batches whatever D.rng is, so one session and two sessions see the same data."""
    def __init__(self, dev, seed=0, holdout_folds=()):
        g = np.random.default_rng(0)
        self.nbits = 16; npool = 40
        self.PFP = torch.as_tensor(np.packbits(g.random((npool, 16)) > 0.5, axis=1))
        self.elem = torch.as_tensor(g.random((npool, 10)), dtype=torch.float32)
        self.sid = np.arange(8); self.sid2pid = np.arange(8); self.fold = np.array(["x"] * 8); self.mode = np.zeros(8, int)
        self.tr = np.arange(8); self.X = g.random((8, 6)).astype(np.float32); self.rng = None
    def sample_ids(self, bs):
        return np.arange(bs) % 8
    def item(self, i, merge_p, aug):
        return dict(x=self.X[i])
    def negatives(self, pos):
        return (pos[:, None] + 1 + np.arange(5)[None, :]) % 40
'''


def _fake_code(tmp_path):
    code = tmp_path / "code"
    (code / "casmi").mkdir(parents=True)
    (code / "casmi/__init__.py").write_text("")
    (code / "casmi/fpnet.py").write_text(FAKE_FPNET)
    (code / "casmi/traindata.py").write_text(FAKE_TRAINDATA)
    (code / "casmi/spectra.py").write_text("def merge_spectra(x):\n    raise NotImplementedError\n")
    kit = tmp_path / "kit"
    kit.mkdir()
    # test-only device swap (the kit hard-codes cuda); the committed file is what the kernel runs
    src = T4.read_text()
    assert src.count("'cuda'") == 5
    src = src.replace("'cuda'", "'cpu'")
    if sys.platform == "darwin":          # macOS: multiprocessing.Queue.qsize() raises NotImplementedError (log line only)
        assert src.count("q.qsize()") == 1
        src = src.replace("q.qsize()", "-1")
    (kit / "train_fpnet_t4.py").write_text(src)
    (kit / "fpnet_resume.py").write_text((KIT / "fpnet_resume.py").read_text())
    return code, kit / "train_fpnet_t4.py"


def _train(script, code, run, extra, timeout=240):
    run.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-u", str(script), "--code", str(code), "--run", str(run), "--holdout", "", "--hours", "1000",
            "--bs", "8", "--workers", "1", "--eval_every", "1000000", "--warmup", "50"] + extra
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                       env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONWARNINGS="ignore"))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="fork")
def test_b2_derived_script_one_session_equals_two(tmp_path):
    code, script = _fake_code(tmp_path)
    N = 6000
    log_a = _train(script, code, tmp_path / "a", ["--max_steps", str(N)])
    assert "ALL_DONE" in log_a and "SESSION_DONE" not in log_a
    log_b1 = _train(script, code, tmp_path / "b", ["--max_steps", str(N), "--session_hours", str(1.0 / 3600)])
    assert "SESSION_DONE" in log_b1 and "ALL_DONE" not in log_b1, log_b1[-2000:]
    ck = FR.load_resume(tmp_path / "b/fpnet_resume.pt")
    k = ck["step"]
    assert 0 < k < N, k
    assert ck["session"] == 1 and ck["args"]["max_steps"] == N
    log_b2 = _train(script, code, tmp_path / "b", ["--max_steps", str(N), "--session", "2",
                                                   "--resume_from", str(tmp_path / "b/fpnet_resume.pt")])
    assert f"step {k} elapsed_s" in log_b2 and "lr continues" in log_b2 and "ALL_DONE" in log_b2
    fa = torch.load(tmp_path / "a/fpnet_final.pt", weights_only=False)
    fb = torch.load(tmp_path / "b/fpnet_final.pt", weights_only=False)
    assert fa["step"] == fb["step"] == N
    assert FR.state_equal(fa["model"], fb["model"]) == []
    # a resumed run must refuse changed kit arguments
    bad = subprocess.run([sys.executable, str(script), "--code", str(code), "--run", str(tmp_path / "b"), "--holdout", "",
                          "--hours", "1000", "--bs", "8", "--workers", "1", "--max_steps", str(N + 1), "--session", "2",
                          "--resume_from", str(tmp_path / "b/fpnet_resume.pt")], capture_output=True, text=True, timeout=120)
    assert bad.returncode != 0 and "max_steps" in bad.stderr
    # session 2 without a resume file is refused
    (tmp_path / "c").mkdir()
    bad2 = subprocess.run([sys.executable, str(script), "--code", str(code), "--run", str(tmp_path / "c"), "--holdout", "",
                           "--workers", "1", "--session", "2"], capture_output=True, text=True, timeout=120)
    assert bad2.returncode != 0 and "AssertionError" in bad2.stderr


# ---------------------------------------------------------------- (c)
def test_c_worker_seed_formula():
    for seed in (0, 1, 7):
        for wid in range(16):
            assert FR.worker_seed(seed, wid, 1, False) == seed * 1000 + wid + 1
            for session in (2, 3):
                assert FR.worker_seed(seed, wid, session, True) == seed * 1000 + wid + 1 + 7919 * session
    assert FR.worker_seed(0, 0, 2, True) == 15839
    assert "    D.rng = np.random.default_rng(FR.worker_seed(args.seed, wid, args.session, G['resumed']))  " + MARK \
        in T4.read_text().splitlines()


# ---------------------------------------------------------------- self-check
def test_selfcheck_ok_and_restores_caller_rng(tmp_path):
    torch.manual_seed(5); np.random.seed(6); random.seed(7)
    before = FR.capture_rng()
    res = FR.selfcheck(Tiny, tiny_batch, tiny_loss, "cpu", torch.float32, tmp_path / "sc.pt", steps=50)
    after = FR.capture_rng()
    assert res["ok"] and res["step_loaded"] == 50 and res["lr_loaded"] == res["lr_saved"]
    assert all(v == [] for v in res["diffs"].values()) and res["scaler_state"]["scale"] > 0
    assert FR.state_equal(before, after) == []
    assert not (tmp_path / "sc.pt").exists()
    # the caller's next draws are what they would have been without the self-check
    x1 = torch.rand(3); torch.manual_seed(5); np.random.seed(6); random.seed(7)
    assert torch.equal(x1, torch.rand(3))


def test_selfcheck_fails_on_mismatch(tmp_path, monkeypatch):
    orig = FR.apply_resume

    def broken(ck, m, o, s, restore_rng_state=True):
        out = orig(ck, m, o, s, restore_rng_state)
        with torch.no_grad():
            next(m.parameters()).add_(1e-3)
        return out
    monkeypatch.setattr(FR, "apply_resume", broken)
    res = FR.selfcheck(Tiny, tiny_batch, tiny_loss, "cpu", torch.float32, tmp_path / "sc.pt", steps=5)
    assert not res["ok"] and res["diffs"]["model"]


# ---------------------------------------------------------------- derived file
def test_orig_is_kit_file_and_t4_is_generated():
    assert hashlib.sha256(ORIG.read_bytes()).hexdigest() == ORIG_SHA
    assert make_t4.derive(ORIG.read_bytes().decode()) == T4.read_text()


def test_t4_differs_only_by_marked_lines():
    import difflib
    a = ORIG.read_bytes().decode().replace("\r\n", "\n").splitlines()
    b = T4.read_text().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    removed = [l[1:] for l in d if l.startswith("-")]
    added = [l[1:] for l in d if l.startswith("+")]
    assert removed == ["    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1)"]
    assert added and all(l.endswith(MARK) for l in added)
    # no kit hyper-parameter default changed: the argparse lines of the kit are all still there verbatim
    for l in a:
        if l.startswith("ap.add_argument"):
            assert l in b, l
