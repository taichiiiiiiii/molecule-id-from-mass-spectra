"""Issue #82: キットの train_fpnet.py の MPS 版（デバイスの選択・途中保存と再開・勾配の累積だけを足したもの）のテスト。

- 原本の写しの sha256 が配布元（dmitriigluzdov/casmi26-fold-safe-fpnet の SHA256SUMS）と一致する
- MPS 版は make_mps.py で原本から機械的に作られ、差分は印（# [port:device] / # [port:resume] / # [port:memory]）の付いた行だけ
- 勾配の累積（torch と casmi のコードがあるときだけ）: 256 件の一括（原本の式の float64 の写し）と
  accum_backward(accum=1 / 2) の勾配とログの値が rtol 1e-5・atol 1e-7 で一致（11/18 昼の事前登録 (b)）
- 再開（torch と casmi のコードがあるときだけ。CI では飛ばす）: 合成データで学習 → SIGTERM → 再開し、
  step と経過時間が戻り、lr の列が途切れず、worker が残らないこと（accum 1 と 2）
- 再開ループ（macOS だけ）と測定のドライバー（CI でも走る）の止まり方
"""
import difflib
import hashlib
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "scripts/train/fpnet_kit"
ORIG = KIT / "train_fpnet_orig.py"
MPS = KIT / "train_fpnet_mps.py"
ORIG_SHA = "46b4dceebb2badeac78e0d5e55849169f5bef8da70a3adf1a7431fe34e603312"
MARKS = ("# [port:device]", "# [port:resume]", "# [port:memory]")
# 原本から消える（置き換わる）行は、この 21 行だけ（autocast の行は評価と学習で 2 回）。
# デバイスの選択・再開で 7 行、勾配の累積（accum_backward に移す）で 14 行
REMOVED_MEMORY = {
    "    target = torch.zeros(args.bs, dtype=torch.long, device=dev)",
    "        inp = tuple(torch.as_tensor(a, device=dev) for a in arrs)",
    "        pos_t = torch.as_tensor(pos, device=dev)",
    "        cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=dev)",
    "            z, el = model(*inp)",
    "        z = z.float(); el = el.float()",
    "        fp = fpnet.unpack_gpu(PFP[cand], D.nbits)",
    "        sc = torch.einsum('bkn,bn->bk', fp, z)",
    "        l_rank = F.cross_entropy(sc, target)",
    "        l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])",
    "        l_el = F.smooth_l1_loss(el, ELEM[pos_t])",
    "        loss = l_rank + args.w_bce * l_bce + args.w_el * l_el",
    "        scaler.scale(loss).backward()",
    "        vals = dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(), acc=(sc.argmax(1) == 0).float().mean().item())",
}
REMOVED = REMOVED_MEMORY | {
    "    D.rng = np.random.default_rng(args.seed * 1000 + wid + 1)",
    "        q.put((tuple(t.numpy() for t in fpnet.collate(items, 'cpu')), pos, D.negatives(pos)))",
    "        inp = fpnet.collate(items, 'cuda')",
    "        with torch.autocast('cuda', dtype=dtype):",
    "    dev = 'cuda'",
    "            log(f\"step {step} lr {lr:.2e} \" + ' '.join(f'{k} {v:.4f}' for k, v in ema.items()) + f\" {step / (time.time() - t0):.2f} it/s queue {q.qsize()}\")",
}


def _diff():
    a = ORIG.read_bytes().decode().replace("\r\n", "\n").splitlines()
    b = MPS.read_text().splitlines()
    d = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if not l.startswith(("---", "+++", "@@"))]
    return [l[1:] for l in d if l.startswith("-")], [l[1:] for l in d if l.startswith("+")]


def test_orig_is_the_kit_file():
    assert hashlib.sha256(ORIG.read_bytes()).hexdigest() == ORIG_SHA


def test_mps_file_is_generated_from_orig(tmp_path):
    out = tmp_path / "gen.py"
    subprocess.run([sys.executable, str(KIT / "make_mps.py"), str(ORIG), str(out)], check=True, capture_output=True)
    assert out.read_text() == MPS.read_text()


def test_diff_is_only_device_and_resume():
    removed, added = _diff()
    assert len(removed) == 21 and set(removed) == REMOVED     # autocast の行は評価と学習で 2 回
    assert removed.count("        with torch.autocast('cuda', dtype=dtype):") == 2
    for l in added:
        assert l.strip() == "" or l.endswith(MARKS), l
    dev = [l for l in added if l.endswith(MARKS[0])]
    assert len(dev) == 10                                      # --device・collate・autocast（評価）・dev・_qsize 5 行・ログ行
    assert len([l for l in added if l.endswith(MARKS[2])]) == 29
    assert all("cuda" not in l for l in added)
    # 学習の繰り返しでは zero_grad → accum_backward → unscale → clip → step の順（2 回の後に 1 回）
    body = MPS.read_text()
    i = [body.index(x) for x in ("        opt.zero_grad(set_to_none=True)", "        vals = accum_backward(",
                                 "        scaler.unscale_(opt)", "        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
                                 "        scaler.step(opt); scaler.update()", "        step += 1")]
    assert i == sorted(i)
    assert "ap.add_argument('--accum', type=int, default=1)" in body
    # 学習の数式（loss・lr・optimizer・引数の既定）の行は原本のまま
    body = MPS.read_text()
    for s in ("lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))",
              "prog = max(step / args.max_steps, (time.time() - t0) / budget)",
              "loss = l_rank + args.w_bce * l_bce + args.w_el * l_el",
              "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
              "lr=args.lr, betas=(0.9, 0.98))",
              "scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)"):
        assert body.count(s) == 1, s


# ------------------------------------------------------------------ 勾配の累積（11/18 昼の事前登録 (b)）

def _load_accum_backward(fpnet, torch, F):
    """MPS 版から accum_backward の関数だけを取り出す（学習のスクリプト全体は argparse と chdir があるので import しない）。"""
    import ast
    import types
    tree = ast.parse(MPS.read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "accum_backward"]
    assert len(fn) == 1
    ns = dict(torch=torch, F=F, np=np, fpnet=fpnet, args=types.SimpleNamespace(w_bce=1.0, w_el=0.1))
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(MPS), "exec"), ns)
    return ns["accum_backward"]


def _ref_full(model, arrs, pos, negs, PFP, ELEM, nbits, fdt, torch, F, fpnet):
    """原本の 125・134〜145・147・152 行の写し（一括、bs 件）。原本との違いは
    `.float()` → `.to(fdt)`、入力・fp・ELEM に `.to(fdt)`（float32 では何もしない）、デバイスは cpu、autocast は enabled=False、
    無効の GradScaler の代わりに loss.backward() だけ。"""
    dev = "cpu"
    target = torch.zeros(len(pos), dtype=torch.long, device=dev)
    inp = tuple(torch.as_tensor(a, device=dev) for a in arrs)
    inp = tuple(t.to(fdt) if t.is_floating_point() else t for t in inp)
    pos_t = torch.as_tensor(pos, device=dev)
    cand = torch.as_tensor(np.concatenate([pos[:, None], negs], 1), device=dev)
    with torch.autocast(dev, dtype=fdt, enabled=False):
        z, el = model(*inp)
    z = z.to(fdt); el = el.to(fdt)
    fp = fpnet.unpack_gpu(PFP[cand], nbits).to(fdt)
    sc = torch.einsum('bkn,bn->bk', fp, z)
    l_rank = F.cross_entropy(sc, target)
    l_bce = F.binary_cross_entropy_with_logits(z, fp[:, 0])
    l_el = F.smooth_l1_loss(el, ELEM[pos_t].to(fdt))
    loss = l_rank + 1.0 * l_bce + 0.1 * l_el
    loss.backward()
    return dict(loss=loss.item(), rank=l_rank.item(), bce=l_bce.item(), el=l_el.item(),
                acc=(sc.argmax(1) == 0).to(fdt).mean().item())


def test_accum_gradient_equals_one_shot():
    torch = pytest.importorskip("torch")
    code = _code_dir()
    if code is None:
        pytest.skip("casmi のコード（FPNET_CODE_DIR）が無い")
    sys.path.insert(0, str(code))
    try:
        from casmi import fpnet
    finally:
        sys.path.remove(str(code))
    import torch.nn.functional as F
    accum_backward = _load_accum_backward(fpnet, torch, F)
    rng = np.random.default_rng(0)
    B, K, nbits, n_pool, fdt = 256, 8, 64, 500, torch.float64
    items = []
    for i in range(B):
        n = int(rng.integers(3, 41)); prec = float(rng.uniform(200, 600))
        mz = np.sort(rng.uniform(50, prec, n)).astype(np.float32); it = rng.uniform(0.05, 1, n).astype(np.float32)
        items.append(dict(mz=mz, it=it, prec=prec, adduct_ix=int(rng.integers(0, 3)), ce=float(rng.uniform(0, 40)),
                          ce_known=float(i % 2), n_merged=int(rng.integers(1, 5)), mode=float(1 - 2 * (i % 3 == 0))))
    arrs = tuple(t.numpy() for t in fpnet.collate(items, "cpu"))
    pos = rng.integers(0, n_pool, B); negs = rng.integers(0, n_pool, (B, K))
    PFP = torch.as_tensor(rng.integers(0, 256, (n_pool, nbits // 8), dtype=np.uint8))
    ELEM = torch.as_tensor(rng.uniform(0, 5, (n_pool, 10)))
    torch.manual_seed(0)
    model = fpnet.FPNet(nbits, d=32, layers=2, heads=4, drop=0.0).double()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    def grads():
        g = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        return g

    model.zero_grad(set_to_none=True)
    v_ref = _ref_full(model, arrs, pos, negs, PFP, ELEM, nbits, fdt, torch, F, fpnet); g_ref = grads()
    out = {}
    for accum in (1, 2):
        target = torch.zeros(B // accum, dtype=torch.long)
        v = accum_backward(model, arrs, pos, negs, PFP, ELEM, nbits, target, accum, "cpu", torch.float32, scaler, fdt=fdt)
        out[accum] = (v, grads())
    assert len(g_ref) == len(list(model.parameters()))
    for accum, (v, g) in out.items():
        assert set(g) == set(g_ref)
        for n in g_ref:
            assert g[n].dtype == torch.float64
            assert torch.allclose(g[n], g_ref[n], rtol=1e-5, atol=1e-7), (accum, n, float((g[n] - g_ref[n]).abs().max()))
        for k in v_ref:
            assert v[k] == pytest.approx(v_ref[k], rel=1e-5, abs=1e-7), (accum, k)
    # 2 つの半分が実際に違う勾配を足している（片方だけでは一括と合わない）ことも見る
    target = torch.zeros(B // 2, dtype=torch.long)
    accum_backward(model, tuple(a[:128] for a in arrs), pos[:128], negs[:128], PFP, ELEM, nbits, target, 1, "cpu",
                   torch.float32, scaler, fdt=fdt)
    g_half = grads()
    assert any(not torch.allclose(g_half[n] / 2, g_ref[n], rtol=1e-5, atol=1e-7) for n in g_ref)
    with pytest.raises(AssertionError):
        accum_backward(model, arrs, pos, negs, PFP, ELEM, nbits, torch.zeros(B, dtype=torch.long), 2, "cpu",
                       torch.float32, scaler, fdt=fdt)


# ------------------------------------------------------------------ 再開（合成データ）

def _code_dir():
    p = os.environ.get("FPNET_CODE_DIR", str(ROOT / "data/casmi26-fpnet-code"))
    return Path(p) if (Path(p) / "casmi/traindata.py").exists() else None


def _make_work(work: Path, rng):
    import pandas as pd
    work.mkdir(parents=True)
    n_pool, nbits, n_struct = 400, 64, 40
    mass = np.sort(rng.uniform(150, 600, n_pool)); mass[1::2] = mass[0::2] + 1e-4      # 窓に 2 つ以上
    keys = [f"K{i:05d}" for i in range(n_pool)]
    pd.DataFrame(dict(key=keys, mass=mass, formula=["C6H12O6"] * n_pool)).to_parquet(work / "pool_meta.parquet")
    np.save(work / "pool_fp.npy", rng.integers(0, 256, (n_pool, nbits // 8), dtype=np.uint8))
    np.save(work / "fp_bits.npy", np.arange(nbits))
    pids = rng.choice(n_pool, n_struct, replace=False)
    pd.DataFrame(dict(sid=np.arange(n_struct), key=[keys[p] for p in pids])).to_parquet(work / "train_structs.parquet")
    pd.DataFrame(dict(sid=[0, 1], fold=["hold", "np"])).to_parquet(work / "split.parquet")
    sid = np.repeat(np.arange(n_struct), 3); n = len(sid)
    npk = rng.integers(5, 20, n); off = np.concatenate([[0], np.cumsum(npk)]).astype(np.int64)
    prec = mass[pids][sid] + 1.007
    mz = np.concatenate([rng.uniform(50, p - 1, k) for p, k in zip(prec, npk)]).astype(np.float32)
    it = rng.uniform(0.01, 1, off[-1]).astype(np.float32)
    np.save(work / "spec_off.npy", off); np.save(work / "spec_mz.npy", mz); np.save(work / "spec_it.npy", it)
    pd.DataFrame(dict(sid=sid, adduct_ix=0, mode=np.where(np.arange(n) % 2, 1, -1), precursor_mz=prec,
                      ce_mean=rng.uniform(10, 40, n), ce_n=1, lib=0)).to_parquet(work / "spec_meta.parquet")


def _cmd(code, run, steps, accum=1):
    return [sys.executable, "-u", str(MPS), "--code", str(code), "--run", str(run), "--holdout", "", "--hours", "1000",
            "--max_steps", str(steps), "--device", "cpu", "--d", "32", "--layers", "1", "--bs", "8", "--workers", "2",
            "--eval_every", "100000", "--eval_groups", "5", "--warmup", "50", "--ckpt_every_min", "0", "--accum", str(accum)]


def _children(pid):
    r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    r = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(r.stdout.strip()) and not r.stdout.strip().startswith("Z")


@pytest.mark.parametrize("accum", [1, 2])
def test_resume_continues_step_elapsed_lr_and_stops_workers(tmp_path, accum):
    torch = pytest.importorskip("torch")
    code = _code_dir()
    if code is None:
        pytest.skip("casmi のコード（FPNET_CODE_DIR）が無い")
    run = tmp_path / "run"; run.mkdir()
    _make_work(run / "work", np.random.default_rng(0))
    steps = 300
    log1 = open(tmp_path / "r1.log", "w")
    p = subprocess.Popen(_cmd(code, run, steps, accum), stdout=log1, stderr=subprocess.STDOUT)
    ck = run / "fpnet_resume.pt"
    t_end = time.time() + 240
    saved = None
    while time.time() < t_end:
        if ck.exists():
            try:
                saved = torch.load(ck, map_location="cpu", weights_only=False)
            except Exception:
                saved = None
            if saved is not None and saved["step"] >= 40:
                break
        assert p.poll() is None, (tmp_path / "r1.log").read_text()
        time.sleep(0.2)
    assert saved is not None and saved["step"] >= 40
    kids = _children(p.pid)
    assert len(kids) >= 2
    p.send_signal(signal.SIGTERM)
    assert p.wait(30) == 128 + signal.SIGTERM
    time.sleep(1.0)
    assert not any(_alive(k) for k in kids)                    # worker は残らない
    last = torch.load(ck, map_location="cpu", weights_only=False)
    assert steps > last["step"] >= saved["step"]
    s_saved, e_saved = last["step"], last["elapsed"]
    r2 = subprocess.run(_cmd(code, run, steps, accum), capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    out = r2.stdout
    line = [l for l in out.splitlines() if "resumed from" in l]
    assert len(line) == 1
    f = line[0].split()
    assert int(f[f.index("step") + 1]) == s_saved
    assert abs(float(f[f.index("elapsed_s") + 1]) - round(e_saved, 1)) < 1e-6
    assert "ALL_DONE" in out
    fin = torch.load(run / "fpnet_resume.pt", map_location="cpu", weights_only=False)
    assert fin["step"] == steps and fin["elapsed"] > e_saved
    final = torch.load(run / "fpnet_final.pt", map_location="cpu", weights_only=False)
    assert final["step"] == steps
    # lr は step（と経過時間）だけの式なので、最後のステップの lr が式の値と一致すれば列は途切れていない
    prog = (steps - 1) / steps
    want = 3e-4 * min(1.0, steps / 50) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
    for g in fin["opt"]["param_groups"]:
        assert g["lr"] == pytest.approx(want, rel=1e-9)
    # optimizer の状態も引き継がれる（AdamW の内部の step がステップ数と一致）
    st = next(iter(fin["opt"]["state"].values()))
    assert int(st["step"]) == steps


# ------------------------------------------------------------------ 再開ループの判定（偽の memguard と偽の学習）

LOOP = KIT / "fpnet_mps_loop.sh"


def _loop(tmp_path, script_body, **env):
    out = tmp_path / "out"; out.mkdir(); run = tmp_path / "run"; (run / "work").mkdir(parents=True)
    (run / "work" / "a.bin").write_bytes(b"x")
    sha = tmp_path / "sha.txt"
    sha.write_text(hashlib.sha256(b"x").hexdigest() + "  a.bin\n")
    guard = tmp_path / "guard.sh"
    guard.write_text('LOG="$1"; shift; "$@" > "$LOG" 2>&1; rc=$?; echo "exit=$rc" >> "$LOG"; exit $rc\n')
    fake = tmp_path / "train_fpnet_mps.py"
    fake.write_text(script_body)
    e = dict(os.environ, MEMGUARD=str(guard), MIN_AVAIL_GB="0", WAIT_STEP_S="1", **env)
    r = subprocess.run(["bash", str(LOOP), str(out), str(run), str(sha), "--", sys.executable, str(fake), "--run", str(run)],
                       env=e, capture_output=True, text=True, timeout=60)
    return r.returncode, (out / "loop.log").read_text(), sha, run


pytestmark_darwin = pytest.mark.skipif(sys.platform != "darwin", reason="vm_stat は macOS だけ")


@pytestmark_darwin
def test_loop_done(tmp_path):
    rc, log, _, _ = _loop(tmp_path, "print('step 200 lr 1e-4'); print('ALL_DONE')\n")
    assert rc == 0 and "decision=done" in log and "sha256 check ok (1 files)" in log


@pytestmark_darwin
def test_loop_resumes_then_done(tmp_path):
    body = ("import os, sys\np = os.path.join(os.path.dirname(__file__), 'count')\n"
            "n = int(open(p).read()) if os.path.exists(p) else 0\nopen(p, 'w').write(str(n + 1))\n"
            "print('step 200 lr 1e-4')\nif n < 2:\n    sys.exit(1)\nprint('ALL_DONE')\n")
    rc, log, _, _ = _loop(tmp_path, body)
    assert rc == 0 and log.count("decision=resume rc_1") == 2 and "round 3 rc=0 decision=done" in log


@pytestmark_darwin
def test_loop_stops_on_nonfinite_and_max_restarts(tmp_path):
    rc, log, _, _ = _loop(tmp_path, "print('non-finite loss at step', 5)\nprint('ALL_DONE')\n")
    assert rc == 2 and "stop non_finite_loss" in log
    (tmp_path / "b").mkdir()
    rc, log, _, _ = _loop(tmp_path / "b", "import sys; sys.exit(99)\n")
    assert rc == 3 and log.count("decision=resume rc_99") == 6 and "stop max_restarts=5" in log


@pytestmark_darwin
def test_loop_stops_on_sha_mismatch(tmp_path):
    body = "print('ALL_DONE')\n"
    out = tmp_path / "out"; out.mkdir(); run = tmp_path / "run"; (run / "work").mkdir(parents=True)
    (run / "work" / "a.bin").write_bytes(b"y")
    sha = tmp_path / "sha.txt"; sha.write_text(hashlib.sha256(b"x").hexdigest() + "  a.bin\n")
    fake = tmp_path / "train_fpnet_mps.py"; fake.write_text(body)
    r = subprocess.run(["bash", str(LOOP), str(out), str(run), str(sha), "--", sys.executable, str(fake)],
                       env=dict(os.environ, MIN_AVAIL_GB="0"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "sha256 check FAILED" in (out / "loop.log").read_text()


def _swapfile(tmp_path, mb):
    f = tmp_path / "swap.txt"
    f.write_text(f"total = 6144.00M  used = {mb:.2f}M  free = 100.00M  (encrypted)\n")
    return f


_SWAP_TRAIN = """import os, signal, sys, time
sw = os.environ['FAKE_SWAP']
p = os.path.join(os.path.dirname(__file__), 'count')
n = int(open(p).read()) if os.path.exists(p) else 0
open(p, 'w').write(str(n + 1))
def term(*a):
    if os.environ.get('FAKE_RESET'):
        open(sw, 'w').write('total = 6144.00M  used = 1000.00M  free = 1.00M\\n')
    os._exit(143)
signal.signal(signal.SIGTERM, term)
print('step 200 lr 1e-4', flush=True)
if n >= 1:
    print('ALL_DONE'); sys.exit(0)
open(sw, 'w').write('total = 6144.00M  used = 4000.00M  free = 1.00M\\n')
time.sleep(60)
"""


@pytestmark_darwin
def test_loop_swap_stop_waits_and_gives_up(tmp_path):
    f = _swapfile(tmp_path, 1000)
    rc, log, _, _ = _loop(tmp_path, _SWAP_TRAIN, SWAP_CMD=f"cat {f}", FAKE_SWAP=str(f), SWAP_CHECK_S="1", WAIT_MAX_S="2")
    assert "swap_base_mb=1000.00" in log
    assert "swap used 4000.00 MB >= 1000.00+2048 -> SIGTERM main" in log
    assert "round 1 rc=143 decision=resume swap" in log
    assert "give up" in log and rc == 5 and "round 2 start" not in log


@pytestmark_darwin
def test_loop_swap_stop_resumes_when_swap_comes_down(tmp_path):
    f = _swapfile(tmp_path, 1000)
    rc, log, _, _ = _loop(tmp_path, _SWAP_TRAIN, SWAP_CMD=f"cat {f}", FAKE_SWAP=str(f), FAKE_RESET="1", SWAP_CHECK_S="1")
    assert "round 1 rc=143 decision=resume swap" in log
    assert rc == 0 and "round 2 rc=0 decision=done" in log


# ------------------------------------------------------------------ 測定のドライバー（偽の memguard と偽の学習。CI でも走る）

SPEED = KIT / "fpnet_mps_speed.sh"

_FAKE_TRAIN = """import json, os, signal, sys, time
mode = os.environ.get('FAKE_MODE', 'normal')
run = sys.argv[sys.argv.index('--run') + 1]
def term(*a):
    os._exit(1 if mode == 'rc1' else 143)
signal.signal(signal.SIGTERM, term)
if mode != 'noresume':
    json.dump(dict(step=7, elapsed=1.5), open(os.path.join(run, 'fpnet_resume.pt'), 'w'))
if mode == 'swap':
    open(os.environ['FAKE_SWAP'], 'w').write('total = 6144.00M  used = 9000.00M  free = 1.00M\\n')
if mode == 'early':
    sys.exit(0)
time.sleep(60)
"""


def _speed(tmp_path, mode="normal", guard_body=None, script_name="train_fpnet_mps.py", **env):
    sp = tmp_path / "speed"; sp.mkdir()
    guard = tmp_path / "guard.sh"
    guard.write_text(guard_body or 'LOG="$1"; shift; "$@" > "$LOG" 2>&1; rc=$?; echo "exit=$rc" >> "$LOG"; exit $rc\n')
    fake = tmp_path / script_name; fake.write_text(_FAKE_TRAIN)
    sw = _swapfile(tmp_path, 1000)
    chk = f"'{sys.executable}' -c \"import json, sys; d = json.load(open(sys.argv[1])); print('resume step', d['step'])\""
    e = dict(os.environ, GUARD=str(guard), T1="3", T2="3", POLL_S="1", PID_LOOKUP_S="1", NO_MONITOR="1",
             AVAIL_CMD="echo 50", SWAP_CMD=f"cat {sw}", RESUME_CHECK=chk, FAKE_MODE=mode, FAKE_SWAP=str(sw), **env)
    r = subprocess.run(["bash", str(SPEED), str(sp), "--", sys.executable, str(fake), "--run", str(sp), "--x", "1"],
                       env=e, capture_output=True, text=True, timeout=120)
    return r.returncode, (sp / "speed_driver.log").read_text()


def test_speed_normal_two_rounds(tmp_path):
    rc, log = _speed(tmp_path)
    assert rc == 0, log
    assert "round 1 memguard rc=143" in log and "round 2 memguard rc=143" in log
    assert "before round 2: resume step 7" in log and "done: 2 rounds" in log


def test_speed_memguard_99_stops_further_rounds(tmp_path):
    g = 'LOG="$1"; shift; "$@" > "$LOG" 2>&1 & p=$!; sleep 0.3; kill -TERM $p; wait $p; exit 99\n'
    rc, log = _speed(tmp_path, guard_body=g)
    assert rc == 8 and "memguard rc=99" in log and "no further rounds" in log
    assert "round 2 start" not in log


def test_speed_early_exit_stops(tmp_path):
    rc, log = _speed(tmp_path, mode="early")
    assert rc == 8 and "round 1 ended before the driver's SIGTERM" in log and "round 2 start" not in log


def test_speed_resume_load_failure_stops(tmp_path):
    rc, log = _speed(tmp_path, mode="noresume")
    assert rc == 4 and "round 2: cannot load" in log and "round 2 start" not in log


def test_speed_swap_increase_stops(tmp_path):
    rc, log = _speed(tmp_path, mode="swap")
    assert rc == 6 and "swap used 9000.00 MB >= 1000.00+2048" in log and "round 2 start" not in log


def test_speed_main_pid_not_found_stops(tmp_path):
    rc, log = _speed(tmp_path, script_name="other_name.py")
    assert rc == 7 and "main training pid not found" in log and "round 2 start" not in log


def test_speed_rc_not_143_after_sigterm_stops(tmp_path):
    rc, log = _speed(tmp_path, mode="rc1")
    assert rc == 9 and "round 1 rc=1 after SIGTERM (expected 143)" in log and "round 2 start" not in log
