"""Issue #82: キットの train_fpnet.py の MPS 版（デバイスの選択と途中保存・再開だけを足したもの）のテスト。

- 原本の写しの sha256 が配布元（dmitriigluzdov/casmi26-fold-safe-fpnet の SHA256SUMS）と一致する
- MPS 版は make_mps.py で原本から機械的に作られ、差分は印（# [port:device] / # [port:resume]）の付いた行だけ
- 再開（torch と casmi のコードがあるときだけ。CI では飛ばす）: 合成データで学習 → SIGTERM → 再開し、
  step と経過時間が戻り、lr の列が途切れず、worker が残らないこと
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
MARKS = ("# [port:device]", "# [port:resume]")
# 原本から消える（置き換わる）行は、この 7 行だけ（autocast の行は 2 回）
REMOVED = {
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
    assert len(removed) == 7 and set(removed) == REMOVED      # autocast の行は評価と学習で 2 回
    assert removed.count("        with torch.autocast('cuda', dtype=dtype):") == 2
    for l in added:
        assert l.strip() == "" or l.endswith(MARKS), l
    dev = [l for l in added if l.endswith(MARKS[0])]
    assert len(dev) == 11                                      # --device・collate・autocast×2・dev・_qsize 5 行・ログ行
    assert all("cuda" not in l for l in added)
    # 学習の数式（loss・lr・optimizer・引数の既定）の行は原本のまま
    body = MPS.read_text()
    for s in ("lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))",
              "prog = max(step / args.max_steps, (time.time() - t0) / budget)",
              "loss = l_rank + args.w_bce * l_bce + args.w_el * l_el",
              "torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)",
              "lr=args.lr, betas=(0.9, 0.98))",
              "scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)"):
        assert body.count(s) == 1, s


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


def _cmd(code, run, steps):
    return [sys.executable, "-u", str(MPS), "--code", str(code), "--run", str(run), "--holdout", "", "--hours", "1000",
            "--max_steps", str(steps), "--device", "cpu", "--d", "32", "--layers", "1", "--bs", "8", "--workers", "2",
            "--eval_every", "100000", "--eval_groups", "5", "--warmup", "50", "--ckpt_every_min", "0"]


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


def test_resume_continues_step_elapsed_lr_and_stops_workers(tmp_path):
    torch = pytest.importorskip("torch")
    code = _code_dir()
    if code is None:
        pytest.skip("casmi のコード（FPNET_CODE_DIR）が無い")
    run = tmp_path / "run"; run.mkdir()
    _make_work(run / "work", np.random.default_rng(0))
    steps = 300
    log1 = open(tmp_path / "r1.log", "w")
    p = subprocess.Popen(_cmd(code, run, steps), stdout=log1, stderr=subprocess.STDOUT)
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
    r2 = subprocess.run(_cmd(code, run, steps), capture_output=True, text=True, timeout=300)
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
