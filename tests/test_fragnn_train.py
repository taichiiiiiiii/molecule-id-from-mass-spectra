"""Issue #120: fragnn_train（本番の学習）の純粋な関数・再開・止め方・漏れの assert・再開ループの判定（合成データ）。

torch が無い CI では学習の核のテストを飛ばす（pytest.importorskip）。再開ループのテストは bash と偽の memguard で動かす。
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fragnn_train", ROOT / "scripts/train/fragnn_train.py")
ft = importlib.util.module_from_spec(_spec)
sys.modules["fragnn_train"] = ft
_spec.loader.exec_module(ft)
LOOP = ROOT / "scripts/train/fragnn_loop.sh"
SPEED = ROOT / "outputs/fragnn_train/speed_report.json"


# ---------------------------------------------------------------- 定数と純粋な関数

def test_constants_match_issue_120():
    c = ft.CONFIG
    assert (c["LR"], c["WD"], c["WARMUP"], c["CLIP"], c["BATCH"], c["K_CAND"], c["SEED"]) == (1e-3, 1e-5, 2000, 5.0,
                                                                                            128, 128, 42)
    assert (c["EVAL_EVERY"], c["STOP_REL"], c["STOP_PATIENCE"], c["MAX_STEPS"]) == (2000, 0.005, 5, 300_000)
    assert (c["PLATEAU_FACTOR"], c["PLATEAU_PATIENCE"], c["MIN_LR"]) == (0.5, 3, 1e-6)
    assert (c["VAL_FRAC"], c["VAL_SEED"], c["VAL_CAND_SEED"], c["PPM"]) == (0.02, 0, 1, 10.0)
    assert (c["CAP_FACTOR"], c["CAP_MAX_H"], c["negatives"]) == (1.5, 96.0, "design")
    assert (ft.EXPECTED_DROP, ft.EXPECTED_DROP_KEY, ft.EXPECTED_POOL_ALLOWED) == (55_369, 55_038, 655_670)
    assert (ft.EXPECTED_TRAIN_STRUCTS, ft.EXPECTED_VAL_STRUCTS) == (214_772, 4_383)
    assert (ft.EXPECTED_TRAIN_SPECTRA, ft.EXPECTED_VAL_SPECTRA) == (1_942_968, 39_437)
    assert ft.SPEED_REPORT_SHA256.startswith("9778ea20")


def test_epoch_batch_is_a_function_of_step():
    a = [ft.epoch_batch(s, 1000, 128, 42).copy() for s in range(20)]
    ft._PERM_CACHE.clear()
    b = [ft.epoch_batch(s, 1000, 128, 42).copy() for s in reversed(range(20))][::-1]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    per = 1000 // 128
    assert len(np.unique(np.concatenate(a[:per]))) == per * 128     # 1 エポックの中で重複しない
    assert not np.array_equal(a[0], a[per])                        # エポックごとに順列が変わる
    r1, r2 = ft.step_rng(7).integers(0, 1 << 30, 5), ft.step_rng(7).integers(0, 1 << 30, 5)
    assert np.array_equal(r1, r2)


def test_early_stop_rule():
    es = ft.EarlyStop(0.005, 5)
    assert es.update(1.0, 2000)["is_best"]
    # 0.5% 以上の改善で bad は 0 に戻る
    u = es.update(0.99, 4000)
    assert u["bad"] == 0 and u["is_best"]
    # 0.5% 未満の改善（best は更新されるが bad は増える）が 5 回で止まる
    outs = [es.update(0.99 * (1 - 0.001 * (i + 1)), 6000 + 2000 * i) for i in range(5)]
    assert [o["bad"] for o in outs] == [1, 2, 3, 4, 5]
    assert [o["stop"] for o in outs] == [False] * 4 + [True]
    assert all(o["is_best"] for o in outs) and es.best_step == 14000
    # 悪化も bad に数える。途中で 0.5% 以上の改善があれば数え直し
    es2 = ft.EarlyStop(0.005, 5)
    for i, v in enumerate([1.0, 1.1, 1.2, 0.9, 0.95]):
        u = es2.update(v, i)
    assert u["bad"] == 1 and es2.best == 0.9 and es2.best_step == 3
    st = es2.state(); es3 = ft.EarlyStop(); es3.load(st)
    assert es3.state() == st


def test_warmup_lr():
    assert ft.warmup_lr(0) == 0.0 and ft.warmup_lr(1000) == pytest.approx(5e-4) and ft.warmup_lr(5000) == 1e-3


def _speed(totals):
    names = ["design/n_union", "design/n_union_val", "allpool/n_union", "allpool/n_union_val"]
    return {"extrapolation": {n: {"28000": {"total_h": t}, "12000": {"total_h": t / 2}} for n, t in zip(names, totals)}}


def test_wallclock_cap_formula():
    assert ft.wallclock_cap_hours(_speed([4.76, 4.77, 6.69, 6.70])) == pytest.approx(1.5 * 6.70)
    assert ft.wallclock_cap_hours(_speed([10, 20, 70, 80])) == 96.0
    bad = _speed([1, 2, 3, 4]); bad["extrapolation"].pop("design/n_union")
    with pytest.raises(ValueError):
        ft.wallclock_cap_hours(bad)


@pytest.mark.skipif(not SPEED.exists(), reason="local only: outputs/fragnn_train/speed_report.json")
def test_wallclock_cap_from_real_speed_report():
    assert ft.sha256_file(SPEED) == ft.SPEED_REPORT_SHA256
    assert ft.wallclock_cap_hours(json.loads(SPEED.read_text())) == pytest.approx(10.04, abs=0.005)


def _leak_inputs(**over):
    d = dict(train_raw=["AAA", "BBB"], train_key=["kA", "kB"], split_rows_raw=[("SPL", 0)], split_rows_key=[("kS", 0)],
             holdout_keys={"v3_c1": ["HOL"]}, neg_keys=["kN1", "kN2"], drop_key={"kS", "kD"}, pos_keys=["kA", "kB"],
             train_neg_in_val_mask=[False, False], design_neg_trainstruct_mask=[True, True],
             train_pos_in_val_mask=[False, False])
    d.update(over)
    return d


def test_leak_counts_zero_passes():
    leak = ft.leak_counts(**_leak_inputs())
    assert set(leak) == {"raw", "key", "neg_in_drop", "pos_in_drop", "train_neg_in_val", "design_neg_not_trainstruct",
                         "train_pos_in_val"}
    assert all(v == 0 for v in leak.values())
    ft.check_leaks(leak)


@pytest.mark.parametrize("over,field", [
    (dict(train_raw=["AAA", "SPL"]), "raw"),                      # 学習の生キーが split にある
    (dict(train_raw=["AAA", "HOL"]), "raw"),                      # 学習の生キーが v3 の集合にある
    (dict(train_key=["kA", "kS"]), "key"),                        # 学習の正準キーが split にある
    (dict(neg_keys=["kN1", "kD"]), "neg_in_drop"),                # 負例が DROP_KEY にある
    (dict(pos_keys=["kA", "kD"]), "pos_in_drop"),                 # 正解が DROP_KEY にある
    (dict(train_neg_in_val_mask=[True, False]), "train_neg_in_val"),
    (dict(design_neg_trainstruct_mask=[True, False]), "design_neg_not_trainstruct"),
    (dict(train_pos_in_val_mask=[False, True]), "train_pos_in_val"),
])
def test_leak_assert_fails(over, field):
    leak = ft.leak_counts(**_leak_inputs(**over))
    assert leak[field] >= 1
    with pytest.raises(AssertionError, match=field):
        ft.check_leaks(leak)


def test_assemble_padding():
    mask = np.zeros((3, 4), bool); mask[:, 0] = True; mask[1, 1:3] = True
    cand = np.arange(12).reshape(3, 4)
    n, p, cs, sp, slot, valid = ft.assemble(cand, mask, np.array([5, 6, 7]), np.array([0] * 5 + [3, 32, 1]), pad_to=8)
    assert (n, p) == (5, 8)
    assert cs[:n].tolist() == [0, 4, 5, 6, 8] and sp[:n].tolist() == [5, 6, 6, 6, 7]
    assert slot[:n].tolist() == [0, 4, 5, 6, 8] and slot[n:].tolist() == [12] * 3
    assert valid.shape == (8, 32) and valid[0].sum() == 3 and valid[1].sum() == 32 and valid[4].sum() == 1
    assert not valid[n:].any()


def test_trim_log_and_curve(tmp_path):
    lp = tmp_path / "log.jsonl"
    lp.write_text("".join(json.dumps({"step": s, "train_loss": 1.0, "val_loss": 2.0, "lr": 1e-3,
                                      "train_elapsed_s": 1.0}) + "\n" for s in (2, 4, 4, 6)))
    assert ft.trim_log(lp, 4) == 2
    rows = [json.loads(x) for x in lp.read_text().splitlines()]
    assert [r["step"] for r in rows] == [2, 4]
    csv = ft.curve_csv(rows)
    assert csv.splitlines()[0] == "step,train_loss,val_loss,lr,train_elapsed_s" and len(csv.splitlines()) == 3
    png = tmp_path / "c.png"
    ft.curve_png(rows, png)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_png_writer_without_matplotlib(tmp_path):
    img = np.zeros((4, 5, 3), np.uint8); img[1, 2] = (255, 0, 0)
    b = ft._png_bytes(img)
    assert b[:8] == b"\x89PNG\r\n\x1a\n" and b[12:16] == b"IHDR"


def test_no_data_root_scripts_on_sys_path(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    ft.assert_code_data_separation(tmp_path)
    monkeypatch.setattr(sys, "path", [str(tmp_path / "scripts" / "train")] + sys.path)
    with pytest.raises(SystemExit):
        ft.assert_code_data_separation(tmp_path)


def test_speed_report_sha_checked(tmp_path):
    src = tmp_path / "report.json"; src.write_text("{}")
    (tmp_path / "out").mkdir(); (tmp_path / "out2").mkdir()
    with pytest.raises(SystemExit, match="sha256"):
        ft.ensure_speed_report(tmp_path / "out", src)
    with pytest.raises(SystemExit, match="not found"):
        ft.ensure_speed_report(tmp_path / "out2", tmp_path / "x.json")


# ---------------------------------------------------------------- 学習の核（torch）

def _synthetic(n_struct=40, n_train_struct=30, n_spec=400, n_train_spec=300, seed=0):
    import fragnn_bench as B
    rng = np.random.default_rng(seed)
    levels = []
    for _ in range(n_struct):
        levels.append([np.sort(rng.uniform(50, 400, rng.integers(1, 4))), np.sort(rng.uniform(30, 300, 12)),
                       np.sort(rng.uniform(20, 250, 20))])
    masses, off = B.levels_to_csr(levels)
    rows = []
    for i in range(n_spec):
        k = int(rng.integers(5, 40))
        rows.append(dict(mz=rng.uniform(20, 400, k), it=rng.uniform(0.01, 1, k), prec=410.0,
                         adduct="[M+H]+" if i % 3 else "[M-H]-", mode=1 if i % 3 else -1, ce=20.0))
    spec = B.pack_spectra(rows)
    tr_i, va_i = np.arange(n_train_spec), np.arange(n_train_spec, n_spec)
    pos_struct = np.where(np.arange(n_spec) < n_train_spec, np.arange(n_spec) % n_train_struct,
                          n_train_struct + np.arange(n_spec) % (n_struct - n_train_struct))
    cid = np.where(np.arange(n_spec) < n_train_spec, pos_struct, pos_struct - n_train_struct)

    def csr(owners, pool):
        offs, flat = [0], []
        for o in owners:
            cands = [x for x in pool if x != o]
            sel = rng.choice(cands, size=int(rng.integers(0, len(cands))), replace=False)
            flat.extend(sel.tolist()); offs.append(len(flat))
        return np.array(offs, np.int64), np.array(flat, np.int64)
    off_tr, flat_tr = csr(range(n_train_struct), range(n_train_struct))
    off_va, flat_va = csr(range(n_train_struct, n_struct), range(n_train_struct))
    return dict(masses=masses, off=off, spec=spec, tr_i=tr_i, va_i=va_i, pos_struct=pos_struct, cid=cid,
                off_tr=off_tr, flat_tr=flat_tr, off_va=off_va, flat_va=flat_va)


META = {"keys_json_sha256": "k", "frag_rescore_sha256": "f", "wallclock_cap_s": 1e9}


def _params(path):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["model"], st


def test_resume_continues_in_the_same_order(tmp_path):
    torch = pytest.importorskip("torch")
    D = _synthetic()
    seen_a, seen_b = {}, {}
    rec = lambda d: (lambda s, bi, cand, mask: d.setdefault(s, (bi.copy(), cand.copy(), mask.copy())))  # noqa: E731
    ra = ft.run_training(D, tmp_path / "a", META, dev="cpu", max_steps=6, eval_every=3, ckpt_every=2,
                         mem_check=lambda: {"avail_gb": None}, on_batch=rec(seen_a))
    assert ra["status"] == "ok" and ra["train"]["stop_reason"] == "max_steps" and ra["train"]["steps"] == 6
    # b: step 5 で落ちる（state は step 4 の ckpt まで）→ 再開して 6 まで
    with pytest.raises(ft._Abort):
        ft.run_training(D, tmp_path / "b", META, dev="cpu", max_steps=6, eval_every=3, ckpt_every=2,
                        mem_check=lambda: {"avail_gb": None}, on_batch=rec(seen_b), abort_at=5)
    _, st = _params(tmp_path / "b/state.pt")
    assert st["step"] == 4
    seen_b2 = {}
    rb = ft.run_training(D, tmp_path / "b", META, dev="cpu", resume=True, max_steps=6, eval_every=3, ckpt_every=2,
                         mem_check=lambda: {"avail_gb": None}, on_batch=rec(seen_b2))
    assert rb["resumed_from_step"] == 4 and rb["status"] == "ok"
    assert sorted(seen_b2) == [4, 5]
    for s in range(6):
        b = seen_b2.get(s, seen_b.get(s))
        assert all(np.array_equal(x, y) for x, y in zip(seen_a[s], b)), s
    pa, _ = _params(tmp_path / "a/state.pt")
    pb, _ = _params(tmp_path / "b/state.pt")
    for k in pa:
        assert torch.equal(pa[k], pb[k]), k
    la = [json.loads(x) for x in (tmp_path / "a/train_log.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (tmp_path / "b/train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in la] == [r["step"] for r in lb] == [3, 6]
    assert [r["val_loss"] for r in la] == [r["val_loss"] for r in lb]
    assert (tmp_path / "a/learning_curve.csv").exists() and (tmp_path / "a/learning_curve.png").exists()
    assert (tmp_path / "a/model_best.pt").exists() and "model_best.pt" in ra["outputs"]
    # 終わった学習は再開しない
    with pytest.raises(SystemExit, match="status ok"):
        ft.run_training(D, tmp_path / "a", META, dev="cpu", resume=True, max_steps=6, eval_every=3)


def test_resume_refuses_meta_mismatch_and_missing_flag(tmp_path):
    pytest.importorskip("torch")
    D = _synthetic()
    with pytest.raises(ft._Abort):
        ft.run_training(D, tmp_path, META, dev="cpu", max_steps=6, eval_every=3, ckpt_every=2,
                        mem_check=lambda: {"avail_gb": None}, abort_at=3)
    with pytest.raises(SystemExit, match="--resume"):
        ft.run_training(D, tmp_path, META, dev="cpu", max_steps=6, eval_every=3)
    with pytest.raises(SystemExit, match="meta"):
        ft.run_training(D, tmp_path, dict(META, keys_json_sha256="other"), dev="cpu", resume=True, max_steps=6,
                        eval_every=3)


def test_stops_on_the_rule(tmp_path):
    pytest.importorskip("torch")
    D = _synthetic()
    vals = {1: 1.0, 2: 0.9, 3: 0.899, 4: 0.898, 5: 0.95, 6: 0.897, 7: 0.896, 8: 0.5}
    r = ft.run_training(D, tmp_path, META, dev="cpu", max_steps=50, eval_every=1, mem_check=lambda: {"avail_gb": None},
                        val_override=lambda s: vals[s])
    # step 2 の改善は 10%（bad 0）、3〜7 は 0.5% 未満が 5 回 → step 7 で止まる。最良は step 7（0.896）
    assert r["status"] == "ok" and r["train"]["stop_reason"] == "converged" and r["train"]["steps"] == 7
    assert r["train"]["best_step"] == 7 and r["train"]["best_val_loss"] == pytest.approx(0.896)
    log = [json.loads(x) for x in (tmp_path / "train_log.jsonl").read_text().splitlines()]
    assert [x["bad"] for x in log] == [0, 0, 1, 2, 3, 4, 5]
    cfg = json.loads((tmp_path / "fragnn_config.json").read_text())
    assert cfg["best_step"] == 7 and cfg["stop_reason"] == "converged"


def test_wallclock_cap_stops_and_survives_resume(tmp_path):
    pytest.importorskip("torch")
    D = _synthetic()
    r = ft.run_training(D, tmp_path, META, dev="cpu", max_steps=50, eval_every=10, cap_s=1e-9,
                        mem_check=lambda: {"avail_gb": None})
    assert r["status"] == "ok" and r["train"]["stop_reason"] == "wallclock_cap" and r["train"]["steps"] == 1
    _, st = _params(tmp_path / "state.pt")
    assert st["step"] == 1 and st["train_elapsed_s"] > 0 and st["wallclock_cap_s"] == 1e-9


def test_elapsed_accumulates_across_resume_and_cap_on_resume(tmp_path):
    pytest.importorskip("torch")
    D = _synthetic()
    with pytest.raises(ft._Abort):
        ft.run_training(D, tmp_path, META, dev="cpu", max_steps=50, eval_every=10, ckpt_every=2,
                        mem_check=lambda: {"avail_gb": None}, abort_at=3)
    _, st = _params(tmp_path / "state.pt")
    e1 = st["train_elapsed_s"]
    assert st["step"] == 2 and e1 > 0
    # 再開で上限を e1 にすると、学習せずに wallclock_cap で終える（経過秒は state から引き継ぐ）
    r = ft.run_training(D, tmp_path, META, dev="cpu", resume=True, max_steps=50, eval_every=10, cap_s=e1,
                        mem_check=lambda: {"avail_gb": None})
    assert r["skipped_training_on_resume"] and r["train"]["stop_reason"] == "wallclock_cap" and r["train"]["steps"] == 2
    assert r["train"]["train_elapsed_s"] >= round(e1, 1)


def test_paused_low_memory_then_resume(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    D = _synthetic()
    monkeypatch.setattr(ft, "MEM_CHECK_EVERY", 2)
    r = ft.run_training(D, tmp_path, META, dev="cpu", max_steps=6, eval_every=3, mem_check=lambda: {"avail_gb": 1.0})
    assert r["status"] == "paused" and r["train"]["steps"] == 2
    r2 = ft.run_training(D, tmp_path, META, dev="cpu", resume=True, max_steps=6, eval_every=3,
                         mem_check=lambda: {"avail_gb": 50.0})
    assert r2["status"] == "ok" and r2["resumed_from_step"] == 2 and r2["train"]["steps"] == 6


# ---------------------------------------------------------------- 再開ループの判定（偽の memguard）

FAKE_GUARD = """#!/bin/bash
LOG="$1"; shift
if [ -n "${FAKE_GUARD_KILL:-}" ]; then echo "memguard: free 1% < 15% -> kill" >> "$LOG"; exit 99; fi
"$@" > "$LOG" 2>&1; rc=$?
echo "2026-01-01 00:00:00 exit=$rc" >> "$LOG"
exit $rc
"""


def _run_loop(tmp_path, name, cmd, env_extra=None, max_restarts=0):
    guard = tmp_path / "guard.sh"
    if not guard.exists():
        guard.write_text(FAKE_GUARD); guard.chmod(guard.stat().st_mode | stat.S_IEXEC)
    rep = tmp_path / f"{name}_report.json"
    env = dict(os.environ, MEMGUARD=str(guard), MAX_RESTARTS=str(max_restarts), WAIT_S="0", LOOP_PY=sys.executable,
               **(env_extra or {}))
    p = subprocess.run(["bash", str(LOOP), str(tmp_path / name), str(rep), "--"] + [c.replace("@REP@", str(rep))
                                                                                     for c in cmd],
                       env=env, capture_output=True, text=True, timeout=60)
    loop = (tmp_path / f"{name}.loop").read_text()
    decisions = [ln.split("decision=")[1].split(" log=")[0] for ln in loop.splitlines() if "decision=" in ln]
    return p.returncode, decisions, loop


W = "import json,sys; json.dump({'status': sys.argv[1], 'train': {'stop_reason': sys.argv[2]}}, open(sys.argv[3], 'w'))"


@pytest.mark.parametrize("name,cmd,env,exp_rc,exp", [
    ("ok_conv", [sys.executable, "-c", W, "ok", "converged", "@REP@"], None, 0, "done converged"),
    ("ok_max", [sys.executable, "-c", W, "ok", "max_steps", "@REP@"], None, 0, "done max_steps"),
    ("ok_cap", [sys.executable, "-c", W, "ok", "wallclock_cap", "@REP@"], None, 0, "done wallclock_cap"),
    ("ok_odd", [sys.executable, "-c", W, "ok", "weird", "@REP@"], None, 2, "stop status_ok_unknown_stop_reason_weird"),
    ("err", [sys.executable, "-c", W, "error", "-", "@REP@"], None, 2, "stop status_error"),
    ("running", [sys.executable, "-c", W, "running", "-", "@REP@"], None, 2, "stop rc0_status_running"),
    ("none", [sys.executable, "-c", "pass"], None, 2, "stop rc0_status_none"),
    ("unread", [sys.executable, "-c", "open(__import__('sys').argv[1], 'w').write('{')", "@REP@"], None, 2,
     "stop rc0_status_unreadable"),
    ("exc", [sys.executable, "-c", "raise RuntimeError('boom')"], None, 2, "stop rc1_other"),
    ("mps", [sys.executable, "-c", "raise SystemExit('set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 before starting')"],
     None, 2, "stop rc1_other"),
    ("meta", [sys.executable, "-c", "raise SystemExit('state の meta が今の入力と違う（再開しない）')"], None, 2,
     "stop rc1_other"),
    ("sigkill", [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGKILL)"], None, 2,
     "stop rc_137"),
    ("rc3", [sys.executable, "-c", "raise SystemExit(3)"], None, 2, "stop rc_3"),
])
def test_loop_terminal_decisions(tmp_path, name, cmd, env, exp_rc, exp):
    rc, dec, _ = _run_loop(tmp_path, name, cmd, env)
    assert dec == [exp] and rc == exp_rc


@pytest.mark.parametrize("name,cmd,env,exp", [
    ("guard", [sys.executable, "-c", "pass"], {"FAKE_GUARD_KILL": "1"}, "resume memguard_kill"),
    ("paused", [sys.executable, "-c", W, "paused", "paused_low_memory", "@REP@"], None, "resume paused_low_memory"),
    ("avail", [sys.executable, "-c",
               "raise SystemExit(\"vm_stat avail(free+speculative+inactive) {'avail_gb': 3.0} < 10.0 GB\")"], None,
     "resume start_low_avail"),
])
def test_loop_resume_decisions_and_max_restarts(tmp_path, name, cmd, env, exp):
    rc, dec, loop = _run_loop(tmp_path, name, cmd, env, max_restarts=2)
    assert dec == [exp] * 3 and rc == 3 and "stop max_restarts=2" in loop


def test_loop_resumes_then_finishes(tmp_path):
    # 1 回目 paused、2 回目 ok（converged）: 再開 1 回で終了 0
    flag = tmp_path / "flag"
    code = ("import json,sys,os; f=sys.argv[2]; st='ok' if os.path.exists(f) else 'paused'; open(f,'w').write('1');"
            " json.dump({'status': st, 'train': {'stop_reason': 'converged' if st == 'ok' else 'paused'}},"
            " open(sys.argv[1], 'w'))")
    rc, dec, _ = _run_loop(tmp_path, "two", [sys.executable, "-c", code, "@REP@", str(flag)], max_restarts=12)
    assert dec == ["resume paused_low_memory", "done converged"] and rc == 0


def test_loop_lock(tmp_path):
    lock = tmp_path / "lock"
    lock.write_text(f"pid={os.getpid()}\n")
    guard = tmp_path / "guard.sh"; guard.write_text(FAKE_GUARD); guard.chmod(0o755)
    env = dict(os.environ, MEMGUARD=str(guard), MAX_RESTARTS="0", WAIT_S="0", LOOP_PY=sys.executable,
               LOCK_FILE=str(lock))
    p = subprocess.run(["bash", str(LOOP), str(tmp_path / "locked"), str(tmp_path / "r.json"), "--", sys.executable,
                        "-c", "print('should not run')"], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 4 and not (tmp_path / "locked.1").exists() and lock.exists()
    lock.unlink()
    p = subprocess.run(["bash", str(LOOP), str(tmp_path / "free"), str(tmp_path / "r2.json"), "--", sys.executable,
                        "-c", W, "ok", "converged", str(tmp_path / "r2.json")], env=env, capture_output=True,
                       text=True, timeout=60)
    assert p.returncode == 0 and not lock.exists()


# ---------------------------------------------------------------- 端から端まで（偽のデータの root、CPU）

FAKE_FRAG = '''import zlib
import numpy as np


def resolve_cfg(cfg=None):
    return {"max_breaks": 2, "max_bonds_2": 100, "max_bonds_3": 40}


def warmup():
    pass


def fragment_levels(smi, max_breaks=2, max_bonds_2=100, max_bonds_3=40):
    if smi == "BAD":
        raise ValueError("unreadable")
    r = np.random.default_rng(zlib.crc32(smi.encode()))
    return [np.array([180.0 + len(smi)]), np.sort(r.uniform(30, 180, 10)), np.sort(r.uniform(20, 150, 15))]
'''


def _fake_roots(tmp_path):
    """train 由来 107 構造（うち 5 が split、2 が v3_c1 の正解）＋外部の 50 行。スペクトルは 1 構造 5 本。"""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    data, code = tmp_path / "data_root", tmp_path / "code_root"
    pool_dir = data / "data/ext/v4g/casmi26-v2-pool"
    for d in (pool_dir, data / "data/ext/foldsafe", data / "data/ext/frag", data / "outputs/align_embed",
              code / "kernels/sanpier_fold0_drop", data / "scripts/train"):
        d.mkdir(parents=True, exist_ok=True)
    (data / "data/ext/frag/frag_rescore.py").write_text(FAKE_FRAG)
    tk = [f"T{i:03d}" for i in range(107)]
    tr_raw = [f"R{i:03d}" for i in range(107)]
    rows = [(k, 200 + i // 6 + (i % 6) * 2e-4, "C" * (3 + i % 7)) for i, k in enumerate(tk)]
    rows += [(f"E{j:03d}", 200 + (j % 18) + 5e-4, "N" * (2 + j % 5)) for j in range(50)]
    rows.sort(key=lambda r: r[1])
    pd.DataFrame({"key": [r[0] for r in rows], "mass": [r[1] for r in rows], "formula": ["C1"] * len(rows),
                  "smiles": [r[2] for r in rows]}).to_parquet(pool_dir / "pool_meta.parquet")
    pd.DataFrame({"inchikey14": tr_raw, "key": tk}).to_parquet(pool_dir / "train_structs.parquet")
    pd.DataFrame({"inchikey14": tr_raw[100:105], "key": tk[100:105], "fold": [0] * 5}).to_parquet(
        data / "data/ext/foldsafe/split.parquet")
    keys = {"v3_c1": tr_raw[105:107], "v3_c2": [], "enveda_v1": [], "v3_c2n": []}
    (code / ft.KEYS_REL).write_text(json.dumps(keys))
    for h in ft.HOLDOUTS:
        (data / "data/holdouts" / h).mkdir(parents=True, exist_ok=True)
        (data / "data/holdouts" / h / "drop_keys.txt").write_text("T105\nT106\n" if h == "v3_c1" else "")
    rng = np.random.default_rng(3)
    spec_ik = [tr_raw[i // 5] for i in range(107 * 5)]
    order = rng.permutation(len(spec_ik))          # train.parquet の行の順は構造の順と違う
    ik_col = [spec_ik[o] for o in order]
    n = len(ik_col)
    pq.write_table(pa.table({
        "inchikey14": ik_col, "adduct": ["[M+H]+" if i % 2 else "[M-H]-" for i in range(n)],
        "ionization_mode": ["positive" if i % 2 else "negative" for i in range(n)],
        "precursor_mz": [1000.0 + i for i in range(n)],
        "ms2_mzs": [list(rng.uniform(20, 200, 8 + i % 30)) for i in range(n)],
        "ms2_normalized_intensities": [list(rng.uniform(0.01, 1, 8 + i % 30)) for i in range(n)],
        "collision_energy_ev": [[20.0, 40.0] if i % 3 else [] for i in range(n)]}),
        data / "data/train.parquet", row_group_size=50)
    pd.DataFrame({"row": np.arange(n), "kind": [0] * n, "inchikey14": ik_col}).to_parquet(
        data / "outputs/align_embed/idx_000.parquet")
    (data / "outputs/align_embed/embed_report.json").write_text(json.dumps({"n_shards": 1}))
    sp = tmp_path / "speed.json"
    sp.write_text(json.dumps(_speed([1.0, 1.0, 2.0, 2.0])))
    return data, code, sp


def _patch_expected(monkeypatch, code, sp):
    for k, v in dict(EXPECTED_DROP=7, EXPECTED_DROP_KEY=9, EXPECTED_POOL_ALLOWED=150, EXPECTED_TRAIN_STRUCTS=98,
                     EXPECTED_VAL_STRUCTS=2, EXPECTED_TRAIN_SPECTRA=490, EXPECTED_VAL_SPECTRA=10,
                     EXPECTED_STRUCTS_DESIGN=100, CODE_ROOT=code, SPEED_REPORT_SHA256=ft.sha256_file(sp)).items():
        monkeypatch.setattr(ft, k, v)


def test_load_spectra_order_and_values(tmp_path):
    import fragnn_bench as B
    data, _, _ = _fake_roots(tmp_path)
    import pyarrow.parquet as pq
    t = pq.read_table(data / "data/train.parquet").to_pydict()
    want = np.array([7, 3, 300, 0, 534, 51], np.int64)
    got = ft.load_spectra(data / "data/train.parquet", want, np.array([t["inchikey14"][i] for i in want], "S14"))
    assert got["prec"].tolist() == [1000.0 + i for i in want]
    ref = B.pack_spectra([dict(mz=t["ms2_mzs"][i], it=t["ms2_normalized_intensities"][i], prec=t["precursor_mz"][i],
                               adduct=t["adduct"][i], mode=1 if t["ionization_mode"][i] == "positive" else -1,
                               ce=float(np.mean(t["collision_energy_ev"][i])) if t["collision_energy_ev"][i] else 0.0)
                          for i in want])
    for k in ref:
        assert np.array_equal(ref[k], got[k]), k
    with pytest.raises(AssertionError, match="inchikey14"):
        ft.load_spectra(data / "data/train.parquet", want, np.array(["XXXXXXXXXXXXXX"] * len(want), "S14"))


@pytest.fixture
def isolated_frag_import(monkeypatch):
    """偽の frag_rescore を読んだあと、sys.path と sys.modules を元に戻す（ほかのテストが本物を読めるように）。"""
    monkeypatch.setattr(sys, "path", list(sys.path))
    prev = sys.modules.pop("frag_rescore", None)
    yield
    sys.modules.pop("frag_rescore", None)
    if prev is not None:
        sys.modules["frag_rescore"] = prev


def test_train_end_to_end_on_fake_roots(tmp_path, monkeypatch, isolated_frag_import):
    pytest.importorskip("torch")
    data, code, sp = _fake_roots(tmp_path)
    _patch_expected(monkeypatch, code, sp)
    out = tmp_path / "out"
    kw = dict(max_steps=4, eval_every=2, mem_check=lambda: {"avail_gb": None})
    with pytest.raises(ft._Abort):
        ft.train(data, out, device="cpu", resume=True, speed_src=sp, abort_at=3, **kw)
    r = ft.train(data, out, device="cpu", resume=True, speed_src=sp, **kw)
    assert r["status"] == "ok", r.get("error")
    assert r["prep_reused"] is True and r["resumed_from_step"] == 2 and r["train"]["steps"] == 4
    d = r["data"]
    assert (d["drop"], d["drop_key"], d["pool_allowed"], d["n_train_structs"], d["n_val_structs"]) == (7, 9, 150, 98, 2)
    assert (d["n_train_spectra"], d["n_val_spectra"], d["n_structs"]) == (490, 10, 100)
    assert all(v == 0 for v in d["leak"].values()) and len(d["leak"]) == 7
    assert r["meta"]["keys_json_sha256"] == ft.sha256_file(code / ft.KEYS_REL)
    assert r["wallclock_cap_h"] == pytest.approx(3.0)
    assert ft.sha256_file(out / "speed_report.json") == ft.sha256_file(sp)
    z = np.load(out / "prep.npz")
    rows = z["struct_rows"]
    import pandas as pd
    keys = pd.read_parquet(data / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet").key.values
    # 負例（design）はすべて train 由来で、split・v3 の正解・内部検証（学習の負例）を含まない
    neg_tr = set(keys[rows[z["flat_tr"]]]); neg_va = set(keys[rows[z["flat_va"]]])
    assert all(k.startswith("T") for k in neg_tr | neg_va)
    assert not ({f"T{i}" for i in range(100, 107)} & (neg_tr | neg_va))
    val_keys = set(keys[rows[z["pos_struct"][z["is_val"]]]])
    assert len(val_keys) == 2 and not (val_keys & neg_tr)
    # DROP の json（コードの側）が変わったら再開しない
    (out / "train_report.json").unlink()
    (code / ft.KEYS_REL).write_text(json.dumps({"v3_c1": ["R105", "R106"], "v3_c2": [], "enveda_v1": [],
                                                 "v3_c2n": []}, indent=1))
    with pytest.raises(SystemExit, match="meta"):
        ft.train(data, out, device="cpu", resume=True, speed_src=sp, **kw)
