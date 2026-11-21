#!/usr/bin/env python3
"""Engine 1 の FP バンク（全 train の FPNet）をキットの train_fpnet.py で学習し直すカーネルを作る（Issue #82）。

    python scripts/train/make_fpnet_full_kernel.py --session 1   # kernels/fpnet_full_train     （セッション 1）
    python scripts/train/make_fpnet_full_kernel.py --session 2   # kernels/fpnet_full_train_s2  （セッション 2、S1 の出力から再開）

事前登録（Issue #82 のコメント 2026-11-21）のとおり:
- キットの train_fpnet.py（sha256 照合）から make_t4.py で train_fpnet_t4.py を作り（セッション区切りの途中保存と再開だけ。
  結果の sha256 がコミットした train_fpnet_t4.py と同じでなければ止まる）、引数は `--holdout '' --hours 1000 --max_steps 82855`
  とセッションの引数（--session / --session_hours 10.5 / --resume_from）だけを渡す。そのほかはキットの既定のまま。
- 学習の前に (1) CUDA の自己検査（別のモデルのコピー・別の乱数状態で 50 step → 保存 → 読み込み → ビット一致・次の loss が有限、
  resume_selfcheck.json）、(2) 結線のリハーサル（派生版を本物のデータで --seed 1 --tag rehearsal、短い区切りで止めて再開。
  出力は /kaggle/working の外に置き、本番の前に消す）。どちらかが不合格なら学習しない。
- 入力ファイルは公開されている sha256 と照合し、合わなければ学習の前に止める。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "scripts/train/fpnet_kit"
SLUG_S1 = "taichiiiii/casmi26-fpnet-full-train"
SLUG_S2 = "taichiiiii/casmi26-fpnet-full-train-s2"
PROD_STEPS = 82855
SESSION_HOURS = 10.5
REHEARSAL_HOURS = 0.03          # 108 s per rehearsal run (wiring check only, output discarded)
REHEARSAL_MIN_STEPS = 20
SELFCHECK_STEPS = 50
SELFCHECK_NBITS = 10226         # nbits of the kit's FP bank ([Data] line of the PR #83 smoke log)
DATASETS = [
    "dmitriigluzdov/casmi26-fold-safe-fpnet",
    "ahmedberatozer/casmi26-v4b-models",
    "ahmedberatozer/casmi26-v2-pool",
    "ahmedberatozer/casmi26-spec-cache",
]
# (データセット, 中の相対パス) -> 公開されている sha256（出典は Issue #82 本文の表）
EXPECT = {
    ("dmitriigluzdov/casmi26-fold-safe-fpnet", "train_fpnet.py"): "46b4dceebb2badeac78e0d5e55849169f5bef8da70a3adf1a7431fe34e603312",
    ("dmitriigluzdov/casmi26-fold-safe-fpnet", "split.parquet"): "dd62549f81b9540890e0da17a06d0dcd216a2781b2c31687bd333445fe57d45e",
    ("dmitriigluzdov/casmi26-fold-safe-fpnet", "fp_bits.npy"): "c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677",
    ("ahmedberatozer/casmi26-v4b-models", "code/casmi/fpnet.py"): "0ea6e372d1f045a877d350234c2ccf5d93fdd98cb23bd4c22376bf7e99547608",
    ("ahmedberatozer/casmi26-v4b-models", "code/casmi/traindata.py"): "cecb329d96362f11d16b40a26f1888f092610ff6202812408c13e2c293c6a2bc",
    ("ahmedberatozer/casmi26-v4b-models", "code/casmi/spectra.py"): "e2481761cdfb2515e8a945751f85f6b9e885680a8e974d7ac35c78bbf279ae74",
    ("ahmedberatozer/casmi26-v4b-models", "code/casmi/chem.py"): "99474f23d8a67040b0c0f8c70e73592a6f7f656f50103e350ca473451e2169f1",
    ("ahmedberatozer/casmi26-v2-pool", "fp_bits.npy"): "c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677",
    ("ahmedberatozer/casmi26-v2-pool", "train_structs.parquet"): "f20aa1cac23af42e430a6633707fda34e48dea97114bdc3e443897f692df3656",
    ("ahmedberatozer/casmi26-spec-cache", "cache/spec_mz.npy"): "482332fec3fabcc46e06ce1b23d79e633af7009b8190a3dcd0a675121b42342b",
    ("ahmedberatozer/casmi26-spec-cache", "cache/spec_it.npy"): "9e127f00eb9d4217403991304e69166adc9a36a5e624ce7fd1ddbf6b1417cc23",
    ("ahmedberatozer/casmi26-spec-cache", "cache/spec_off.npy"): "af2be108a532a278ec4563c023808b7fa19e40fdf4ea7dc888e980bb6f618e6c",
    ("ahmedberatozer/casmi26-spec-cache", "cache/spec_meta.parquet"): "4f968592dc75b9bf2b6256dda394be42bad3cd07fd243ff0078101e066a44376",
}
# sha256 を計算して記録するだけのもの（公開の値が見つかっていない）
RECORD_ONLY = [
    ("ahmedberatozer/casmi26-v2-pool", "pool_meta.parquet"),
    ("ahmedberatozer/casmi26-v2-pool", "pool_fp.npy"),
]
# work/ に置く名前 -> (データセット, 相対パス)
WORK_LINKS = {
    "spec_off.npy": ("ahmedberatozer/casmi26-spec-cache", "cache/spec_off.npy"),
    "spec_mz.npy": ("ahmedberatozer/casmi26-spec-cache", "cache/spec_mz.npy"),
    "spec_it.npy": ("ahmedberatozer/casmi26-spec-cache", "cache/spec_it.npy"),
    "spec_meta.parquet": ("ahmedberatozer/casmi26-spec-cache", "cache/spec_meta.parquet"),
    "split.parquet": ("dmitriigluzdov/casmi26-fold-safe-fpnet", "split.parquet"),
    "train_structs.parquet": ("ahmedberatozer/casmi26-v2-pool", "train_structs.parquet"),
    "pool_meta.parquet": ("ahmedberatozer/casmi26-v2-pool", "pool_meta.parquet"),
    "pool_fp.npy": ("ahmedberatozer/casmi26-v2-pool", "pool_fp.npy"),
    "fp_bits.npy": ("ahmedberatozer/casmi26-v2-pool", "fp_bits.npy"),
}
# キットへ渡す引数はこれだけ（ハイパラは渡さない＝既定のまま）
FIXED_ARGS = ["--holdout", "", "--hours", "1000", "--max_steps", str(PROD_STEPS)]

CELL_MAIN = r'''
# ---- Issue #82: Engine 1 FP bank rebuilt on all train with the fold-safe kit's train_fpnet.py, T4 session SESSION ----
import json, os, sys, time, hashlib, shutil, subprocess, threading, signal, platform
from pathlib import Path
T0 = time.time()
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
OUT = Path('/kaggle/working'); RUN = OUT / 'run'; WORK = RUN / 'work'; CODE = OUT / 'code'; KITD = CODE / 'kit'
REH = Path('/tmp/fpnet_rehearsal')
WORK.mkdir(parents=True, exist_ok=True)
REPORT = dict(issue=82, recipe='kit train_fpnet.py + session-boundary save/resume only (train_fpnet_t4.py)', session=SESSION,
              fixed_args=FIXED_ARGS, session_hours=SESSION_HOURS, status='started')
REPORT_PATH = OUT / f'fpnet_full_report_s{SESSION}.json'
def write_report():
    REPORT['seconds_total_notebook'] = round(time.time() - T0, 1)
    REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=str))

def ds_root(slug):
    owner, name = slug.split('/')
    for p in (Path('/kaggle/input') / name, Path('/kaggle/input/datasets') / owner / name):
        if p.is_dir():
            return p
    hits = [p for p in Path('/kaggle/input').rglob(name) if p.is_dir()]
    assert len(hits) == 1, (slug, hits)
    return hits[0]

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def path_of(slug, rel):
    p = ds_root(slug) / rel
    assert p.is_file(), p
    return p

# 1) inputs: published sha256 must match before training
t = time.time(); hashes = {}
for (slug, rel), want in EXPECT.items():
    got = sha256(path_of(slug, rel)); hashes[f'{slug}/{rel}'] = got
    assert got == want, (slug, rel, got, want)
for slug, rel in RECORD_ONLY:
    hashes[f'{slug}/{rel}'] = sha256(path_of(slug, rel))
REPORT['input_sha256'] = hashes; REPORT['seconds_hash_inputs'] = round(time.time() - t, 1)
log('INPUTS OK', len(EXPECT), 'checked,', len(RECORD_ONLY), 'recorded', REPORT['seconds_hash_inputs'], 's')

# 2) code: copy of v4b code/casmi (writable for numba cache); train_fpnet_t4.py derived from the kit file here
def link_work(work):
    work.mkdir(parents=True, exist_ok=True)
    for name, (slug, rel) in WORK_LINKS.items():
        dst = work / name
        if not dst.exists():
            os.symlink(path_of(slug, rel), dst)
link_work(WORK)
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(ds_root('ahmedberatozer/casmi26-v4b-models') / 'code' / 'casmi', CODE / 'casmi')
for rel in ('fpnet.py', 'traindata.py', 'spectra.py', 'chem.py'):
    assert sha256(CODE / 'casmi' / rel) == EXPECT[('ahmedberatozer/casmi26-v4b-models', 'code/casmi/' + rel)], rel
KITD.mkdir(parents=True)
ns = {'__name__': 'make_t4'}
exec(compile(MAKE_T4_SRC, 'make_t4', 'exec'), ns)
t4_src = ns['derive'](path_of('dmitriigluzdov/casmi26-fold-safe-fpnet', 'train_fpnet.py').read_bytes().decode())
assert hashlib.sha256(t4_src.encode()).hexdigest() == T4_SHA, 'derived train_fpnet_t4.py differs from the committed one'
(KITD / 'train_fpnet_t4.py').write_text(t4_src); (KITD / 'fpnet_resume.py').write_text(FR_SRC)
assert hashlib.sha256((KITD / 'fpnet_resume.py').read_bytes()).hexdigest() == FR_SHA
TRAIN = KITD / 'train_fpnet_t4.py'
REPORT['code_sha256'] = dict(train_fpnet_t4=T4_SHA, fpnet_resume=FR_SHA)

# 3) environment
import torch
REPORT['env'] = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     gpu_capability=list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
                     gpu_count=torch.cuda.device_count(), vcpu_os_cpu_count=os.cpu_count(),
                     vcpu_affinity=len(os.sched_getaffinity(0)), workers=16, workers_note='kit default, not passed',
                     platform=platform.platform(), torch_config=torch.__config__.show())
try:
    REPORT['env']['mem_total_gb'] = round(int(open('/proc/meminfo').read().split('MemTotal:')[1].split()[0]) / 2**20, 2)
except Exception as e:
    REPORT['env']['mem_total_gb'] = repr(e)
log('ENV', {k: v for k, v in REPORT['env'].items() if k != 'torch_config'})

# 4) CUDA self-check: a separate FPNet copy (same shape, own RNG state) -> 50 steps -> save -> load -> bitwise -> 1 step
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(KITD))
import numpy as np
import torch.nn.functional as F
import fpnet_resume as FR
from casmi import fpnet
dev = 'cuda'; dtype = fpnet.autocast_dtype(dev)
def make_model():
    return fpnet.FPNet(SELFCHECK_NBITS, d=512, layers=8)
def make_batch(step, gen):
    items = []
    for _ in range(8):
        n = 40
        prec = float(150 + 600 * torch.rand(1, generator=gen).item())
        mz = np.sort((50 + (prec - 50) * torch.rand(n, generator=gen)).numpy()).astype(np.float32)
        it = torch.rand(n, generator=gen).numpy().astype(np.float32)
        items.append(dict(mz=mz, it=it, prec=prec, adduct_ix=int(torch.randint(0, fpnet.N_ADDUCT, (1,), generator=gen)),
                          ce=30.0, ce_known=1.0, n_merged=1, mode=1.0))
    y = (torch.rand(8, SELFCHECK_NBITS, generator=gen) > 0.9).float().to(dev)
    return dict(inp=fpnet.collate(items, dev), y=y, e=torch.rand(8, 10, generator=gen).to(dev))
def loss_fn(out, b):
    z, el = out
    return F.binary_cross_entropy_with_logits(z.float(), b['y']) + 0.1 * F.smooth_l1_loss(el.float(), b['e'])
t = time.time()
sc = FR.selfcheck(make_model, make_batch, loss_fn, dev, dtype, '/tmp/selfcheck_resume.pt', steps=SELFCHECK_STEPS)
sc['seconds'] = round(time.time() - t, 1); sc['dtype'] = str(dtype); sc['nbits'] = SELFCHECK_NBITS
(OUT / 'resume_selfcheck.json').write_text(json.dumps(sc, indent=2, default=str))
REPORT['selfcheck'] = dict(ok=sc['ok'], seconds=sc['seconds'], next_loss=sc['next_loss'], dtype=str(dtype))
log('SELFCHECK', json.dumps(REPORT['selfcheck']))
torch.cuda.empty_cache()
if not sc['ok']:
    REPORT['status'] = 'selfcheck_failed'; write_report()
    raise SystemExit('self-check failed: no training')

# 5) memory sampler (every 5 s): GPU via nvidia-smi, CPU = RSS sum of the training process tree and system used
import psutil
def sampler(pid, peak, stop):
    while not stop.is_set():
        try:
            g = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10).stdout.split()
            peak['gpu_mem_used_mib'] = max([peak['gpu_mem_used_mib']] + [int(x) for x in g])
        except Exception:
            pass
        try:
            root = psutil.Process(pid); procs = [root] + root.children(recursive=True); rss = 0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            peak['proc_tree_rss_gb'] = max(peak['proc_tree_rss_gb'], rss / 2**30)
        except psutil.Error:
            pass
        vm = psutil.virtual_memory()
        peak['system_used_gb'] = max(peak['system_used_gb'], (vm.total - vm.available) / 2**30)
        peak['samples'] += 1
        stop.wait(5)

def run_train(run, extra, log_path, timeout_s=None):
    cmd = [sys.executable, '-u', str(TRAIN), '--code', str(CODE), '--run', str(run)] + FIXED_ARGS + extra
    peak = dict(gpu_mem_used_mib=0, proc_tree_rss_gb=0.0, system_used_gb=0.0, samples=0); stop = threading.Event()
    lt = time.localtime(); start_sec = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec; t_start = time.time(); killed = False
    with open(log_path, 'w') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                                env=dict(os.environ, NUMBA_CACHE_DIR=str(OUT / 'numba_cache')))
        th = threading.Thread(target=sampler, args=(proc.pid, peak, stop), daemon=True); th.start()
        while proc.poll() is None:
            time.sleep(5)
            if timeout_s and time.time() - t_start > timeout_s:
                killed = True; os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        try:
            os.killpg(proc.pid, signal.SIGKILL)   # daemon data workers left behind after a normal exit
        except ProcessLookupError:
            pass
    stop.set(); th.join(timeout=20)
    return dict(cmd=cmd, returncode=proc.returncode, killed=killed, seconds=round(time.time() - t_start, 1), start_sec=start_sec,
                peak={k: (round(v, 3) if isinstance(v, float) else v) for k, v in peak.items()}, text=Path(log_path).read_text())

# 6) wiring rehearsal: the derived script itself on real data, --seed 1 --tag rehearsal, outside /kaggle/working, deleted
shutil.rmtree(REH, ignore_errors=True); link_work(REH / 'work')
common = ['--seed', '1', '--tag', 'rehearsal', '--session_hours', str(REHEARSAL_HOURS)]
ra = run_train(REH, common, REH / 'a.log', timeout_s=1200)
ka = FR.load_resume(REH / 'rehearsal_resume.pt')['step'] if (REH / 'rehearsal_resume.pt').exists() else -1
rb = run_train(REH, common + ['--session', '2', '--resume_from', str(REH / 'rehearsal_resume.pt')], REH / 'b.log', timeout_s=1200)
ckb = FR.load_resume(REH / 'rehearsal_resume.pt') if (REH / 'rehearsal_resume.pt').exists() else None
reh = dict(a_rc=ra['returncode'], a_seconds=ra['seconds'], a_step=ka, a_session_done='SESSION_DONE' in ra['text'],
           b_rc=rb['returncode'], b_seconds=rb['seconds'], b_resumed_line=f"step {ka} elapsed_s" in rb['text'],
           b_lr_continues='lr continues' in rb['text'], b_session_done='SESSION_DONE' in rb['text'],
           b_step=ckb['step'] if ckb else -1, b_session=ckb['session'] if ckb else -1,
           a_peak=ra['peak'], b_peak=rb['peak'], a_tail=ra['text'][-1500:], b_tail=rb['text'][-1500:])
reh['ok'] = bool(reh['a_rc'] == 0 and reh['a_session_done'] and ka >= REHEARSAL_MIN_STEPS and reh['b_rc'] == 0
                 and reh['b_resumed_line'] and reh['b_lr_continues'] and reh['b_session_done'] and reh['b_step'] > ka
                 and reh['b_session'] == 2 and not ra['killed'] and not rb['killed'])
del ckb
REPORT['rehearsal'] = reh
shutil.rmtree(REH, ignore_errors=True)
log('REHEARSAL', json.dumps({k: v for k, v in reh.items() if not k.endswith('_tail')}))
if not reh['ok']:
    REPORT['status'] = 'rehearsal_failed'; write_report()
    print(reh['a_tail']); print(reh['b_tail'])
    raise SystemExit('rehearsal failed: no training')

# 7) session 2: resume file from session 1's output (kernel_sources), sha256 checked against session 1's report
extra = ['--session_hours', str(SESSION_HOURS)]
if SESSION >= 2:
    reps = [p for p in Path('/kaggle/input').rglob(f'fpnet_full_report_s{SESSION - 1}.json')]
    assert len(reps) == 1, reps
    prev = json.loads(reps[0].read_text())
    assert prev['status'] == 'session_done' and prev['session'] == SESSION - 1, (prev['status'], prev['session'])
    src_ck = reps[0].parent / 'fpnet_resume.pt'
    got = sha256(src_ck)
    assert got == prev['resume_sha256'], (got, prev['resume_sha256'])
    shutil.copy(src_ck, RUN / 'fpnet_resume.pt')
    REPORT['resumed_from'] = dict(report=str(reps[0]), resume_sha256=got, step=prev['resume_step'])
    extra += ['--session', str(SESSION), '--resume_from', str(RUN / 'fpnet_resume.pt')]
del torch

# 8) train (kit hyper-parameters at their defaults; seed 0 from the kit default in session 1)
LOG = OUT / f'train_fpnet_s{SESSION}.log'
r = run_train(RUN, extra, LOG)
text = r['text']; print(text[-3000:])
REPORT['command'] = r['cmd']; REPORT['returncode'] = r['returncode']; REPORT['seconds_train_process'] = r['seconds']
REPORT['peak'] = r['peak']
ns = {}
exec(compile(REPORT_SRC, 'fpnet_full_report', 'exec'), ns)
REPORT['summary'] = ns['summarize'](text, r['start_sec'])
done, sdone = 'ALL_DONE' in text, 'SESSION_DONE' in text
if r['returncode'] == 0 and done:
    REPORT['status'] = 'ok'
elif r['returncode'] == 0 and sdone and (RUN / 'fpnet_resume.pt').exists():
    REPORT['status'] = 'session_done'
else:
    REPORT['status'] = 'failed'
if (RUN / 'fpnet_history.json').exists():
    shutil.copy(RUN / 'fpnet_history.json', OUT / 'fpnet_history.json')
for name in ('fpnet_final.pt', 'fpnet_last.pt') + (('fpnet_resume.pt',) if REPORT['status'] == 'session_done' else ()):
    if (RUN / name).exists():
        shutil.copy(RUN / name, OUT / name)
        REPORT[name.replace('.pt', '') + '_sha256'] = sha256(OUT / name)
if REPORT['status'] == 'session_done':
    import torch
    ck = torch.load(OUT / 'fpnet_resume.pt', map_location='cpu', weights_only=False)
    REPORT['resume_sha256'] = REPORT.pop('fpnet_resume_sha256'); REPORT['resume_step'] = ck['step']
    REPORT['resume_lr'] = ck['lr']; REPORT['resume_elapsed_s'] = ck['elapsed']; del ck
shutil.rmtree(CODE, ignore_errors=True); shutil.rmtree(OUT / 'numba_cache', ignore_errors=True)
shutil.rmtree(RUN, ignore_errors=True)   # work/ = symlinks only; .pt files were copied to /kaggle/working above
write_report()
log('REPORT', json.dumps({k: v for k, v in REPORT.items() if k not in ('input_sha256', 'env', 'command', 'rehearsal')}, default=str)[:4000])
assert REPORT['status'] in ('ok', 'session_done'), REPORT['status']
'''


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_notebook(session: int) -> dict:
    assert session in (1, 2), session
    report_src = (ROOT / "scripts/train/fpnet_full_report.py").read_text()
    make_t4_src = (KIT / "make_t4.py").read_text()
    fr_src = (KIT / "fpnet_resume.py").read_text()
    consts = (f"SESSION = {session}\nPROD_STEPS = {PROD_STEPS}\nSESSION_HOURS = {SESSION_HOURS}\n"
              f"REHEARSAL_HOURS = {REHEARSAL_HOURS}\nREHEARSAL_MIN_STEPS = {REHEARSAL_MIN_STEPS}\n"
              f"SELFCHECK_STEPS = {SELFCHECK_STEPS}\nSELFCHECK_NBITS = {SELFCHECK_NBITS}\n"
              f"FIXED_ARGS = {FIXED_ARGS!r}\nEXPECT = {EXPECT!r}\nRECORD_ONLY = {RECORD_ONLY!r}\n"
              f"WORK_LINKS = {WORK_LINKS!r}\nT4_SHA = {_sha(KIT / 'train_fpnet_t4.py')!r}\nFR_SHA = {_sha(KIT / 'fpnet_resume.py')!r}\n"
              f"REPORT_SRC = {report_src!r}\nMAKE_T4_SRC = {make_t4_src!r}\nFR_SRC = {fr_src!r}\n")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            f"# casmi26 FPNet full-train rebuild, T4 session {session} (Issue #82)\n",
            "Fold-safe kit `train_fpnet.py` + session-boundary save/resume only (`train_fpnet_t4.py`, derived and sha-checked "
            "in the kernel), `--holdout '' --hours 1000 --max_steps 82855`; other arguments at the kit defaults. "
            "CUDA self-check and a wiring rehearsal run before training. Not for v3 evaluation (trains on v3 answers).\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": consts},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CELL_MAIN.lstrip("\n")},
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def metadata(session: int) -> dict:
    slug = SLUG_S1 if session == 1 else SLUG_S2
    return {"id": slug, "title": slug.split("/")[1].replace("-", " "), "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": [] if session == 1 else [SLUG_S1], "model_sources": []}


def kernel_dir(session: int) -> Path:
    return ROOT / ("kernels/fpnet_full_train" if session == 1 else "kernels/fpnet_full_train_s2")


def push_size(nb: dict) -> int:
    """Bytes of the notebook source as the CLI sends it (outputs empty, sources joined, json.dumps default)."""
    cells = [dict(c, source="".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in nb["cells"]]
    return len(json.dumps(dict(nb, cells=cells)).encode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    nb = build_notebook(a.session)
    size = push_size(nb)
    assert size < 1_000_000, size
    kdir = kernel_dir(a.session); kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "notebook.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    (kdir / "kernel-metadata.json").write_text(json.dumps(metadata(a.session), indent=2) + "\n")
    print("wrote", kdir, "push size", size, "B")


if __name__ == "__main__":
    main()
