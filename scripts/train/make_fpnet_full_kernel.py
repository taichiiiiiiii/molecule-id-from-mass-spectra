#!/usr/bin/env python3
"""Engine 1 の FP バンク（全 train の FPNet）をキットの train_fpnet.py で学習し直すカーネルを作る（Issue #82）。

    python scripts/train/make_fpnet_full_kernel.py            # 試走（SMOKE_STEPS = 2000）
    python scripts/train/make_fpnet_full_kernel.py --prod     # 本番（82,855 ステップ。今日は push しない）

出力: kernels/fpnet_full_train/notebook.ipynb, kernel-metadata.json

事前登録（Issue #82 のコメント）のとおり、キットの train_fpnet.py は中身を変えずに
`--holdout '' --hours 1000 --max_steps <N>` だけを渡して動かす。そのほかの引数は渡さない（キットの既定のまま）。
入力ファイルは公開されている sha256 と照合し、合わなければ学習の前に止める。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLUG = "taichiiiii/casmi26-fpnet-full-train"
SMOKE_STEPS = 2000
PROD_STEPS = 82855
GUARD_SECONDS = 28 * 60
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
FIXED_ARGS = ["--holdout", "", "--hours", "1000"]

CELL_MAIN = r'''
# ---- Issue #82: Engine 1 FP bank rebuilt on all train with the fold-safe kit's train_fpnet.py (plan B) ----
import json, os, sys, time, hashlib, shutil, subprocess, threading, signal, platform
from pathlib import Path
T0 = time.time()
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)
SMOKE = SMOKE_STEPS > 0
STEPS = SMOKE_STEPS if SMOKE else PROD_STEPS
OUT = Path('/kaggle/working'); RUN = OUT / 'run'; WORK = RUN / 'work'; CODE = OUT / 'code'
WORK.mkdir(parents=True, exist_ok=True)
REPORT = dict(issue=82, recipe='plan B: fold-safe kit train_fpnet.py unchanged', smoke_steps=SMOKE_STEPS, max_steps=STEPS,
              fixed_args=FIXED_ARGS, prod_steps=PROD_STEPS, status='started')

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

# 2) run dir: work/ links to the published files; code = copy of v4b code/casmi (writable for numba cache)
for name, (slug, rel) in WORK_LINKS.items():
    dst = WORK / name
    if not dst.exists():
        os.symlink(path_of(slug, rel), dst)
if CODE.exists():
    shutil.rmtree(CODE)
shutil.copytree(ds_root('ahmedberatozer/casmi26-v4b-models') / 'code' / 'casmi', CODE / 'casmi')
for rel in ('fpnet.py', 'traindata.py', 'spectra.py', 'chem.py'):
    assert sha256(CODE / 'casmi' / rel) == EXPECT[('ahmedberatozer/casmi26-v4b-models', 'code/casmi/' + rel)], rel
TRAIN = path_of('dmitriigluzdov/casmi26-fold-safe-fpnet', 'train_fpnet.py')

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
del torch

# 4) memory sampler (every 5 s): GPU via nvidia-smi, CPU = RSS sum of the training process tree and system used
import psutil
PEAK = dict(gpu_mem_used_mib=0, proc_tree_rss_gb=0.0, system_used_gb=0.0, samples=0)
STOP_SAMPLER = threading.Event()
def sampler(pid):
    while not STOP_SAMPLER.is_set():
        try:
            g = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10).stdout.split()
            PEAK['gpu_mem_used_mib'] = max([PEAK['gpu_mem_used_mib']] + [int(x) for x in g])
        except Exception:
            pass
        try:
            root = psutil.Process(pid); procs = [root] + root.children(recursive=True); rss = 0
            for p in procs:
                try:
                    rss += p.memory_info().rss
                except psutil.Error:
                    pass
            PEAK['proc_tree_rss_gb'] = max(PEAK['proc_tree_rss_gb'], rss / 2**30)
        except psutil.Error:
            pass
        vm = psutil.virtual_memory()
        PEAK['system_used_gb'] = max(PEAK['system_used_gb'], (vm.total - vm.available) / 2**30)
        PEAK['samples'] += 1
        STOP_SAMPLER.wait(5)

# 5) train (kit script unchanged; only --holdout '' --hours 1000 --max_steps N)
cmd = [sys.executable, '-u', str(TRAIN), '--code', str(CODE), '--run', str(RUN)] + FIXED_ARGS + ['--max_steps', str(STEPS)]
REPORT['command'] = cmd
LOG = OUT / 'train_fpnet.log'
lt = time.localtime(); start_sec = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec; t_start = time.time()
with open(LOG, 'w') as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                            env=dict(os.environ, NUMBA_CACHE_DIR=str(OUT / 'numba_cache')))
    th = threading.Thread(target=sampler, args=(proc.pid,), daemon=True); th.start()
    guard_hit = False
    while proc.poll() is None:
        time.sleep(10)
        if SMOKE and time.time() - T0 > GUARD_SECONDS:
            guard_hit = True; log('GUARD: stopping the training process group at', round(time.time() - T0), 's')
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
            break
    try:
        os.killpg(proc.pid, signal.SIGKILL)   # daemon data workers left behind after a normal exit
    except ProcessLookupError:
        pass
STOP_SAMPLER.set(); th.join(timeout=20)
REPORT['returncode'] = proc.returncode; REPORT['guard_hit'] = guard_hit
REPORT['seconds_train_process'] = round(time.time() - t_start, 1)
REPORT['peak'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in PEAK.items()}
text = LOG.read_text()
print(text[-3000:])

# 6) summary + extrapolation (pre-registered formula, see fpnet_full_report)
ns = {}
exec(compile(REPORT_SRC, 'fpnet_full_report', 'exec'), ns)
REPORT['summary'] = ns['summarize'](text, start_sec)
REPORT['seconds_total_notebook'] = round(time.time() - T0, 1)
done = 'ALL_DONE' in text
REPORT['status'] = 'ok' if (done and proc.returncode == 0 and not guard_hit) else ('guard_stopped' if guard_hit else 'failed')
for name in ('fpnet_history.json',):
    if (RUN / name).exists():
        shutil.copy(RUN / name, OUT / name)
if SMOKE:
    for p in RUN.glob('fpnet_*.pt'):
        p.unlink()      # smoke weights are not kept (pre-registered)
else:
    for name in ('fpnet_final.pt', 'fpnet_last.pt'):
        if (RUN / name).exists():
            shutil.copy(RUN / name, OUT / name)
            REPORT[f'{name}_sha256'] = sha256(OUT / name)
shutil.rmtree(CODE, ignore_errors=True); shutil.rmtree(OUT / 'numba_cache', ignore_errors=True)
shutil.rmtree(WORK, ignore_errors=True)   # only symlinks to /kaggle/input; keep them out of the kernel output
(OUT / ('fpnet_smoke_report.json' if SMOKE else 'fpnet_full_report.json')).write_text(json.dumps(REPORT, indent=2, default=str))
log('REPORT', json.dumps({k: v for k, v in REPORT.items() if k not in ('input_sha256', 'env', 'command')}, default=str)[:4000])
assert REPORT['status'] in ('ok', 'guard_stopped'), REPORT['status']
'''


def build_notebook(smoke_steps: int) -> dict:
    report_src = (ROOT / "scripts/train/fpnet_full_report.py").read_text()
    consts = (f"SMOKE_STEPS = {smoke_steps}   # 0 = production run\n"
              f"PROD_STEPS = {PROD_STEPS}\nGUARD_SECONDS = {GUARD_SECONDS}\n"
              f"FIXED_ARGS = {FIXED_ARGS!r}\nEXPECT = {EXPECT!r}\nRECORD_ONLY = {RECORD_ONLY!r}\n"
              f"WORK_LINKS = {WORK_LINKS!r}\nREPORT_SRC = {report_src!r}\n")
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# casmi26 FPNet full-train rebuild (Issue #82)\n",
            "Fold-safe kit `train_fpnet.py` unchanged, `--holdout '' --hours 1000 --max_steps N`; other arguments at the kit "
            "defaults. Inputs are checked against published sha256 before training. Not for v3 evaluation (trains on v3 answers).\n"]},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": consts},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CELL_MAIN.lstrip("\n")},
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def metadata() -> dict:
    return {"id": SLUG, "title": "casmi26 fpnet full train", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod", action="store_true", help="production run (82,855 steps); not pushed on 11/12")
    a = ap.parse_args()
    kdir = ROOT / "kernels/fpnet_full_train"; kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "notebook.ipynb").write_text(json.dumps(build_notebook(0 if a.prod else SMOKE_STEPS), indent=1) + "\n")
    (kdir / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print("wrote", kdir)


if __name__ == "__main__":
    main()
