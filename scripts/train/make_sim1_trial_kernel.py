#!/usr/bin/env python3
"""段 1（TC `simulate_v1ctx.py`）を Kaggle T4 で短く試走するカーネルを作る（Issue #88）。

    python scripts/train/make_sim1_trial_kernel.py \
        --spec-readme <casmi26-spec-cache の docs/README.md> \
        --v4g-manifest <casmi26-attributed-v4g-reproduction-assets の MANIFEST.json> \
        --v4b-manifest <casmi26-v4b-models の MANIFEST.json>

出力: kernels/casmi26-v4b-sim1-trial/{notebook.ipynb, kernel-metadata.json}

引数は TC 既定（`launch_sim1x.sh`）に `--shard 0 --nshard 6 --max_structs 40` だけを足す。照合に使う sha256 は
上の 3 つの配布元の記録から完全な値を取り、ノートに埋め込む（カーネルの中でも v4b MANIFEST と spec README を
マウントから読み直して同じか確かめる）。1 つでも違えば段 1 を回さず trial_report.json に記録して終える。

`--two-shards`（2026-11-14 夜）: kernels/casmi26-v4b-sim1-trial2/ に、シャード 0 を GPU 0・シャード 1 を GPU 1 で
同時に回すノートを作る（照合までは昼と同じコード、段 1 は 45 分・ノート全体 60 分で打ち切り、部品は
`sim1_trial2.py`）。この引数を付けないときの生成物は昼のものとバイト同一。
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "kernels" / "casmi26-v4b-sim1-trial"
SLUG = "taichiiiii/casmi26-v4b-sim1-trial"
DATASETS = [
    "ahmedberatozer/casmi26-training-code",
    "ahmedberatozer/casmi26-spec-cache",
    "ahmedberatozer/casmi26-v2-pool",
    "ahmedberatozer/casmi26-v4b-models",
]
# TC 既定（launch_sim1x.sh / simulate_v1ctx.py の既定）＋試走で足す 3 つ
TC_ARGS = ["--out", "work/analysis/v1fe/sim1x/rows", "--ctx", "work/analysis/v1fe/sim1x/ctx",
           "--folds", "np,nplib,syn,plusk", "--regimes", "c1,c2,c3", "--seed", "21", "--part_every", "40",
           "--engine_models", "work/fpnet_a.pt"]
TRIAL_ARGS = ["--shard", "0", "--nshard", "6", "--max_structs", "40"]
TC_ENV = {"PYTHONIOENCODING": "utf-8", "NUMBA_NUM_THREADS": "2", "OMP_NUM_THREADS": "2",
          "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
PIP = ["rdkit==2026.03.3", "lightgbm==4.7.0"]

RUNNER = r'''
# ---- Issue #88: TC 段 1 (simulate_v1ctx.py) の T4 試走。ノート全体 25 分で打ち切り ----
import os, sys, time, json, glob, shutil, signal, subprocess, threading, platform, shlex
from pathlib import Path
T0 = time.time()
BUDGET = 25 * 60
PY = shlex.quote(sys.executable)
OUT = Path('/kaggle/working/out'); OUT.mkdir(parents=True, exist_ok=True)
R = Path('/tmp/R')                      # 作業場所は /tmp（/kaggle/working に 4 GB の symlink を置かない）
REPORT = dict(issue=88, slug=SLUG, started_unix=T0, budget_s=BUDGET, args=TC_ARGS + TRIAL_ARGS, env=TC_ENV,
              status='started', steps=[])
log = lambda *a: print(*a, flush=True)
el = lambda: round(time.time() - T0, 1)

def save():
    REPORT['elapsed_s'] = el()
    (OUT / 'trial_report.json').write_text(dumps(REPORT))

def step(name, **kw):
    REPORT['steps'].append(dict(name=name, t=el(), **kw)); save(); log('STEP', name, el(), kw if kw else '')

def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as e:
        return f'ERR {e!r}'

def main():
    # 1. 環境
    REPORT['env_kaggle'] = dict(python=sys.version.split()[0], platform=platform.platform(),
                                nvidia_smi=sh('nvidia-smi'), lscpu=sh('lscpu'), nproc=sh('nproc'),
                                meminfo=sh('head -3 /proc/meminfo'), df=sh('df -h /tmp /kaggle/working'))
    step('env')
    pip = {}
    for spec in PIP:
        o = sh(f'{PY} -m pip install -q {spec}', timeout=300)
        pip[spec] = o[-1500:] if o else 'ok'
    REPORT['pip'] = pip
    vers = {}
    for mod, expr in (('torch', "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"),
                      ('rdkit', "from rdkit import rdBase; print(rdBase.rdkitVersion)"),
                      ('numpy', "import numpy; print(numpy.__version__)"), ('pandas', "import pandas; print(pandas.__version__)"),
                      ('pyarrow', "import pyarrow; print(pyarrow.__version__)"), ('numba', "import numba; print(numba.__version__)"),
                      ('lightgbm', "import lightgbm; print(lightgbm.__version__)")):
        vers[mod] = sh(f'{PY} -c "{expr}"', timeout=120)[-300:]
    REPORT['versions'] = vers
    REPORT['tc_env'] = 'TC README §4: python 3.14.5, torch 2.14.0+cu130, lightgbm 4.7.0, rdkit 2026.03.3（ここと違う版は versions を見る）'
    step('pip', versions=REPORT['versions'])

    # 2. 入力の場所
    inp = Path('/kaggle/input')
    def one(pattern, must=None):
        hits = [p for p in inp.rglob(pattern) if must is None or must(p)]
        assert len(hits) == 1, (pattern, [str(h) for h in hits])
        return hits[0]
    TC = one('simulate_v1ctx.py').parent.parent                 # TC の根（fe_v4/ の親）
    SPEC = one('spec_off.npy').parent                            # cache/
    POOL = one('pool_fp.npy').parent
    V4B = one('fe_D.pt').parent
    REPORT['mounts'] = dict(tc=str(TC), spec=str(SPEC), pool=str(POOL), v4b=str(V4B))
    # 配布元の記録をマウントから読み直し、ノートに埋め込んだ値と同じか
    doc = {}
    m_v4b = json.loads((V4B / 'MANIFEST.json').read_text())
    readme = (SPEC.parent / 'docs' / 'README.md').read_text()
    doc['v4b_manifest_sha256'] = sha256_file(V4B / 'MANIFEST.json')
    doc['spec_readme_sha256'] = sha256_file(SPEC.parent / 'docs' / 'README.md')
    doc['v4b_manifest_same_as_embedded'] = expected_code(m_v4b) == EXP_CODE and all(
        m_v4b['files'].get(EXP_DATA[w]['rel']) == EXP_DATA[w]['sha256'] for w, _ in MODEL_FILES)
    _sp = parse_spec_readme(readme)
    doc['spec_readme_same_as_embedded'] = all(_sp.get(n) == (EXP_DATA[n]['bytes'], EXP_DATA[n]['sha256']) for n in SPEC_FILES)
    doc['embedded_sources'] = EMBED_SOURCES
    REPORT['records'] = doc
    step('records', **{k: v for k, v in doc.items() if k.endswith('embedded')})

    # 3. README §4 の配置（コードは写す、データは symlink）
    if R.exists():
        shutil.rmtree(R)
    for src, dst in (('fe_v4', 'work/analysis/fe_v4'), ('v1fe_code_casmi', 'work/analysis/v1fe/code/casmi'),
                     ('engine_v6_casmi', 'work/analysis/engine_v6/code/casmi'), ('casmi', 'src/casmi')):
        shutil.copytree(TC / src, R / dst)
    W = R / 'work'
    srcs = {}
    for n in SPEC_FILES:
        srcs[n] = SPEC / n
    for n in POOL_FILES:
        srcs[n] = POOL / n
    for w, name in MODEL_FILES:
        srcs[w] = V4B / name
    for w, p in srcs.items():
        (W / w).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(p, W / w)
    step('layout')

    # 4. 照合（データ 15 本＋コード 23 本）。1 つでも違えば段 1 を回さない
    act = {}
    for w, p in srcs.items():
        act[w] = dict(sha256=sha256_file(p), bytes=p.stat().st_size)
    act_code = {rel: dict(sha256=sha256_file(R / rel)) for rel in EXP_CODE}
    bad = compare(EXP_DATA, act) + compare({k: dict(sha256=v) for k, v in EXP_CODE.items()}, act_code)
    REPORT['inputs'] = {w: dict(EXP_DATA[w], actual_sha256=act[w]['sha256'], actual_bytes=act[w]['bytes']) for w in EXP_DATA}
    REPORT['code_checked'] = len(act_code)
    # 記録の無い TC のコード（段 1 の入口 simulate_v1ctx.py ほか）は sha256 だけを残す
    REPORT['code_unrecorded_sha256'] = {str(f.relative_to(R)): sha256_file(f) for f in sorted((R / 'work').rglob('*.py'))
                                        if str(f.relative_to(R)) not in EXP_CODE}
    REPORT['mismatch'] = bad
    step('verify', n_data=len(act), n_code=len(act_code), n_bad=len(bad))
    if bad:
        REPORT['status'] = 'stopped_sha256_mismatch'
        return

    # 5. 段 1
    tmo = stage_timeout(time.time() - T0)
    REPORT['stage1_timeout_s'] = tmo
    if tmo <= 60:
        REPORT['status'] = 'skipped_no_time'
        return
    env = dict(os.environ, **TC_ENV)
    cmd = [sys.executable, 'work/analysis/fe_v4/simulate_v1ctx.py'] + TC_ARGS + TRIAL_ARGS
    if os.path.exists('/usr/bin/time') and sh('/usr/bin/time -v true').find('Maximum resident') >= 0:
        cmd = ['/usr/bin/time', '-v'] + cmd   # GNU time の最大 RSS も取る（2 秒おきの監視と別の測り方）
    REPORT['stage1_cmd'] = cmd
    lines = []
    peak = dict(rss_mb=0.0, gpu_mem_mb=0.0, gpu_util=0.0, min_mem_available_mb=None)
    t_start = time.time()
    p = subprocess.Popen(cmd, cwd=str(R), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, bufsize=1)
    def reader():
        with open(OUT / 'sim1_stdout.log', 'w') as fh:
            for ln in p.stdout:
                lines.append(ln.rstrip('\n')); fh.write(ln); fh.flush(); print(ln, end='', flush=True)
    def tree_rss_mb(root):
        try:
            import psutil
            pr = psutil.Process(root)
            return sum(q.memory_info().rss for q in [pr] + pr.children(recursive=True)) / 2**20
        except ImportError:
            pass
        except Exception:
            return 0.0
        tot, todo, seen = 0, [root], set()
        while todo:
            q = todo.pop()
            if q in seen:
                continue
            seen.add(q)
            try:
                for l in open(f'/proc/{q}/status'):
                    if l.startswith('VmRSS:'):
                        tot += int(l.split()[1])
                for t in os.listdir(f'/proc/{q}/task'):
                    try:
                        todo += [int(c) for c in open(f'/proc/{q}/task/{t}/children').read().split()]
                    except OSError:
                        pass
            except OSError:
                pass
        return tot / 1024
    def monitor():
        while p.poll() is None:
            peak['rss_mb'] = max(peak['rss_mb'], tree_rss_mb(p.pid))
            o = sh('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits', timeout=10)
            try:
                mu, ut = [float(x) for x in o.splitlines()[0].split(',')]
                peak['gpu_mem_mb'] = max(peak['gpu_mem_mb'], mu); peak['gpu_util'] = max(peak['gpu_util'], ut)
            except Exception:
                pass
            try:
                ma = [int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:')][0] / 1024
                peak['min_mem_available_mb'] = ma if peak['min_mem_available_mb'] is None else min(peak['min_mem_available_mb'], ma)
            except Exception:
                pass
            time.sleep(2)
    th_r = threading.Thread(target=reader, daemon=True); th_r.start()
    th_m = threading.Thread(target=monitor, daemon=True); th_m.start()
    timed_out = False
    try:
        rc = p.wait(timeout=tmo)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            rc = p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = p.wait()
    th_r.join(timeout=10)
    wall = time.time() - t_start
    parsed = parse_sim_log(lines)
    for ln in lines:
        if 'Maximum resident set size' in ln:
            peak['gnu_time_max_rss_mb'] = round(int(ln.split(':')[-1].strip()) / 1024, 1)
    REPORT['stage1'] = dict(returncode=rc, timed_out=timed_out, wall_s=round(wall, 1), peak=peak,
                            header=parsed['header'], done=parsed['done'], progress=parsed['progress'],
                            tail=lines[-40:])
    step('stage1', rc=rc, timed_out=timed_out, wall_s=round(wall, 1))

    # 6. 出力の形
    sim = R / 'work/analysis/v1fe/sim1x'
    files = {str(f.relative_to(sim)): f.stat().st_size for f in sorted(sim.rglob('*')) if f.is_file()}
    REPORT['outputs'] = files
    shape = {}
    rows_q = None
    try:
        import numpy as np, pandas as pd
        parts = sorted(sim.glob('rows_0_p[0-9][0-9][0-9].npz'))
        if parts:
            with np.load(parts[0], allow_pickle=True) as z:
                X, y, g, feats = z['X'], z['y'], z['g'], list(z['features'])
            shape.update(part=parts[0].name, n_rows=int(X.shape[0]), n_cols=int(X.shape[1]), n_features=len(feats),
                         same_as_94=bool(X.shape[1] == 94 and len(feats) == 94), dtype=str(X.dtype),
                         n_groups=int(len(np.unique(g))), n_pos=int(y.sum()), first_features=feats[:5], last_features=feats[-5:])
            m = pd.read_parquet(str(parts[0]).replace('.npz', '_meta.parquet'))
            rows_q = int(len(m))
            shape['meta'] = dict(n=int(len(m)), regime=m.regime.value_counts().to_dict(), has_truth=int(m.has_truth.sum()),
                                 n_query_mean=float(m.n_query.mean()), n_cand_mean=float(m.n_cand.mean()),
                                 n_cand_max=int(m.n_cand.max()), t_query_mean=float(m.t_query.mean()),
                                 t_query_median=float(m.t_query.median()), t_query_max=float(m.t_query.max()),
                                 fold=m.fold.value_counts().to_dict(), n_structs=int(m.sid.nunique()))
            shutil.copy2(parts[0], OUT / parts[0].name)
            shutil.copy2(str(parts[0]).replace('.npz', '_meta.parquet'), OUT / parts[0].name.replace('.npz', '_meta.parquet'))
        ctxs = sorted((sim / 'ctx').glob('ctx_s0_c*.pkl'))
        if ctxs:
            shape['ctx'] = dict(files=len(ctxs), bytes=int(sum(c.stat().st_size for c in ctxs)),
                                bytes_per_query=(sum(c.stat().st_size for c in ctxs) / rows_q) if rows_q else None,
                                sha256_first=sha256_file(ctxs[0]))
    except Exception as e:
        shape['error'] = repr(e)
    REPORT['rows'] = shape
    S = summarize(parsed, rows_meta_queries=rows_q)
    REPORT['summary'] = S
    REPORT['extrapolation'] = extrapolate(S['s_per_query'], S['s_per_struct'])
    REPORT['author_shard0_first40'] = AUTHOR_FIRST40
    REPORT['status'] = 'ok' if (S['structs'] or 0) >= 1 and S['s_per_struct'] else 'no_progress'
    step('summary', **S)

try:
    main()
except Exception as e:
    import traceback
    REPORT['status'] = 'error'
    REPORT['error'] = traceback.format_exc()[-4000:]
    log(REPORT['error'])
finally:
    save()
    log('SIM1_TRIAL', REPORT['status'], el(), 's')
    log(dumps({k: REPORT.get(k) for k in ('status', 'summary', 'rows', 'extrapolation', 'mismatch')}))
'''


def module_source(name="sim1_trial.py"):
    src = (ROOT / "scripts" / "train" / name).read_text()
    assert "from __future__" not in src
    return src


# ---- 2 シャード同時（--two-shards、2026-11-14 夜、#88）。既定（1 シャード）の生成物は変えない ----
KDIR2 = ROOT / "kernels" / "casmi26-v4b-sim1-trial2"
SLUG2 = "taichiiiii/casmi26-v4b-sim1-trial2"
# 昼の試走（taichiiiii/casmi26-v4b-sim1-trial v1）のシャード 0 の rows_0_p000_meta.parquet から（s の分母）
DAY_SHARD0 = dict(kernel="taichiiiii/casmi26-v4b-sim1-trial v1",
                  meta_sha256="253c729adfcc27817a027398872dd59dd3cf183f65300ec181d2734b90bcb51c",
                  n_queries=200, n_structs=40, t_query_sum=928.8429119586945,
                  query_signature="6c64b7d4df0129725e3387e963463b0702475b1c8b4785a966449c16feca61f1",
                  cpu="Intel(R) Xeon(R) CPU @ 2.00GHz", vcpus=4)
_STAGE1_MARK = "    # 5. 段 1\n"

RUNNER2_STAGE = r'''    # 5. 段 1 を 2 シャード同時に（シャード 0 → GPU 0、シャード 1 → GPU 1）
    tmo = stage2_timeout(time.time() - T0)
    REPORT['stage1_timeout_s'] = tmo
    REPORT['day_shard0'] = DAY_SHARD0
    if tmo <= 60:
        REPORT['status'] = 'skipped_no_time'
        return
    use_time = os.path.exists('/usr/bin/time') and sh('/usr/bin/time -v true').find('Maximum resident') >= 0
    procs, lines, cmds = {}, {}, {}
    t_start = time.time()
    for shard, gpu in SHARD_GPU:
        cmd = [sys.executable, 'work/analysis/fe_v4/simulate_v1ctx.py'] + TC_ARGS + shard_args(shard)
        if use_time:
            cmd = ['/usr/bin/time', '-v'] + cmd
        cmds[str(shard)] = dict(cmd=cmd, CUDA_VISIBLE_DEVICES=str(gpu))
        lines[shard] = []
        procs[shard] = subprocess.Popen(cmd, cwd=str(R), env=shard_env(os.environ, TC_ENV, gpu), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, start_new_session=True, bufsize=1)
    REPORT['stage1_cmds'] = cmds
    REPORT['stage1_start_gap_s'] = round(time.time() - t_start, 2)
    step('stage1_started', shards=[s for s, _ in SHARD_GPU])

    def reader(shard):
        p = procs[shard]
        with open(OUT / f'sim1_stdout_s{shard}.log', 'w') as fh:
            for ln in p.stdout:
                lines[shard].append(ln.rstrip('\n')); fh.write(ln); fh.flush(); print(f'[s{shard}] ' + ln, end='', flush=True)

    def tree_rss_mb(root):
        try:
            import psutil
            pr = psutil.Process(root)
            return sum(q.memory_info().rss for q in [pr] + pr.children(recursive=True)) / 2**20
        except ImportError:
            pass
        except Exception:
            return 0.0
        tot, todo, seen = 0, [root], set()
        while todo:
            q = todo.pop()
            if q in seen:
                continue
            seen.add(q)
            try:
                for l in open(f'/proc/{q}/status'):
                    if l.startswith('VmRSS:'):
                        tot += int(l.split()[1])
                for t in os.listdir(f'/proc/{q}/task'):
                    try:
                        todo += [int(c) for c in open(f'/proc/{q}/task/{t}/children').read().split()]
                    except OSError:
                        pass
            except OSError:
                pass
        return tot / 1024

    peak = dict(rss_mb={str(s): 0.0 for s, _ in SHARD_GPU}, rss_total_mb=0.0, gpu={}, min_mem_available_mb=None,
                loadavg1_max=0.0, n_samples=0)
    ends = {}   # シャードごとの終了を見た時刻（2 秒おきの監視。2 本が重なっていた時間を出す）
    def monitor():
        while any(p.poll() is None for p in procs.values()):
            for s, p in procs.items():
                if p.poll() is not None and s not in ends:
                    ends[s] = time.time()
            tot = 0.0
            for s, p in procs.items():
                v = tree_rss_mb(p.pid) if p.poll() is None else 0.0
                peak['rss_mb'][str(s)] = max(peak['rss_mb'][str(s)], v); tot += v
            peak['rss_total_mb'] = max(peak['rss_total_mb'], tot)
            o = sh('nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits', timeout=10)
            update_gpu_peak(peak['gpu'], parse_gpu_query(o))
            try:
                ma = [int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemAvailable:')][0] / 1024
                peak['min_mem_available_mb'] = ma if peak['min_mem_available_mb'] is None else min(peak['min_mem_available_mb'], ma)
                peak['loadavg1_max'] = max(peak['loadavg1_max'], float(open('/proc/loadavg').read().split()[0]))
            except Exception:
                pass
            peak['n_samples'] += 1
            time.sleep(2)
    ths = [threading.Thread(target=reader, args=(s,), daemon=True) for s in procs]
    for th in ths:
        th.start()
    th_m = threading.Thread(target=monitor, daemon=True); th_m.start()

    def kill_one(p):
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return p.wait()
    def wait_one(p, timeout):
        return p.wait(timeout=timeout)
    res = wait_all(procs, t_start + tmo, time.time, wait_one, kill_one)
    for th in ths:
        th.join(timeout=10)
    wall = time.time() - t_start
    for s, p in procs.items():
        ends.setdefault(s, time.time())
    REPORT['stage1_end_s'] = {str(s): round(ends[s] - t_start, 1) for s in procs}
    REPORT['stage1_overlap_s'] = round(min(ends.values()) - t_start, 1)
    st1 = {}
    for s, _ in SHARD_GPU:
        parsed = parse_sim_log(lines[s])
        gnu = None
        for ln in lines[s]:
            if 'Maximum resident set size' in ln:
                gnu = round(int(ln.split(':')[-1].strip()) / 1024, 1)
        st1[str(s)] = dict(returncode=res[s][0], timed_out=res[s][1], header=parsed['header'], done=parsed['done'],
                           progress=parsed['progress'], gnu_time_max_rss_mb=gnu, tail=lines[s][-40:], _parsed=parsed)
    REPORT['stage1'] = {k: {kk: vv for kk, vv in v.items() if kk != '_parsed'} for k, v in st1.items()}
    REPORT['stage1_wall_s'] = round(wall, 1)
    REPORT['stage1_peak'] = peak
    step('stage1', **{f'rc{k}': v['returncode'] for k, v in st1.items()}, wall_s=round(wall, 1),
         timed_out=[k for k, v in st1.items() if v['timed_out']])

    # 6. 出力の形（シャードごと）と s
    import numpy as np, pandas as pd
    sim = R / 'work/analysis/v1fe/sim1x'
    REPORT['outputs'] = {str(f.relative_to(sim)): f.stat().st_size for f in sorted(sim.rglob('*')) if f.is_file()}
    shards = {}
    for s, _ in SHARD_GPU:
        d = {}
        try:
            parts = sorted(sim.glob(f'rows_{s}_p[0-9][0-9][0-9].npz'))
            rows_q = None
            if parts:
                with np.load(parts[0], allow_pickle=True) as z:
                    X, y, g, feats = z['X'], z['y'], z['g'], list(z['features'])
                d.update(part=parts[0].name, n_parts=len(parts), n_rows=int(X.shape[0]), n_cols=int(X.shape[1]),
                         n_features=len(feats), same_as_94=bool(X.shape[1] == 94 and len(feats) == 94), dtype=str(X.dtype),
                         all_finite=bool(np.isfinite(X).all()), n_groups=int(len(np.unique(g))), n_pos=int(y.sum()))
                mp = str(parts[0]).replace('.npz', '_meta.parquet')
                m = pd.read_parquet(mp)
                rows_q = int(len(m))
                tq = t_query_sum(m.t_query)
                d['meta'] = dict(n=int(len(m)), n_structs=int(m.sid.nunique()), regime=m.regime.value_counts().to_dict(),
                                 has_truth=int(m.has_truth.sum()), t_query=tq, t_query_median=float(m.t_query.median()),
                                 t_query_max=float(m.t_query.max()),
                                 t_query_mean_by_regime={k: float(v) for k, v in m.groupby('regime').t_query.mean().items()},
                                 query_signature=query_signature(m.sid, m.regime, m.qidx), sha256=sha256_file(mp))
                shutil.copy2(mp, OUT / Path(mp).name)
            ctxs = sorted((sim / 'ctx').glob(f'ctx_s{s}_c*.pkl'))
            d['ctx'] = dict(files=len(ctxs), bytes=int(sum(c.stat().st_size for c in ctxs)))
        except Exception as e:
            d['error'] = repr(e)
        d['summary'] = summarize(st1[str(s)]['_parsed'], rows_meta_queries=rows_q)
        shards[str(s)] = d
    REPORT['shards'] = shards
    m0 = shards.get('0', {}).get('meta')
    sv = dict(definition='同時実行のシャード 0 の t_query 合計 ÷ 昼の試走のシャード 0 の t_query 合計',
              day_sum=DAY_SHARD0['t_query_sum'])
    if m0:
        sv.update(concurrent_sum=m0['t_query']['sum'], s=slowdown(m0['t_query']['sum'], DAY_SHARD0['t_query_sum']),
                  same_queries_as_day=(m0['query_signature'] == DAY_SHARD0['query_signature']),
                  n_queries=m0['n'], day_n_queries=DAY_SHARD0['n_queries'])
    model, ncpu = cpu_model(REPORT.get('env_kaggle', {}).get('lscpu', ''))
    sv['cpu_this_session'] = dict(model=model, vcpus=ncpu)
    sv['cpu_day_session'] = dict(model=DAY_SHARD0['cpu'], vcpus=DAY_SHARD0['vcpus'])
    sv['note'] = ('別のセッションなのでホストの CPU の違いが s に混ざる。numba の JIT キャッシュは 2 本で同じ /tmp/R を使う。'
                  'シャード 1 が先に終わるとシャード 0 の後半は単独で走る（stage1_end_s・stage1_overlap_s を見る）')
    n_gpu = len([l for l in REPORT.get('env_kaggle', {}).get('nvidia_smi', '').splitlines() if 'Tesla' in l or 'T4' in l])
    sv['n_gpu_lines_in_nvidia_smi'] = n_gpu
    sv['gpus_seen_by_monitor'] = sorted(peak['gpu'])
    sv['s_valid'] = bool(sv.get('s') is not None and sv.get('same_queries_as_day') and len(peak['gpu']) >= 2
                         and all(st1[str(x)]['returncode'] == 0 and not st1[str(x)]['timed_out']
                                 and (shards[str(x)]['summary'].get('structs') or 0) >= 40 for x, _ in SHARD_GPU))
    REPORT['slowdown'] = sv
    ok = all((shards[str(s)]['summary'].get('structs') or 0) >= 40 and st1[str(s)]['returncode'] == 0
             and not st1[str(s)]['timed_out'] for s, _ in SHARD_GPU)
    REPORT['status'] = 'ok' if ok else 'incomplete'
    step('summary', s=sv.get('s'), status=REPORT['status'])

try:
    main()
except Exception as e:
    import traceback
    REPORT['status'] = 'error'
    REPORT['error'] = traceback.format_exc()[-4000:]
    log(REPORT['error'])
finally:
    save()
    log('SIM1_TRIAL2', REPORT['status'], el(), 's')
    log(dumps({k: REPORT.get(k) for k in ('status', 'slowdown', 'stage1_wall_s', 'stage1_peak', 'mismatch')}))
'''


def runner2():
    """昼の RUNNER の 1〜4（環境・pip・記録・配置・照合）をそのまま使い、5 以降を 2 シャード同時に差し替える。"""
    assert RUNNER.count(_STAGE1_MARK) == 1
    head = RUNNER.split(_STAGE1_MARK)[0]
    old_h = "# ---- Issue #88: TC 段 1 (simulate_v1ctx.py) の T4 試走。ノート全体 25 分で打ち切り ----"
    new_h = "# ---- Issue #88: TC 段 1 (simulate_v1ctx.py) の 2 シャード同時の T4 試走。ノート全体 60 分・段 1 は 45 分で打ち切り ----"
    assert head.count(old_h) == 1 and head.count("BUDGET = 25 * 60\n") == 1
    head = head.replace(old_h, new_h).replace("BUDGET = 25 * 60\n", "BUDGET = BUDGET2_S\n")
    return head + RUNNER2_STAGE


def build(spec_readme: Path, v4g_manifest: Path, v4b_manifest: Path, two_shards: bool = False):
    """two_shards=False は昼の 1 シャードの生成物と同じ（バイト同一）。True で 2 シャード同時のノート。"""
    import sys
    sys.path.insert(0, str(ROOT / "scripts" / "train"))
    import sim1_trial as st
    m4g = json.loads(v4g_manifest.read_text())
    m4b = json.loads(v4b_manifest.read_text())
    exp_data = st.expected_inputs(spec_readme.read_text(), m4g, m4b)
    exp_code = st.expected_code(m4b)
    assert all(e["sha256"] and len(e["sha256"]) == 64 for e in exp_data.values()), exp_data
    sources = {
        "casmi26-spec-cache/docs/README.md": hashlib.sha256(spec_readme.read_bytes()).hexdigest(),
        "casmi26-attributed-v4g-reproduction-assets/MANIFEST.json": hashlib.sha256(v4g_manifest.read_bytes()).hexdigest(),
        "casmi26-v4b-models/MANIFEST.json": hashlib.sha256(v4b_manifest.read_bytes()).hexdigest(),
    }
    slug = SLUG2 if two_shards else SLUG
    consts_lines = [
        "# ---- 定数（scripts/train/make_sim1_trial_kernel.py が配布元の記録から埋め込む） ----",
        f"SLUG = {slug!r}",
        f"TC_ARGS = {TC_ARGS!r}",
        f"TRIAL_ARGS = {TRIAL_ARGS!r}",
        f"TC_ENV = {TC_ENV!r}",
        f"PIP = {PIP!r}",
        f"EXP_DATA = json.loads({json.dumps(exp_data, ensure_ascii=False)!r})",
        f"EXP_CODE = json.loads({json.dumps(exp_code)!r})",
        f"EMBED_SOURCES = {sources!r}",
    ]
    if two_shards:
        consts_lines.append(f"DAY_SHARD0 = {DAY_SHARD0!r}")
    consts = "\n".join(consts_lines)
    md = ("# casmi26 v4b 段 1 の T4 試走（Issue #88）\n\n"
          "TC `ahmedberatozer/casmi26-training-code` の `fe_v4/simulate_v1ctx.py` を、TC 既定の引数に "
          "`--shard 0 --nshard 6 --max_structs 40` だけを足して回し、構造あたりの秒数・出力の形・GPU メモリ・RAM を "
          "`/kaggle/working/out/trial_report.json` に書く。入力の sha256 が配布元の記録と 1 つでも違えば段 1 を回さない。"
          "ノート全体 25 分で打ち切り。生成: `scripts/train/make_sim1_trial_kernel.py`。")
    code = lambda src: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}
    if two_shards:
        md = ("# casmi26 v4b 段 1 の 2 シャード同時の T4 試走（Issue #88、2026-11-14 夜）\n\n"
              "昼の試走（`taichiiiii/casmi26-v4b-sim1-trial` v1）と同じ TC 既定の引数＋`--nshard 6 --max_structs 40` で、"
              "シャード 0 を GPU 0（`CUDA_VISIBLE_DEVICES=0`）、シャード 1 を GPU 1 で同時に回し、減速倍率 "
              "s = 同時実行のシャード 0 の t_query 合計 ÷ 昼のシャード 0 の t_query 合計 を "
              "`/kaggle/working/out/trial_report.json` に書く。入力の照合は昼と同じ（1 つでも違えば段 1 を回さない）。"
              "段 1 は 45 分、ノート全体 60 分で打ち切り。生成: `scripts/train/make_sim1_trial_kernel.py --two-shards`。")
        cells = [{"cell_type": "markdown", "metadata": {}, "source": md}, code(module_source()),
                 code(module_source("sim1_trial2.py")), code(consts), code(runner2().lstrip("\n"))]
    else:
        cells = [{"cell_type": "markdown", "metadata": {}, "source": md}, code(module_source()), code(consts),
                 code(RUNNER.lstrip("\n"))]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    meta = {
        "id": slug, "title": "casmi26 v4b sim1 trial2" if two_shards else "casmi26 v4b sim1 trial", "code_file": "notebook.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu", "chemistry"], "dataset_sources": DATASETS,
        "kernel_sources": [], "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    return nb, meta


def check(nb):
    """コードセルを compile()（ノートの中と同じ順で 1 つのソースにまとめて）。"""
    code = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for i, s in enumerate(code):
        compile(s, f"<cell{i}>", "exec")
    compile("\n".join(code), "<notebook>", "exec")
    return len(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-readme", type=Path, required=True)
    ap.add_argument("--v4g-manifest", type=Path, required=True)
    ap.add_argument("--v4b-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--two-shards", action="store_true", help="2 シャード同時の試走（kernels/casmi26-v4b-sim1-trial2）")
    a = ap.parse_args()
    if a.out is None:
        a.out = KDIR2 if a.two_shards else KDIR
    nb, meta = build(a.spec_readme, a.v4g_manifest, a.v4b_manifest, two_shards=a.two_shards)
    n = check(nb)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")
    (a.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", a.out, "code cells", n)


if __name__ == "__main__":
    main()
