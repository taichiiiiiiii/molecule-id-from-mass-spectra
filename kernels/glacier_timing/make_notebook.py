#!/usr/bin/env python3
"""Issue #242: GLACIER の時間測定カーネル（GPU、提出しない）のノートブックを作る。

    .venv/bin/python kernels/glacier_timing/make_notebook.py      # notebook.ipynb を書き、セルを compile で確かめる

カーネルの中身: 入力 CSV（private dataset taichiiiii/casmi26-glacier-timing-input、2,000 本）を、1 行 = 1 候補・
1 区分の擬似分子（問い合わせはダミーの 1 ピーク。点の値は使わない）にして、ahmedberatozer/casmi26-glacier の
gl_runner.py をそのまま subprocess で呼ぶ（source = pool / tier の 2 回）。秒/本 = meta の t_pred / n_pred
（重みの読み込み・wheel の準備を除く）。最初の chunk を除いた値は stderr の chunk 行から出す。全体は 3,600 s で打ち切る。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELL = r'''
import os, sys, json, time, glob, re, csv, subprocess, hashlib
T0 = time.time()
HARD_LIMIT_S = 3600.0          # カーネル全体の打ち切り（計画: 3,600 s）
RUN_BUDGET_S = 1400.0          # 1 回（pool / tier）あたりのランナーの予算
GRACE_S = 120.0

def tag(name, **kw):
    print('TAG ' + json.dumps(dict(tag=name, t=round(time.time() - T0, 1), **kw)), flush=True)

def find_one(pattern):
    hits = sorted({os.path.realpath(h) for h in glob.glob(pattern, recursive=True)})
    assert len(hits) == 1, (pattern, hits)
    return hits[0]

GL_RUNNER = find_one('/kaggle/input/**/casmi26-glacier/**/gl_runner.py')
GL_PKG = os.path.dirname(GL_RUNNER)
IN_CSV = find_one('/kaggle/input/**/glacier_timing_input.csv')
WHEELS = os.path.dirname(find_one('/kaggle/input/**/casmi26-iceberg/**/wheels/rdkit-*.whl.ice'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'
tag('env', gpu=gpu, torch=torch.__version__, python=sys.version.split()[0], gl_pkg=GL_PKG,
    gl_runner_sha256=sha(GL_RUNNER), ckpt_sha256=sha(os.path.join(GL_PKG, 'ckpt', 'glacier.pt')),
    input_sha256=sha(IN_CSV))
rows = list(csv.DictReader(open(IN_CSV)))
assert len(rows) == 2000, len(rows)

def items_for(src):
    out = []
    for r in rows:
        if r['source'] != src:
            continue
        sp = dict(mz=[100.0], it=[1.0], prec=1000.0, adduct=r['adduct'], mode='positive', instr=r['instr'],
                  ce=float(r['ce']))
        out.append(dict(mid='r%04d' % int(r['row']), cands=[r['smiles']], spectra=[sp]))
    return out

CHUNK_RE = re.compile(r'\[gl\s+([0-9.]+)s\] chunk: (\d+) mols (\d+) jobs ([0-9.]+)s')
report = dict(gpu=gpu, runs={})
SITE = '/kaggle/working/ice_site'      # wheel の入れ方で落ちたら、以後は 'none'
for src in ('pool', 'tier'):
    left = HARD_LIMIT_S - (time.time() - T0) - GRACE_S - 60.0
    if left < 120:
        tag('skip', source=src, reason='hard limit', left=round(left, 1))
        continue
    budget = min(RUN_BUDGET_S, left)
    wd = '/kaggle/working/gl_' + src
    os.makedirs(wd, exist_ok=True)
    fin, fout = os.path.join(wd, 'in.json'), os.path.join(wd, 'out.json')
    items = items_for(src)
    json.dump(items, open(fin, 'w'))
    cmd = [sys.executable, GL_RUNNER, GL_PKG, fin, fout, '--device', 'cuda', '--budget', str(budget),
           '--site', SITE, '--wheels', WHEELS, '--any-rdkit', '--chunk-jobs', '250']
    def run_once(c, b):
        try:
            r = subprocess.run(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=b + GRACE_S)
            txt = r.stdout
        except subprocess.TimeoutExpired as e:
            txt = 'TIMEOUT ' + str(e.stdout or '')
        m = json.load(open(fout + '.meta.json')) if os.path.exists(fout + '.meta.json') else {}
        return txt, m
    t1 = time.time()
    logtxt, meta = run_once(cmd, budget)
    site_mode = 'wheels' if SITE != 'none' else 'none'
    left = HARD_LIMIT_S - (time.time() - T0) - GRACE_S - 60.0
    if (SITE != 'none' and meta.get('status') == 'error' and not meta.get('n_pred')
            and 'install' in json.dumps(meta.get('errors')) and left >= 120):
        # wheel の入れ方で落ちたときだけ、イメージの環境のまま 1 回やり直す（同じ push の中）
        tag('retry_site_none', source=src, errors=(meta.get('errors') or [])[-1:])
        site_mode = SITE = 'none'
        b2 = min(budget, left)
        cmd2 = [x if x != '/kaggle/working/ice_site' else 'none' for x in cmd]
        cmd2[cmd2.index('--budget') + 1] = str(b2)
        logtxt2, meta = run_once(cmd2, b2)
        logtxt = logtxt + '\n--- retry --site none ---\n' + logtxt2
    wall = time.time() - t1
    open(os.path.join(wd, 'runner.log'), 'w').write(logtxt)
    chunks = [dict(t=float(a), mols=int(b), jobs=int(c), secs=float(d)) for a, b, c, d in CHUNK_RE.findall(logtxt)]
    n_pred, t_pred = meta.get('n_pred') or 0, meta.get('t_pred') or 0.0
    rest = chunks[1:]
    rest_jobs = sum(c['jobs'] for c in rest)
    rest_secs = sum(c['secs'] for c in rest)
    res = dict(n_rows=len(items), site_mode=site_mode, status=meta.get('status'), device=meta.get('device'), rdkit=meta.get('rdkit'),
               n_pred=n_pred, n_pred_fail=meta.get('n_pred_fail'), n_feat_fail=meta.get('n_feat_fail'),
               n_bad_smiles=meta.get('n_bad_smiles'), t_pred=t_pred, t_load=meta.get('t_load'),
               seconds=meta.get('seconds'), wall=round(wall, 1), max_mem_gb=meta.get('max_mem_gb'),
               bs_final=meta.get('bs_final'),
               sec_per_pred=round(t_pred / n_pred, 6) if n_pred else None,
               sec_per_pred_excl_first_chunk=round(rest_secs / rest_jobs, 6) if rest_jobs else None,
               chunks=chunks, errors=(meta.get('errors') or [])[:5])
    report['runs'][src] = res
    tag('run', source=src, **{k: v for k, v in res.items() if k not in ('chunks', 'errors')})
    print(logtxt[-3000:])
tot_n = sum(r['n_pred'] for r in report['runs'].values())
tot_t = sum(r['t_pred'] for r in report['runs'].values())
report.update(n_pred=tot_n, t_pred=round(tot_t, 2), sec_per_pred=round(tot_t / tot_n, 6) if tot_n else None,
              total_wall=round(time.time() - T0, 1))
json.dump(report, open('/kaggle/working/glacier_timing_report.json', 'w'), indent=1)
tag('done', gpu=gpu, n_pred=tot_n, t_pred=report['t_pred'], sec_per_pred=report['sec_per_pred'],
    total_wall=report['total_wall'])
'''


def main():
    src = CELL.strip('\n')
    compile(src, 'glacier_timing_cell', 'exec')
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                            language_info=dict(name='python')),
              cells=[dict(cell_type='markdown', id='md0', metadata={},
                          source=['# GLACIER timing (#242)\n', 'GPU timing only; never submitted.']),
                     dict(cell_type='code', id='c1', metadata={}, execution_count=None, outputs=[],
                          source=[l + '\n' for l in src.split('\n')[:-1]] + [src.split('\n')[-1]])])
    p = HERE / 'notebook.ipynb'
    p.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    back = json.loads(p.read_text())
    for c in back['cells']:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), 'cell', 'exec')
    print('ok', p, p.stat().st_size)


if __name__ == '__main__':
    main()
