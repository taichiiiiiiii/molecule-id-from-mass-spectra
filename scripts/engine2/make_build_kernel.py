"""Engine 2 の自前 2 表を作る Kaggle CPU カーネルと、生成コードの dataset を組み立てる（#74）。

    .venv/bin/python scripts/engine2/make_build_kernel.py --code-dir <scratch>/e2code
出力:
- <code-dir>/: scripts/engine2（build_tables.py・base_fs.py・eng/）、engine2_table_schema.py、drop_keys/<h>/drop_keys.txt、
  dataset-metadata.json（taichiiiii/casmi26-engine2-build-code）
- kernels/engine2_build/{trial,run1,run2}/: notebook.ipynb + kernel-metadata.json（CPU、private）。中身はセル 1 の
  N_MOL・REPEAT だけが違う（trial 50・2、run1/run2 2000・1）。run1 と run2 は notebook が同一
"""
import argparse
import json
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE_SLUG = "taichiiiii/casmi26-engine2-build-code"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
SOURCES = ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets", "dmitriigluzdov/casmi26-fold-safe-fpnet",
           "dmitriigluzdov/casmi26-identity-disjoint-validation-folds", "metric/rdkit-2026-3-3-wheel", CODE_SLUG]
HOLDOUTS = ["v3_c1", "v3_c2", "enveda_v1", "v3_c2n"]
KERNELS = {"trial": (50, 2), "run1": (2000, 1), "run2": (2000, 1)}

CELL_SETUP = r'''# 入力の場所（明示パス）と RDKit 2026.03.3
import os, sys, json, glob, subprocess, time, resource
from pathlib import Path
T0 = time.time()
IN = Path('/kaggle/input')
def one(pattern, must=None):
    hits = sorted(p for p in IN.rglob(pattern) if must is None or must in str(p))
    assert len(hits) >= 1, (pattern, must)
    return min(hits, key=lambda p: len(str(p)))
V4G = one('casmi26-v3-models/code/casmi/chem.py', 'casmi26-attributed-v4g-reproduction-assets').parents[3]
FS = one('fpnet_fold1.pt', 'casmi26-fold-safe-fpnet').parent
FOLDS = one('structures.parquet', 'casmi26-identity-disjoint-validation-folds').parent
CODE = one('build_tables.py', 'casmi26-engine2-build-code').parent
COMP = min((p for p in IN.rglob('train.parquet') if 'molecule-id-mass-spectra' in str(p).lower()), key=lambda p: len(str(p))).parent
import shutil
shutil.copytree(CODE, '/kaggle/working/e2code', dirs_exist_ok=True)   # numba の cache を書ける場所で動かす
CODE = Path('/kaggle/working/e2code')
TABLES_PRESENT = sorted(str(p) for p in IN.rglob('*') if p.is_file() and (p.name == 'rank_train.npz' or p.name.startswith('sim_rank_rows')))
print('original Engine 2 tables under /kaggle/input (v4g の写し、読まない):', TABLES_PRESENT)
assert all('casmi26-attributed-v4g-reproduction-assets' in p for p in TABLES_PRESENT), TABLES_PRESENT
from importlib.metadata import version as _v
try:
    installed = _v('rdkit')
except Exception:
    installed = ''
if installed.replace('.03.', '.3.') != '2026.3.3':
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheels = sorted(IN.rglob(f'rdkit-2026.3.3-{tag}-*.whl'))
    assert wheels, f'RDKit 2026.3.3 wheel for {tag}'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '-q', str(wheels[0])])
ASSETS = dict(coco_dir=str(V4G / 'coconut-casmi26-candidates'), bio_dir=str(V4G / 'casmi-chebi-lmsd-structures'),
              pc_dir=str(V4G / 'casmi26-pubchemlite-cc0-candidate-fingerprints'), fs_models_dir=str(FS),
              fs_bits=str(FS / 'fp_bits.npy'), casmi_code=str(V4G / 'casmi26-v3-models' / 'code'),
              split=str(FS / 'split.parquet'), structures=str(FOLDS / 'structures.parquet'),
              drop_keys_dir=str(CODE.parent / 'drop_keys') if (CODE.parent / 'drop_keys').is_dir() else str(CODE / 'drop_keys'))
for k, v in ASSETS.items():
    assert os.path.exists(v), (k, v)
json.dump(ASSETS, open('/kaggle/working/assets.json', 'w'), indent=1)
print(json.dumps(ASSETS, indent=1))
'''

CELL_RUN = r'''# 生成（サブプロセス。PYTHONHASHSEED=0、ログは build.log）
OUT = '/kaggle/working/own'
cmd = [sys.executable, str(CODE / 'build_tables.py'), '--train', str(COMP / 'train.parquet'), '--assets', '/kaggle/working/assets.json',
       '--out', OUT, '--n-mol', str(N_MOL), '--repeat', str(REPEAT)]
print(cmd, flush=True)
with open('/kaggle/working/build.log', 'w') as lf:
    rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(CODE),
                        env={**os.environ, 'PYTHONHASHSEED': '0', 'OPENBLAS_NUM_THREADS': '1'}).returncode
PEAK_GB = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6   # Linux は KB
log = open('/kaggle/working/build.log').read()
print(log[-6000:])
print('child peak RSS GB', round(PEAK_GB, 2), 'rc', rc)
assert rc == 0, rc
'''

CELL_CHECK = r'''# 確認（形・正例・恒等式・決定性・読んだファイル）
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('sch', str(CODE / 'engine2_table_schema.py'))
SCH = importlib.util.module_from_spec(spec); spec.loader.exec_module(SCH)
rep = json.load(open(f'{OUT}/build_report.json'))
chk = dict(n_mol=N_MOL, repeat=REPEAT, child_peak_rss_gb=round(PEAK_GB, 2), opened_forbidden=rep['opened_forbidden'], deterministic=rep.get('deterministic'),
           runs=[], wall_secs=round(time.time() - T0, 1))
for r in rep['runs']:
    for path, blocks in ((r['files'][0], None), (r['files'][1], ['base'])):
        d = SCH.describe(path, blocks)
        fr = {k: v.get('frac', v.get('status')) for k, v in d['identities'].items()}
        chk['runs'].append(dict(file=os.path.basename(path), n_rows=d['n_rows'], n_feat=d['n_feat'], blocks=d['blocks'],
                                pos_per_group=d['pos_per_group'], n_groups=d['n_groups'],
                                identities_all_one=all(v in (1.0, 'all_zero') for v in fr.values()), identities=fr))
json.dump(chk, open('/kaggle/working/check.json', 'w'), indent=1)
print(json.dumps(chk, indent=1)[:6000])
print('report:', json.dumps({k: rep[k] for k in rep if k not in ('opened_files', 'runs')}, indent=1)[:4000])
for r in rep['runs']:
    print('run', {k: r[k] for k in r if k != 'queries'})
'''


def nb(n_mol, repeat):
    def code(src):
        return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src)
    md = dict(cell_type="markdown", metadata={}, source=(
        "# CASMI26 Engine 2 の自前 2 表の生成（#74）\n"
        "生成コードは `taichiiiii/casmi26-engine2-build-code`（リポジトリの `scripts/engine2/`、#72）。"
        "v4g の中の元の 2 表は読まない（明示パス、ファイル名ガード、読んだファイルの一覧）。"))
    params = f"N_MOL = {n_mol}\nREPEAT = {repeat}\n"
    return dict(cells=[md, code(params), code(CELL_SETUP), code(CELL_RUN), code(CELL_CHECK)],
                metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                              language_info=dict(name="python")), nbformat=4, nbformat_minor=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code-dir", required=True)
    args = ap.parse_args()
    cd = args.code_dir
    if os.path.isdir(cd):
        shutil.rmtree(cd)
    os.makedirs(os.path.join(cd, "eng"))
    for f in ("build_tables.py", "base_fs.py"):
        shutil.copy2(os.path.join(ROOT, "scripts", "engine2", f), cd)
    for f in ("casmi_engine.py", "pv.py", "pv_fp.py", "NOTICE.md"):
        shutil.copy2(os.path.join(ROOT, "scripts", "engine2", "eng", f), os.path.join(cd, "eng"))
    shutil.copy2(os.path.join(ROOT, "scripts", "research", "engine2_table_schema.py"), cd)
    for h in HOLDOUTS:
        os.makedirs(os.path.join(cd, "drop_keys", h))
        shutil.copy2(os.path.join(ROOT, "data", "holdouts", h, "drop_keys.txt"), os.path.join(cd, "drop_keys", h))
    json.dump(dict(title="casmi26 engine2 build code", id=CODE_SLUG, licenses=[dict(name="other")]),
              open(os.path.join(cd, "dataset-metadata.json"), "w"), indent=1)
    for name, (n_mol, rep) in KERNELS.items():
        kd = os.path.join(ROOT, "kernels", "engine2_build", name)
        os.makedirs(kd, exist_ok=True)
        json.dump(nb(n_mol, rep), open(os.path.join(kd, "notebook.ipynb"), "w"), indent=1, ensure_ascii=False)
        slug = f"casmi26-engine2-build-{name}"
        meta = dict(id=f"taichiiiii/{slug}", title=f"casmi26 engine2 build {name}", code_file="notebook.ipynb",
                    language="python", kernel_type="notebook", is_private=True, enable_gpu=False, enable_tpu=False,
                    enable_internet=False, dataset_sources=SOURCES, kernel_sources=[],
                    competition_sources=["enveda-CASMI26-molecule-id-mass-spectra"], model_sources=[], docker_image=DOCKER)
        json.dump(meta, open(os.path.join(kd, "kernel-metadata.json"), "w"), indent=2)
        print("wrote", kd)
    print("code dataset", cd, sorted(os.listdir(cd)))


if __name__ == "__main__":
    main()
