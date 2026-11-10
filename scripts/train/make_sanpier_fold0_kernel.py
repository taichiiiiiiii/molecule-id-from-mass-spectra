#!/usr/bin/env python3
"""sanpier キットを fold 0 と検証の正解構造を外して学習するカーネルを作る（Issue #54）。

    python scripts/train/make_sanpier_fold0_kernel.py

出力:
  kernels/sanpier_fold0/notebook.ipynb, kernel-metadata.json  … Kaggle GPU（T4）で学習するノートブック
  kernels/sanpier_fold0_drop/                                 … 4 検証の正解 inchikey14（private dataset 用）

config はキットの config.json のまま（パス 3 つだけ置き換える）。キットへの変更は scripts/train/sanpier_fold0.py の 2 点だけ。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
EXPECTED_DROP = 55369
SLUG = "taichiiiii/casmi26-sanpier-fold0"
DROP_DS = "taichiiiii/casmi26-sanpier-fold0-drop"
KEYS_FILE = "sanpier_fold0_holdout_keys.json"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01"

CELL_SETUP = r'''
# ---- Issue #54: sanpier kit trained without fold-safe hold/np structures and v3 holdout answers ----
import json, sys, time, hashlib, shutil, subprocess, threading
from pathlib import Path
T0 = time.time()
log = lambda *a: print(*a, flush=True)
def _heartbeat():
    while True:
        time.sleep(300); log(f'HEARTBEAT {round(time.time() - T0)}s')
threading.Thread(target=_heartbeat, daemon=True).start()
import torch
log('ENV python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
try:
    from rdkit import rdBase
except ImportError:
    wh = sorted(Path('/kaggle/input').rglob('rdkit-*.whl')); assert wh, 'rdkit wheel dataset not attached'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(wh[0].parent), 'rdkit==2026.03.3'], check=True)
    from rdkit import rdBase
log('RDKIT', rdBase.rdkitVersion)
assert rdBase.rdkitVersion == '2026.03.3', rdBase.rdkitVersion  # same as v80 and the kit's environment.json
'''

CELL_PATCH = r'''
hits = sorted(Path('/kaggle/input').rglob('casmi_spectrum_model_v1/train.py')); assert len(hits) == 1, hits
KIT = hits[0].parent
STAGE = Path('/kaggle/working/kit_staged'); shutil.copytree(KIT, STAGE, dirs_exist_ok=True)
for _p in STAGE.glob('*.pt'):
    _p.unlink()  # published checkpoints are not used here
(STAGE / 'sanpier_fold0.py').write_text(SANPIER_FOLD0_SRC)
sys.path.insert(0, str(STAGE))
import sanpier_fold0 as sf
_src = (STAGE / 'prepare.py').read_text()
(STAGE / 'prepare.py').write_text(sf.patch_prepare_source(_src))
log('KIT_DELTA prepare.py sha256 before', hashlib.sha256(_src.encode()).hexdigest()[:16],
    'after', hashlib.sha256((STAGE / 'prepare.py').read_bytes()).hexdigest()[:16])
import prepare, train, features
'''

CELL_DROP = r'''
import pandas as pd, pyarrow.parquet as pq
_tp = sorted(Path('/kaggle/input').rglob('train.parquet')); assert len(_tp) == 1, _tp
TRAIN = _tp[0]
_sp = [p for p in Path('/kaggle/input').rglob('split.parquet') if 'fold-safe' in str(p)]; assert len(_sp) == 1, _sp
SPLIT = pd.read_parquet(_sp[0], columns=['inchikey14', 'fold'])
_kf = sorted(Path('/kaggle/input').rglob(KEYS_FILE)); assert len(_kf) == 1, _kf
HOLDOUT_KEYS = json.loads(_kf[0].read_text())
assert sorted(HOLDOUT_KEYS) == sorted(HOLDOUTS), sorted(HOLDOUT_KEYS)
DROP_IK = sf.drop_inchikey14(SPLIT.inchikey14, HOLDOUT_KEYS)
log('DROP_IK', len(DROP_IK), 'split rows', len(SPLIT), SPLIT.fold.value_counts().to_dict())
assert len(DROP_IK) == EXPECTED_DROP, len(DROP_IK)
# kit structure keys (stereo-stripped, tautomer-canonical) of every dropped row, so a tautomer twin is dropped too
_smi = set()
for _b in pq.ParquetFile(TRAIN).iter_batches(batch_size=65536, columns=['inchikey14', 'normalized_smiles']):
    _d = _b.to_pandas()
    _smi.update(_d.normalized_smiles[_d.inchikey14.isin(DROP_IK)].astype(str))
DROP_KIT, _bad = set(), 0
for _s in sorted(_smi):
    try:
        DROP_KIT.add(features.structure(_s, 2048)[0])
    except (ValueError, RuntimeError):
        _bad += 1
log('DROP_KIT', len(DROP_KIT), 'from smiles', len(_smi), 'invalid', _bad, round(time.time() - T0), 's')
prepare.DROP_IK.update(DROP_IK); prepare.DROP_KIT.update(DROP_KIT)
'''

CELL_TRAIN = r'''
cfg = json.loads((KIT / 'config.json').read_text())
cfg.update(train_path=str(TRAIN), work='/kaggle/working/cache_fold0', output='/kaggle/working/casmi_spectrum_model_fold0')
log('CFG', json.dumps(cfg))
audit = prepare.prepare(cfg)
WORK, OUT = Path(cfg['work']), Path(cfg['output']); OUT.mkdir(parents=True, exist_ok=True)
shutil.copy2(WORK / 'data_audit.json', OUT / 'data_audit.json'); log('PREP_DONE', round(time.time() - T0), 's')
_kitkeys = set(pd.read_parquet(WORK / 'molecules.parquet', columns=['key']).key)
result = train.run(cfg)
log('TRAIN_RESULT', json.dumps(result))
'''

CELL_AUDIT = r'''
TRAIN_IK = set(prepare.TRAIN_IK)
(OUT / 'train_keys.txt').write_text('\n'.join(sorted(TRAIN_IK)) + '\n')
(OUT / 'drop_keys_union.txt').write_text('\n'.join(sorted(DROP_IK)) + '\n')
LEAK = sf.leak_audit(TRAIN_IK, zip(SPLIT.inchikey14, SPLIT.fold), HOLDOUT_KEYS)
DROP_AUDIT = dict(issue=54, drop_inchikey14=len(DROP_IK), drop_kitkey=len(DROP_KIT), train_inchikey14=len(TRAIN_IK),
                  train_kitkey=len(_kitkeys), kitkey_overlap=len(_kitkeys & DROP_KIT), rejected=audit['rejected'],
                  leak=LEAK, seconds=round(time.time() - T0))
(OUT / 'drop_audit.json').write_text(json.dumps(DROP_AUDIT, indent=2))
(OUT / 'config_used.json').write_text(json.dumps(cfg, indent=2))
for _n in ('model.py', 'features.py', 'inference.py', 'prepare.py', 'sanpier_fold0.py'):
    shutil.copy2(STAGE / _n, OUT / _n)
log('DROP_AUDIT', json.dumps(DROP_AUDIT))
assert LEAK['total'] == 0 and DROP_AUDIT['kitkey_overlap'] == 0, DROP_AUDIT
for _p in sorted(OUT.iterdir()):
    log('ARTIFACT', _p.name, _p.stat().st_size, hashlib.sha256(_p.read_bytes()).hexdigest()[:16])
shutil.rmtree(WORK, ignore_errors=True); shutil.rmtree(STAGE, ignore_errors=True)
log(f'SANPIER_FOLD0_DONE {round(time.time() - T0)}s')
'''


def holdout_keys() -> dict:
    out = {}
    for h in HOLDOUTS:
        keys = (ROOT / "data/holdouts" / h / "drop_keys.txt").read_text().split()
        out[h] = sorted(set(keys))
    return out


def cell(src: str) -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook() -> dict:
    module = (ROOT / "scripts/train/sanpier_fold0.py").read_text()
    consts = (f"SANPIER_FOLD0_SRC = {module!r}\nHOLDOUTS = {list(HOLDOUTS)!r}\n"
              f"EXPECTED_DROP = {EXPECTED_DROP}\nKEYS_FILE = {KEYS_FILE!r}\n")
    md = {"cell_type": "markdown", "metadata": {},
          "source": ["# sanpier kit, fold-safe retrain (Issue #54)\n",
                     "Kit config verbatim; prepare.py gets the normalized_smiles alias and drops rows whose inchikey14 is in "
                     "fold-safe split.parquet (hold+np) or the v3 holdout answers, plus their tautomer kit keys.\n"]}
    cells = [md, cell(consts + CELL_SETUP), cell(CELL_PATCH), cell(CELL_DROP), cell(CELL_TRAIN), cell(CELL_AUDIT)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def metadata() -> dict:
    return {"id": SLUG, "title": "casmi26 sanpier fold0", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False, "keywords": ["gpu", "chemistry"],
            "dataset_sources": ["sanpier/casmi-spectrum-model-v1", "dmitriigluzdov/casmi26-fold-safe-fpnet", DROP_DS,
                                "metric/rdkit-2026-3-3-wheel"],
            "kernel_sources": [], "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4", "docker_image": DOCKER}


def main() -> None:
    kdir = ROOT / "kernels/sanpier_fold0"; kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n")
    (kdir / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    ddir = ROOT / "kernels/sanpier_fold0_drop"; ddir.mkdir(parents=True, exist_ok=True)
    (ddir / KEYS_FILE).write_text(json.dumps(holdout_keys(), indent=1) + "\n")
    (ddir / "dataset-metadata.json").write_text(json.dumps(
        {"title": "casmi26 sanpier fold0 drop", "id": DROP_DS, "licenses": [{"name": "CC0-1.0"}]}, indent=2) + "\n")
    print("wrote", kdir, ddir)


if __name__ == "__main__":
    main()
