#!/usr/bin/env python3
"""整列チャネルの速度の試走カーネルを作る（Issue #99、計画 2026-11-14 昼 項目 5）。

    python scripts/train/make_align_speed_kernel.py

出力: kernels/casmi26-align-speed-trial/notebook.ipynb, kernel-metadata.json
  T4、インターネット有り、private。新しい dataset は上げない（既存の公開データセット
  dmitriigluzdov/casmi26-fold-safe-fpnet の split.parquet と競技データだけを読む）。
  本体は scripts/train/align_speed_trial.py をそのまま文字列で埋め込み、ノートの中でファイルに書いて import する。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "kernels/casmi26-align-speed-trial"
SLUG = "taichiiiii/casmi26-align-speed-trial"
MODULE = ROOT / "scripts/train/align_speed_trial.py"

CELL_RUN = r'''
import json, sys, time, shutil
from pathlib import Path
T0 = time.time()
WORK = Path('/kaggle/tmp_align'); WORK.mkdir(parents=True, exist_ok=True)
(WORK / 'align_speed_trial.py').write_text(ALIGN_SRC)
sys.path.insert(0, str(WORK))
import align_speed_trial as A
assert A.CKPT_MD5 == '4deee91dd4486c2b6cbcde1f200b8f50' and A.DREAMS_COMMIT.startswith('dbec3a0')
rep = A.run(out_dir='/kaggle/working', work_dir=str(WORK), input_root='/kaggle/input', t0=T0)
print(json.dumps({k: rep.get(k) for k in ('status', 'stage', 'error', 'summary', 'elapsed_s')}, indent=1, default=str))
shutil.rmtree(WORK, ignore_errors=True)
print('OUTPUTS', sorted(p.name for p in Path('/kaggle/working').iterdir()))
'''


def cell(src: str) -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def build_notebook() -> dict:
    src = MODULE.read_text()
    md = {"cell_type": "markdown", "metadata": {},
          "source": ["# casmi26 align speed trial (Issue #99)\n",
                     "Speed only (not used for evaluation): frozen DreaMS `ssl_model.ckpt` (Zenodo 10997887, md5 checked) "
                     "forward on 10,000 fold-safe train spectra and one epoch of a provisional alignment MLP on T4. "
                     "Writes `align_speed_report.json` only.\n"]}
    cells = [md, cell(f"ALIGN_SRC = {src!r}\n"), cell(CELL_RUN)]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def metadata() -> dict:
    return {"id": SLUG, "title": "casmi26 align speed trial", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": ["gpu", "chemistry"],
            "dataset_sources": ["dmitriigluzdov/casmi26-fold-safe-fpnet"],
            "kernel_sources": [], "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}


def main() -> None:
    KDIR.mkdir(parents=True, exist_ok=True)
    (KDIR / "notebook.ipynb").write_text(json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n")
    (KDIR / "kernel-metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print("wrote", KDIR)


if __name__ == "__main__":
    main()
