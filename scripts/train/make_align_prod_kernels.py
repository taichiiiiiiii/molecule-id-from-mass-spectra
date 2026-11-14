#!/usr/bin/env python3
"""整列チャネルの埋め込みの**予備**の Kaggle カーネルを作る（push はしない。Issue #109）。

本番はローカル（scripts/train/align_local_embed.py → align_train.py）。ローカルの MPS が遅すぎたときだけ、埋め込みをこの
カーネルで作り、出力（約 4.1 GB）を outputs/align_embed/ に取って、学習はローカルの align_train.py で回す。

    python scripts/train/make_align_prod_kernels.py            # 初回
    python scripts/train/make_align_prod_kernels.py --resume   # 埋め込みが時間切れで partial のとき（前の版の出力を付けて再開）

出力:
  kernels/casmi26-align-embed/  … T4、インターネット有り（Zenodo の重みと GitHub の DreaMS のコード）。DreaMS fp32 の埋め込み
本体は scripts/train/ の .py をそのまま文字列で埋め込み、ノートの中でファイルに書いて import する。
v3 の 4 集合の正解キーは #54 と同じファイル（kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json）を埋め込む。
新しい dataset は上げない。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "scripts/train"
EMBED_KDIR = ROOT / "kernels/casmi26-align-embed"
EMBED_SLUG = "taichiiiii/casmi26-align-embed"
KEYS = ROOT / "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json"
MANIFEST = TRAIN_DIR / "align_query_manifest.json"
EMBED_MODULES = ("align_speed_trial.py", "sanpier_fold0.py", "align_query_manifest.py", "align_embed.py")
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

CELL_WRITE = r'''
import json, sys, time, threading
from pathlib import Path
T0 = time.time()
WORK = Path('/kaggle/tmp_align'); WORK.mkdir(parents=True, exist_ok=True)
for _n, _s in SRC.items():
    (WORK / _n).write_text(_s)
sys.path.insert(0, str(WORK))
def _heartbeat():
    while True:
        time.sleep(600); print(f'HEARTBEAT {round(time.time() - T0)}s', flush=True)
threading.Thread(target=_heartbeat, daemon=True).start()
'''

CELL_EMBED = r'''
import align_speed_trial as A, align_embed as E
assert A.CKPT_MD5 == '4deee91dd4486c2b6cbcde1f200b8f50' and A.DREAMS_COMMIT.startswith('dbec3a0') and E.PEAKS == 100
import os; assert os.environ.get(E.FAKE_ENV) is None
rep = E.run(out_dir='/kaggle/working', work_dir=str(WORK), input_root='/kaggle/input', t0=T0,
            holdout_keys=HOLDOUT_KEYS, manifest=MANIFEST, src_dir=str(WORK))
print(json.dumps({k: rep.get(k) for k in ('status', 'error', 'n_shards', 'shards_done', 'resumed_shards', 'n_train_spectra',
                                          'n_train_structures', 'n_query_embedded', 'leak_audit', 'totals', 'dtypes',
                                          'elapsed_s')}, indent=1, default=str))
print('OUTPUTS', len(list(Path('/kaggle/working').iterdir())), 'files')
'''

def cell(src: str) -> dict:
    lines = src.strip("\n").splitlines(keepends=True)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def _nb(md_lines: list[str], cells: list[dict]) -> dict:
    md = {"cell_type": "markdown", "metadata": {}, "source": md_lines}
    return {"cells": [md] + cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                               "name": "python3"},
                                                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def _consts(modules) -> str:
    src = {n: (TRAIN_DIR / n).read_text() for n in modules}
    keys = json.loads(KEYS.read_text())
    return f"SRC = {src!r}\nHOLDOUT_KEYS = {keys!r}\n"


def build_embed_notebook() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    consts = _consts(EMBED_MODULES) + f"MANIFEST = {manifest!r}\n"
    return _nb(["# casmi26 align embed\n",
                "Frozen DreaMS `ssl_model.ckpt` (Zenodo 10997887, md5 checked), fp32, 100 peaks: embeds fold-safe train "
                "spectra (split hold/np and the 4 v3 answers excluded) and the 4,756 v3 query spectra. Writes fp16 shards "
                "(one per train.parquet row group) + index + done marker to /kaggle/working; re-runs resume from finished "
                "shards.\n"], [cell(consts), cell(CELL_WRITE), cell(CELL_EMBED)])


def embed_metadata(resume: bool = False) -> dict:
    return {"id": EMBED_SLUG, "title": "casmi26 align embed", "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": True, "keywords": ["gpu", "chemistry"],
            "dataset_sources": ["dmitriigluzdov/casmi26-fold-safe-fpnet"],
            "kernel_sources": [EMBED_SLUG] if resume else [], "competition_sources": [COMP], "model_sources": [],
            "machine_shape": "NvidiaTeslaT4"}


def write(kdir: Path, nb: dict, meta: dict) -> None:
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true", help="embed の前の版の出力を kernel_sources に付ける")
    args = ap.parse_args()
    write(EMBED_KDIR, build_embed_notebook(), embed_metadata(args.resume))
    print("wrote", EMBED_KDIR, "(resume)" if args.resume else "")


if __name__ == "__main__":
    main()
