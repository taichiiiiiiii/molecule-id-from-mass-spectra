#!/usr/bin/env python3
"""kernels/lb479 のノートブックから、ホールドアウト採点用のノートブック kernels/holdout を作る。

変更点（それ以外はそのまま）:
1. 先頭にセルを 1 つ足す: ホールドアウトのデータセットを探し、CASMI_TEST_PATH / CASMI_SAMPLE_PATH を設定し、
   パイプラインの find('test.parquet') がこのデータセットを選ぶことを確かめる。
2. 外側のセルの COMP の候補の先頭に、ホールドアウトのディレクトリを足す。

    python scripts/cv/make_holdout_kernel.py --dataset taichiiiii/casmi26-holdout-v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("kernels/lb479")
DST = Path("kernels/holdout")

GUARD = '''# ── Holdout evaluation guard (taichiiiii) ──────────────────────────────────────
# このノートブックは隠れテストではなく、正解付きホールドアウトで公開パイプラインを採点するためのもの。
import glob, os
from pathlib import Path
HOLD = next(Path(p).parent for p in sorted(glob.glob('/kaggle/input/**/HOLDOUT.json', recursive=True), key=len))
os.environ['CASMI_TEST_PATH'] = str(HOLD / 'test.parquet')
os.environ['CASMI_SAMPLE_PATH'] = str(HOLD / 'sample_submission.csv')
_hits = sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)
print('HOLD', HOLD, '| test.parquet candidates', _hits)
assert Path(_hits[0]).parent == HOLD, 'pipeline find() would not pick the holdout test.parquet'
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--slug", default="casmi26-holdout-eval")
    args = ap.parse_args()

    nb = json.loads((SRC / "notebook.ipynb").read_text())
    old = "COMP = next((p for p in (INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',"
    new = "COMP = next((p for p in (HOLD, INPUT / 'competitions/enveda-CASMI26-molecule-id-mass-spectra',"
    hits = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if old in src:
            c["source"] = src.replace(old, new, 1)
            hits += 1
    assert hits == 1, f"COMP line found {hits} times"
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": GUARD})

    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"taichiiiii/{args.slug}", title=args.slug.replace("-", " "), is_private=True)
    meta["dataset_sources"] = [args.dataset] + [d for d in meta["dataset_sources"] if d != args.dataset]

    DST.mkdir(parents=True, exist_ok=True)
    (DST / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    (DST / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", DST, meta["id"], meta["dataset_sources"][:2])


if __name__ == "__main__":
    main()
