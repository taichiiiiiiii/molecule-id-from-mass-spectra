#!/usr/bin/env python3
"""
Issue #34: ランカー提出用の Kaggle データセット（コード＋モデル）とノートブックを作る。

  python scripts/submit/make_ranker_kernel.py
出力:
  kernels/casmi26_ranker_code/   → 非公開データセット taichiiiii/casmi26-ranker-code（scripts/、model_c2p.txt、frag_rescore.py、
                                    リハーサル用 v3_c1 の問い合わせと正解）
  kernels/casmi26_ranker/        → ノートブック taichiiiii/casmi26-ranker-submit（CPU、インターネット無し）

ノートブックは /kaggle/working/repo にリポジトリと同じ配置を作り（入力はシンボリックリンク）、
scripts/submit/ranker_submission.py をそのまま実行する（評価と同じコード経路）。
入力の対応表（必要ファイル → Kaggle データセット）:
  data/train.parquet, test.parquet, sample_submission.csv  → competition enveda-CASMI26-molecule-id-mass-spectra
  data/ext/v4g/{casmi26-v2-pool,casmi26-v3-models,casmi26-glacier}  → dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets
  data/ext/pop/pool_popularity.csv                          → dmitriigluzdov/casmi26-pubchem-popularity-prior
  data/ext/pubchem_tier/{pc_mass,pc_off,pc_smiles}.npy      → ahmedberatozer/casmi26-pubchem-tier
  data/ext/foldsafe/fpnet_fold0.pt                          → dmitriigluzdov/casmi26-fold-safe-fpnet
  data/ext/frag/frag_rescore.py、scripts/、model_c2p.txt     → taichiiiii/casmi26-ranker-code（このスクリプトが作る）
  rdkit                                                     → metric/rdkit-2026-3-3-wheel（import できないときだけ入れる）
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_DS = "taichiiiii/casmi26-ranker-code"
KERNEL_ID = "taichiiiii/casmi26-ranker-submit"
SOURCES = ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets", "dmitriigluzdov/casmi26-pubchem-popularity-prior",
           "ahmedberatozer/casmi26-pubchem-tier", "dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel", CODE_DS]
CODE_FILES = ["scripts/local_eval/run.py", "scripts/local_eval/pubchem.py", "scripts/local_eval/library.py",
              "scripts/ranker/ranker_features.py", "scripts/ranker/train_eval.py", "scripts/cv/local_fold0_logits.py",
              "scripts/validate/mrr25.py", "scripts/submit/ranker_submission.py", "data/ext/frag/frag_rescore.py",
              "outputs/ranker/model_c2p.txt",
              "data/holdouts/v3_c1/test.parquet", "data/holdouts/v3_c1/drop_keys.txt", "data/holdouts/v3_c1/hide_rule.json",
              "data/holdouts/v3_c1/sample_submission.csv", "outputs/holdouts/v3_c1/truth.csv"]
REHEARSAL_EXPECTED = 0.869   # train_eval（result_c2p.json）の v3_c1。ずれは記録するだけで、提出は止めない

SETUP = r'''
# 入力を探し、/kaggle/working/repo にリポジトリと同じ配置を作る（#34）
import glob, os, shutil, subprocess, sys, json, time
from pathlib import Path
T0 = time.time()
def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    if len(hits) > 1:
        print("WARNING: 複数見つかった", pattern, hits[:5])
    return Path(hits[0])
def find_comp():
    """train / test / sample_submission の 3 つが揃うディレクトリ（コンペの入力）。competitions/ を優先。"""
    dirs = sorted({Path(p).parent for p in glob.glob("/kaggle/input/**/train.parquet", recursive=True)
                   if (Path(p).parent / "test.parquet").is_file() and (Path(p).parent / "sample_submission.csv").is_file()},
                  key=lambda d: ("competitions" not in str(d) and "enveda" not in str(d).lower(), str(d)))
    if not dirs:
        raise FileNotFoundError("competition dir")
    if len(dirs) > 1:
        print("WARNING: コンペ入力の候補が複数", dirs)
    return dirs[0]
R = Path("/kaggle/working/repo")
code = find("scripts/submit/ranker_submission.py").parents[2]
shutil.copytree(code, R, dirs_exist_ok=True)
comp = find_comp()
v4g = find("casmi26-v2-pool/pool_meta.parquet").parents[1]
links = {"data/train.parquet": comp / "train.parquet",
         "data/ext/v4g": v4g,
         "data/ext/pop/pool_popularity.csv": find("pool_popularity.csv"),
         "data/ext/pubchem_tier": find("pc_smiles.npy").parent,
         "data/ext/foldsafe/fpnet_fold0.pt": find("fpnet_fold0.pt")}
for dst, src in links.items():
    d = R / dst
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        os.symlink(src, d)
    print(dst, "->", src)
TEST, SAMPLE = comp / "test.parquet", comp / "sample_submission.csv"
os.environ["NUMBA_CACHE_DIR"] = "/kaggle/working/numba_cache"
try:
    import rdkit  # noqa
except ImportError:
    whl = sorted(glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True))
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", whl[0]], check=True)
import lightgbm, rdkit, torch
print("lightgbm", lightgbm.__version__, "rdkit", rdkit.__version__, "torch", torch.__version__)
'''

SUBMIT = r'''
# 提出ファイル（評価と同じ経路: logits → run.py → ranker_features → model_c2p）
try:
    r = subprocess.run([sys.executable, "scripts/submit/ranker_submission.py", "--test", str(TEST), "--sample", str(SAMPLE),
                        "--name", "sub_test", "--out", "/kaggle/working/submission.csv"], cwd=R, timeout=7 * 3600)
    rc = r.returncode
except subprocess.TimeoutExpired:
    rc = "timeout"
if rc != 0:
    # 予備: ランカー経路が落ちたら score+prior の並び、それも無ければ sample_submission。空の行は CCO で埋め、形式を確かめる
    import pandas as pd
    sys.path.insert(0, str(R / "scripts/submit")); sys.path.insert(0, str(R / "scripts/validate"))
    import ranker_submission as RS
    fb = R / "outputs/local_eval/sub_test/submission_score+prior.csv"
    shutil.copy(fb if fb.is_file() else SAMPLE, "/kaggle/working/submission.csv")
    print("FALLBACK used:", rc, fb if fb.is_file() else SAMPLE,
          RS.fill_and_validate(Path("/kaggle/working/submission.csv"), list(pd.read_csv(SAMPLE).molecule_id)))
print(open("/kaggle/working/submission.json").read() if os.path.exists("/kaggle/working/submission.json") else "no report")
print(f"elapsed {time.time() - T0:.0f}s")
'''

REHEARSAL = r'''
# リハーサル: v3_c1 の写しを同じ経路に通し、MRR が train_eval の値（EXPECTED）と合うか記録する（提出は止めない）
EXPECTED = {expected}
# 隠れテストの再実行では誰も結果を見ないので飛ばす。時間が掛かっていても飛ばす（9 時間上限を守る）
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN") or time.time() - T0 > 3 * 3600:
    print("rehearsal skipped", bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN")), f"{{time.time() - T0:.0f}}s")
else:
  try:
    subprocess.run([sys.executable, "scripts/submit/ranker_submission.py", "--like", "v3_c1", "--name", "chk_v3_c1",
                    "--out", "/kaggle/working/rehearsal/chk_v3_c1.csv"], cwd=R, check=True, timeout=int(8 * 3600 - (time.time() - T0)))
    rep = json.load(open("/kaggle/working/rehearsal/chk_v3_c1.json"))
    print("rehearsal mrr25", rep.get("mrr25"), "expected", EXPECTED, "diag", rep.get("diagnostics"))
  except Exception as e:
    print("rehearsal failed:", repr(e))
print(f"elapsed {{time.time() - T0:.0f}}s")
'''


def cell(src: str) -> dict:
    return dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=src.strip("\n").splitlines(True))


def main() -> None:
    ds = ROOT / "kernels/casmi26_ranker_code"
    if ds.exists():
        shutil.rmtree(ds)
    for f in CODE_FILES:
        (ds / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, ds / f)
    (ds / "dataset-metadata.json").write_text(json.dumps(dict(title="casmi26-ranker-code", id=CODE_DS, licenses=[dict(name="other")]), indent=2))
    kd = ROOT / "kernels/casmi26_ranker"
    kd.mkdir(parents=True, exist_ok=True)
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
              cells=[cell(SETUP), cell(SUBMIT), cell(REHEARSAL.format(expected=REHEARSAL_EXPECTED))])
    (kd / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    meta = json.loads((ROOT / "kernels/lb479/kernel-metadata.json").read_text())
    meta.update(id=KERNEL_ID, title="casmi26 ranker submit", enable_gpu=False, enable_internet=False, dataset_sources=SOURCES,
                keywords=["chemistry"])
    meta.pop("machine_shape", None)
    (kd / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", ds, kd, sum(p.stat().st_size for p in ds.rglob("*") if p.is_file()), "bytes")


if __name__ == "__main__":
    main()
