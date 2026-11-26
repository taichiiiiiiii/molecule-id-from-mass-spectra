#!/usr/bin/env python3
"""
#189 のパスの確認用の CPU カーネル（GPU 0、提出しない）を作る: taichiiiii/casmi26-btables-pathcheck。

- 入力は #189 の診断ノート（make_lb487_btables.build()）と同じ dataset_sources・kernel_sources・competition_sources
- 診断ノートの凍結セル 32 に入る「表の決め方」のコード（PATCH_BEGIN〜PATCH_END、eng_runner.py の書き込みを含む）を
  そのまま実行し、2 本の npz がちょうど 1 つずつ見え sha256 が #189 §1 の値と一致することを確かめる
- 書かれた eng_runner.py の verify_own_tables でもう一度照合する（runner 側の照合）
- b_run1 の出力のファイル名ごとに、/kaggle/input の下の同名ファイルを数え、元のノートが名前で探すファイル
  （下の LOOKUP）が b_run1 の下に無いこと、b_run1 の下の名前が他の入力の名前と重なるときは sorted(key=len)[0] が
  b_run1 の下でないこと（＝元のノートの find の 1 番目が変わらない）を確かめる
- 結果を /kaggle/working/pathcheck.json に書き、`PATHCHECK {json}` を 1 行出す。どれか外れたら最後に例外で止める

    python scripts/submit/make_btables_pathcheck.py [--out-dir kernels/btables_pathcheck]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_lb487_btables", ROOT / "scripts" / "submit" / "make_lb487_btables.py")
B = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_lb487_btables", B)
_spec.loader.exec_module(B)

SLUG = "casmi26-btables-pathcheck"
# 元のノート（0.487 写し）が名前で探すファイル（find / glob / rglob の名前。make_lb487_btables のテスト (k) と同じ調べ方で
# 拾ったもの）。これらが b_run1 の下に見つかったら、元の動きが変わりうるので止める
LOOKUP = ["coco_fp.npy", "coco_meta.pkl", "fp_bits.npy", "bio_fp.npy", "bio_meta.pkl", "pubchemlite_fp.npy",
          "pubchemlite_meta.pkl", "pool_meta.parquet", "pc_smiles.npy", "test.parquet", "train.parquet",
          "sample_submission.csv", "pool_lsid.npy", "pool_lpmid.npy", "pool_popularity.csv", "final_model.pt",
          "dreams_np_mass.npy", "ext_meta.parquet", "ext_mz.npy", "ext_in.npy", "ext_mass.parquet", "spectra.parquet",
          "candidates_lotus.parquet", "engine.py", "MANIFEST.json", "ranker_0.pkl", "ice_runner.py", "gl_runner.py",
          "fpnet_full1.pt", "rank_train.npz", "sim_rank_rows_nofp.npz"]

CHECK = r'''
import os, sys, json, glob, hashlib, time
T0 = time.time()
OUT = {'slug': SLUG, 'status': 'started'}
ENG_DIR = '/kaggle/working/eng'; os.makedirs(ENG_DIR, exist_ok=True)
IS_RERUN = False
# ---- 診断ノートの凍結セル 32 の表の決め方（そのまま） ----
exec(compile(LOADER, 'frozen_cell_32_loader', 'exec'))
OUT['own'] = _E2_OWN
fails = []
for k, rel, sha, size in (('sim', SIM_REL, SIM_SHA, SIM_SIZE), ('rt', RT_REL, RT_SHA, RT_SIZE)):
    p = _E2_OWN[k]
    got = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    OUT[k] = dict(path=p, sha256=got, size=os.path.getsize(p), sha_ok=got == sha, size_ok=os.path.getsize(p) == size)
    if got != sha or os.path.getsize(p) != size:
        fails.append(('table', k, p, got, os.path.getsize(p)))
# ---- runner 側の照合（書かれた eng_runner.py の verify_own_tables） ----
rg = {'__name__': 'eng_runner_check'}
exec(compile(open(os.path.join(ENG_DIR, 'eng_runner.py')).read(), 'eng_runner.py', 'exec'), rg)
OUT['runner_verify'] = rg['verify_own_tables'](_E2_OWN)
# ---- 名前の重なり ----
by_name = {}
for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    for f in files:
        by_name.setdefault(f, []).append(os.path.join(root, f))
def in_b(p):
    return KSRC_NAME in p.split('/')[:-1]
b_root = sorted({p.split('/' + KSRC_NAME + '/')[0] + '/' + KSRC_NAME for ps in by_name.values() for p in ps if in_b(p)})
OUT['b_run1_roots'] = b_root
if len(b_root) != 1:
    fails.append(('b_run1 root not unique', b_root))
b_names = sorted({os.path.basename(p) for ps in by_name.values() for p in ps if in_b(p)})
OUT['b_run1_files'] = sum(1 for ps in by_name.values() for p in ps if in_b(p))
overlap = {}
for n in b_names:
    others = [p for p in by_name[n] if not in_b(p)]
    if others:
        first = sorted(by_name[n], key=len)[0]
        overlap[n] = dict(others=len(others), first_is_b_run1=in_b(first))
        if in_b(first):
            fails.append(('find() first hit moves into b_run1', n, first))
OUT['name_overlap'] = overlap
lk = {}
for n in LOOKUP:
    hits = by_name.get(n, [])
    lk[n] = dict(total=len(hits), in_b_run1=sum(in_b(p) for p in hits))
    if any(in_b(p) for p in hits):
        fails.append(('lookup name inside b_run1', n))
OUT['lookup'] = lk
fp_pt = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
OUT['engine_fp_models'] = fp_pt
if any(in_b(p) for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True)):
    fails.append(('fp_*.pt inside b_run1',))
OUT['input_top'] = sorted(os.listdir('/kaggle/input'))
OUT['fails'] = fails
OUT['status'] = 'ok' if not fails else 'fail'
OUT['secs'] = round(time.time() - T0, 1)
json.dump(OUT, open('/kaggle/working/pathcheck.json', 'w'), indent=1, default=str)
print('PATHCHECK ' + json.dumps(dict(status=OUT['status'], sim=OUT['sim'], rt=OUT['rt'], fails=fails,
                                     b_run1_roots=b_root, name_overlap=overlap), default=str), flush=True)
assert not fails, fails
'''


def loader_source() -> str:
    """診断ノートの凍結セル 32 に入る表の決め方のコード（生成した診断ノートから取り出す）。"""
    nb, _, _ = B.build()
    cell32 = "".join(json.loads(B.split_frozen("".join(nb["cells"][12]["source"]))[2])["cells"][B.CHANGED_FROZEN]["source"])
    a = cell32.index(B.PATCH_BEGIN)
    b = cell32.index(B.PATCH_END) + len(B.PATCH_END)
    return cell32[a:b]


def build() -> tuple[dict, dict]:
    head = "\n".join([
        f"SLUG = {SLUG!r}",
        f"KSRC_NAME = {B.KSRC_NAME!r}",
        f"SIM_REL = {B.SIM_REL!r}", f"SIM_SHA = {B.SIM_SHA!r}", f"SIM_SIZE = {B.SIM_SIZE!r}",
        f"RT_REL = {B.RT_REL!r}", f"RT_SHA = {B.RT_SHA!r}", f"RT_SIZE = {B.RT_SIZE!r}",
        f"LOOKUP = {LOOKUP!r}",
        f"LOADER = {loader_source()!r}",
    ])
    src = head + "\n" + CHECK
    compile(src, "pathcheck", "exec")
    nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": "# #189 パスの確認（CPU、GPU 0、提出しない）\n"
         "診断ノート（casmi26-lb487-btables）の凍結セル 32 の表の決め方をそのまま実行し、2 本の npz の sha256 と名前の重なりを確かめる。"},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src},
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}
    _, dmeta, _ = B.build()
    meta = dict(id=f"taichiiiii/{SLUG}", title=SLUG.replace("-", " "), code_file="notebook.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                dataset_sources=dmeta["dataset_sources"], kernel_sources=dmeta["kernel_sources"],
                competition_sources=dmeta["competition_sources"], model_sources=[],
                docker_image=dmeta["docker_image"])
    return nb, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "kernels" / "btables_pathcheck"))
    args = ap.parse_args()
    nb, meta = build()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
