#!/usr/bin/env python3
"""
Issue #274: Kaggle 上の端から端の時間測定のノートブック（CPU、提出しない）と、その小さな入力データセットを作る。

- コード（下の CODE_FILES）は 1 つの base85(xz(zip, 無圧縮)) の塊にしてノートに埋め込む。展開すると
  /kaggle/working/repo にリポジトリと同じ配置ができ、scripts/kaggle/e2e_timing.py を実行する
- LambdaRank のモデル・特徴の一覧・NP らしさの fold 0 のモデルは private dataset INPUTS_DS（≤ 20 MB）に置く
- 生成のときに、ノートのコードのセルと塊の中の .py をすべて compile() し、送信サイズ（kaggle CLI と同じく
  出力を空にし source をつないだ json.dumps）が 1,000,000 B 未満であることを assert する

    python scripts/kaggle/make_e2e_timing_kernel.py                       # kernels/e2e_timing/ にノートと metadata
    python scripts/kaggle/make_e2e_timing_kernel.py --workers auto --check 40   # #278: PubChem 層の並列と一致の確かめ
    python scripts/kaggle/make_e2e_timing_kernel.py --inputs-dir <dir> \\
        --ranker-dir outputs/local_eval/ranker/submit_e2e_274            # 入力データセットのディレクトリも作る
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import lzma
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_ID = "taichiiiii/casmi26-e2e-timing-274"
INPUTS_DS = "taichiiiii/casmi26-e2e-timing-inputs"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
SOURCES = ["dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets", "dmitriigluzdov/casmi26-pubchem-popularity-prior",
           "ahmedberatozer/casmi26-pubchem-tier", "dmitriigluzdov/casmi26-fold-safe-fpnet", "metric/rdkit-2026-3-3-wheel",
           INPUTS_DS]
# kernels/casmi26_ranker（#34、Kaggle で完走した CPU ノート）と同じ固定のイメージ
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
CODE_FILES = [
    "scripts/kaggle/e2e_timing.py",
    "scripts/cv/local_fold0_logits.py",
    "scripts/local_eval/run.py", "scripts/local_eval/pubchem.py", "scripts/local_eval/library.py",
    "scripts/local_eval/instrument.py",   # #287: library の H2 の関数が読む（既定の動きでは読まない）
    "scripts/local_eval/selpred.py", "scripts/local_eval/elem_rule.py",
    "scripts/features/dump_features.py", "scripts/features/mass_feasibility.py", "scripts/features/relativize.py",
    "scripts/features/np_likeness.py", "scripts/features/contrastive_coverage.py",
    "scripts/features/add_contrastive_coverage.py",
    "scripts/ranker/lambdarank_oof.py", "scripts/ranker/evaluate.py",
    "scripts/validate/mrr25.py",
    "data/ext/frag/frag_rescore.py",   # MetFrag-lite（学習なし、出どころは凍結セル 51 のコメント）。Kaggle 上の置き場所が無いので埋め込む
]
INPUT_FILES = {"ranker_submit.txt": "ranker", "features.json": "ranker", "np_model_fold0.txt": "np"}
NP_MODEL = "outputs/local_eval/np_likeness/model_fold0.txt"
MAX_PUSH_BYTES = 1_000_000
ZIP_DATE = (2026, 11, 27, 0, 0, 0)

SETUP = r'''
# #274: 埋め込みのコードを展開し、入力を /kaggle/working/repo のリポジトリと同じ配置にリンクする
import base64, glob, hashlib, io, json, lzma, os, subprocess, sys, time, zipfile
from pathlib import Path
T0 = time.time()
BLOB = "__BLOB__"
MANIFEST = __MANIFEST__
R = Path("/kaggle/working/repo")
with zipfile.ZipFile(io.BytesIO(lzma.decompress(base64.b85decode(BLOB)))) as z:
    names = sorted(z.namelist())
    assert names == sorted(MANIFEST), "塊の中のファイルが一覧と違う"
    for n in names:
        b = z.read(n)
        assert hashlib.sha256(b).hexdigest() == MANIFEST[n], n
        (R / n).parent.mkdir(parents=True, exist_ok=True)
        (R / n).write_bytes(b)
print("code files", len(names))
def find(pattern):
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    if len(hits) > 1:
        print("WARNING: 複数見つかった", pattern, hits[:5])
    return Path(hits[0])
def find_comp():
    dirs = sorted({Path(p).parent for p in glob.glob("/kaggle/input/**/train.parquet", recursive=True)
                   if (Path(p).parent / "test.parquet").is_file() and (Path(p).parent / "sample_submission.csv").is_file()},
                  key=lambda d: ("competitions" not in str(d) and "enveda" not in str(d).lower(), str(d)))
    if not dirs:
        raise FileNotFoundError("competition dir")
    return dirs[0]
COMP = find_comp()
INP = find("ranker_submit.txt").parent
links = {"data/train.parquet": COMP / "train.parquet",
         "data/ext/v4g": find("casmi26-v2-pool/pool_meta.parquet").parents[1],
         "data/ext/pop/pool_popularity.csv": find("pool_popularity.csv"),
         "data/ext/pubchem_tier": find("pc_smiles.npy").parent,
         "data/ext/foldsafe/fpnet_fold0.pt": find("fpnet_fold0.pt"),
         "outputs/local_eval/np_likeness/model_fold0.txt": INP / "np_model_fold0.txt"}
for dst, src in links.items():
    d = R / dst
    d.parent.mkdir(parents=True, exist_ok=True)
    if not d.exists():
        os.symlink(src, d)
    print(dst, "->", src)
for f, h in json.loads((INP / "MANIFEST.json").read_text())["files"].items():
    got = hashlib.sha256((INP / f).read_bytes()).hexdigest()
    assert got == h, f"入力 {f} の sha256 が違う"
os.environ["NUMBA_CACHE_DIR"] = "/kaggle/working/numba_cache"
try:
    import rdkit  # noqa
except ImportError:
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    whl = sorted(glob.glob(f"/kaggle/input/**/rdkit*-{tag}-*.whl", recursive=True))
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", whl[0]], check=True)
import lightgbm, rdkit, torch
print("lightgbm", lightgbm.__version__, "rdkit", rdkit.__version__, "torch", torch.__version__, "cpu", os.cpu_count())
print("TAG " + json.dumps(dict(stage="setup", secs=round(time.time() - T0, 2))), flush=True)
'''

RUN = r'''
# #274: 段ごとの秒を測る（提出はしない。submission.csv は形式の点検用）。#278: PubChem 層の段の worker の数と一致の確かめ
OUT = Path("/kaggle/working")
CODE_COMMIT = "__COMMIT__"
print("TAG " + json.dumps(dict(stage="code", commit=CODE_COMMIT)), flush=True)
cmd = [sys.executable, "-u", "scripts/kaggle/e2e_timing.py", "--test", str(COMP / "test.parquet"),
       "--sample", str(COMP / "sample_submission.csv"), "--ranker", str(INP / "ranker_submit.txt"),
       "--features", str(INP / "features.json"), "--out-dir", str(OUT),
       "--pubchem-workers", "__WORKERS__", "--pubchem-check", "__CHECK__"]
with open(OUT / "e2e.log", "w") as log:
    p = subprocess.Popen(cmd, cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, CASMI_CODE_COMMIT=CODE_COMMIT))
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = p.wait()
print("rc", rc, f"elapsed {time.time() - T0:.0f}s")
if (OUT / "timing.json").is_file():
    print(open(OUT / "timing.json").read())
assert rc == 0, f"e2e_timing.py が rc {rc} で止まった（ログ e2e.log の TAG で、どの段まで進んだかを見る）"
'''


def cell(src: str, kind: str = "code") -> dict:
    c = dict(cell_type=kind, metadata={}, source=src.strip("\n").splitlines(True))
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def pack(files: list[str], root: Path = ROOT) -> tuple[str, dict]:
    """base85(xz(zip 無圧縮))。名前順・日付固定なので同じ入力から同じ塊になる。"""
    buf = io.BytesIO()
    manifest = {}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for f in sorted(files):
            b = (root / f).read_bytes()
            manifest[f] = hashlib.sha256(b).hexdigest()
            z.writestr(zipfile.ZipInfo(f, date_time=ZIP_DATE), b)
    blob = base64.b85encode(lzma.compress(buf.getvalue(), preset=9)).decode("ascii")
    return blob, manifest


def unpack(blob: str) -> dict:
    with zipfile.ZipFile(io.BytesIO(lzma.decompress(base64.b85decode(blob)))) as z:
        return {n: z.read(n) for n in z.namelist()}


def push_bytes(nb: dict) -> int:
    """kaggle CLI が送る大きさ（出力を空に、source をつないだ json.dumps）。"""
    nb2 = json.loads(json.dumps(nb))
    for c in nb2["cells"]:
        if isinstance(c.get("source"), list):
            c["source"] = "".join(c["source"])
        if c.get("cell_type") == "code":
            c["outputs"] = []
    return len(json.dumps(nb2).encode("utf-8"))


def code_commit(root: Path = ROOT) -> str:
    """埋め込むコードの commit（未コミットの変更があれば +dirty）。git が無ければ unknown。"""
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        tracked = [f for f in CODE_FILES if not f.startswith("data/")]   # data/ext はリンクの先で git の外
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *tracked], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_notebook(root: Path = ROOT, workers: str = "1", check: int = 0, commit: str | None = None) -> tuple[dict, dict]:
    """#278: workers（整数か auto）と check（直列と並列の一致を確かめる分子の数）を e2e_timing.py に渡す。"""
    assert workers == "auto" or int(workers) >= 1
    assert int(check) >= 0
    commit = commit if commit is not None else code_commit(root)
    assert '"' not in commit and "\\" not in commit
    blob, manifest = pack(CODE_FILES, root)
    setup = SETUP.replace("__MANIFEST__", json.dumps(manifest, indent=0)).replace("__BLOB__", blob)
    run = RUN.replace("__COMMIT__", commit).replace("__WORKERS__", str(workers)).replace("__CHECK__", str(int(check)))
    md = ("# casmi26 e2e timing (#274)\n自前の最小版（プール＋PubChem 層 → exp003 の 64 列 → LambdaRank）の段ごとの秒を Kaggle の CPU で測る。"
          "提出しない。LambdaRank は dev の全行で学習した時間測定用の 1 本で、評価には使わない。")
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
              cells=[cell(md, "markdown"), cell(setup), cell(run)])
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    for n, b in unpack(blob).items():
        assert hashlib.sha256(b).hexdigest() == manifest[n]
        if n.endswith(".py"):
            compile(b.decode("utf-8"), n, "exec")
    size = push_bytes(nb)
    assert size < MAX_PUSH_BYTES, f"送信サイズ {size} B が 1 MB 以上"
    return nb, dict(push_bytes=size, blob_chars=len(blob), n_files=len(manifest), manifest=manifest, commit=commit,
                    workers=str(workers), check=int(check))


def run_params(nb: dict) -> dict:
    """生成したノートの RUN のセルから commit・workers・check を読み戻す（コミットしたノートの点検用）。"""
    import re
    src = "".join(nb["cells"][-1]["source"])
    commit = re.search(r'CODE_COMMIT = "([^"]*)"', src).group(1)
    workers = re.search(r'"--pubchem-workers", "([^"]*)"', src).group(1)
    check = int(re.search(r'"--pubchem-check", "([^"]*)"', src).group(1))
    return dict(commit=commit, workers=workers, check=check)


def kernel_metadata() -> dict:
    return dict(id=KERNEL_ID, title="casmi26 e2e timing 274", code_file="notebook.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                keywords=[], dataset_sources=SOURCES, kernel_sources=[], competition_sources=[COMPETITION],
                model_sources=[], docker_image=DOCKER_IMAGE)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_inputs(inputs_dir: Path, ranker_dir: Path, root: Path = ROOT) -> dict:
    """入力データセットのディレクトリ（ranker_submit.txt・features.json・np_model_fold0.txt・MANIFEST.json）。"""
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True)
    shutil.copy(ranker_dir / "ranker_submit.txt", inputs_dir / "ranker_submit.txt")
    shutil.copy(ranker_dir / "features.json", inputs_dir / "features.json")
    shutil.copy(root / NP_MODEL, inputs_dir / "np_model_fold0.txt")
    files = {f: sha256_file(inputs_dir / f) for f in INPUT_FILES}
    sizes = {f: (inputs_dir / f).stat().st_size for f in INPUT_FILES}
    meta = json.loads((ranker_dir / "meta.json").read_text())
    assert files["ranker_submit.txt"] == meta["model_sha256"] and files["features.json"] == meta["features_json_sha256"]
    man = dict(issue=274, files=files, sizes=sizes, ranker_meta=meta,
               np_model=dict(source=NP_MODEL, note="#241 の fold 0 のモデル（holdout と同じ）"))
    (inputs_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    (inputs_dir / "dataset-metadata.json").write_text(json.dumps(
        dict(title="casmi26-e2e-timing-inputs", id=INPUTS_DS, licenses=[dict(name="MIT")]), indent=2))
    total = sum(p.stat().st_size for p in inputs_dir.iterdir())
    assert total <= 20 * 1024 * 1024, f"入力データセットが 20 MB を超える: {total}"
    return dict(dir=str(inputs_dir), files=files, sizes=sizes, total_bytes=total)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "kernels/e2e_timing")
    ap.add_argument("--inputs-dir", type=Path, default=None)
    ap.add_argument("--ranker-dir", type=Path, default=ROOT / "outputs/local_eval/ranker/submit_e2e_274")
    ap.add_argument("--workers", default="1", help="#278: PubChem 層の段の worker の数（整数か auto）")
    ap.add_argument("--check", type=int, default=0, help="#278: 直列と並列の一致を確かめる分子の数")
    a = ap.parse_args(argv)
    nb, info = build_notebook(workers=a.workers, check=a.check)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (a.out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    rep = dict(kernel=str(a.out_dir), **{k: v for k, v in info.items() if k != "manifest"})
    if a.inputs_dir:
        rep["inputs"] = build_inputs(a.inputs_dir, a.ranker_dir)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
