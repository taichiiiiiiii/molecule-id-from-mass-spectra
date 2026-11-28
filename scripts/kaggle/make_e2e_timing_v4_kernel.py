#!/usr/bin/env python3
"""
Issue #302: 版 A の時間測定 版 4（GLACIER の段だけ、Kaggle GPU T4、提出しない）のノートブックを作る。

make_e2e_timing_v3_kernel.py（版 3、#294）と同じ形: コードは 1 つの base85(xz(zip, 無圧縮)) の塊にして埋め込み、
生成のときにセルと塊の中の .py をすべて compile() し、送信サイズ < 1,000,000 B を assert する。
入力は GLACIER の重み（casmi26-glacier）と依存の wheel（casmi26-iceberg）だけ（コンペのデータ・private dataset は使わない）。
ノートの中の上限: 開始から 3,000 s（50 分）でランナーの deadline、子のプロセスは 3,300 s で打ち切る。

    python scripts/kaggle/make_e2e_timing_v4_kernel.py          # kernels/e2e_timing_v4/ にノートと metadata
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, m)
    spec.loader.exec_module(m)
    return m


MK = _load("make_e2e_timing_kernel")
KERNEL_ID = "taichiiiii/casmi26-e2e-timing-v4"
SOURCES = ["ahmedberatozer/casmi26-glacier", "ahmedberatozer/casmi26-iceberg"]
MACHINE_SHAPE = "NvidiaTeslaT4"
CODE_FILES = [
    "scripts/kaggle/e2e_timing_v4.py",
    "scripts/kaggle/e2e_timing_v3.py",
    "scripts/bench/glacier_l60.py",
    "docs/bench/gl_dev_jobs_head4000.csv.gz",
]
GL_JOBS = 4000
DEADLINE_SECS = 3000            # 50 分
CHILD_TIMEOUT_SECS = 3300       # 55 分（カーネル全体 1 h の内）

SETUP = r'''
# #302: 埋め込みのコードを展開する（入力は GLACIER の重みと wheel だけ）
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
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "cpu", os.cpu_count())
print("TAG " + json.dumps(dict(stage="setup", secs=round(time.time() - T0, 2))), flush=True)
'''

RUN = r'''
# #302: 版 4。GLACIER の予測の段だけを GPU で測る（提出しない）
OUT = Path("/kaggle/working")
CODE_COMMIT = "__COMMIT__"
print("TAG " + json.dumps(dict(stage="code", commit=CODE_COMMIT)), flush=True)
cmd = [sys.executable, "-u", "scripts/kaggle/e2e_timing_v4.py", "--out-dir", str(OUT), "--device", "cuda",
       "--gl-jobs", "__GLJOBS__", "--deadline", str(T0 + __DEADLINE__), "--gl-site", "/kaggle/working/gl_site"]
with open(OUT / "e2e_v4.log", "w") as log:
    p = subprocess.Popen(cmd, cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, CASMI_CODE_COMMIT=CODE_COMMIT))
    import threading
    killer = threading.Timer(max(1.0, __TIMEOUT__ - (time.time() - T0)), p.kill)   # 出力が止まっても打ち切る
    killer.start()
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = p.wait()
    killer.cancel()
if rc == -9:
    print("KILLED: 子のプロセスが上限（開始から __TIMEOUT__ s）を超えた", flush=True)
print("rc", rc, f"elapsed {time.time() - T0:.0f}s")
if (OUT / "timing_v4.json").is_file():
    print(open(OUT / "timing_v4.json").read())
assert rc == 0, f"e2e_timing_v4.py が rc {rc} で止まった（timing_v4.json の error と e2e_v4.log を見る）"
'''


def code_commit(root: Path = ROOT) -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *CODE_FILES], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_notebook(root: Path = ROOT, commit: str | None = None, gl_jobs: int = GL_JOBS):
    commit = commit if commit is not None else code_commit(root)
    assert '"' not in commit and "\\" not in commit
    assert int(gl_jobs) >= 1
    blob, manifest = MK.pack(CODE_FILES, root)
    setup = SETUP.replace("__MANIFEST__", json.dumps(manifest, indent=0)).replace("__BLOB__", blob)
    run = (RUN.replace("__COMMIT__", commit).replace("__GLJOBS__", str(int(gl_jobs)))
           .replace("__DEADLINE__", str(DEADLINE_SECS)).replace("__TIMEOUT__", str(CHILD_TIMEOUT_SECS)))
    md = ("# casmi26 e2e timing v4 (#302)\n版 A の GLACIER の予測の段だけを Kaggle の GPU（T4）で測る。提出しない。"
          "dev の固定の 4,000 本（#294 と同じ）、device cuda（無ければ止める）、batch はランナーの既定。")
    nb = dict(nbformat=4, nbformat_minor=5,
              metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")),
              cells=[MK.cell(md, "markdown"), MK.cell(setup), MK.cell(run)])
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "<cell>", "exec")
    for n, b in MK.unpack(blob).items():
        assert hashlib.sha256(b).hexdigest() == manifest[n]
        if n.endswith(".py"):
            compile(b.decode("utf-8"), n, "exec")
    size = MK.push_bytes(nb)
    assert size < MK.MAX_PUSH_BYTES, f"送信サイズ {size} B が 1 MB 以上"
    return nb, dict(push_bytes=size, blob_chars=len(blob), n_files=len(manifest), manifest=manifest, commit=commit,
                    gl_jobs=int(gl_jobs))


def run_params(nb: dict) -> dict:
    import re
    src = "".join(nb["cells"][-1]["source"])
    return dict(commit=re.search(r'CODE_COMMIT = "([^"]*)"', src).group(1),
                gl_jobs=int(re.search(r'"--gl-jobs", "([^"]*)"', src).group(1)))


def kernel_metadata() -> dict:
    return dict(id=KERNEL_ID, title="casmi26 e2e timing v4", code_file="notebook.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=False,
                keywords=[], dataset_sources=SOURCES, kernel_sources=[], competition_sources=[], model_sources=[],
                machine_shape=MACHINE_SHAPE, docker_image=MK.DOCKER_IMAGE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "kernels/e2e_timing_v4")
    ap.add_argument("--gl-jobs", type=int, default=GL_JOBS)
    a = ap.parse_args(argv)
    nb, info = build_notebook(gl_jobs=a.gl_jobs)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (a.out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    print(json.dumps(dict(kernel=str(a.out_dir), **{k: v for k, v in info.items() if k != "manifest"}), indent=1,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
