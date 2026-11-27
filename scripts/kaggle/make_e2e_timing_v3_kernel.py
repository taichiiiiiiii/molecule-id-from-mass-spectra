#!/usr/bin/env python3
"""
Issue #294: 版 A の Kaggle 時間測定 版 3（GLACIER・N1 の段、CPU、提出しない）のノートブックを作る。

make_e2e_timing_kernel.py（版 2、#274/#278）と同じ形: コードは 1 つの base85(xz(zip, 無圧縮)) の塊にして埋め込み、
セットアップのセルは版 2 と同じもの（展開・入力のリンク・rdkit）。実行のセルは scripts/kaggle/e2e_timing_v3.py all。
生成のときにセルと塊の中の .py をすべて compile() し、送信サイズ < 1,000,000 B を assert する。

    python scripts/kaggle/make_e2e_timing_v3_kernel.py          # kernels/e2e_timing_v3/ にノートと metadata
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


def _load_mk():
    spec = importlib.util.spec_from_file_location("make_e2e_timing_kernel", Path(__file__).with_name("make_e2e_timing_kernel.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("make_e2e_timing_kernel", m)
    spec.loader.exec_module(m)
    return m


MK = _load_mk()
KERNEL_ID = "taichiiiii/casmi26-e2e-timing-v3"
GLACIER_DS = "ahmedberatozer/casmi26-glacier"     # gl_runner.py・ckpt/glacier.pt（#242・#290 と同じ sha256）
ICEBERG_DS = "ahmedberatozer/casmi26-iceberg"     # GLACIER の依存の wheel（*.whl.ice、#242 の T4 のカーネルと同じ）
SOURCES = MK.SOURCES + [GLACIER_DS, ICEBERG_DS]
CODE_FILES = MK.CODE_FILES + [
    "scripts/kaggle/e2e_timing_v3.py",
    "scripts/features/neighbour_ref.py",
    "scripts/bench/glacier_l60.py", "scripts/bench/glacier_counts.py",
    "docs/bench/gl_dev_jobs_head4000.csv.gz",      # dev の glacier_l60/jobs.parquet の先頭 4,000 本（#294）
]
N_MOL = 40
GL_JOBS = 4000

RUN = r'''
# #294: 版 3。GLACIER と N1 の段だけを測る（提出はしない。base の submission.csv は形式の点検用）
OUT = Path("/kaggle/working")
CODE_COMMIT = "__COMMIT__"
print("TAG " + json.dumps(dict(stage="code", commit=CODE_COMMIT)), flush=True)
cmd = [sys.executable, "-u", "scripts/kaggle/e2e_timing_v3.py", "all", "--test", str(COMP / "test.parquet"),
       "--sample", str(COMP / "sample_submission.csv"), "--ranker", str(INP / "ranker_submit.txt"),
       "--features", str(INP / "features.json"), "--out-dir", str(OUT),
       "--n-mol", "__NMOL__", "--gl-jobs", "__GLJOBS__", "--gl-site", "/kaggle/working/gl_site"]
with open(OUT / "e2e_v3.log", "w") as log:
    p = subprocess.Popen(cmd, cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, CASMI_CODE_COMMIT=CODE_COMMIT))
    for line in p.stdout:
        print(line, end="", flush=True)
        log.write(line)
    rc = p.wait()
print("rc", rc, f"elapsed {time.time() - T0:.0f}s")
if (OUT / "timing_v3.json").is_file():
    print(open(OUT / "timing_v3.json").read())
assert rc == 0, f"e2e_timing_v3.py が rc {rc} で止まった（timing_v3.json の errors と各段のログを見る）"
'''


def code_commit(root: Path = ROOT) -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        tracked = [f for f in CODE_FILES if not f.startswith("data/")]
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *tracked], cwd=root, capture_output=True,
                               text=True, check=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_notebook(root: Path = ROOT, commit: str | None = None, n_mol: int = N_MOL, gl_jobs: int = GL_JOBS):
    commit = commit if commit is not None else code_commit(root)
    assert '"' not in commit and "\\" not in commit
    assert int(n_mol) >= 1 and int(gl_jobs) >= 1
    blob, manifest = MK.pack(CODE_FILES, root)
    setup = MK.SETUP.replace("__MANIFEST__", json.dumps(manifest, indent=0)).replace("__BLOB__", blob)
    run = RUN.replace("__COMMIT__", commit).replace("__NMOL__", str(int(n_mol))).replace("__GLJOBS__", str(int(gl_jobs)))
    md = ("# casmi26 e2e timing v3 (#294)\n版 A で増える 2 段（GLACIER の予測・N1 の近傍の参照の照合）を Kaggle の CPU で測る。"
          "提出しない。見える test の先頭 40 分子の候補（版 2 と同じ経路）で N1、dev の固定の 4,000 本で GLACIER。")
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
                    n_mol=int(n_mol), gl_jobs=int(gl_jobs))


def run_params(nb: dict) -> dict:
    import re
    src = "".join(nb["cells"][-1]["source"])
    return dict(commit=re.search(r'CODE_COMMIT = "([^"]*)"', src).group(1),
                n_mol=int(re.search(r'"--n-mol", "([^"]*)"', src).group(1)),
                gl_jobs=int(re.search(r'"--gl-jobs", "([^"]*)"', src).group(1)))


def kernel_metadata() -> dict:
    return dict(id=KERNEL_ID, title="casmi26 e2e timing v3", code_file="notebook.ipynb", language="python",
                kernel_type="notebook", is_private=True, enable_gpu=False, enable_tpu=False, enable_internet=False,
                keywords=[], dataset_sources=SOURCES, kernel_sources=[], competition_sources=[MK.COMPETITION],
                model_sources=[], docker_image=MK.DOCKER_IMAGE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "kernels/e2e_timing_v3")
    ap.add_argument("--n-mol", type=int, default=N_MOL)
    ap.add_argument("--gl-jobs", type=int, default=GL_JOBS)
    a = ap.parse_args(argv)
    nb, info = build_notebook(n_mol=a.n_mol, gl_jobs=a.gl_jobs)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "notebook.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (a.out_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata(), indent=2))
    print(json.dumps(dict(kernel=str(a.out_dir), **{k: v for k, v in info.items() if k != "manifest"}), indent=1,
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
