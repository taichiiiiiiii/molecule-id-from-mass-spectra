"""#82: scripts/submit/make_lb487_ownfpnet.py — 0.487 写しの Engine 1 の FP バンクだけを作り直した FPNet に替える生成器の検査。"""
from __future__ import annotations

import ast
import base64
import contextlib
import difflib
import glob
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "submit"))
import make_lb487_ownfpnet as M  # noqa: E402

ORIG_NB = ROOT / "kernels" / "lb487" / "casmi26-lb487-copy.ipynb"
OUT_DIR = ROOT / "kernels" / "lb487_ownfpnet"


@pytest.fixture(scope="module")
def built():
    return M.build()


def _src(c):
    return "".join(c["source"])


def _frozen(nb):
    c12 = _src(nb["cells"][12])
    _, _, raw = M.split_frozen(c12)
    return c12, raw, json.loads(raw)


def test_source_notebook_is_the_0487_copy():
    assert hashlib.sha256(ORIG_NB.read_bytes()).hexdigest() == M.SRC_NB_SHA


def test_old_block_occurs_exactly_once_in_frozen():
    _, raw, fr = _frozen(json.loads(ORIG_NB.read_bytes()))
    assert hashlib.sha256(raw).hexdigest() == M.FROZEN_SHA_OLD
    hits = [(k, _src(c).count(M.OLD_BLOCK)) for k, c in enumerate(fr["cells"]) if M.OLD_BLOCK in _src(c)]
    assert hits == [(37, 1)]


def test_frozen_diff_is_only_the_cell37_block(built):
    nb, _, rep = built
    _, _, fr0 = _frozen(json.loads(ORIG_NB.read_bytes()))
    c12, raw, fr = _frozen(nb)
    assert len(fr["cells"]) == len(fr0["cells"]) == 59
    changed = [k for k, (a, b) in enumerate(zip(fr0["cells"], fr["cells"])) if _src(a) != _src(b) or a != {**b, "source": a["source"]}]
    assert changed == [37]
    a, b = _src(fr0["cells"][37]).splitlines(), _src(fr["cells"][37]).splitlines()
    removed = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("-") and not l.startswith("---")]
    added = [l[1:] for l in difflib.unified_diff(a, b, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    assert removed == M.OLD_BLOCK.splitlines()
    assert added == M.NEW_BLOCK.splitlines()
    assert {k: v for k, v in fr.items() if k != "cells"} == {k: v for k, v in fr0.items() if k != "cells"}
    for k, c in enumerate(fr["cells"]):
        if c["cell_type"] == "code":
            compile(_src(c), f"frozen_{k}", "exec")


def test_new_block_has_no_silent_fallback():
    code = "\n".join(l for l in M.NEW_BLOCK.splitlines() if not l.lstrip().startswith("#"))
    for bad in ("fpnet_full1", "fpnet_0", "FALLBACK", "V4MOD"):
        assert bad not in code, bad
    assert "except" not in code
    assert M.NEW_BLOCK.count("print('OWN_FPNET ok'") == 1
    assert M.OWN_SHA in M.NEW_BLOCK and M.OWN_PATH in M.NEW_BLOCK


def test_frozen_sha_and_metadata_match(built):
    nb, _, rep = built
    c12, raw, _ = _frozen(nb)
    new_sha = hashlib.sha256(raw).hexdigest()
    assert new_sha == rep["frozen_sha_new"] != M.FROZEN_SHA_OLD
    assert c12.count(f"FROZEN_SHA256 = '{new_sha}'") == 1 and M.FROZEN_SHA_OLD not in c12
    cm = nb["metadata"]["casmi"]
    assert cm["frozen_source_sha256"] == new_sha
    assert cm["upstream_casmi_metadata"]["frozen_source_sha256"] == new_sha


def test_embedded_literal_and_outer_cells_compile(built):
    nb, _, _ = built
    c12 = _src(nb["cells"][12])
    tree = ast.parse(c12)
    vals = {t.targets[0].id: ast.literal_eval(t.value) for t in tree.body
            if isinstance(t, ast.Assign) and isinstance(t.targets[0], ast.Name) and t.targets[0].id in ("FROZEN_B64", "FROZEN_SHA256")}
    assert hashlib.sha256(base64.b64decode(vals["FROZEN_B64"])).hexdigest() == vals["FROZEN_SHA256"]
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(_src(c), f"outer_{k}", "exec")


def test_outer_diff_only_commit_mode_and_frozen(built):
    nb, _, _ = built
    orig = json.loads(ORIG_NB.read_bytes())
    assert len(orig["cells"]) == len(nb["cells"])
    changed = [k for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])) if a != b]
    assert changed == [5, 12]
    d5 = [l for l in difflib.unified_diff(_src(orig["cells"][5]).splitlines(), _src(nb["cells"][5]).splitlines(), lineterm="", n=0)
          if l[:1] in "+-" and not l.startswith(("---", "+++"))]
    assert len(d5) == 2 and d5[0].startswith("-COMMIT_MODE = 'fast'") and d5[1] == "+" + d5[0][1:].replace("'fast'", "'smoke'", 1)
    d12 = [l for l in difflib.unified_diff(_src(orig["cells"][12]).splitlines(), _src(nb["cells"][12]).splitlines(), lineterm="", n=0)
           if l[:1] in "+-" and not l.startswith(("---", "+++"))]
    assert sorted(l[1:].split(" = ")[0] for l in d12) == ["FROZEN_B64", "FROZEN_B64", "FROZEN_SHA256", "FROZEN_SHA256"]
    m0, m1 = json.loads(json.dumps(orig["metadata"])), json.loads(json.dumps(nb["metadata"]))
    for m in (m0, m1):
        m["casmi"].pop("frozen_source_sha256"); m["casmi"]["upstream_casmi_metadata"].pop("frozen_source_sha256")
    assert m0 == m1
    assert {k: v for k, v in orig.items() if k not in ("cells", "metadata")} == {k: v for k, v in nb.items() if k not in ("cells", "metadata")}


def test_kernel_metadata_diff(built):
    _, meta, _ = built
    old = json.loads((ROOT / "kernels" / "lb487" / "kernel-metadata.json").read_text())
    assert "id_no" in old and "id_no" not in meta
    assert meta["id"] == "taichiiiii/casmi26-lb487-ownfpnet"
    assert meta["title"] == "casmi26 lb487 ownfpnet" and meta["code_file"] == "notebook.ipynb" and meta["is_private"] is True
    assert meta["kernel_sources"] == old["kernel_sources"] + ["taichiiiii/casmi26-fpnet-full-train-s2"]
    changed = {k for k in set(old) | set(meta) if old.get(k) != meta.get(k)}
    assert changed == {"id", "id_no", "title", "code_file", "kernel_sources"}


def test_push_size_under_limit(built):
    nb, _, _ = built
    assert M.push_source_bytes(nb) < 1_000_000


def test_committed_files_match_build(built):
    nb, meta, rep = built
    assert json.loads((OUT_DIR / "notebook.ipynb").read_text()) == nb
    assert json.loads((OUT_DIR / "kernel-metadata.json").read_text()) == meta
    info = json.loads((OUT_DIR / "build_info.json").read_text())
    assert {k: info[k] for k in rep} == rep and info["push_source_bytes"] == M.push_source_bytes(nb)


# ---- 置き換えた塊の動き（パスと sha256 だけ一時ファイルに向け、fpnet は偽物） ----
class _FakeBank:
    def __init__(self, paths, device="cpu", nbits=10226):
        self.paths = list(paths); self.models = [object() for _ in paths]; self.nbits = nbits


def _run_block(tmp_path, explicit, sha, glob_root, nbits=10226):
    code = (M.NEW_BLOCK.replace(M.OWN_PATH, str(explicit)).replace(M.OWN_SHA, sha)
            .replace("'/kaggle/input/**/casmi26-fpnet-full-train-s2/**/fpnet_final.pt'",
                     repr(str(glob_root) + "/**/casmi26-fpnet-full-train-s2/**/fpnet_final.pt")))
    calls = []

    class _fp:
        @staticmethod
        def ModelBank(paths, device):
            calls.append(list(paths)); return _FakeBank(paths, device, nbits)

    ns = dict(os=os, glob=glob, fpnet=_fp, dev="cpu", V4MOD=str(tmp_path / "v4mod"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "block", "exec"), ns)
    return ns, calls, buf.getvalue()


def _weights(p, data=b"weights"):
    p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_block_explicit_route(tmp_path):
    w = tmp_path / "in" / "notebooks" / "casmi26-fpnet-full-train-s2" / "fpnet_final.pt"
    sha = _weights(w)
    ns, calls, out = _run_block(tmp_path, w, sha, tmp_path / "in")
    assert calls == [[str(w)]] and ns["_FULL"] == [str(w)]
    assert f"OWN_FPNET ok {sha}" in out and "route=explicit" in out and f"engine FP bank: {w}" in out


def test_block_glob_route_exactly_once(tmp_path):
    w = tmp_path / "in" / "x" / "casmi26-fpnet-full-train-s2" / "fpnet_final.pt"
    sha = _weights(w)
    ns, calls, out = _run_block(tmp_path, tmp_path / "missing.pt", sha, tmp_path / "in")
    assert calls == [[str(w)]] and "route=glob" in out
    _weights(tmp_path / "in" / "y" / "casmi26-fpnet-full-train-s2" / "fpnet_final.pt")
    with pytest.raises(RuntimeError, match="exactly once"):
        _run_block(tmp_path, tmp_path / "missing.pt", sha, tmp_path / "in")


def test_block_missing_raises_without_fallback(tmp_path):
    (tmp_path / "v4mod").mkdir(); (tmp_path / "v4mod" / "fpnet_0.pt").write_bytes(b"A")
    with pytest.raises(RuntimeError, match="exactly once"):
        _run_block(tmp_path, tmp_path / "missing.pt", "0" * 64, tmp_path / "empty")


def test_block_sha_mismatch_raises(tmp_path):
    w = tmp_path / "in" / "notebooks" / "casmi26-fpnet-full-train-s2" / "fpnet_final.pt"
    _weights(w)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _run_block(tmp_path, w, "0" * 64, tmp_path / "in")


def test_block_wrong_nbits_raises(tmp_path):
    w = tmp_path / "in" / "notebooks" / "casmi26-fpnet-full-train-s2" / "fpnet_final.pt"
    sha = _weights(w)
    with pytest.raises(AssertionError, match="bank shape"):
        _run_block(tmp_path, w, sha, tmp_path / "in", nbits=6930)
