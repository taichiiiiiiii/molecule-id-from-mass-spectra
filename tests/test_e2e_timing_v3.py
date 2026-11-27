"""Issue #294: Kaggle 時間測定 版 3（GLACIER・N1 の段）のノート生成と、外挿・判定の式。"""
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MK3 = _load("make_e2e_timing_v3_kernel", "scripts/kaggle/make_e2e_timing_v3_kernel.py")
V3 = _load("e2e_timing_v3", "scripts/kaggle/e2e_timing_v3.py")
FRAG = "data/ext/frag/frag_rescore.py"


def _root(tmp_path) -> Path:
    r = tmp_path / "repo"
    for f in MK3.CODE_FILES:
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / f
        if src.is_file():
            shutil.copy(src, r / f)
        else:
            assert f == FRAG, f"リポジトリに無いコード: {f}"
            (r / f).write_text("def score_molecule(spectra, smiles, cfg=None, cache=None):\n    return [None] * len(smiles)\n")
    return r


def test_notebook_size_compile_roundtrip(tmp_path):
    r = _root(tmp_path)
    nb, info = MK3.build_notebook(r, commit="abc")
    assert info["push_bytes"] < MK3.MK.MAX_PUSH_BYTES == 1_000_000
    assert sorted(info["manifest"]) == sorted(MK3.CODE_FILES)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 2
    for c in code:
        compile("".join(c["source"]), "<cell>", "exec")
        assert c["outputs"] == [] and c["execution_count"] is None
    setup = "".join(code[0]["source"])
    blob = setup.split('BLOB = "', 1)[1].split('"', 1)[0]
    files = MK3.MK.unpack(blob)
    for f in MK3.CODE_FILES:
        assert files[f] == (r / f).read_bytes()
    assert MK3.run_params(nb) == dict(commit="abc", n_mol=40, gl_jobs=4000)
    run = "".join(code[1]["source"])
    assert '"scripts/kaggle/e2e_timing_v3.py", "all"' in run and "CASMI_CODE_COMMIT=CODE_COMMIT" in run


def test_size_gate_fires(tmp_path, monkeypatch):
    r = _root(tmp_path)
    monkeypatch.setattr(MK3.MK, "MAX_PUSH_BYTES", 1000)
    with pytest.raises(AssertionError, match="送信サイズ"):
        MK3.build_notebook(r, commit="x")


def test_compile_gate_fires(tmp_path):
    r = _root(tmp_path)
    (r / "scripts/features/neighbour_ref.py").write_text("def broken(:\n")
    with pytest.raises(SyntaxError):
        MK3.build_notebook(r, commit="x")


def test_kernel_metadata_cpu_private_glacier_sources():
    m = MK3.kernel_metadata()
    assert m["is_private"] is True and m["enable_gpu"] is False and m["enable_internet"] is False
    assert m["id"] == "taichiiiii/casmi26-e2e-timing-v3" and "id_no" not in m and "machine_shape" not in m
    assert m["dataset_sources"][:len(MK3.MK.SOURCES)] == MK3.MK.SOURCES
    assert "ahmedberatozer/casmi26-glacier" in m["dataset_sources"] and "ahmedberatozer/casmi26-iceberg" in m["dataset_sources"]
    assert m["docker_image"] == MK3.MK.DOCKER_IMAGE


def test_committed_kernel_matches_generator():
    d = ROOT / "kernels/e2e_timing_v3"
    assert json.loads((d / "kernel-metadata.json").read_text()) == MK3.kernel_metadata()
    nb = json.loads((d / "notebook.ipynb").read_text())
    assert MK3.MK.push_bytes(nb) < MK3.MK.MAX_PUSH_BYTES
    if (ROOT / FRAG).is_file():
        assert nb == MK3.build_notebook(**MK3.run_params(nb))[0]


def test_gl_jobs_file_is_fixed_head_4000():
    p = ROOT / V3.GL_JOBS_FILE
    raw = gzip.decompress(p.read_bytes()).decode()
    lines = raw.splitlines()
    assert lines[0] == "row,smiles,adduct,instr,ce" and len(lines) == 4001
    assert [int(x.split(",", 1)[0]) for x in lines[1:]] == list(range(4000))
    j = V3.read_gl_jobs(p, 4000)
    assert len(j) == 4000 and j.ce.notna().all()


def test_dev_jobs_per_mol():
    assert V3.DEV_JOBS == 1_218_100 and round(V3.DEV_JOBS_PER_MOL, 2) == 312.25


def test_verdict_bands():
    assert V3.verdict_of(16_200.0) == "関門内"
    assert V3.verdict_of(16_200.1) == "上限内・関門外"
    assert V3.verdict_of(32_400.0) == "上限内・関門外"
    assert V3.verdict_of(32_400.1) == "上限外"


def test_extrapolate_formula():
    # GLACIER 40 ms/本、test 300 本/分子（< dev 312.25 → dev を使う）、固定 60 s。N1 400 s / 40 分子
    ex = V3.extrapolate(160.0, 4000, 60.0, 300.0, 400.0, 40)
    assert ex["gl_ms_per_job"] == 40.0 and ex["jobs_per_mol_used"] == 312.25
    assert ex["gl_1200"] == round(0.04 * V3.DEV_JOBS_PER_MOL * 1200 + 60.0, 1)
    assert ex["n1_secs_per_mol"] == 10.0 and ex["n1_1200"] == 12_000.0
    assert ex["total"] == round(12_448.5 + 0.04 * V3.DEV_JOBS_PER_MOL * 1200 + 60.0 + 12_000.0, 1)
    assert ex["verdict"] == "上限外" and ex["main_cause"] == "glacier"
    # test の方が多ければ test を使う
    ex2 = V3.extrapolate(4.0, 4000, 0.0, 400.0, 4.0, 40)
    assert ex2["jobs_per_mol_used"] == 400.0 and ex2["verdict"] == "関門内" and ex2["main_cause"] is None
    # 数えられなければ dev だけ
    assert V3.extrapolate(4.0, 4000, 0.0, None, 4.0, 40)["jobs_per_mol_used"] == 312.25


def test_find_glacier_pkg_needs_both_sha(tmp_path, monkeypatch):
    d = tmp_path / "a" / "casmi26-glacier"
    (d / "ckpt").mkdir(parents=True)
    (d / "gl_runner.py").write_text("x")
    (d / "ckpt/glacier.pt").write_bytes(b"w")
    monkeypatch.setattr(V3, "GL_RUNNER_SHA", hashlib.sha256(b"x").hexdigest())
    monkeypatch.setattr(V3, "GL_CKPT_SHA", hashlib.sha256(b"w").hexdigest())
    assert V3.find_glacier_pkg(str(tmp_path / "**/gl_runner.py")) == d.resolve()
    monkeypatch.setattr(V3, "GL_CKPT_SHA", "0" * 64)
    with pytest.raises(FileNotFoundError):
        V3.find_glacier_pkg(str(tmp_path / "**/gl_runner.py"))
