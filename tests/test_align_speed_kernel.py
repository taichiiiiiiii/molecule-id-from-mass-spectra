"""Issue #99: 整列チャネルの速度の試走（scripts/train/align_speed_trial.py, make_align_speed_kernel.py）。

torch は CI に無いので、純粋な関数・生成物の形・md5 の照合・stub の import・止まり方だけを確かめる。
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))

import align_speed_trial as A  # noqa: E402
import make_align_speed_kernel as M  # noqa: E402


# ---- 生成物

def test_notebook_is_deterministic_and_embeds_module():
    nb1, nb2 = M.build_notebook(), M.build_notebook()
    assert nb1 == nb2
    code = [c for c in nb1["cells"] if c["cell_type"] == "code"]
    assert len(code) == 2
    scope: dict = {}
    exec(compile("".join(code[0]["source"]), "cell0", "exec"), scope)
    assert scope["ALIGN_SRC"] == (ROOT / "scripts/train/align_speed_trial.py").read_text()
    compile(scope["ALIGN_SRC"], "align_speed_trial.py", "exec")
    compile("".join(code[1]["source"]), "cell1", "exec")


def test_committed_kernel_matches_generator():
    kdir = ROOT / "kernels/casmi26-align-speed-trial"
    nb = json.loads((kdir / "notebook.ipynb").read_text())
    assert nb == M.build_notebook()
    assert json.loads((kdir / "kernel-metadata.json").read_text()) == M.metadata()


def test_metadata_shape():
    m = M.metadata()
    assert m["id"] == "taichiiiii/casmi26-align-speed-trial"
    assert m["is_private"] is True and m["enable_gpu"] is True and m["enable_internet"] is True
    assert m["machine_shape"] == "NvidiaTeslaT4"
    assert m["competition_sources"] == ["enveda-CASMI26-molecule-id-mass-spectra"]
    assert m["dataset_sources"] == ["dmitriigluzdov/casmi26-fold-safe-fpnet"]  # 新しい dataset を上げない
    assert "id_no" not in m


def test_constants_are_the_preregistered_ones():
    assert A.CKPT_MD5 == "4deee91dd4486c2b6cbcde1f200b8f50" and len(A.CKPT_MD5) == 32
    assert A.CKPT_SIZE == 1_392_710_212
    assert A.DREAMS_COMMIT == "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
    assert A.N_SPECTRA == 10_000 and A.EMB_BATCH == 32 and A.PEAKS == (60, 100)
    assert (A.MLP_BATCH, A.MLP_K_NEG, A.MLP_STEPS_PROD) == (128, 128, 30_000)
    assert (A.PROD_TRAIN_SPECTRA, A.PROD_TRAIN_STRUCTS, A.PROD_QUERY_SPECTRA) == (1_982_405, 220_443, 4_756)
    assert A.LIMIT_HOURS == 6.0 and A.BUDGET_S == 1800 and A.STAGE_CUTOFF_S < A.BUDGET_S


# ---- md5

def test_md5_stream_accepts_exact_value():
    data = [b"abc", b"def"]
    want = hashlib.md5(b"abcdef").hexdigest()
    info = A.md5_stream(iter(data), want, 6)
    assert info["md5"] == want and info["size"] == 6


def test_md5_stream_stops_on_wrong_value_or_size():
    want = hashlib.md5(b"abcdef").hexdigest()
    with pytest.raises(A.Md5Mismatch):
        A.md5_stream(iter([b"abcdeX"]), want, 6)
    with pytest.raises(A.Md5Mismatch):
        A.md5_stream(iter([b"abcdef"]), want, 7)
    with pytest.raises(ValueError):
        A.md5_stream(iter([b"abcdef"]), want[:8], 6)   # 途中までの値では照合しない


def test_verify_md5_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"0123456789" * 1000)
    good = hashlib.md5(p.read_bytes()).hexdigest()
    assert A.verify_md5_file(p, good, 10_000)["md5"] == good
    with pytest.raises(A.Md5Mismatch):
        A.verify_md5_file(p, "0" * 32, 10_000)


# ---- 外挿

def test_extrapolate_formula_and_decision():
    r = A.extrapolate({60: 400.0, 100: 200.0}, mlp_steps_per_s=10.0, read_s=1.0, rows_read=8192, fixed_s=300.0)
    t_read = 1.0 * 2_539_608 / 8192
    t_embed = (1_982_405 + 4_756) / 200.0
    t_mlp = 30_000 / 10.0
    assert r["worst_peaks"] == 100
    assert r["T_total_s"] == pytest.approx(t_read + t_embed + t_mlp + 300.0)
    assert r["decision"] == "candidate_for_next_plan"  # ~3.7 h
    slow = A.extrapolate({60: 50.0, 100: 40.0}, 10.0, 1.0, 8192, 300.0)
    assert slow["T_total_hours"] > 6 and slow["decision"] == "stop_production"


def test_extrapolate_boundary_is_inclusive_at_6h():
    # T_total ちょうど 6 時間は候補（「6 時間以下 → 候補」）
    sps = (1_982_405 + 4_756) / (6 * 3600)
    r = A.extrapolate({60: sps, 100: sps}, mlp_steps_per_s=1e18, read_s=0.0, rows_read=1, fixed_s=0.0)
    assert r["T_total_hours"] == pytest.approx(6.0)
    assert r["decision"] == "candidate_for_next_plan"


# ---- stub の import

def test_stub_finder_makes_importable_placeholders():
    name = "zz_fake_pkg_for_align_test"
    f = A.StubFinder([name])
    sys.meta_path.insert(0, f)
    try:
        mod = importlib.import_module(f"{name}.sub.deep")
        from zz_fake_pkg_for_align_test import njit, Spectrum  # noqa: F401

        @njit
        def add(a, b):
            return a + b

        @njit(cache=True)
        def mul(a, b):
            return a * b

        assert add(2, 3) == 5 and mul(2, 3) == 6

        class Child(Spectrum):  # 基底クラスとして使える
            pass

        assert isinstance(mod, types.ModuleType)
        assert f"{name}.sub.deep" in f.made
    finally:
        sys.meta_path.remove(f)
        for k in [k for k in sys.modules if k.split(".")[0] == name]:
            del sys.modules[k]


def test_install_stubs_skips_already_imported():
    f = A.install_stubs(("json", "zz_never_imported_pkg"))
    try:
        assert "json" not in f.tops and "zz_never_imported_pkg" in f.tops
    finally:
        sys.meta_path.remove(f)


def test_stub_list_covers_no_core_packages():
    for core in ("torch", "numpy", "pandas", "pyarrow", "dreams"):
        assert core not in A.STUB_TOP


# ---- 止まり方（torch の代わりを入れて run を回す）

def _fake_torch(monkeypatch):
    t = types.ModuleType("torch")
    t.__version__ = "fake"
    t.version = types.SimpleNamespace(cuda=None)
    t.cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", t)


def test_run_stops_on_md5_mismatch_without_numbers(tmp_path, monkeypatch):
    _fake_torch(monkeypatch)

    def bad(dest, *a, **k):
        raise A.Md5Mismatch(json.dumps({"expected_md5": A.CKPT_MD5, "md5": "0" * 32}))
    monkeypatch.setattr(A, "download_ckpt", bad)
    called = []
    monkeypatch.setattr(A, "fetch_dreams_code", lambda *a, **k: called.append(1))
    rep = A.run(out_dir=tmp_path / "out", work_dir=tmp_path / "work", input_root=None)
    assert rep["status"] == "md5_mismatch" and rep["md5_ok"] is False
    assert rep["stage"] == "download_ckpt" and not called
    assert "extrapolation" not in rep and "summary" not in rep
    on_disk = json.loads((tmp_path / "out/align_speed_report.json").read_text())
    assert on_disk["status"] == "md5_mismatch"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["align_speed_report.json"]


def test_run_records_stage_and_error(tmp_path, monkeypatch):
    _fake_torch(monkeypatch)
    monkeypatch.setattr(A, "download_ckpt", lambda dest, *a, **k: {"seconds": 1.0})
    monkeypatch.setattr(A, "fetch_dreams_code", lambda *a, **k: (_ for _ in ()).throw(OSError("github down")))
    rep = A.run(out_dir=tmp_path / "out", work_dir=tmp_path / "work", input_root=None)
    assert rep["status"] == "error" and rep["stage"] == "dreams_code"
    assert "github down" in rep["error"] and "elapsed_s" in rep and "extrapolation" not in rep


def test_run_does_not_start_stage_after_cutoff(tmp_path, monkeypatch):
    _fake_torch(monkeypatch)
    import time as _t
    rep = A.run(out_dir=tmp_path / "out", work_dir=tmp_path / "work", input_root=None,
                t0=_t.time() - A.STAGE_CUTOFF_S - 1)
    assert rep["status"] == "timeout" and "not starting stage env" in rep["error"]
