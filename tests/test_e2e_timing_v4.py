"""Issue #302: 版 4（GLACIER の段だけ、Kaggle GPU）のノート生成・外挿の式・make_predict_fn の device。"""
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MK4 = _load("make_e2e_timing_v4_kernel", "scripts/kaggle/make_e2e_timing_v4_kernel.py")
V4 = _load("e2e_timing_v4", "scripts/kaggle/e2e_timing_v4.py")
sys.path.insert(0, str(ROOT / "scripts/bench"))
import glacier_l60 as GL  # noqa: E402


def _root(tmp_path) -> Path:
    r = tmp_path / "repo"
    for f in MK4.CODE_FILES:
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / f, r / f)
    return r


# ---------------------------------------------------------------- ノート
def test_notebook_size_compile_roundtrip(tmp_path):
    r = _root(tmp_path)
    nb, info = MK4.build_notebook(r, commit="abc")
    assert info["push_bytes"] < 1_000_000 and info["n_files"] == len(MK4.CODE_FILES)
    src = "".join(nb["cells"][-1]["source"])
    assert '"--device", "cuda"' in src and "T0 + 3000" in src and "3300" in src
    assert MK4.run_params(nb) == dict(commit="abc", gl_jobs=4000)
    blob = "".join(nb["cells"][1]["source"]).split('BLOB = "', 1)[1].split('"', 1)[0]
    files = MK4.MK.unpack(blob)
    assert sorted(files) == sorted(MK4.CODE_FILES)
    for n, b in files.items():
        assert b == (r / n).read_bytes()


def test_size_gate_fires(tmp_path, monkeypatch):
    r = _root(tmp_path)
    monkeypatch.setattr(MK4.MK, "MAX_PUSH_BYTES", 1000)
    with pytest.raises(AssertionError):
        MK4.build_notebook(r, commit="x")


def test_compile_gate_fires(tmp_path):
    r = _root(tmp_path)
    (r / "scripts/kaggle/e2e_timing_v4.py").write_text("def (:\n")
    with pytest.raises(SyntaxError):
        MK4.build_notebook(r, commit="x")


def test_kernel_metadata_gpu_private():
    m = MK4.kernel_metadata()
    assert m["enable_gpu"] is True and m["is_private"] is True and m["enable_internet"] is False
    assert m["machine_shape"] == "NvidiaTeslaT4" and m["competition_sources"] == []
    assert m["dataset_sources"] == ["ahmedberatozer/casmi26-glacier", "ahmedberatozer/casmi26-iceberg"]


def test_committed_kernel_matches_generator():
    d = ROOT / "kernels/e2e_timing_v4"
    assert json.loads((d / "kernel-metadata.json").read_text()) == MK4.kernel_metadata()
    nb = json.loads((d / "notebook.ipynb").read_text())
    assert MK4.MK.push_bytes(nb) < MK4.MK.MAX_PUSH_BYTES
    assert nb == MK4.build_notebook(**MK4.run_params(nb))[0]


# ---------------------------------------------------------------- 外挿と判定
def test_extrapolate_v4_matches_daily_estimate():
    # 日報 Q2 の推定（#242 の 20.74 ms/本・固定費 39.8 s）: 5,145.4 + 3,261.5 + 4,300.3 = 12,707.2
    ex = V4.extrapolate_v4(20.74, 39.8)
    assert ex["v2_part"] == round(12_448.5 / 3 * 1.24, 1) == 5145.4
    assert ex["glacier"] == round((0.02074 * V4.JOBS_PER_MOL * 400 + 39.8) * 1.24, 1)
    assert ex["n1"] == round(8.67 * 400 * 1.24, 1) == 4300.3
    assert ex["total"] == 12_707.2 and ex["verdict"] == "関門内"
    assert ex["gate"] == 16_200.0 and ex["limit"] == 32_400.0


def test_extrapolate_v4_bands():
    # 版 3 の CPU の 100.418 ms/本・39.8 s → 上限内・関門外
    assert V4.extrapolate_v4(100.418, 39.8)["verdict"] == "上限内・関門外"
    assert V4.extrapolate_v4(250.0, 0.0)["verdict"] == "上限外"


def test_old_b5_reference_matches_v3_formula():
    # 版 3 の CPU の 100.418 ms/本・39.8 s → #294 の 60,519.2 s（test の本数 < dev なので dev の 312.25）
    ref = V4.old_b5_reference(100.418, 39.8)
    assert abs(ref["total"] - 60_519.2) <= 0.2 and ref["verdict"] == "上限外"


def test_require_device_never_falls_back(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError):
        V4.require_device("cuda")
    assert V4.require_device("cpu") == "cpu"


# ---------------------------------------------------------------- make_predict_fn の device
FAKE_RUNNER = '''
import numpy as np
SEEN = []
_CANON = {}
def setup_env(a, log):
    return "fake-rdkit"
def load_model(pkg, dev, log):
    SEEN.append(("load_model", dev))
    return "model"
def smiles_ok(s):
    return (s != "bad"), None
class Predictor:
    def __init__(self, model, dev, bs, log):
        SEEN.append(("Predictor", dev, bs))
        self.cache = {}; self.n_pred = 0; self.n_fail = 0; self.n_feat_fail = 0; self.t_pred = 0.0; self.bs = bs
    def predict(self, jobs, deadline):
        SEEN.append(("deadline", deadline))
        for j in sorted(jobs):
            self.cache[j] = (np.array([len(j[0]) + 0.5]), np.array([1.0]))
            self.n_pred += 1
        return deadline > 0
'''


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "ckpt").mkdir(parents=True)
    (pkg / "gl_runner.py").write_text(FAKE_RUNNER)
    (pkg / "ckpt/glacier.pt").write_bytes(b"w")
    fake_torch = types.SimpleNamespace(__version__="fake", set_num_threads=lambda n: None, get_num_threads=lambda: 1,
                                       use_deterministic_algorithms=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delitem(sys.modules, "gl_runner", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    return pkg, tmp_path / "site"


KEYS = [("CCO", "[M+H]+", "QTOF", 20.0), ("bad", "[M+H]+", "QTOF", 20.0), ("CCCO", "[M+Na]+", "QTOF", 40.0)]


def test_make_predict_fn_default_cpu_both_places(fake_env):
    pkg, site = fake_env
    fn = GL.make_predict_fn(pkg, site, 0, lambda *x: None)
    preds, fail, stats = fn(KEYS)
    R = sys.modules["gl_runner"]
    assert ("load_model", "cpu") in R.SEEN and ("Predictor", "cpu", 32) in R.SEEN
    assert ("deadline", float("inf")) in R.SEEN
    assert "finished" not in stats                       # 既定の stats の鍵は前と同じ
    assert preds[1] is None and fail[1] == GL.FAIL_SMILES and fail[0] == 0
    assert preds[0][0].dtype == np.float32 and preds[0][0][0] == 3.5


def test_make_predict_fn_same_device_both_places(fake_env):
    pkg, site = fake_env
    fn = GL.make_predict_fn(pkg, site, 0, lambda *x: None, device="cuda")
    fn(KEYS)
    R = sys.modules["gl_runner"]
    devs = [s[1] for s in R.SEEN if s[0] in ("load_model", "Predictor")]
    assert devs == ["cuda", "cuda"]


def test_make_predict_fn_default_output_equals_explicit_cpu(fake_env):
    pkg, site = fake_env
    a = GL.make_predict_fn(pkg, site, 0, lambda *x: None)(KEYS)
    b = GL.make_predict_fn(pkg, site, 0, lambda *x: None, device="cpu")(KEYS)
    assert a[2] == b[2] and np.array_equal(a[1], b[1])
    for x, y in zip(a[0], b[0]):
        assert (x is None and y is None) or all(np.array_equal(p, q) for p, q in zip(x, y))


def test_make_predict_fn_deadline_reports_unfinished(fake_env):
    pkg, site = fake_env
    _, _, stats = GL.make_predict_fn(pkg, site, 0, lambda *x: None, deadline=-1.0)(KEYS)   # fake: deadline ≤ 0 → 途中
    assert stats["finished"] is False
    _, _, stats = GL.make_predict_fn(pkg, site, 0, lambda *x: None, deadline=1e18)(KEYS)
    assert stats["finished"] is True
