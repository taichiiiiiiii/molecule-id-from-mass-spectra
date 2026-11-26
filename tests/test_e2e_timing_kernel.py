"""Issue #274: Kaggle の端から端の時間測定のノート（生成・送信サイズ・compile）と、測定の部品。"""
import csv
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


MK = _load("make_e2e_timing_kernel", "scripts/kaggle/make_e2e_timing_kernel.py")
E2E = _load("e2e_timing", "scripts/kaggle/e2e_timing.py")
FRAG = "data/ext/frag/frag_rescore.py"   # git の外（元のフォルダの data/ext）。CI では代わりの小さなファイルで作る


def _root(tmp_path) -> Path:
    r = tmp_path / "repo"
    for f in MK.CODE_FILES:
        (r / f).parent.mkdir(parents=True, exist_ok=True)
        src = ROOT / f
        if src.is_file():
            shutil.copy(src, r / f)
        else:
            assert f == FRAG, f"リポジトリに無いコード: {f}"
            (r / f).write_text("def score_molecule(spectra, smiles, cfg=None, cache=None):\n    return [None] * len(smiles)\n")
    return r


def test_notebook_size_compile_and_roundtrip(tmp_path):
    r = _root(tmp_path)
    nb, info = MK.build_notebook(r)
    assert info["push_bytes"] < MK.MAX_PUSH_BYTES
    assert MK.push_bytes(nb) == info["push_bytes"]
    assert sorted(info["manifest"]) == sorted(MK.CODE_FILES)
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code) == 2
    for c in code:
        compile("".join(c["source"]), "<cell>", "exec")
        assert c["outputs"] == [] and c["execution_count"] is None
    setup = "".join(code[0]["source"])
    blob = setup.split('BLOB = "', 1)[1].split('"', 1)[0]
    files = MK.unpack(blob)
    for f in MK.CODE_FILES:
        assert files[f] == (r / f).read_bytes()
        assert hashlib.sha256(files[f]).hexdigest() == info["manifest"][f]
    assert "__BLOB__" not in setup and "__MANIFEST__" not in setup


def test_pack_is_deterministic(tmp_path):
    r = _root(tmp_path)
    assert MK.pack(MK.CODE_FILES, r) == MK.pack(MK.CODE_FILES, r)


def test_size_gate_fires(tmp_path, monkeypatch):
    r = _root(tmp_path)
    monkeypatch.setattr(MK, "MAX_PUSH_BYTES", 1000)
    with pytest.raises(AssertionError, match="送信サイズ"):
        MK.build_notebook(r)


def test_compile_gate_fires(tmp_path):
    r = _root(tmp_path)
    (r / "scripts/local_eval/run.py").write_text("def broken(:\n")
    with pytest.raises(SyntaxError):
        MK.build_notebook(r)


@pytest.mark.skipif(not (ROOT / FRAG).is_file(), reason="frag_rescore.py は手元の data/ext にだけある")
def test_real_notebook_under_limit():
    nb, info = MK.build_notebook()
    assert info["push_bytes"] < MK.MAX_PUSH_BYTES and info["n_files"] == len(MK.CODE_FILES)


def test_kernel_metadata_cpu_private_no_submit():
    m = MK.kernel_metadata()
    assert m["is_private"] is True and m["enable_gpu"] is False and m["enable_internet"] is False
    assert m["competition_sources"] == [MK.COMPETITION] and MK.INPUTS_DS in m["dataset_sources"]
    assert m["code_file"] == "notebook.ipynb" and "machine_shape" not in m
    assert "id_no" not in m


def test_committed_kernel_matches_generator():
    """コミットしたノートと metadata が生成器の出力と同じ（frag_rescore.py がある手元だけ中身まで比べる）。"""
    d = ROOT / "kernels/e2e_timing"
    assert json.loads((d / "kernel-metadata.json").read_text()) == MK.kernel_metadata()
    nb = json.loads((d / "notebook.ipynb").read_text())
    assert MK.push_bytes(nb) < MK.MAX_PUSH_BYTES
    if (ROOT / FRAG).is_file():
        assert nb == MK.build_notebook()[0]


def test_extrapolate_factor_and_over():
    ex = E2E.extrapolate({"a": 100.0, "b": 6000.0}, 400)
    assert ex["factor"] == 3.0 and ex["stages"] == {"a": 300.0, "b": 18000.0} and ex["total"] == 18300.0
    assert ex["limit_secs"] == 16200.0
    assert ex["over"] == [dict(stage="b", secs=18000.0, over_by=1800.0), dict(stage="total", secs=18300.0, over_by=2100.0)]
    assert E2E.extrapolate({"a": 1.0}, 400)["over"] == []


def test_within_counts_calls():
    import types
    M = types.ModuleType("m")
    M.f = lambda x: x + 1
    w = E2E.Within()
    w.wrap(M, "f")
    assert M.f(1) == 2 and M.f(2) == 3
    rep = w.report()
    assert rep["m.f"]["calls"] == 2 and rep["m.f"]["secs"] >= 0


def test_write_submission_fallback_and_valid(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    from mrr25 import validate_submission
    out = tmp_path / "s.csv"
    c = E2E.write_submission(out, ["a", "b"], {"a": [f"C{'C' * i}" for i in range(30)]})
    assert c == dict(ranker=1, fallback_cco=1)
    got = validate_submission(list(csv.reader(open(out))), expected_ids={"a", "b"})
    assert len(got["a"]) == 25 and got["b"] == ["CCO"]


def test_prepare_hides_nothing_and_limit(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(E2E, "ROOT", tmp_path)
    test, sample = tmp_path / "t.parquet", tmp_path / "s.csv"
    pd.DataFrame(dict(molecule_id=["m1", "m1", "m2", "m3"], x=[1, 2, 3, 4])).to_parquet(test)
    pd.DataFrame(dict(molecule_id=["m1", "m2", "m3"], smiles="CCO")).to_csv(sample, index=False)
    ids = E2E.prepare(test, sample, 2)
    assert ids == ["m1", "m2"]
    hd = tmp_path / f"data/holdouts/{E2E.NAME}"
    assert json.loads((hd / "hide_rule.json").read_text())["pool"] == "keep"
    assert (hd / "drop_keys.txt").read_text() == ""
    assert len(pd.read_parquet(hd / "test.parquet")) == 3
    assert list(pd.read_csv(tmp_path / f"outputs/local_eval/{E2E.NAME}/truth.csv").molecule_id) == ["m1", "m2"]


def test_submit_ranker_features_are_exp003():
    TR = _load("train_submit_ranker", "scripts/kaggle/train_submit_ranker.py")
    f = TR.submit_features()
    assert len(f) == 64 and "prior_unverified_pop" not in f and f[-2:] == ("cc", "cc_z")
    assert TR.EXPECT_FEATURES_SHA == E2E.EXPECT_FEATURES_SHA
