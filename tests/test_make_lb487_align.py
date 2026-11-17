"""Issue #127: 0.487 写しに整列チャネルを足したノート（kernels/lb487_align）と、その段（scripts/submit/align0487_runner.py）。

確かめること（計画 2026-11-17 昼 項目 1 の受け入れ条件）:
- 生成したノートは、外側セル 15 の neural_score の 1 行と足したセル（6 番目）以外、0.487 写しとセル単位でバイト一致
- 生成したノートの全コードセルが compile() を通る
- 足したセルが埋め込むファイルは sha256 が合い、リポジトリのファイル（runner など）・PR #110 の align_train.py と同じ
- 式の値: a が分子の全候補で有限な分子だけ (z(sanpier) + z(v80) + z(a)) / 3、ほかは今と同じ 0.5 z + 0.5 z
- runner が fp_mismatch・失敗なら例外で止まり、report の status に書く（黙って 2 項に戻らない）
torch・RDKit は使わない（CI に torch が無い）。ALIGN0487_MIX の中の `import torch` は偽のモジュールで置き換える。
"""
from __future__ import annotations

import ast
import base64
import difflib
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/submit"))

import align0487_runner as AR  # noqa: E402
import make_lb487_align as M  # noqa: E402

BASE = ROOT / "kernels/lb487/casmi26-lb487-copy.ipynb"
NEW = ROOT / "kernels/lb487_align/notebook.ipynb"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture(scope="module")
def base():
    raw = BASE.read_bytes()
    assert sha(raw) == M.SRC_SHA256
    return json.loads(raw)


@pytest.fixture(scope="module")
def new():
    return json.loads(NEW.read_bytes())


def added_source(nb) -> str:
    return "".join(nb["cells"][M.ADDED_INDEX]["source"])


def literal(cell: str, name: str):
    for node in ast.parse(cell).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


# ---------------------------------------------------------------- 1 行だけ変わる・他セル一致

def test_only_one_line_and_one_added_cell_differ(base, new):
    assert len(base["cells"]) == M.N_CELLS == 31
    assert len(new["cells"]) == 32
    assert added_source(new).startswith(M.MARK)
    for i, c in enumerate(base["cells"]):
        j = i if i < M.ADDED_INDEX else i + 1
        if i == M.NEURAL_CELL:
            continue
        assert json.dumps(new["cells"][j], sort_keys=True) == json.dumps(c, sort_keys=True), f"cell {i} changed"
    # ノートの metadata・nbformat も同じ
    assert {k: v for k, v in new.items() if k != "cells"} == {k: v for k, v in base.items() if k != "cells"}


def test_neural_cell_changes_exactly_the_neural_score_line(base, new):
    old_c, new_c = base["cells"][M.NEURAL_CELL], new["cells"][M.NEURAL_CELL + 1]
    assert {k: v for k, v in old_c.items() if k != "source"} == {k: v for k, v in new_c.items() if k != "source"}
    old_l = "".join(old_c["source"]).splitlines(keepends=True)
    new_l = "".join(new_c["source"]).splitlines(keepends=True)
    assert len(old_l) == len(new_l)
    diff = [(a, b) for a, b in zip(old_l, new_l) if a != b]
    assert diff == [(M.OLD_LINE, M.NEW_LINE)]
    ops = [o for o in difflib.SequenceMatcher(a=old_l, b=new_l).get_opcodes() if o[0] != "equal"]
    assert len(ops) == 1 and ops[0][0] == "replace" and ops[0][2] - ops[0][1] == 1
    # 新しい行は今の 2 項の z の式をそのまま渡している（fillna(0.0) を残す、#127 固定すること (2)）
    za = "_z(_a['neural_score'].fillna(0.0))"
    zb = "_z(_b['neural_score'].fillna(0.0))"
    assert za in M.OLD_LINE and zb in M.OLD_LINE and za in M.NEW_LINE and zb in M.NEW_LINE
    assert "neural_cos" not in M.NEW_LINE


def test_generated_file_equals_build(base, new):
    nb = M.build(base, added_source(new))
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    assert text.encode() == NEW.read_bytes()
    info = json.loads((ROOT / "kernels/lb487_align/build_info.json").read_text())
    assert info["notebook_sha256"] == sha(NEW.read_bytes())
    assert info["base_sha256"] == M.SRC_SHA256


def test_every_code_cell_compiles(base, new):
    n = 0
    for i, c in enumerate(new["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.lstrip().startswith(("%", "!")):
            continue
        compile(src, f"cell{i}", "exec")
        n += 1
    n_code = sum(c["cell_type"] == "code" for c in new["cells"])
    assert n == n_code == sum(c["cell_type"] == "code" for c in base["cells"]) + 1   # 1 つも飛ばしていない


def test_build_refuses_when_line_not_unique(base):
    bad = json.loads(json.dumps(base))
    bad["cells"][M.NEURAL_CELL]["source"] = ["x = 1\n"]
    with pytest.raises(AssertionError):
        M.build(bad, "pass\n")


# ---------------------------------------------------------------- 埋め込んだファイル

def test_embedded_files_match_repo_and_sha(new):
    cell = added_source(new)
    files = {k: base64.b64decode(v) for k, v in literal(cell, "_A0487_FILES").items()}
    shas = literal(cell, "_A0487_SHA256")
    assert set(files) == {"align0487_runner.py", "align_speed_trial.py", "base_fs.py", "align_train.py",
                          "align_config.json"}
    for k, b in files.items():
        assert sha(b) == shas[k], k
    assert files["align0487_runner.py"] == (ROOT / "scripts/submit/align0487_runner.py").read_bytes()
    assert files["align_speed_trial.py"] == (ROOT / "scripts/train/align_speed_trial.py").read_bytes()
    assert files["base_fs.py"] == (ROOT / "scripts/engine2/base_fs.py").read_bytes()
    assert sha(files["align_train.py"]) == M.ALIGN_TRAIN_SHA256
    cfg = json.loads(files["align_config.json"])
    assert cfg["pool_meta_sha256"] == AR.POOL_META_SHA256 and cfg["best_step"] == 24000
    dz = base64.b64decode(literal(cell, "_A0487_DREAMS_ZIP"))
    assert sha(dz) == literal(cell, "_A0487_DREAMS_SHA256")
    assert literal(cell, "_A0487_DREAMS_COMMIT") == M.DREAMS_COMMIT
    info = json.loads((ROOT / "kernels/lb487_align/build_info.json").read_text())
    assert info["embedded_sha256"] == shas and info["dreams_zip_sha256"] == sha(dz)


def test_embedded_align_train_matches_pr110_branch_if_available():
    r = subprocess.run(["git", "-C", str(ROOT), "show", M.ALIGN_TRAIN_REF], capture_output=True)
    if r.returncode != 0:
        pytest.skip("origin/feat/109-align-local-train is not fetched here")
    assert sha(r.stdout) == M.ALIGN_TRAIN_SHA256


def test_kernel_metadata():
    base = json.loads((ROOT / "kernels/lb487/kernel-metadata.json").read_text())
    meta = json.loads((ROOT / "kernels/lb487_align/kernel-metadata.json").read_text())
    assert meta["id"] == M.SLUG and "id_no" not in meta and meta["code_file"] == "notebook.ipynb"
    assert meta["dataset_sources"] == base["dataset_sources"] + [M.WEIGHTS_DS]
    for k in ("kernel_sources", "competition_sources", "docker_image", "machine_shape", "enable_gpu",
              "enable_internet", "is_private"):
        assert meta.get(k) == base.get(k), k
    assert meta["is_private"] is True and meta["enable_internet"] is False


# ---------------------------------------------------------------- 式の値

def nb_z(s: pd.Series) -> pd.Series:
    """外側セル 15 の _z と同じ。"""
    g = s.groupby(level=0)
    return (s - g.transform("mean")) / g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)


def synth(seed=0):
    rng = np.random.default_rng(seed)
    mids = ["m1"] * 4 + ["m2"] * 3 + ["m3"] * 5
    smi = [f"C{i}" for i in range(len(mids))]
    idx = pd.MultiIndex.from_arrays([mids, smi], names=["molecule_id", "smiles"])
    za = nb_z(pd.Series(rng.normal(size=len(idx)), index=idx))
    zb = nb_z(pd.Series(rng.normal(size=len(idx)), index=idx))
    a = rng.uniform(-0.5, 0.8, size=len(idx)).astype(np.float32)
    a[5] = np.nan     # m2 は 2 項に戻る
    return idx, za, zb, a


def test_z_by_molecule_equals_notebook_z():
    idx, za, zb, a = synth()
    x = np.arange(len(idx), dtype=float) ** 1.5
    assert np.allclose(AR.z_by_molecule(x, idx.get_level_values(0).values),
                       nb_z(pd.Series(x, index=idx)).values, rtol=0, atol=1e-12)
    # std = 0 の分子は 1e-6 で割る（0 になる）
    assert np.all(AR.z_by_molecule(np.ones(3), np.array(["m", "m", "m"], object)) == 0)


def test_mix_rows_formula():
    idx, za, zb, a = synth()
    mids = idx.get_level_values(0).values
    out, ok = AR.mix_rows(mids, za.values, zb.values, a, AR.z_by_molecule)
    assert ok == {"m1", "m3"}
    for m in ("m1", "m3"):
        i = mids == m
        za_, zb_, a_ = za.values[i], zb.values[i], a[i].astype(np.float64)
        za3 = (a_ - a_.mean()) / max(a_.std(), 1e-6)
        assert np.allclose(out[i], (za_ + zb_ + za3) / 3, rtol=0, atol=1e-12)
    i = mids == "m2"
    assert np.array_equal(out[i], 0.5 * za.values[i] + 0.5 * zb.values[i])


def test_runner_pure_functions():
    assert np.isnan(AR.ce_value(None)) and np.isnan(AR.ce_value([])) and AR.ce_value([20.0, 40.0]) == 20.0
    assert AR.ce_value(35) == 35.0 and np.isnan(AR.ce_value(["x"])) and np.isnan(AR.ce_value([float("inf")]))
    e = np.array([[1.0001, -2.5], [3.3, 0.1]], np.float32)
    assert np.array_equal(AR.fp16_like_training(e), e.astype(np.float16).astype(np.float32))
    big = np.array([[70000.0, 1.0]], np.float32)
    assert np.array_equal(AR.fp16_like_training(big), big)
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    v = AR.molecule_vector(g)
    assert np.allclose(v, [2 ** -0.5, 2 ** -0.5])
    assert AR.molecule_vector(np.zeros((0, 2))) is None and AR.molecule_vector(np.array([[1.0, 0], [-1.0, 0]])) is None
    h = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert np.allclose(AR.scores(h, v), h @ v)
    assert np.isnan(AR.scores(h, None)).all()


# ---------------------------------------------------------------- 足したセルの ALIGN0487_MIX（偽の runner）

@pytest.fixture()
def cell_ns(new, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    comp = tmp_path / "comp"
    comp.mkdir()
    ns = {"ROOT": tmp_path, "COMP": comp, "__name__": "t"}
    exec(compile(added_source(new), "added_cell", "exec"), ns)
    assert ns["COMMIT_MODE"] == "smoke"
    ns["_A0487_DIR"] = tmp_path / "align0487"
    ns["_A0487_DIR"].mkdir()
    ns["_a0487_materialise"] = lambda: tmp_path / "dreams"
    ns["_a0487_one"] = lambda pattern, must: tmp_path / "in" / must / pattern
    return ns


def fake_runner(ns, a, status="ok", rc=0):
    def run(cmd, capture_output, text, timeout):
        assert timeout == 3600 and "--expect-rdkit" in cmd and cmd[cmd.index("--expect-rdkit") + 1] == "2026.03.3"
        out = Path(cmd[cmd.index("--out") + 1])
        c = pd.read_parquet(cmd[cmd.index("--cands") + 1])
        if status == "ok":
            c.assign(a_align=a).to_parquet(out, index=False)
        Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps({"status": status, "seconds": 1.0}))
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")
    ns["_a_sp"] = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)


def test_mix_cell_ok(cell_ns):
    idx, za, zb, a = synth(1)
    fake_runner(cell_ns, a)
    out = cell_ns["ALIGN0487_MIX"](idx, za, zb, nb_z)
    ref, ok = AR.mix_rows(idx.get_level_values(0).values, za.values, zb.values, a, AR.z_by_molecule)
    assert list(out.index) == list(idx)
    assert np.allclose(out.values, ref, rtol=0, atol=1e-12)
    rep = json.loads((cell_ns["ROOT"] / "align0487_report.json").read_text())
    assert rep["status"] == "ok" and rep["molecules"] == 3 and rep["three_term_molecules"] == 2
    assert rep["two_term_molecules"] == 1 and abs(rep["two_term_frac"] - 1 / 3) < 1e-12
    # 尾の枠・空き枠の記録（読むだけ）
    cell_ns["PARALLAX_DIAGNOSTICS"] = {"status": "ok", "tail_changed": 2}
    cell_ns["MERGE_REPORT"] = {"status": "ok", "outsider_fills": 7}
    cell_ns["_a0487_downstream"]()
    rep = json.loads((cell_ns["ROOT"] / "align0487_report.json").read_text())
    assert rep["downstream"] == {"tail_slot": {"status": "ok", "tail_changed": 2},
                                 "empty_slot": {"status": "ok", "outsider_fills": 7}}
    assert rep["status"] == "ok"


@pytest.mark.parametrize("status,rc,want", [("fp_mismatch", AR.EXIT_FP_MISMATCH, "fp_mismatch"),
                                            ("error", 2, "error"), ("ok", 1, "error")])
def test_mix_cell_stops_loudly(cell_ns, status, rc, want):
    idx, za, zb, a = synth(2)
    fake_runner(cell_ns, a, status=status, rc=rc)
    with pytest.raises(RuntimeError):
        cell_ns["ALIGN0487_MIX"](idx, za, zb, nb_z)
    rep = json.loads((cell_ns["ROOT"] / "align0487_report.json").read_text())
    assert rep["status"] == want and "error" in rep


def test_downstream_hook_is_quiet_without_report(cell_ns):
    cell_ns["_a0487_downstream"]()      # report が無くても例外を出さない
    assert not (cell_ns["ROOT"] / "align0487_report.json").exists()
