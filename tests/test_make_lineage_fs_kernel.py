"""make_lineage_fs_kernel（骨組み）の単体テスト（合成のノート）。"""
import base64
import hashlib
import importlib.util
import json
import os
import pickle
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("make_lineage_fs_kernel", os.path.join(HERE, "..", "scripts", "cv", "make_lineage_fs_kernel.py"))
K = importlib.util.module_from_spec(spec); sys.modules["make_lineage_fs_kernel"] = K; spec.loader.exec_module(K)


def _code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def _nb():
    inner = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# frozen"]},
                       _code("bank = ModelBank([_FULL[0] if _FULL else 'fpnet_0.pt'])\nx = 1\n"),
                       _code("!pip install nothing\ny = x + 1\n")],
             "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    raw = K.encode_frozen(inner)
    c12 = ("MODULES = {}\nMODULES['m1'] = 'def f():\\n    return 1\\n'\n"
           f"FROZEN_SHA256 = '{hashlib.sha256(raw).hexdigest()}'\nFROZEN_B64 = '{base64.b64encode(raw).decode()}'\nprint(len(FROZEN_B64))\n")
    cells = [_code("a = 1\n")] * 12 + [_code(c12), _code("%time b = 2\n")]
    return {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, inner


def test_replace_exactly_once():
    assert K.replace_exactly_once("a b c", "b", "B") == "a B c"
    with pytest.raises(ValueError, match="0 回"):
        K.replace_exactly_once("a b c", "z", "Z", "none")
    with pytest.raises(ValueError, match="2 回"):
        K.replace_exactly_once("b b", "b", "B", "twice")


def test_frozen_roundtrip_and_sha():
    nb, inner = _nb()
    got, h = K.get_frozen(nb)
    assert got == inner and h == hashlib.sha256(K.encode_frozen(inner)).hexdigest()
    h2 = K.set_frozen(nb, got)
    assert h2 == h                                            # 差し替え無しなら同じバイト列
    # sha の食い違いは止める
    bad = json.loads(json.dumps(nb)); src = K.cell_source(bad["cells"][12])
    K.set_cell_source(bad["cells"][12], src.replace(h, "0" * 64))
    with pytest.raises(ValueError, match="sha256"):
        K.get_frozen(bad)


def test_apply_patches_changes_only_listed_cells_and_rewrites_sha():
    nb, _ = _nb()
    P = (K.Patch("frozen", 1, "fpnet_0.pt", "fpnet_fold0.pt", "§2-1"), K.Patch("outer", 0, "a = 1", "a = 3", "outer"))
    out, rep = K.apply_patches(nb, P)
    inner, h = K.get_frozen(out)
    assert "fpnet_fold0.pt" in K.cell_source(inner["cells"][1]) and rep["frozen_sha256_after"] == h != rep["frozen_sha256_before"]
    assert rep["changed_frozen"] == [1] and rep["changed_outer"] == [0]
    assert K.cell_source(nb["cells"][0]) == "a = 1\n"         # 元は変えない
    with pytest.raises(ValueError):
        K.apply_patches(nb, (K.Patch("frozen", 1, "not there", "x", "miss"),))


def test_compile_all_catches_syntax_errors_everywhere():
    nb, inner = _nb()
    assert K.compile_all(nb) == dict(outer=14, modules=1, frozen=2)    # ! と % の行は飛ばす
    broken = json.loads(json.dumps(nb)); broken["cells"][0] = _code("def (:\n")
    with pytest.raises(SyntaxError):
        K.compile_all(broken)
    inner2 = json.loads(json.dumps(inner)); inner2["cells"][1] = _code("x = = 1\n")
    nb2 = json.loads(json.dumps(nb)); K.set_frozen(nb2, inner2)
    with pytest.raises(SyntaxError):
        K.compile_all(nb2)
    nb3 = json.loads(json.dumps(nb)); s = K.cell_source(nb3["cells"][12])
    K.set_cell_source(nb3["cells"][12], s.replace("return 1", "return (1"))
    with pytest.raises(SyntaxError):
        K.compile_all(nb3)


def test_forbidden_list_and_matching():
    names = [p for p, _ in K.FORBIDDEN]
    for need in ("fpnet_full1.pt", "fpnet_0.pt", "fpnet_1.pt", "fe_A.pt", "fe_B.pt", "fe_D.pt", "final_model.pt", "fp_single_s2.pt",
                 "fp_merged_m1.pt", "rank_train.npz", "sim_rank_rows*.npz", "fpnet_fold1.pt", "fpnet_fold2.pt", "fpnet_fold3.pt", "fpnet_fold4.pt"):
        assert need in names
    assert K.is_forbidden("/kaggle/input/casmi26-v4b-models/fe_D.pt")
    assert K.is_forbidden("/kaggle/input/x/sim_rank_rows_nofp.npz")
    assert K.is_forbidden("/kaggle/input/casmi-spectrum-model-v1/final_model.pt")
    assert not K.is_forbidden("/kaggle/input/casmi26-sanpier-fold0/casmi_spectrum_model_fold0/final_model.pt")   # #54
    assert not K.is_forbidden("/kaggle/input/casmi26-fold-safe-fpnet/fpnet_fold0.pt")
    assert not K.is_forbidden("/kaggle/working/own_rank_train.npz")                                              # #74 の自前表


def _guarded_targets():
    """監視のセルが差し替える (モジュール, 属性)。入っていないパッケージは飛ばす。"""
    import importlib
    out = [(np, "load"), (pickle, "load")]
    for mod, attr in (("torch", "load"), ("joblib", "load"), ("lightgbm", "Booster")):
        try:
            out.append((importlib.import_module(mod), attr))
        except ImportError:
            pass
    return out


@pytest.fixture
def restore_guarded():
    saved = [(m, a, getattr(m, a)) for m, a in _guarded_targets()]
    yield
    for m, a, f in saved:
        setattr(m, a, f)


def test_guard_cell_blocks_and_logs(tmp_path, monkeypatch, restore_guarded):
    ok = tmp_path / "ok.npy"; np.save(ok, np.arange(3))
    bad = tmp_path / "fpnet_full1.pt"; bad.write_bytes(pickle.dumps(1))
    ns: dict = {}
    exec(compile(K.guard_cell_source(), "<guard>", "exec"), ns)
    monkeypatch.chdir(tmp_path)
    assert list(np.load(ok)) == [0, 1, 2]
    with open(bad, "rb") as f, pytest.raises(RuntimeError, match="forbidden"):
        pickle.load(f)
    assert [d["path"] for d in ns["OPENED_FILES"]] == [str(ok), str(bad)]
    assert ns["OPENED_FORBIDDEN"] == [str(bad)]


def test_guard_is_restored_by_fixture():
    # 前のテストの差し替えが残っていない（restore_guarded）
    for m, a in _guarded_targets():
        assert not getattr(getattr(m, a), "_lineage_wrapped", False), (m.__name__, a)


def test_guard_joblib_and_lgb_booster_with_stubs(tmp_path, monkeypatch, restore_guarded):
    """joblib.load と lgb.Booster（model_file を位置引数でもキーワードでも）を包む。CI に無いパッケージなので差し替えの模型で見る。"""
    import types
    jl = types.ModuleType("joblib"); jl.load = lambda filename, mmap_mode=None: ("loaded", filename)
    lg = types.ModuleType("lightgbm")

    class Booster:
        def __init__(self, params=None, train_set=None, model_file=None, model_str=None):
            self.model_file = model_file
    lg.Booster = Booster
    monkeypatch.setitem(sys.modules, "joblib", jl)
    monkeypatch.setitem(sys.modules, "lightgbm", lg)
    monkeypatch.chdir(tmp_path)
    ns: dict = {}
    exec(compile(K.guard_cell_source(), "<guard>", "exec"), ns)
    assert jl.load("ok.pkl") == ("loaded", "ok.pkl")
    with pytest.raises(RuntimeError, match="forbidden"):
        jl.load("/x/rank_train.npz")
    with pytest.raises(RuntimeError, match="forbidden"):
        lg.Booster(None, None, "/x/fe_D.pt")                       # model_file を位置引数で
    with pytest.raises(RuntimeError, match="forbidden"):
        lg.Booster(model_file="/x/fe_A.pt")
    assert lg.Booster(model_file="/x/ranker_ok.txt").model_file == "/x/ranker_ok.txt"
    lg.Booster(params={})                                           # model_file 無し（学習用）は記録しない
    assert ns["OPENED_FORBIDDEN"] == ["/x/rank_train.npz", "/x/fe_D.pt", "/x/fe_A.pt"]
    assert [d["kind"] for d in ns["OPENED_FILES"]] == ["joblib.load", "joblib.load", "lgb.Booster", "lgb.Booster", "lgb.Booster"]


def test_real_notebook_byte_roundtrip():
    """c3d9992 の本物のノートで、復号 → 再エンコードが元のバイト列（FROZEN_SHA256）に戻る。CI（浅い clone）では飛ばす。"""
    import subprocess
    try:
        nb, sha = K.load_base(None)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pytest.skip(f"{K.BASE_COMMIT}:{K.BASE_PATH} が無い（浅い clone）")
    assert sha == K.BASE_SHA256
    inner, fsha = K.get_frozen(nb)
    assert fsha == K.FROZEN_SHA256_BASE
    assert hashlib.sha256(K.encode_frozen(inner)).hexdigest() == fsha
    out, rep = K.apply_patches(nb, ())
    assert out == nb and rep["frozen_sha256_after"] is None              # 差し替え無しなら何も変えない
    nb2 = json.loads(json.dumps(nb))
    assert K.set_frozen(nb2, inner) == fsha                                # 書き戻しても同じ sha256
    assert K.cell_source(nb2["cells"][K.FROZEN_CELL]) == K.cell_source(nb["cells"][K.FROZEN_CELL])   # セル 12 の文字列も同じ
    assert K.compile_all(K.insert_guard(nb, K.FROZEN_CELL + 1))["frozen"] > 0
