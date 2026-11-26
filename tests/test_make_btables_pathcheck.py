"""scripts/submit/make_btables_pathcheck.py の検証（#189 のパスの確認の CPU カーネル）。"""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC_NB = os.path.join(ROOT, "kernels", "lb487", "casmi26-lb487-copy.ipynb")
pytestmark = pytest.mark.skipif(not os.path.exists(SRC_NB), reason="kernels/lb487 の写しのノートが無い")

_spec = importlib.util.spec_from_file_location("make_btables_pathcheck",
                                               os.path.join(ROOT, "scripts", "submit", "make_btables_pathcheck.py"))
P = importlib.util.module_from_spec(_spec); sys.modules["make_btables_pathcheck"] = P; _spec.loader.exec_module(P)


def test_committed_matches_generator():
    nb, meta = P.build()
    d = os.path.join(ROOT, "kernels", "btables_pathcheck")
    assert open(os.path.join(d, "notebook.ipynb"), encoding="utf-8").read() == json.dumps(nb, ensure_ascii=False, indent=1)
    assert json.load(open(os.path.join(d, "kernel-metadata.json"))) == meta


def test_cpu_only_same_sources_as_diagnostic():
    _, meta = P.build()
    _, dmeta, _ = P.B.build()
    assert meta["enable_gpu"] is False and meta["enable_internet"] is False and "machine_shape" not in meta
    assert "timing" not in meta["id"] and meta["id"] == "taichiiiii/casmi26-btables-pathcheck"
    for k in ("dataset_sources", "kernel_sources", "competition_sources", "docker_image"):
        assert meta[k] == dmeta[k], k


def test_loader_is_the_diagnostic_cell32_segment():
    nb, _ = P.build()
    src = nb["cells"][1]["source"]
    loader = P.loader_source()
    assert repr(loader) in src
    assert loader.startswith(P.B.PATCH_BEGIN) and loader.endswith(P.B.PATCH_END)
    # 時間測定の版のセル 32 にも同じ表の決め方が入る
    dnb_t = P.B.build("casmi26-lb487-btables-timing400", 400)[0]
    c32 = "".join(json.loads(P.B.split_frozen("".join(dnb_t["cells"][12]["source"]))[2])["cells"][32]["source"])
    assert loader in c32
    compile(src, "pathcheck", "exec")
