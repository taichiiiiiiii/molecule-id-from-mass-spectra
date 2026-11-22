"""scripts/submit/make_lb487_btables.py の検証（#189 = #158 (B) の診断、計画 2026-11-22 昼 項目 5）。

0.487 写しのノート（kernels/lb487）と生成物（kernels/lb487_btables・kernels/lb487_btables_timing400）だけを使う。
Kaggle の資産・RDKit・torch は要らない。写しのノートが無いときは全体を skip する。
"""
import ast
import difflib
import glob
import hashlib
import importlib.util
import json
import os
import re
import sys

import pytest

REAL_GLOB = glob.glob
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "submit", "make_lb487_btables.py")
SRC_NB = os.path.join(ROOT, "kernels", "lb487", "casmi26-lb487-copy.ipynb")
SRC_META = os.path.join(ROOT, "kernels", "lb487", "kernel-metadata.json")
FILES_JSON = os.path.join(ROOT, "kernels", "lb487_btables", "b_run1_v3_files.json")
LOCAL_OWN = os.path.join(ROOT, "outputs", "engine2_build_b", "run1_v3", "own")

pytestmark = pytest.mark.skipif(not os.path.exists(SRC_NB), reason="kernels/lb487 の写しのノートが無い")

_spec = importlib.util.spec_from_file_location("make_lb487_btables", SCRIPT)
B = importlib.util.module_from_spec(_spec); sys.modules["make_lb487_btables"] = B; _spec.loader.exec_module(B)

# #189 §1（#180 の判定で通った b_run1 version 3 の表）。生成器の定数がこれと違えば落ちる
EXPECT_SIM_SHA = "9ee9011314e158ef410e499368137e96aec8eafc13d3999594cfa70b3650d013"
EXPECT_RT_SHA = "74c0a37545d8c3cc578ef14d24ac79cfc1c9cbdae7eb65fe8c76ccf4633bf269"


def _src(c):
    return "".join(c["source"])


def _frozen(nb):
    return json.loads(B.split_frozen(_src(nb["cells"][12]))[2])


@pytest.fixture(scope="module")
def built():
    nb, meta, rep = B.build()
    orig = json.load(open(SRC_NB, encoding="utf-8"))
    return nb, meta, rep, orig


@pytest.fixture(scope="module")
def built_timing():
    return B.build("casmi26-lb487-btables-timing400", 400)


def _runner_literals(cell32):
    tree = ast.parse(cell32)
    lits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
                and isinstance(node.func.value, ast.Call) and node.args):
            opener = node.func.value
            if (getattr(opener.func, "id", None) == "open" and opener.args
                    and "eng_runner.py" in ast.unparse(opener.args[0])):
                lits.append((node.lineno, ast.literal_eval(node.args[0])))
    assert len(lits) == 2, len(lits)
    lits.sort()
    return lits[0][1], lits[1][1]


# (a) 表の sha256・パス・出どころが #189 §1 の値
def test_a_constants_match_issue():
    assert B.SIM_SHA == EXPECT_SIM_SHA and B.RT_SHA == EXPECT_RT_SHA
    assert B.SIM_REL == "own/own_sim_rank_rows.npz" and B.RT_REL == "own/own_rank_train.npz"
    assert B.KSRC == "taichiiiii/casmi26-engine2-build-b-run1"


def test_a2_local_tables_match_if_present():
    """クローンに取り出した表があれば（CI には無い）、ファイルの sha256 と大きさが定数と一致。"""
    if not os.path.isdir(LOCAL_OWN):
        pytest.skip("取り出した表が無い（CI）")
    for rel, sha, size in ((B.SIM_REL, B.SIM_SHA, B.SIM_SIZE), (B.RT_REL, B.RT_SHA, B.RT_SIZE)):
        p = os.path.join(LOCAL_OWN, os.path.basename(rel))
        assert os.path.getsize(p) == size
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha


# (b) 凍結コード: セル 32 だけが変わり、変化は「追加」だけ（元のセル 32 の行は順番どおり残る）
def test_b_frozen_only_cell32_added(built):
    nb, _, rep, orig = built
    f0, f1 = _frozen(orig), _frozen(nb)
    assert len(f0["cells"]) == len(f1["cells"]) == 59
    diff = [k for k, (a, b) in enumerate(zip(f0["cells"], f1["cells"])) if _src(a) != _src(b)
            or a["cell_type"] != b["cell_type"]]
    assert diff == [32]
    old32, new32 = _src(f0["cells"][32]), _src(f1["cells"][32])
    patch = B.PATCH.format(ksrc_name=B.KSRC_NAME, sim_sha=B.SIM_SHA, rt_sha=B.RT_SHA, sim_rel=B.SIM_REL,
                           rt_rel=B.RT_REL, runner=B.B_RUNNER)
    assert new32.replace(patch, "", 1).replace(B.EXC_ADD, "", 1).replace(B.RUN_MARK + B.RUN_ADD, B.RUN_MARK, 1) == old32
    for n in (orig, nb):
        c12 = _src(n["cells"][12])
        assert f"FROZEN_SHA256 = '{hashlib.sha256(B.split_frozen(c12)[2]).hexdigest()}'" in c12


# (c) 推論の FP は元のまま: base_fs・fold-safe FPNet・score_molecules の差し替えが無い
def test_c_no_fp_inference_change(built):
    cell32 = _src(_frozen(built[0])["cells"][32])
    a = cell32.index(B.PATCH_BEGIN); b = cell32.index(B.PATCH_END)
    added = cell32[a:b] + B.EXC_ADD + B.RUN_ADD
    for bad in ("base_fs", "fold-safe", "fpnet_fold", "score_molecules", "zlog_fs", "casmi26-engine2-own-tables",
                "casmi26-engine2-build-code"):
        assert bad not in added, bad
        assert bad not in B.B_RUNNER, bad


# (d) runner: 元の runner との差は表の読み込み・照合・レポートだけ
def test_d_runner_diff_is_tables_only(built):
    cell32 = _src(_frozen(built[0])["cells"][32])
    orig_runner, b_runner = _runner_literals(cell32)
    assert b_runner == B.B_RUNNER
    compile(orig_runner, "eng_runner_orig", "exec"); compile(b_runner, "eng_runner_b", "exec")
    ol, nl = orig_runner.split("\n"), b_runner.split("\n")
    removed, added = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=ol, b=nl, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            removed += ol[i1:i2]
        if tag in ("replace", "insert"):
            added += nl[j1:j2]
    # 消えた行: 先頭の docstring と、元の名前で 2 表を探す E.main の 1 行だけ
    assert len(removed) == 2, removed
    assert removed[0].startswith('"""Our engine')
    assert "E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz')" in removed[1]
    # E.main の他の引数は同じ
    assert "    subs, recs, S = E.main(test, train, sample, OWN['sim'], OWN['rt'], fp_models," in added
    # 足した行は照合・レポート・ガードに関するものだけ（FP・ランカーの重みに触る行が無い）
    for ln in added:
        assert not re.search(r"W_A|SEEDS|N_ANALOG|w_pv|blend_scores|fp_models = ", ln), ln
    assert "    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200" in nl and nl.count("    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200") == 1


# (e) 外側セルは 5・12 と frozen_source_sha256 の宣言だけが違う
def test_e_outer_cells(built):
    nb, _, rep, orig = built
    diff = [k for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])) if a != b]
    assert diff == [5, 12]
    c5o, c5n = _src(orig["cells"][5]), _src(nb["cells"][5])
    assert c5n == c5o.replace("COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'")
    mo, mn = json.loads(json.dumps(orig["metadata"])), json.loads(json.dumps(nb["metadata"]))
    assert mn["casmi"]["frozen_source_sha256"] == rep["frozen_sha_new"]
    mn["casmi"]["frozen_source_sha256"] = mo["casmi"]["frozen_source_sha256"]
    if "upstream_casmi_metadata" in mn["casmi"]:
        mn["casmi"]["upstream_casmi_metadata"]["frozen_source_sha256"] = \
            mo["casmi"]["upstream_casmi_metadata"]["frozen_source_sha256"]
    assert mo == mn


# (f) kernel-metadata: dataset_sources は元のまま、kernel_sources に b_run1 を足すだけ、id_no なし
def test_f_metadata(built, built_timing):
    src = json.load(open(SRC_META))
    for (nb, meta, rep), slug in ((built[:3], "casmi26-lb487-btables"),
                                  (built_timing, "casmi26-lb487-btables-timing400")):
        assert "id_no" not in meta and meta["id"] == f"taichiiiii/{slug}" and meta["is_private"] is True
        assert meta["dataset_sources"] == src["dataset_sources"]
        assert meta["kernel_sources"] == src["kernel_sources"] + [B.KSRC]
        assert meta["code_file"] == "notebook.ipynb"
        for k in ("competition_sources", "docker_image", "machine_shape", "enable_gpu", "enable_internet"):
            assert meta[k] == src[k], k


# (g) 時間測定の版は凍結セル 30 の SMOKE_N だけが提出の版と違う。slug の取り違えは止める
def test_g_timing_variant(built, built_timing):
    f_sub, f_tim = _frozen(built[0]), _frozen(built_timing[0])
    diff = [k for k, (a, b) in enumerate(zip(f_sub["cells"], f_tim["cells"])) if _src(a) != _src(b)]
    assert diff == [30]
    assert _src(f_tim["cells"][30]) == _src(f_sub["cells"][30]).replace("SMOKE_N = 12", "SMOKE_N = 400")
    assert built_timing[2]["timing_only"] is True and built_timing[2]["smoke_n"] == 400
    with pytest.raises(AssertionError):
        B.build("casmi26-lb487-btables", 400)
    with pytest.raises(AssertionError):
        B.build("casmi26-lb487-btables-timing400", None)


# (h) 生成物（リポジトリの kernels/ の下）が生成器の出力と同じ
@pytest.mark.parametrize("d,slug,smoke", [("lb487_btables", "casmi26-lb487-btables", None),
                                           ("lb487_btables_timing400", "casmi26-lb487-btables-timing400", 400)])
def test_h_committed_matches_generator(d, slug, smoke):
    out = os.path.join(ROOT, "kernels", d)
    nb, meta, rep = B.build(slug, smoke)
    assert open(os.path.join(out, "notebook.ipynb"), encoding="utf-8").read() == json.dumps(nb, ensure_ascii=False, indent=1)
    assert json.load(open(os.path.join(out, "kernel-metadata.json"))) == meta
    assert json.load(open(os.path.join(out, "build_info.json"))) == json.loads(json.dumps(rep, ensure_ascii=False))


# (i) 全セル compile、送るソース < 1,000,000 B
def test_i_compile_and_size(built, built_timing):
    for nb, _, rep in (built[:3], built_timing):
        for k, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                compile(_src(c), f"outer_{k}", "exec")
        for k, c in enumerate(_frozen(nb)["cells"]):
            if c["cell_type"] == "code":
                compile(_src(c), f"frozen_{k}", "exec")
        assert rep["push_source_bytes"] == B.push_source_bytes(nb) < 1_000_000


# (j) セル 32 の表の決め方を偽の入力で exec: ちょうど 1 つ・b_run1 の下・sha256 一致のときだけ通る
def _loader(sim_sha, rt_sha):
    patch = B.PATCH.format(ksrc_name=B.KSRC_NAME, sim_sha=sim_sha, rt_sha=rt_sha, sim_rel=B.SIM_REL,
                           rt_rel=B.RT_REL, runner=B.B_RUNNER)
    return patch


def _fake(tmp_path, where="notebooks/taichiiiii/casmi26-engine2-build-b-run1"):
    inp = tmp_path / "input"
    sh = {}
    for rel, body in ((B.SIM_REL, b"sim-table"), (B.RT_REL, b"rank-train-table")):
        p = inp / where / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(body)
        sh[rel] = hashlib.sha256(body).hexdigest()
    return inp, sh


def _run(loader, inp, tmp_path, monkeypatch, rerun=False):
    real_glob = REAL_GLOB

    def fake_glob(pat, recursive=False):
        assert pat.startswith("/kaggle/input/"), pat
        return real_glob(str(inp) + pat[len("/kaggle/input"):], recursive=recursive)
    monkeypatch.setattr(glob, "glob", fake_glob)
    monkeypatch.delenv("E2_OWN", raising=False)
    eng = tmp_path / "eng"; eng.mkdir(exist_ok=True)
    g = {"os": os, "ENG_DIR": str(eng), "IS_RERUN": rerun, "__name__": "frozen_cell_32"}
    exec(compile(loader, "loader", "exec"), g)
    return g, eng


def test_j_loader_ok(tmp_path, monkeypatch):
    inp, sh = _fake(tmp_path)
    g, eng = _run(_loader(sh[B.SIM_REL], sh[B.RT_REL]), inp, tmp_path, monkeypatch)
    own = g["_E2_OWN"]
    assert own["sim"].endswith("/casmi26-engine2-build-b-run1/own/own_sim_rank_rows.npz")
    assert own["rt"].endswith("/casmi26-engine2-build-b-run1/own/own_rank_train.npz")
    assert g["_E2_FAIL"] is None and json.loads(os.environ["E2_OWN"]) == own
    assert (eng / "eng_runner.py").read_text() == B.B_RUNNER
    rg = {"__name__": "eng_runner_test"}
    exec(compile(B.B_RUNNER, "eng_runner_b", "exec"), rg)
    assert rg["verify_own_tables"](own) == {"sim": own["sim_sha"], "rt": own["rt_sha"]}
    bad = dict(own, rt_sha="0" * 64)
    with pytest.raises(ValueError):
        rg["verify_own_tables"](bad)


def test_j_loader_rejects(tmp_path, monkeypatch):
    inp, sh = _fake(tmp_path)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _run(_loader("0" * 64, sh[B.RT_REL]), inp, tmp_path, monkeypatch)
    # 再実行では止めずに _E2_FAIL に記録（v4g のリストへ戻る側）
    g, _ = _run(_loader("0" * 64, sh[B.RT_REL]), inp, tmp_path, monkeypatch, rerun=True)
    assert g["_E2_FAIL"] is not None and g["_E2_SIM"] is None
    # 同じ相対パスが b_run1 の下に 2 つ → 止める
    inp2, sh2 = _fake(tmp_path / "dup")
    _fake(tmp_path / "dup", where="notebooks/taichiiiii/casmi26-engine2-build-b-run1/copy")
    with pytest.raises(RuntimeError, match="exactly once"):
        _run(_loader(sh2[B.SIM_REL], sh2[B.RT_REL]), inp2, tmp_path / "dup", monkeypatch)
    # b_run1 の外にしか無い → 止める
    inp3, sh3 = _fake(tmp_path / "out", where="datasets/someone/other")
    with pytest.raises(RuntimeError, match="exactly once"):
        _run(_loader(sh3[B.SIM_REL], sh3[B.RT_REL]), inp3, tmp_path / "out", monkeypatch)


# (k) b_run1 の出力のファイル名を、ノートが名前で探していない（ENG_DIR に書く 3 本を除く）
def _all_sources(nb):
    out = [_src(c) for k, c in enumerate(nb["cells"]) if k != 12]
    out += [_src(c) for c in _frozen(nb)["cells"]]
    return out


def test_k_no_name_overlap_with_b_run1(built):
    names = json.load(open(FILES_JSON))["names"]
    assert "own_rank_train.npz" in names and len(names) >= 30
    orig = json.load(open(SRC_NB, encoding="utf-8"))
    text = "\n".join(_all_sources(orig))
    hits = sorted(n for n in set(names) if re.search(r"(?<![\w.\-])" + re.escape(n), text))
    # 元のノートで名前が出るのは、セル 32 が ENG_DIR に書いて import する Engine 2 の 3 本だけ（探すのではない）
    assert hits == ["casmi_engine.py", "pv.py", "pv_fp.py"], hits
    cell32 = _src(_frozen(orig)["cells"][32])
    for n in hits:
        assert f"'{n}': " in cell32.split("\n")[2]          # ENG_FILES = {'pv.py': ..., } の鍵
