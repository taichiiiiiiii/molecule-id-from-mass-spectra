"""scripts/submit/make_lb487_owntables.py の検証（#77、計画 2026-11-12 昼 項目 3 の (a)〜(e)）。

0.487 写しのノート（kernels/lb487、リポジトリに入っている）だけを使う。Kaggle の資産・RDKit・torch は要らない。
写しのノートが無いとき（sparse checkout など）は全体を skip する。
"""
import ast
import base64
import glob
import hashlib
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SCRIPT = os.path.join(ROOT, "scripts", "submit", "make_lb487_owntables.py")
SRC_NB = os.path.join(ROOT, "kernels", "lb487", "casmi26-lb487-copy.ipynb")
SRC_META = os.path.join(ROOT, "kernels", "lb487", "kernel-metadata.json")

pytestmark = pytest.mark.skipif(not os.path.exists(SRC_NB), reason="kernels/lb487 の写しのノートが無い")

_spec = importlib.util.spec_from_file_location("make_lb487_owntables", SCRIPT)
M = importlib.util.module_from_spec(_spec); sys.modules["make_lb487_owntables"] = M; _spec.loader.exec_module(M)

SIM_SHA = "1" * 64
RT_SHA = "2" * 64


def _src(c):
    return "".join(c["source"])


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _frozen(nb):
    return json.loads(M.split_frozen(_src(nb["cells"][12]))[2])


@pytest.fixture(scope="module")
def built():
    nb, meta, rep = M.build(SIM_SHA, RT_SHA)
    orig = json.load(open(SRC_NB, encoding="utf-8"))
    return nb, meta, rep, orig


@pytest.fixture(scope="module")
def cell32(built):
    return _src(_frozen(built[0])["cells"][M.CHANGED_FROZEN])


def _runner_literal(cell32):
    """セル 32 の 2 つ目の eng_runner.py の書き込み（自前の runner）の文字列リテラルを ast.literal_eval で取り出す。"""
    tree = ast.parse(cell32)
    lits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "write"
                and isinstance(node.func.value, ast.Call) and node.args):
            opener = node.func.value
            if (getattr(opener.func, "id", None) == "open" and opener.args
                    and "eng_runner.py" in ast.unparse(opener.args[0])):
                lits.append((node.lineno, ast.literal_eval(node.args[0])))
    assert len(lits) == 2, len(lits)      # 元の runner と自前の runner
    lits.sort()
    return lits[0][1], lits[1][1]


# ------------------------------------------------------------------------------------------------
# (a) 凍結コード 59 セル中、セル 32 以外がセルごとの sha256 で kernels/lb487 と一致（58 セル）
# ------------------------------------------------------------------------------------------------
def test_a_frozen_cells_identical_except_32(built):
    nb, _, rep, orig = built
    f0, f1 = _frozen(orig), _frozen(nb)
    assert len(f0["cells"]) == len(f1["cells"]) == 59
    same = [k for k, (a, b) in enumerate(zip(f0["cells"], f1["cells"])) if _sha(_src(a)) == _sha(_src(b))
            and a["cell_type"] == b["cell_type"]]
    assert len(same) == 58 and set(range(59)) - set(same) == {32}
    # 凍結コードの宣言 sha256 は復号したバイト列と一致（元も新も）
    for n in (orig, nb):
        c12 = _src(n["cells"][12])
        raw = M.split_frozen(c12)[2]
        assert f"FROZEN_SHA256 = '{hashlib.sha256(raw).hexdigest()}'" in c12
    assert rep["frozen_sha_new"] != rep["frozen_sha_old"]
    # セル 32 の変更は「追加だけ」: 元のセル 32 の行は順番どおりすべて残る
    old32 = _src(f0["cells"][32]); new32 = _src(f1["cells"][32])
    assert new32.replace(M.PATCH.format(own_ds_name=M.OWN_DS_NAME, sim_sha=SIM_SHA, rt_sha=RT_SHA,
                                        sim_rel="own_sim_rank_rows.npz", rt_rel="own_rank_train.npz",
                                        runner=M.OWN_RUNNER), "", 1).replace(M.EXC_ADD, "", 1) == old32


# ------------------------------------------------------------------------------------------------
# (b) 全セル compile()、直した文字列リテラル（自前の runner）は ast.literal_eval で取り出して compile
# ------------------------------------------------------------------------------------------------
def test_b_all_cells_compile_and_runner_literal(built, cell32):
    nb = built[0]
    for k, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            compile(_src(c), f"outer_{k}", "exec")
    for k, c in enumerate(_frozen(nb)["cells"]):
        if c["cell_type"] == "code":
            compile(_src(c), f"frozen_{k}", "exec")
    orig_runner, own_runner = _runner_literal(cell32)
    assert own_runner == M.OWN_RUNNER
    compile(own_runner, "eng_runner_own", "exec")
    compile(orig_runner, "eng_runner_orig", "exec")
    # 元の 2 表の名前で探す行が自前の runner に無い（元の runner には有る）
    assert "E.find('rank_train.npz')" in orig_runner and "E.find('rank_train.npz')" not in own_runner
    assert "sim_rank_rows_nofp.npz" not in own_runner
    # ENG_FILES（凍結コードのエンジン本体）は変えていない
    assert cell32.count("ENG_FILES = {") == 1


# ------------------------------------------------------------------------------------------------
# (c) 外側セルは 5（COMMIT_MODE が 'smoke'）・12（FROZEN_B64・FROZEN_SHA256）と notebook メタデータの宣言以外が同一
# ------------------------------------------------------------------------------------------------
def test_c_outer_cells_and_metadata(built):
    nb, _, rep, orig = built
    assert len(nb["cells"]) == len(orig["cells"]) == 31
    diff = [k for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])) if a != b]
    assert diff == [5, 12]
    c5o, c5n = _src(orig["cells"][5]), _src(nb["cells"][5])
    assert "COMMIT_MODE = 'fast'" in c5o and c5n == c5o.replace("COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'")
    c12o, c12n = _src(orig["cells"][12]), _src(nb["cells"][12])
    io, jo, _ = M.split_frozen(c12o); i_n, jn, _ = M.split_frozen(c12n)
    assert (c12n[:i_n] + c12n[jn:]).replace(rep["frozen_sha_new"], rep["frozen_sha_old"]) == c12o[:io] + c12o[jo:]
    # notebook メタデータ: frozen_source_sha256 の宣言だけが変わる
    mo, mn = json.loads(json.dumps(orig["metadata"])), json.loads(json.dumps(nb["metadata"]))
    assert mn["casmi"]["frozen_source_sha256"] == rep["frozen_sha_new"]
    assert "casmi.frozen_source_sha256" in rep["changed_metadata"]
    for path in rep["changed_metadata"]:
        d_o, d_n = mo, mn
        keys = path.split(".")
        for key in keys[:-1]:
            d_o, d_n = d_o[key], d_n[key]
        assert d_o[keys[-1]] == rep["frozen_sha_old"] and d_n[keys[-1]] == rep["frozen_sha_new"]
        d_n[keys[-1]] = d_o[keys[-1]]
    assert mo == mn
    # ノートの他のトップレベルの鍵も同じ
    assert {k: v for k, v in orig.items() if k not in ("cells", "metadata")} == \
           {k: v for k, v in nb.items() if k not in ("cells", "metadata")}


# ------------------------------------------------------------------------------------------------
# (d) セル 32 の読み込み部分を偽の入力で exec: sha256 が違う表・自前表の dataset 外のパスで例外
# ------------------------------------------------------------------------------------------------
def _fake_input(tmp_path, tables_in="casmi26-engine2-own-tables"):
    inp = tmp_path / "input"
    files = {}
    for name, body in (("own_sim_rank_rows.npz", b"sim-table"), ("own_rank_train.npz", b"rank-train-table")):
        p = inp / tables_in / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(body)
        files[name] = hashlib.sha256(body).hexdigest()
    for rel in ["casmi26-attributed-v4g-reproduction-assets/casmi26-v3-models/code/casmi/chem.py",
                "casmi26-engine2-build-code/base_fs.py",
                "casmi26-fold-safe-fpnet/fp_bits.npy"] + [f"casmi26-fold-safe-fpnet/fpnet_fold{k}.pt" for k in range(5)]:
        p = inp / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# fake\n")
    return inp, files


def _exec_loader(cell32, inp, tmp_path, monkeypatch):
    a = cell32.index(M.PATCH_BEGIN); b = cell32.index(M.PATCH_END) + len(M.PATCH_END)
    loader = cell32[a:b]
    real_glob = glob.glob

    def fake_glob(pat, recursive=False):
        assert pat.startswith("/kaggle/input/"), pat
        return real_glob(str(inp) + pat[len("/kaggle/input"):], recursive=recursive)
    monkeypatch.setattr(glob, "glob", fake_glob)
    monkeypatch.delenv("E2_OWN", raising=False)
    eng = tmp_path / "eng"; eng.mkdir(exist_ok=True)
    g = {"os": os, "ENG_DIR": str(eng), "__name__": "frozen_cell_32"}
    exec(compile(loader, "frozen_cell_32_loader", "exec"), g)
    return g, eng


def _cell32_with(sim_sha, rt_sha):
    nb, _, _ = M.build(sim_sha, rt_sha)
    return _src(_frozen(nb)["cells"][M.CHANGED_FROZEN])


def test_d_loader_accepts_matching_tables(tmp_path, monkeypatch):
    inp, sh = _fake_input(tmp_path)
    cell = _cell32_with(sh["own_sim_rank_rows.npz"], sh["own_rank_train.npz"])
    g, eng = _exec_loader(cell, inp, tmp_path, monkeypatch)
    own = g["_E2_OWN"]
    assert own["sim"] == str(inp / "casmi26-engine2-own-tables" / "own_sim_rank_rows.npz")
    assert own["rt"] == str(inp / "casmi26-engine2-own-tables" / "own_rank_train.npz")
    assert json.loads(os.environ["E2_OWN"]) == own
    assert (eng / "eng_runner.py").read_text() == M.OWN_RUNNER and (eng / "base_fs.py").exists()
    assert len(own["fs_models"]) == 5 and own["casmi_code"].endswith("casmi26-v3-models/code")
    # runner 側の確認も同じ入力で通る
    rg = {"__name__": "eng_runner_test"}
    exec(compile(M.OWN_RUNNER, "eng_runner_own", "exec"), rg)
    assert rg["verify_own_tables"](own) == {"sim": own["sim_sha"], "rt": own["rt_sha"]}


@pytest.mark.parametrize("which", ["sim", "rt"])
def test_d_loader_rejects_wrong_sha(tmp_path, monkeypatch, which):
    inp, sh = _fake_input(tmp_path)
    s, r = sh["own_sim_rank_rows.npz"], sh["own_rank_train.npz"]
    cell = _cell32_with("0" * 64 if which == "sim" else s, "0" * 64 if which == "rt" else r)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _exec_loader(cell, inp, tmp_path, monkeypatch)
    assert not (tmp_path / "eng" / "eng_runner.py").exists()     # Engine 2 の runner を書く前に止まる


@pytest.mark.parametrize("ds", ["other-dataset", "casmi26-engine2-own-tables-old", "casmi26-fp-models-v2"])
def test_d_loader_rejects_tables_outside_own_dataset(tmp_path, monkeypatch, ds):
    inp, sh = _fake_input(tmp_path, tables_in=ds)                  # 同名・同じ中身の表が別の dataset にだけある
    cell = _cell32_with(sh["own_sim_rank_rows.npz"], sh["own_rank_train.npz"])
    with pytest.raises(RuntimeError, match="not found under casmi26-engine2-own-tables"):
        _exec_loader(cell, inp, tmp_path, monkeypatch)


def test_d_runner_verify_rejects(tmp_path):
    rg = {"__name__": "eng_runner_test"}
    exec(compile(M.OWN_RUNNER, "eng_runner_own", "exec"), rg)
    verify = rg["verify_own_tables"]
    good = tmp_path / "casmi26-engine2-own-tables"; good.mkdir()
    bad = tmp_path / "elsewhere"; bad.mkdir()
    for d in (good, bad):
        (d / "own_sim_rank_rows.npz").write_bytes(b"s"); (d / "own_rank_train.npz").write_bytes(b"r")
    hs, hr = hashlib.sha256(b"s").hexdigest(), hashlib.sha256(b"r").hexdigest()
    own = dict(sim=str(good / "own_sim_rank_rows.npz"), sim_sha=hs, rt=str(good / "own_rank_train.npz"), rt_sha=hr)
    assert verify(own) == {"sim": hs, "rt": hr}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify(dict(own, rt_sha="0" * 64))
    with pytest.raises(ValueError, match="outside"):
        verify(dict(own, sim=str(bad / "own_sim_rank_rows.npz")))


def test_d_bad_arguments_rejected():
    with pytest.raises(ValueError):
        M.build("abc", RT_SHA)
    with pytest.raises(ValueError):
        M.build(SIM_SHA, RT_SHA, sim_rel="../own_sim_rank_rows.npz")
    with pytest.raises(ValueError):
        M.build(SIM_SHA, RT_SHA, rt_rel="rank_train.npz")


# ------------------------------------------------------------------------------------------------
# (e) kernel-metadata.json: dataset_sources は元の 10 件＋自前表＋生成コードの 2 件、id_no が無い
# ------------------------------------------------------------------------------------------------
def test_e_kernel_metadata(built, tmp_path):
    _, meta, rep, _ = built
    src = json.load(open(SRC_META))
    assert len(src["dataset_sources"]) == 10
    assert meta["dataset_sources"] == src["dataset_sources"] + [M.OWN_DS, M.CODE_DS]
    assert "id_no" not in meta and meta["id"] == "taichiiiii/casmi26-lb487-owntables"
    assert meta["code_file"] == "notebook.ipynb" and meta["is_private"] is True
    for k in ("kernel_sources", "competition_sources", "docker_image", "machine_shape", "enable_gpu", "enable_internet"):
        assert meta[k] == src[k], k
    # 書き出し: 仮の値は kernels/ の下に書けない。scratch には書けて provisional: true が残る
    with pytest.raises(SystemExit):
        M.write(os.path.join(ROOT, "kernels", "lb487_owntables"), *built[:3], provisional=True)
    out = M.write(tmp_path / "nb", *built[:3], provisional=True)
    info = json.load(open(out / "build_info.json"))
    assert info["provisional"] is True and info["sim_sha"] == SIM_SHA
    assert json.load(open(out / "kernel-metadata.json")) == meta
    assert json.load(open(out / "notebook.ipynb")) == built[0]
