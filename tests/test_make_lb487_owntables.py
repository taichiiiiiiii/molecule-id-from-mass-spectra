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
                                        runner=M.OWN_RUNNER), "", 1).replace(M.EXC_ADD, "", 1) \
        .replace(M.RUN_MARK + M.RUN_ADD, M.RUN_MARK, 1) == old32


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
    g = {"os": os, "ENG_DIR": str(eng), "IS_RERUN": False, "__name__": "frozen_cell_32"}   # commit（止める側）
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
    aux = rg["aux_sha256"](own, str(eng))          # 記録だけ（base_fs.py の写し、FPNet 5 本、fp_bits.npy）
    assert len(aux) == 7 and set(aux.values()) == {hashlib.sha256(b"# fake\n").hexdigest()}


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


def test_f_timing_variant_differs_only_in_smoke_n(built):
    """時間測定の版（提出しない）: 提出版との違いは凍結セル 30 の SMOKE_N だけ。slug に timing が無ければ作れない。"""
    nb_t, meta_t, rep_t = M.build(SIM_SHA, RT_SHA, slug="casmi26-lb487-owntables-timing400", smoke_n=400)
    fa, fb = _frozen(built[0]), _frozen(nb_t)
    diff = [k for k, (a, b) in enumerate(zip(fa["cells"], fb["cells"])) if _src(a) != _src(b)]
    assert diff == [30]
    assert _src(fb["cells"][30]) == _src(fa["cells"][30]).replace("SMOKE_N = 12", "SMOKE_N = 400")
    assert rep_t["timing_only"] is True and rep_t["smoke_n"] == 400 and "timing" in meta_t["id"]
    assert "timing_only" not in built[2]
    with pytest.raises(AssertionError):
        M.build(SIM_SHA, RT_SHA, smoke_n=400)


# ------------------------------------------------------------------------------------------------
# (g) #77（#108 09:19Z の 2）: Engine 2 が時間切れ・失敗したとき、commit では止め、再実行では v4g のリストに戻る
#     セル 32 全体を、偽の Engine 2（subprocess.run の差し替え）と偽の /kaggle で exec する
# ------------------------------------------------------------------------------------------------
import subprocess as _real_subprocess
import textwrap
import types

import pandas as pd

N_MOL = 12
MIDS = [f"M{k:03d}" for k in range(N_MOL)]


def _fake_kaggle(tmp_path, tables_in="casmi26-engine2-own-tables"):
    """偽の /kaggle/input（自前表など）・/kaggle/working・COMP（test.parquet 12 分子）。"""
    inp, sh = _fake_input(tmp_path, tables_in=tables_in)
    work = tmp_path / "working"; work.mkdir(exist_ok=True)
    comp = tmp_path / "comp"; comp.mkdir(exist_ok=True)
    rows = [dict(molecule_id=m, spectrum_id=f"{m}_s{j}") for m in MIDS for j in range(2)]
    pd.DataFrame(rows).to_parquet(comp / "test.parquet")
    return inp, sh, work, comp


class FakeEngine2:
    """Engine 2 のサブプロセスの代わり。mode ごとに eng_lists.json・report を書き、returncode を返すか例外を出す。"""

    def __init__(self, work, mode):
        self.work, self.mode, self.calls = work, mode, 0

    def _lists(self, mids):
        return {m: dict(smiles=[f"C{'C' * (i % 5)}N"], keys=[f"K{m}"]) for i, m in enumerate(mids)}

    def run(self, args, **kw):
        self.calls += 1
        assert kw.get("timeout") == 4 * 3600
        rp = self.work / "engine2_owntables_report.json"; out = self.work / "eng_lists.json"
        own = json.loads(kw["env"]["E2_OWN"])
        rp.write_text(json.dumps(dict(status="started", own=own)))
        m = self.mode
        if m == "timeout":
            raise _real_subprocess.TimeoutExpired(args, kw["timeout"])
        if m == "crash_no_lists":            # returncode != 0 で eng_lists.json が無い
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Traceback ...")
        if m == "crash_after_lists":         # eng_lists.json を書いた後に落ちて status = started のまま
            out.write_text(json.dumps(self._lists(MIDS)))
            return types.SimpleNamespace(returncode=1, stdout="", stderr="AssertionError CALLS")
        if m == "rc_nonzero_only":           # report も lists もそろっているのに returncode != 0
            out.write_text(json.dumps(self._lists(MIDS)))
            rp.write_text(json.dumps(dict(status="ok", own=own, tables_sha256={"sim": own["sim_sha"], "rt": own["rt_sha"]})))
            return types.SimpleNamespace(returncode=3, stdout="", stderr="")
        if m == "partial_cover":             # status ok だが ENG が active な分子を覆わない
            out.write_text(json.dumps(self._lists(MIDS[:-2])))
            rp.write_text(json.dumps(dict(status="ok", own=own)))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if m == "no_report":                 # lists はあるが report が消えた（読めない）
            out.write_text(json.dumps(self._lists(MIDS)))
            rp.unlink()
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        assert m == "ok", m
        out.write_text(json.dumps(self._lists(MIDS)))
        rp.write_text(json.dumps(dict(status="ok", own=own, tables_sha256={"sim": own["sim_sha"], "rt": own["rt_sha"]},
                                      zlog_fs_molecules=N_MOL, base_fs_molecules=N_MOL)))
        return types.SimpleNamespace(returncode=0, stdout="engine lists 12", stderr="")


def _exec_cell32(cell32, inp, work, comp, tmp_path, monkeypatch, is_rerun, fake):
    """セル 32 全体を exec（/kaggle/working を tmp に、/kaggle/input を glob で偽の入力に、subprocess を偽の Engine 2 に）。"""
    src = cell32.replace("/kaggle/working", str(work))
    real_glob = glob.glob

    def fake_glob(pat, recursive=False):
        assert pat.startswith("/kaggle/input/"), pat
        return real_glob(str(inp) + pat[len("/kaggle/input"):], recursive=recursive)
    monkeypatch.setattr(glob, "glob", fake_glob)
    monkeypatch.delenv("E2_OWN", raising=False)
    fake_sp = types.SimpleNamespace(run=fake.run, TimeoutExpired=_real_subprocess.TimeoutExpired)
    g = {"os": os, "sys": sys, "json": json, "time": __import__("time"), "pd": pd, "subprocess": fake_sp,
         "COMP": str(comp), "IS_RERUN": is_rerun, "__name__": "frozen_cell_32"}
    exec(compile(src, "frozen_cell_32_full", "exec"), g)
    return g


FAIL_MODES = ["timeout", "crash_no_lists", "crash_after_lists", "sha_mismatch", "rc_nonzero_only", "partial_cover",
              "no_report"]
EXPECT_REASON = {"timeout": "exception", "crash_no_lists": "returncode", "crash_after_lists": "report status",
                 "sha_mismatch": "own tables", "rc_nonzero_only": "returncode",
                 "partial_cover": "ENG does not cover active molecules", "no_report": "report unreadable"}


def _cell_for(mode, sh):
    s, r = sh["own_sim_rank_rows.npz"], sh["own_rank_train.npz"]
    return _cell32_with("0" * 64 if mode == "sha_mismatch" else s, r)


@pytest.mark.parametrize("mode", FAIL_MODES)
def test_g_commit_stops_on_engine2_failure(tmp_path, monkeypatch, mode):
    """commit（IS_RERUN = False）: どの失敗でも例外で止まる（smoke で見逃さない）。"""
    inp, sh, work, comp = _fake_kaggle(tmp_path)
    fake = FakeEngine2(work, mode)
    if mode == "sha_mismatch":
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            _exec_cell32(_cell_for(mode, sh), inp, work, comp, tmp_path, monkeypatch, False, fake)
        assert fake.calls == 0                               # Engine 2 は起動しない
    else:
        with pytest.raises(AssertionError, match="did not finish") as ei:
            _exec_cell32(_cell_for(mode, sh), inp, work, comp, tmp_path, monkeypatch, False, fake)
        assert EXPECT_REASON[mode] in str(ei.value)
        assert fake.calls == 1


@pytest.mark.parametrize("mode", FAIL_MODES)
def test_g_rerun_falls_back_on_engine2_failure(tmp_path, monkeypatch, mode):
    """再実行（IS_RERUN = True）: 止めずに、active な全分子に空のリストを入れ、report に fallback: true と理由。"""
    inp, sh, work, comp = _fake_kaggle(tmp_path)
    fake = FakeEngine2(work, mode)
    g = _exec_cell32(_cell_for(mode, sh), inp, work, comp, tmp_path, monkeypatch, True, fake)
    assert fake.calls == (0 if mode == "sha_mismatch" else 1)
    assert g["ENG"] == {m: dict(smiles=[], keys=[]) for m in MIDS}       # Engine 2 の出力は使わない（一部も残さない）
    rep = json.loads((work / "engine2_owntables_report.json").read_text())
    assert rep["fallback"] is True and rep["fallback_molecules"] == N_MOL
    assert "#108 09:19Z" in rep["fallback_note"] and "do not read" in rep["fallback_note"]
    assert any(EXPECT_REASON[mode] in json.dumps(r) for r in rep["fallback_reasons"]), rep["fallback_reasons"]


@pytest.mark.parametrize("is_rerun", [False, True])
def test_g_normal_run_keeps_engine2_lists(tmp_path, monkeypatch, is_rerun):
    """正常: ENG は Engine 2 の出力のまま、report に fallback: false（commit でも再実行でも）。"""
    inp, sh, work, comp = _fake_kaggle(tmp_path)
    fake = FakeEngine2(work, "ok")
    g = _exec_cell32(_cell_for("ok", sh), inp, work, comp, tmp_path, monkeypatch, is_rerun, fake)
    assert fake.calls == 1
    assert g["ENG"] == json.loads((work / "eng_lists.json").read_text()) and set(g["ENG"]) == set(MIDS)
    rep = json.loads((work / "engine2_owntables_report.json").read_text())
    assert rep["fallback"] is False and rep["status"] == "ok" and "fallback_reasons" not in rep
    assert rep["tables_sha256"] == {"sim": sh["own_sim_rank_rows.npz"], "rt": sh["own_rank_train.npz"]}


def test_g_cell32_changes_are_additions_only(cell32):
    """セル 32 は元の行を順番どおりすべて残し、足すのは PATCH・try の先頭の 1 行（RUN_ADD）・except の後（EXC_ADD）だけ。"""
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    old32 = _src(_frozen(nb)["cells"][M.CHANGED_FROZEN])
    assert cell32.count(M.RUN_MARK + M.RUN_ADD) == 1 and cell32.count(M.EXC_OLD + M.EXC_ADD) == 1
    # 時間切れの上限は元のまま
    assert cell32.count("timeout=4 * 3600") == 1 and old32.count("timeout=4 * 3600") == 1
    # 再実行の守りは commit で止める形を残す
    assert "if _e2why and not IS_RERUN:\n    raise AssertionError" in M.EXC_ADD
    assert "    if not IS_RERUN:\n        raise\n" in M.PATCH


# ------------------------------------------------------------------------------------------------
# (h) 戻った後の ENG（全分子が空のリスト）で、ENG を読む後段（凍結セル 41 の ICE_UNION、45、47・48、49、50、51、52、
#     53、54）の assert と枝が IS_RERUN = True の偽のスコープ・12 分子で通る。依存の多いセルは ENG を読む行を
#     凍結コードから一字一句そのまま抜き出して（存在を assert してから）実行する
# ------------------------------------------------------------------------------------------------
def _lines(src, first, last):
    """凍結セルの src から、first で始まる行から last を含む行までを取り出し dedent する（どちらも 1 回だけ現れる）。"""
    assert src.count(first) == 1, first
    a = src.index(first); a = src.rindex("\n", 0, a) + 1
    assert src.count(last, a) >= 1, last
    b = src.index(last, a); b = src.find("\n", b); b = len(src) if b < 0 else b
    return textwrap.dedent(src[a:b]) + "\n"


SMI = ["CCN", "CCCN", "CCCCN", "CCCCCN", "CCCCCCN", "CC(C)N", "CC(C)CN", "c1ccccc1N", "c1ccccc1O", "CCOC", "CCCOC", "OCCN"]


def test_h_downstream_cells_pass_with_fallback_eng(tmp_path, monkeypatch):
    pytest.importorskip("rdkit")
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    F = [_src(c) for c in _frozen(nb)["cells"]]
    # 戻りの ENG はセル 32 の EXC_ADD と同じ式で作る
    ENG = {m: dict(smiles=[], keys=[]) for m in MIDS}
    monkeypatch.chdir(tmp_path)
    te = pd.DataFrame(dict(molecule_id=MIDS, nm=[100.0 + k for k in range(N_MOL)], adduct="[M+H]+"))
    sub0 = pd.DataFrame(dict(molecule_id=MIDS, smiles=[";".join([SMI[k], SMI[(k + 1) % N_MOL]]) for k in range(N_MOL)]))
    sub0.to_csv("submission.csv", index=False)
    comp = tmp_path / "comp"; comp.mkdir()
    sub0[["molecule_id"]].assign(smiles="CCO").to_csv(comp / "sample_submission.csv", index=False)
    for d in ("ice_work", "gl_work"):
        os.makedirs(d); (tmp_path / d / f"{d[:-5]}_out.json.meta.json").write_text(json.dumps({"status": "ok"}))
    chem = types.SimpleNamespace(score_key=lambda s: s)
    g = {"__name__": "frozen", "os": os, "json": json, "glob": glob, "np": __import__("numpy"), "pd": pd,
         "time": __import__("time"), "te": te, "ENG": ENG, "IS_RERUN": True, "SMOKE_N": 12, "COMP": str(comp),
         "chem": chem, "ICE_SCORES": {}, "n_err": 0, "PC": {}, "BASE_TAIL_CANDIDATES": {}, "T0": 0.0,
         "rdkit": types.SimpleNamespace(__version__="2026.03.3"), "display": lambda *a, **k: None,
         "gl_of": lambda m: {}, "sha256": lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()}

    # セル 41（ICE_UNION）: 空のリストの分子は飛ばされ、足される候補は 0
    ice_union = _lines(F[41], "    if ENG:                              # ADDED (ICE_UNION)",
                       "print('ICE_UNION: engine candidates added to ICE input:', n_add, flush=True)")
    g41 = dict(g, mids=list(MIDS), BASE={m: ([SMI[0]], ["k"], 0.5, None, ["C2H7N"]) for m in MIDS}, TOPN=10,
               cands={m: [] for m in MIDS}, ice_fuse=types.SimpleNamespace(ice_candidates=lambda *a: pytest.fail("called")))
    exec(compile(ice_union, "frozen_41_ice_union", "exec"), g41)
    assert g41["n_add"] == 0 and g41["cands"] == {m: [] for m in MIDS}

    # セル 45 全体: if ENG: の枝に入るが、どの分子も空なので v4g のリストのまま書き直される
    exec(compile(F[45], "frozen_45", "exec"), g)
    sub45 = pd.read_csv("submission.csv")
    assert sub45.equals(sub0) and g["FUSION_OVERFLOW"] == {}
    assert g["PREFUSION_LISTS"] == {m: s.split(";") for m, s in zip(sub0.molecule_id, sub0.smiles)}

    # セル 47・48 全体: 凍結セル 48 の assert _active <= set(map(str, ENG)) と IS_RERUN の守りを通る
    exec(compile(F[47], "frozen_47", "exec"), g)
    assert "assert _active <= set(map(str, ENG)), 'Engine 2 did not cover all active molecules'" in F[48]
    exec(compile(F[48], "frozen_48", "exec"), g)
    sub48 = pd.read_csv("submission.csv")
    assert list(sub48.molecule_id) == MIDS and sub48.smiles.str.split(";").map(len).between(1, 25).all()
    assert [s.split(";")[:2] for s in sub48.smiles] == [s.split(";") for s in sub0.smiles]   # 元の順は保たれる

    # セル 49 全体: Engine 2 が空なので融合の枝に入らず、変化 0
    exec(compile(F[49], "frozen_49", "exec"), g)
    assert g["_report"]["changed_molecules"] == 0

    # セル 50（E004 postflight）の ENG を読む行（再実行の側）
    e50 = (_lines(F[50], "base, eng, pc = scope['BASE'], scope['ENG'], scope['PC']",
                  "assert processed <= set(map(str, eng))")
           + _lines(F[50], "if not full:\n        assert all(v[0] for v in base.values())",
                    "assert all(eng[mid]['smiles'] and pc.get(mid, {}).get('pc') for mid in processed)")
           + "engine2 = {mid: eng[mid] for mid in sorted(processed)}\n"
           + "n2 = sum(bool(eng[mid]['smiles']) for mid in processed)\n")
    assert "'engine2': {mid: eng[mid] for mid in sorted(processed)}," in F[50]
    assert "engine2_nonempty=sum(bool(eng[mid]['smiles']) for mid in processed)," in F[50]
    g50 = dict(scope=dict(g, BASE={m: ([SMI[0]],) for m in MIDS}), processed=set(MIDS), full=True, pc={})
    exec(compile(e50, "frozen_50_eng", "exec"), g50)
    assert g50["n2"] == 0 and len(g50["engine2"]) == N_MOL

    # セル 51（FRAG_GAP）: ENG を読む 2 行
    e51 = _lines(F[51], "_e = ENG.get(_mid) or {}", "_c = [s for s in dict.fromkeys(list(_lst) + list(_e.get('smiles', []))) if s and s != 'CCO']")
    for mid, lst in g["PREFUSION_LISTS"].items():
        g51 = dict(ENG=ENG, _mid=mid, _lst=lst); exec(compile(e51, "frozen_51_eng", "exec"), g51)
        assert g51["_c"] == lst

    # セル 52（factorial overlay）: ENG が空なら元のリストを返す枝
    e52 = ("def _f(scope, mid, original):\n"
           + textwrap.indent(_lines(F[52], 'engine = scope["ENG"].get(mid)',
                                    'if mid not in scope["PREFUSION_LISTS"] or not engine or not engine.get("smiles"):'), "    ")
           + "        return 'original'\n    return 'fused'\n")
    g52 = {}; exec(compile(e52, "frozen_52_eng", "exec"), g52)
    assert all(g52["_f"](g, m, []) == "original" for m in MIDS)

    # セル 53 全体（rank-2 の挿入）: Engine 2 の先頭が無いので挿入 0、status ok、ファイルは変わらない
    before53 = open("submission.csv", "rb").read()
    exec(compile(F[53], "frozen_53", "exec"), g)
    assert g["RANK2_REPORT"]["status"] == "ok" and g["RANK2_REPORT"]["inserted"] == 0
    assert open("submission.csv", "rb").read().replace(b"\r\n", b"\n") == before53.replace(b"\r\n", b"\n")

    # セル 54（receipt）の ENG を読む 2 つの式
    assert "engine2_lists=len(scope['ENG'])" in F[54]
    assert "engine2_nonempty=sum(bool(scope['ENG'].get(mid, {}).get('smiles')) for mid in processed)," in F[54]
    scope = g; processed = set(MIDS)
    assert len(scope['ENG']) == N_MOL
    assert sum(bool(scope['ENG'].get(mid, {}).get('smiles')) for mid in processed) == 0
