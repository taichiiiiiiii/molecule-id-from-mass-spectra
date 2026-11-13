import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sim1_trial", ROOT / "scripts/train/sim1_trial.py")
st = importlib.util.module_from_spec(_spec)
sys.modules["sim1_trial"] = st
_spec.loader.exec_module(st)
_spec2 = importlib.util.spec_from_file_location("make_sim1_trial_kernel", ROOT / "scripts/train/make_sim1_trial_kernel.py")
mk = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(mk)

H = lambda c: c * 64
README = """# x
| file | bytes | SHA256 |
|---|---|---|
| cache/spec_mz.npy | 532,912,560 | %s |
| cache/spec_it.npy | 532,912,560 | %s |
| cache/spec_off.npy | 20,317,000 | %s |
| cache/spec_meta.parquet | 54,036,239 | %s |
| cache/split.parquet | 9,641,941 | %s |
""" % (H("1"), H("2"), H("3"), H("4"), H("5"))


def _v4g():
    return {"files": {f"v2_pool/{n}": {"bytes": i + 10, "sha256": H("a")} for i, n in enumerate(st.POOL_FILES)}}


def _v4b():
    f = {"fe_A.pt": H("b"), "fe_B.pt": H("c"), "fe_D.pt": H("d"), "ranker_v4b.pkl": H("e")}
    for i in range(st.N_V1_CODE):
        f[f"code/casmi/m{i}.py"] = H("f")
    for n in st.FE_V4_CODE:
        f[f"code/fe_v4/{n}"] = H("0")
    f["code/fe_v4/fams/frag2.py"] = H("9")   # 段 1 では使わない
    return {"files": f}


def test_parse_spec_readme_reads_full_hashes_and_bytes():
    sp = st.parse_spec_readme(README)
    assert sp["spec_mz.npy"] == (532912560, H("1")) and sp["split.parquet"] == (9641941, H("5")) and len(sp) == 5


def test_expected_inputs_cover_stage1_files_with_sources():
    e = st.expected_inputs(README, _v4g(), _v4b())
    assert len(e) == 15
    assert e["spec_off.npy"]["sha256"] == H("3") and e["spec_off.npy"]["rel"] == "cache/spec_off.npy"
    assert e["pool_fp.npy"]["dataset"] == "ahmedberatozer/casmi26-v2-pool"
    assert e["analysis/dreams_ft/s2/pilot_d1_step20k_ema.pt"]["rel"] == "fe_D.pt"
    assert e["fpnet_a.pt"]["sha256"] == H("b")


def test_missing_record_is_marked_not_stopped():
    g = _v4g(); del g["files"]["v2_pool/fp_bits.npy"]
    e = st.expected_inputs(README, g, _v4b())
    assert e["fp_bits.npy"]["sha256"] is None and e["fp_bits.npy"]["record"] == "記録なし"
    assert st.compare(e, {k: dict(sha256="zz", bytes=1) for k in e if k == "fp_bits.npy"}) != []  # 他は見つからない
    assert not any(b["name"] == "fp_bits.npy" for b in st.compare(e, {"fp_bits.npy": dict(sha256="zz", bytes=1)}))


def test_expected_code_is_23_files_under_readme_layout():
    c = st.expected_code(_v4b())
    assert len(c) == 23
    assert "work/analysis/v1fe/code/casmi/m0.py" in c and "work/analysis/fe_v4/ctxlib.py" in c
    assert not any("fams" in k for k in c)


def test_compare_flags_hash_and_size():
    exp = {"a": dict(sha256=H("1"), bytes=5), "b": dict(sha256=None, bytes=7)}
    assert st.compare(exp, {"a": dict(sha256=H("1"), bytes=5), "b": dict(sha256="x", bytes=7)}) == []
    bad = st.compare(exp, {"a": dict(sha256=H("2"), bytes=5), "b": dict(sha256="x", bytes=8)})
    assert [b["name"] for b in bad] == ["a", "b"]


def test_stage_timeout_caps_at_20min_and_respects_budget():
    assert st.stage_timeout(60) == 20 * 60
    assert st.stage_timeout(20 * 60) == 25 * 60 - 20 * 60 - 120
    assert st.stage_timeout(24 * 60) <= 0


LOG = """[Library] 2,539,608 spectra, 275,810 structures, 133,228,108 peaks (0.6s)
[sim1x] shard 0: 40 structures, 0 done, seed 21, init 4s
  shard 0: 10/40 structs, 50 queries, 64s (1.28s/q, ETA 1.57h)
  shard 0: 20/40 structs, 100 queries, 179s (1.79s/q, ETA 2.17h)
"""


def test_parse_and_summarize_timed_out_run_uses_last_progress():
    p = st.parse_sim_log(LOG.splitlines())
    assert p["header"]["structs"] == 40 and p["header"]["init_s"] == 4 and p["done"] is None
    s = st.summarize(p)
    assert s["structs"] == 20 and s["queries"] == 100 and s["s_per_struct"] == 179 / 20 and "打ち切り" in s["source"]


def test_summarize_completed_run_uses_done_line_and_meta_queries():
    lines = LOG.splitlines() + ["[sim1x] done shard 0: 200 new queries 641s"]
    s = st.summarize(st.parse_sim_log(lines), rows_meta_queries=198)
    assert s["structs"] == 40 and s["queries"] == 198 and s["seconds"] == 641.0 and s["s_per_struct"] == 641 / 40


def test_summarize_without_progress_reports_zero():
    s = st.summarize(st.parse_sim_log(["Traceback ..."]))
    assert s["structs"] == 0 and s["s_per_struct"] is None


def test_extrapolate_formula():
    e = st.extrapolate(2.0, 6.0)
    assert e["by_parallel"]["1"]["by_query_h"] == round(2.0 * 15701 / 3600, 3)
    assert e["by_parallel"]["2"]["by_struct_h"] == round(6.0 * 5350 / 2 / 3600, 3)
    corr = (36099 / 15701) / (641 / 200)
    assert abs(e["correction"] - corr) < 1e-4
    assert e["by_parallel"]["6"]["by_query_corrected_h"] == round(2.0 * corr * 15701 / 6 / 3600, 3)


def test_generated_notebook_compiles_and_metadata(tmp_path):
    r = tmp_path / "README.md"; r.write_text(README)
    g = tmp_path / "g.json"; g.write_text(json.dumps(_v4g()))
    b = tmp_path / "b.json"; b.write_text(json.dumps(_v4b()))
    nb, meta = mk.build(r, g, b)
    assert mk.check(nb) == 3
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["dataset_sources"] == mk.DATASETS and "docker_image" not in meta
    src = "\n".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert "--max_structs" in src and "'40'" in src and "sha256_file" in src
    # 足すのは --shard 0 --nshard 6 --max_structs 40 だけ
    assert mk.TRIAL_ARGS == ["--shard", "0", "--nshard", "6", "--max_structs", "40"]


# ---- 2 シャード同時（2026-11-14 夜） ----
_spec3 = importlib.util.spec_from_file_location("sim1_trial2", ROOT / "scripts/train/sim1_trial2.py")
st2 = importlib.util.module_from_spec(_spec3)
_spec3.loader.exec_module(st2)


def _inputs(tmp_path):
    r = tmp_path / "README.md"; r.write_text(README)
    g = tmp_path / "g.json"; g.write_text(json.dumps(_v4g()))
    b = tmp_path / "b.json"; b.write_text(json.dumps(_v4b()))
    return r, g, b


def test_default_build_keeps_committed_cells_except_constants(tmp_path):
    """既定（1 シャード）の生成物は昼にコミットしたノートと同じ（定数のセルは実の記録から作るので除く）。"""
    nb, meta = mk.build(*_inputs(tmp_path))
    committed = json.loads((ROOT / "kernels/casmi26-v4b-sim1-trial/notebook.ipynb").read_text())
    assert len(nb["cells"]) == len(committed["cells"]) == 4
    for i in (0, 1, 3):
        assert nb["cells"][i] == committed["cells"][i], i
    c_new, c_old = nb["cells"][2]["source"].splitlines(), committed["cells"][2]["source"].splitlines()
    assert [l for l in c_new if not l.startswith(("EXP_DATA", "EXP_CODE", "EMBED_SOURCES"))] == \
           [l for l in c_old if not l.startswith(("EXP_DATA", "EXP_CODE", "EMBED_SOURCES"))]
    assert nb["metadata"] == committed["metadata"]
    assert meta == json.loads((ROOT / "kernels/casmi26-v4b-sim1-trial/kernel-metadata.json").read_text())


def test_two_shard_notebook_compiles_and_metadata(tmp_path):
    nb, meta = mk.build(*_inputs(tmp_path), two_shards=True)
    assert mk.check(nb) == 4
    assert meta["id"] == "taichiiiii/casmi26-v4b-sim1-trial2" and meta["is_private"] and meta["enable_gpu"]
    assert meta["machine_shape"] == "NvidiaTeslaT4" and meta["dataset_sources"] == mk.DATASETS
    run = nb["cells"][4]["source"]
    # 照合までは昼と同じコード、段 1 から先だけが違う
    day_head = mk.RUNNER.lstrip("\n").split("    # 5. 段 1\n")[0]
    assert run.split("    # 5. 段 1 を 2 シャード同時に")[0].replace("BUDGET = BUDGET2_S", "BUDGET = 25 * 60") \
        .replace("の 2 シャード同時の T4 試走。ノート全体 60 分・段 1 は 45 分で打ち切り", "の T4 試走。ノート全体 25 分で打ち切り") == day_head
    assert "CUDA_VISIBLE_DEVICES" in nb["cells"][2]["source"] and "DAY_SHARD0 = " in nb["cells"][3]["source"]
    # 埋め込みの定数・部品を実際に実行し、2 シャードの引数を確かめる
    ns = {}
    exec(compile(nb["cells"][1]["source"] + "\n" + nb["cells"][2]["source"], "<mods>", "exec"), ns)
    assert ns["shard_args"](0) == mk.TRIAL_ARGS and ns["shard_args"](1) == ["--shard", "1", "--nshard", "6", "--max_structs", "40"]
    assert ns["SHARD_GPU"] == ((0, 0), (1, 1))


def test_shard_env_sets_gpu_and_tc_threads():
    e = st2.shard_env({"PATH": "/x", "CUDA_VISIBLE_DEVICES": "0,1"}, mk.TC_ENV, 1)
    assert e["CUDA_VISIBLE_DEVICES"] == "1" and e["OMP_NUM_THREADS"] == "2" and e["PATH"] == "/x"


def test_stage2_timeout_45min_cap_and_60min_budget():
    assert st2.stage2_timeout(60) == 45 * 60
    assert st2.stage2_timeout(20 * 60) == 60 * 60 - 20 * 60 - 180
    assert st2.stage2_timeout(59 * 60) <= 60


def test_gpu_query_parse_and_peak():
    s = st2.parse_gpu_query("0, 1397, 31\n1, 1200, 45\nbad line\n")
    assert s == {0: (1397.0, 31.0), 1: (1200.0, 45.0)}
    p = st2.update_gpu_peak({}, s)
    p = st2.update_gpu_peak(p, {0: (1000.0, 50.0)})
    assert p == {"0": dict(mem_mb=1397.0, util=50.0), "1": dict(mem_mb=1200.0, util=45.0)}


def test_wait_all_kills_only_late_process():
    clock = [0.0]
    class P:
        def __init__(self, dur): self.dur = dur
    def wait_one(p, timeout):
        if p.dur > clock[0] + timeout:
            clock[0] += timeout
            raise TimeoutError
        clock[0] = max(clock[0], p.dur)
        return 0
    res = st2.wait_all({0: P(10), 1: P(100)}, 50, lambda: clock[0], wait_one, lambda p: -15)
    assert res == {0: (0, False), 1: (-15, True)}


def test_slowdown_and_signature():
    assert st2.slowdown(1857.6858239173890, 928.8429119586945) == 2.0
    assert st2.slowdown(None, 1.0) is None
    a = st2.query_signature([6, 6], ["c1", "c2"], [[1, 2], [3]])
    assert a == st2.query_signature([6, 6], ["c1", "c2"], [(1, 2), (3,)]) != st2.query_signature([6, 6], ["c2", "c1"], [[1, 2], [3]])
    assert st2.t_query_sum([1.0, 2.0, 3.0]) == dict(n=3, sum=6.0, mean=2.0)


def test_cpu_model_from_lscpu():
    t = "Architecture: x86_64\nCPU(s):  4\nModel name:   Intel(R) Xeon(R) CPU @ 2.00GHz\nNUMA node0 CPU(s): 0-3\n"
    assert st2.cpu_model(t) == ("Intel(R) Xeon(R) CPU @ 2.00GHz", 4)
