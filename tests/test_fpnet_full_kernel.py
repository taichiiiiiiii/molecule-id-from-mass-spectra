"""Issue #82: 全 train の FPNet を作り直すカーネルの生成と、ログからの外挿。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
import fpnet_full_report as fr  # noqa: E402
import make_fpnet_full_kernel as mk  # noqa: E402

# キットの train_fold0.log と同じ形の行（数は作りもの）
LOG = """[Data] train spectra 2,539,608 groups 371,921 pool 710,701 nbits 10226
10:00:30 {'hold': 3000, 'np': 499, 'train': 3000} train spectra 2539608
10:00:40 parameters 49686780 mean bit frequency 0.0
10:01:20 step 200 lr 3.00e-05 loss 5.3268 rank 4.6200 bce 0.6929 el 0.1392 acc 0.0788 4.53 it/s queue 64
10:04:00 step 1000 lr 1.50e-04 loss 3.5767 rank 2.8833 bce 0.6893 el 0.0420 acc 0.3283 4.35 it/s queue 12
10:07:20 step 2000 lr 3.00e-04 loss 3.0087 rank 2.3177 bce 0.6875 el 0.0350 acc 0.4546 4.49 it/s queue 3
10:07:30 EVAL {"step": 2000, "hours": 0.1, "final": true, "hold_n": 3000}
10:07:31 ALL_DONE
"""


def test_parse_log_reads_kit_lines():
    p = fr.parse_log(LOG)
    assert p["data"] == dict(train_spectra=2539608, groups=371921, pool=710701, nbits=10226)
    assert p["sets"]["train_spectra"] == 2539608
    assert [s[1] for s in p["steps"]] == [200, 1000, 2000]
    assert p["steps"][1][3] == 12 and p["eval"]["step"] == 2000


def test_real_kit_log_format_if_present():
    log = ROOT / "data/ext/foldsafe/train_fold0.log"
    if not log.exists():  # data/ は git に無い（CI では飛ばす）
        return
    p = fr.parse_log(log.read_text())
    assert p["data"]["train_spectra"] == 1982405 and p["steps"][-1][1] == 82800
    assert p["eval"]["step"] == 82855 and p["eval"]["final"] is True


def test_steady_rate_and_preregistered_formula():
    p = fr.parse_log(LOG)
    r = fr.steady_rate(p["steps"])
    assert abs(r - 1000 / 200) < 1e-12          # step 1000 -> 2000 in 200 s
    s = fr.summarize(LOG, 10 * 3600)            # process started 10:00:00
    e = s["extrapolation"]
    assert e["setup_seconds"] == 40 and e["eval_seconds"] == 10
    assert abs(e["total_seconds"] - (40 + 82855 / 5 + 16 * 10)) < 1e-9
    assert e["one_session"] is True and e["lower_bound"] is False
    assert s["min_queue"] == 3


def test_guard_stop_without_eval_is_lower_bound_and_midnight_wrap():
    log = "\n".join(l for l in LOG.splitlines() if "EVAL" not in l and "ALL_DONE" not in l)
    log = log.replace("10:04:00 step 1000", "23:59:50 step 1000").replace("10:07:20 step 2000", "00:03:10 step 2000")
    s = fr.summarize(log, 10 * 3600)
    assert s["extrapolation"]["lower_bound"] is True and s["extrapolation"]["eval_seconds"] == 0
    assert abs(s["extrapolation"]["steady_it_per_s"] - 5.0) < 1e-12


def test_session_limit_over_11h():
    e = fr.extrapolate(0, 82855 / (12 * 3600), 0)
    assert e["one_session"] is False and abs(e["total_hours"] - 12) < 1e-9


def test_notebook_compiles_and_passes_only_fixed_args():
    nb = mk.build_notebook(mk.SMOKE_STEPS)
    src = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for s in src:
        compile(s, "cell", "exec")
    ns = {}
    exec(src[0], ns)
    assert ns["SMOKE_STEPS"] == 2000 and ns["PROD_STEPS"] == 82855
    assert ns["FIXED_ARGS"] == ["--holdout", "", "--hours", "1000"]
    compile(ns["REPORT_SRC"], "report", "exec")
    main = src[1]
    for flag in ("--bs", "--lr", "--wd", "--warmup", "--merge_p", "--w_bce", "--w_el", "--workers", "--seed", "--layers",
                 "--d'", "--eval_every", "--tag", "--bias_init", "--segment_minutes", "--smoke_steps"):
        assert flag not in main, flag
    assert "'--max_steps', str(STEPS)" in main
    assert "check_inputs" not in main          # no hand-made gate json
    assert mk.build_notebook(0)["cells"][1]["source"].startswith("SMOKE_STEPS = 0")


def test_expected_hashes_match_issue_table():
    e = mk.EXPECT
    assert e[("dmitriigluzdov/casmi26-fold-safe-fpnet", "train_fpnet.py")].startswith("46b4dceebb2b")
    assert e[("ahmedberatozer/casmi26-v4b-models", "code/casmi/fpnet.py")] == \
        "0ea6e372d1f045a877d350234c2ccf5d93fdd98cb23bd4c22376bf7e99547608"   # MODEL_INPUTS.json fpnet_code_sha256
    assert all(len(v) == 64 for v in e.values())
    assert set(mk.WORK_LINKS) == {"spec_off.npy", "spec_mz.npy", "spec_it.npy", "spec_meta.parquet", "split.parquet",
                                  "train_structs.parquet", "pool_meta.parquet", "pool_fp.npy", "fp_bits.npy"}
    md = mk.metadata()
    assert md["is_private"] and md["enable_gpu"] and md["id"] == mk.SLUG
    assert sorted(md["dataset_sources"]) == sorted({s for s, _ in mk.WORK_LINKS.values()} |
                                                   {"ahmedberatozer/casmi26-v4b-models"})


def test_committed_kernel_matches_generator():
    kdir = ROOT / "kernels/fpnet_full_train"
    nb = json.loads((kdir / "notebook.ipynb").read_text())
    assert nb == json.loads(json.dumps(mk.build_notebook(mk.SMOKE_STEPS)))
    assert json.loads((kdir / "kernel-metadata.json").read_text()) == mk.metadata()
