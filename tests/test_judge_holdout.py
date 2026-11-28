"""Issue #249: scripts/local_eval/judge_holdout.py（12/6・12/11 の holdout の判定）のテスト。合成の per_molecule だけを使う。"""
import builtins
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import judge_holdout as J  # noqa: E402

SETS = [s for s, _ in J.HOLDOUT_SETS]


# ---------- 合成の入力 ----------

def _mk_holdout(root: Path, shifts, n_by_set=None, n_msg1_by_set=None, msg1_shift=0.0, seed=0):
    """基準 rr はランダム、版 A の 3 本 = 基準 + shift（msg1 の分子は + msg1_shift）。差が定数なので区間 = shift。"""
    rng = np.random.default_rng(seed)
    n_by_set = n_by_set or {s: 200 for s in SETS}
    n_msg1_by_set = n_msg1_by_set or {s: 10 for s in SETS}
    bdir = root / "base"
    bdir.mkdir(parents=True)
    rows, m0, m1 = [], [], []
    for s in SETS:
        n, k = n_by_set[s], n_msg1_by_set[s]
        mids = [f"{s[:3]}_{i:05d}" for i in range(n)]
        rb = rng.choice([0.0, 0.25, 0.5, 1.0], size=n)   # 3 本の平均が丸めずに元に戻る値
        pd.DataFrame({"molecule_id": mids, "rr": rb}).to_csv(bdir / f"{s}_rr.csv", index=False)
        for i, (m, r) in enumerate(zip(mids, rb)):
            is1 = i < k
            (m1 if is1 else m0).append((s, m))
            rows.append((s, m, r + shifts[s] + (msg1_shift if is1 else 0.0)))
    cands = []
    for sd in J.SEEDS:
        d = root / f"cand_s{sd}"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"seed": sd, "features_sha256": "f" * 64}))
        pd.DataFrame(rows, columns=["set", "molecule_id", "rr"]).to_csv(d / "per_molecule.csv", index=False)
        cands.append(d)
    pd.DataFrame(m0, columns=["set", "molecule_id"]).to_csv(root / "m0.csv", index=False)
    pd.DataFrame(m1, columns=["set", "molecule_id"]).to_csv(root / "m1.csv", index=False)
    pins = {s: J.sha256_file(bdir / f"{s}_rr.csv") for s in SETS}
    mpins = {"msg0": J.sha256_file(root / "m0.csv"), "msg1": J.sha256_file(root / "m1.csv")}
    return dict(cands=cands, bdir=bdir, m0=root / "m0.csv", m1=root / "m1.csv", pins=pins, mpins=mpins)


def _run(root, h, log, label="interim_1206"):
    return J.run_holdout(h["cands"], h["bdir"], h["m0"], h["m1"], label, root / f"out_{label}", reads_log=log,
                         baseline_pins=h["pins"], msg_pins=h["mpins"])


# ---------- (i) 規則どおりの合否 ----------

@pytest.mark.parametrize("shifts,ok,fail_i,pos", [
    ({"v3_c1u_t10": 0.01, "enveda_v1u_t10": 0.0, "v3_c2_t10": 0.0, "v3_c2nu_t10": 0.0}, True, [], ["v3_c1u_t10"]),
    ({"v3_c1u_t10": 0.05, "enveda_v1u_t10": -0.03, "v3_c2_t10": 0.0, "v3_c2nu_t10": 0.0}, False, ["enveda_v1u_t10"],
     ["v3_c1u_t10"]),
    ({"v3_c1u_t10": 0.0, "enveda_v1u_t10": 0.0, "v3_c2_t10": 0.0, "v3_c2nu_t10": -0.01}, False, [], []),
    ({"v3_c1u_t10": -0.01, "enveda_v1u_t10": -0.01, "v3_c2_t10": 0.03, "v3_c2nu_t10": -0.015}, True, [],
     ["v3_c2_t10"]),
])
def test_rule_pass_fail(tmp_path, shifts, ok, fail_i, pos):
    h = _mk_holdout(tmp_path, shifts)
    res, text = _run(tmp_path, h, tmp_path / "reads.jsonl")
    assert res["pass_"] is ok
    assert res["cond_i_fail"] == fail_i
    assert res["cond_ii_sets"] == pos
    for r in res["sets"]:
        assert r["lo"] == pytest.approx(shifts[r["set"]], abs=1e-12)
        assert r["hi"] == pytest.approx(shifts[r["set"]], abs=1e-12)
    assert ("判定: 条件を満たす\n" in text) is ok
    if pos == ["v3_c2_t10"]:
        assert "（v3_c2_t10 だけ）" in text


def test_rule_boundary_unrounded():
    """下限ちょうど −0.02 は満たす。−0.02 より僅かに下（丸めると −0.0200）は満たさない。下限 0 は L > 0 ではない。"""
    n = 256
    idx = lambda s: [(s, f"m{i}") for i in range(n)]  # noqa: E731
    def go(vals):
        allidx = [m for s in SETS for m in idx(s)]
        base = pd.DataFrame({"rr": 0.0}, index=pd.MultiIndex.from_tuples(allidx))
        cand = pd.DataFrame({"rr": np.concatenate([np.full(n, vals[s]) for s in SETS])},
                            index=pd.MultiIndex.from_tuples(allidx))
        groups = [(s, s, idx(s)) for s in SETS]
        return J.judge_groups(groups, cand, base, set(allidx), set(), exact_lists=False)
    r = go({"v3_c1u_t10": -0.02, "enveda_v1u_t10": 0.0, "v3_c2_t10": 0.001, "v3_c2nu_t10": 0.0})
    assert r["cond_i"] and r["cond_ii_sets"] == ["v3_c2_t10"] and r["pass_"]
    r = go({"v3_c1u_t10": -0.02000001, "enveda_v1u_t10": 0.0, "v3_c2_t10": 0.001, "v3_c2nu_t10": 0.0})
    assert r["cond_i_fail"] == ["v3_c1u_t10"] and not r["pass_"]
    r = go({s: 0.0 for s in SETS})
    assert r["cond_i"] and not r["cond_ii"] and not r["pass_"]


def test_constants_fixed():
    import evaluate
    assert J.GATE == -0.02 and J.MIN_REMAIN == 150 and J.MAX_READS == 2 and J.SEEDS == (0, 1, 2)
    assert J.N_BOOT == evaluate.N_BOOT == 2000 and J.BOOT_SEED == evaluate.BOOT_SEED == 0
    assert J.paired_bootstrap is evaluate.paired_bootstrap
    assert SETS == ["v3_c1u_t10", "enveda_v1u_t10", "v3_c2_t10", "v3_c2nu_t10"]
    lists = json.loads((ROOT / "docs/experiments/holdout_msg_lists.json").read_text())
    assert J.HOLDOUT_MSG_SHA256["msg0"] == lists["holdout_msg0_eval.csv"]["sha256"]
    assert J.HOLDOUT_MSG_SHA256["msg1"] == lists["holdout_msg1_exclude.csv"]["sha256"]
    assert lists["min_remain"] == J.MIN_REMAIN


# ---------- (ii) 150 分子未満の集合は全集合 ----------

def test_min_remain_switch(tmp_path):
    """v3_c2nu_t10: 全 300・msg1 151 → msg0 149 < 150 で全集合。v3_c2_t10: 全 300・msg1 150 → msg0 150 で部分集合。"""
    shifts = {s: 0.01 for s in SETS}
    n = {s: 300 for s in SETS}
    k = {"v3_c1u_t10": 3, "enveda_v1u_t10": 0, "v3_c2_t10": 150, "v3_c2nu_t10": 151}
    h = _mk_holdout(tmp_path, shifts, n, k, msg1_shift=-0.05)
    res, text = _run(tmp_path, h, tmp_path / "reads.jsonl")
    by = {r["set"]: r for r in res["sets"]}
    assert by["v3_c2_t10"]["judge_on"] == "subset" and by["v3_c2_t10"]["n"] == 150
    assert by["v3_c2_t10"]["delta"] == pytest.approx(0.01)
    assert by["v3_c2nu_t10"]["judge_on"] == "full" and by["v3_c2nu_t10"]["n"] == 300
    assert by["v3_c2nu_t10"]["n_msg1"] == 151
    assert by["v3_c2nu_t10"]["delta"] == pytest.approx(0.01 - 0.05 * 151 / 300)
    assert by["enveda_v1u_t10"]["judge_on"] == "subset" and by["enveda_v1u_t10"]["n"] == 300
    assert "msg0 が 149 分子で 150 未満なので全集合で判定・MSG との重なり 151" in text


def test_lists_must_partition(tmp_path):
    h = _mk_holdout(tmp_path, {s: 0.0 for s in SETS})
    d = pd.read_csv(h["m0"]).iloc[1:]          # 1 分子をどちらの一覧からも落とす
    d.to_csv(h["m0"], index=False)
    h["mpins"]["msg0"] = J.sha256_file(h["m0"])
    with pytest.raises(J.JudgeError, match="msg0 にも msg1 にも無い"):
        _run(tmp_path, h, tmp_path / "reads.jsonl")


# ---------- (iii) 3 回目の読み出しで止まる ----------

def test_third_read_stops_before_opening(tmp_path, monkeypatch):
    h = _mk_holdout(tmp_path, {s: 0.01 for s in SETS})
    log = tmp_path / "reads.jsonl"
    r1, _ = _run(tmp_path, h, log, "interim_1206")
    r2, _ = _run(tmp_path, h, log, "final_1211")
    assert (r1["read_no"], r2["read_no"]) == (1, 2)
    assert J.count_reads(log) == 2
    recs = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["label"] for x in recs] == ["interim_1206", "final_1211"]

    def boom(*a, **k):
        raise AssertionError("3 回目に holdout の入力を開いた")
    for f in ("load_list", "load_baseline", "load_runs", "sha256_file"):
        monkeypatch.setattr(J, f, boom)
    opened = []
    real_open = builtins.open
    def spy(p, *a, **k):
        opened.append(str(p))
        return real_open(p, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    with pytest.raises(J.ReadLimitError):
        _run(tmp_path, h, log, "third")
    monkeypatch.setattr(builtins, "open", real_open)
    assert J.count_reads(log) == 2
    assert not any(str(h["bdir"]) in p or "cand_s" in p for p in opened)
    assert not (tmp_path / "out_third").exists()


def test_bad_baseline_sha_stops_and_counts(tmp_path):
    h = _mk_holdout(tmp_path, {s: 0.0 for s in SETS})
    h["pins"]["v3_c2_t10"] = "0" * 64
    log = tmp_path / "reads.jsonl"
    with pytest.raises(J.JudgeError, match="sha256"):
        _run(tmp_path, h, log)
    assert J.count_reads(log) == 1             # 開いた後に止まった回も 1 回（保守側）


def test_missing_input_does_not_count(tmp_path):
    h = _mk_holdout(tmp_path, {s: 0.0 for s in SETS})
    (h["bdir"] / "v3_c2_t10_rr.csv").unlink()
    log = tmp_path / "reads.jsonl"
    with pytest.raises(J.JudgeError, match="入力が無い"):
        _run(tmp_path, h, log)
    assert not log.exists()


def test_production_pins_reject_synthetic(tmp_path):
    """引数で固定値を外せない: CLI の経路（固定の sha256）では合成の基準は通らない。"""
    h = _mk_holdout(tmp_path, {s: 0.0 for s in SETS})
    with pytest.raises(J.JudgeError, match="sha256"):
        J.run_holdout(h["cands"], h["bdir"], h["m0"], h["m1"], "x", tmp_path / "o", reads_log=tmp_path / "r.jsonl")


# ---------- holdout のパスを与えないと holdout を読めない ----------

def test_cli_without_holdout_paths_refuses(tmp_path, monkeypatch):
    log = tmp_path / "reads.jsonl"
    monkeypatch.setattr(J, "MAIN_CLONE", tmp_path)
    (tmp_path / ".git").mkdir()
    for argv in (["--cand", "a", "b", "c", "--msg0", "m0", "--msg1", "m1"],
                 ["--cand", "a", "b", "c", "--msg0", "m0", "--msg1", "m1", "--label", "x", "--out", "o"],
                 ["--cand", "a", "b", "c", "--base", "x", "y", "z", "--msg0", "m0", "--msg1", "m1"],
                 ["--dry-run-dev", "--cand", "a", "b", "c", "--base", "x", "y", "z", "--msg0", "m0", "--msg1", "m1",
                  "--baseline-dir", "outputs/local_eval/baseline_1206"]):
        with pytest.raises(SystemExit):
            J.main(argv)
    assert not (tmp_path / J.READS_REL).exists()
    src = (ROOT / "scripts/local_eval/judge_holdout.py").read_text()
    assert "baseline_1206\"" not in src.split("HOLDOUT_MARKERS")[0]   # 既定のパスとして持たない


# ---------- (iv) dev の 5 型の試し ----------

def _mk_dev(root: Path, n_per_type=150, prefix="b8_f1_"):
    rng = np.random.default_rng(1)
    rows, m0 = [], []
    for t, _ in J.DEV_TYPES:
        for i in range(n_per_type):
            s, m = f"{prefix}{t}_t10", f"mf{t}_{i:05d}"
            rows.append((s, m, t, float(rng.choice([0, 0.5, 1.0]))))
            m0.append((s, m))
    dirs = {}
    for name, shift in (("cand", 0.02), ("base", 0.0)):
        ds = []
        for sd in J.SEEDS:
            d = root / f"dev_{name}_s{sd}"
            d.mkdir()
            sets = {f"{prefix}{t}_t10": t for t, _ in J.DEV_TYPES}
            (d / "meta.json").write_text(json.dumps({"seed": sd, "sets": sets}))
            pm = pd.DataFrame(rows, columns=["set", "molecule_id", "type", "rr"])
            pm["rr"] = pm["rr"] + shift
            pm.to_csv(d / "per_molecule.csv", index=False)
            ds.append(d)
        dirs[name] = ds
    pd.DataFrame(m0, columns=["set", "molecule_id"]).to_csv(root / "dev_m0.csv", index=False)
    pd.DataFrame([("b8_f9_a_t10", "zz")], columns=["set", "molecule_id"]).to_csv(root / "dev_m1.csv", index=False)
    return dirs, root / "dev_m0.csv", root / "dev_m1.csv"


LINE = re.compile(r"^- \S+（.+）: n ＝ \d+（.+）、基準( score\+prior)? ＝ \d\.\d{4}、版 A ＝ -?\d\.\d{4}、Δ ＝ [+-]\d\.\d{4}（記録用）、"
                  r"95% 区間 \[[+-]\d\.\d{4} , [+-]\d\.\d{4}\]、L ≥ −0\.02: (満たす|満たさない)、L > 0: (はい|いいえ)$")


def test_dev_dry_run_same_format(tmp_path, monkeypatch):
    log = tmp_path / "reads.jsonl"
    monkeypatch.setattr(J, "MAIN_CLONE", tmp_path)
    (tmp_path / ".git").mkdir()
    dirs, m0, m1 = _mk_dev(tmp_path)
    opened = []
    real_open = builtins.open
    def spy(p, *a, **k):
        opened.append(str(p))
        return real_open(p, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    res, text = J.run_dev(dirs["cand"], dirs["base"], m0, m1, tmp_path / "dev_out")
    monkeypatch.setattr(builtins, "open", real_open)
    assert res["mode"] == "dry-run-dev"
    assert [r["set"] for r in res["sets"]] == ["a", "b", "c", "d", "e"]
    assert all(r["judge_on"] == "subset" and r["n"] == 150 for r in res["sets"])
    assert res["pass_"] and res["cond_ii_sets"] == ["a", "b", "c", "d", "e"]
    lines = text.splitlines()
    assert lines[0].startswith("【dev の試し。採否には使わない")
    assert sum(bool(LINE.match(x)) for x in lines) == 5
    assert any(x.startswith("- (i) 5 型すべてで L ≥ −0.02: 満たす") for x in lines)
    assert "- 判定: 条件を満たす" in lines
    # holdout の書式と同じ行の形（合成の holdout と比べる）
    h = _mk_holdout(tmp_path / "h", {s: 0.01 for s in SETS})
    _, htext = _run(tmp_path / "h", h, tmp_path / "h" / "reads.jsonl")
    assert sum(bool(LINE.match(x)) for x in htext.splitlines()) == 4
    norm = lambda x: re.sub(r"\d", "", x.split(":")[0])  # noqa: E731
    assert [norm(x) for x in htext.splitlines()[-5:]] == [norm(x).replace("型", "集合") for x in lines[-5:]]
    # reads.jsonl を作らない・holdout らしいパスを開かない
    assert not (tmp_path / J.READS_REL).exists()
    assert not any(any(k in p for k in J.HOLDOUT_MARKERS) for p in opened)
    assert (tmp_path / "dev_out" / "judge.md").read_text() == text


def test_dev_dry_run_rejects_judge_sets(tmp_path):
    dirs, m0, m1 = _mk_dev(tmp_path)
    meta = json.loads((dirs["cand"][0] / "meta.json").read_text())
    meta["sets"]["v3_c1u_t10"] = "a"
    (dirs["cand"][0] / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(J.JudgeError, match="dev（b8_）以外"):
        J.run_dev(dirs["cand"], dirs["base"], m0, m1, None)
    with pytest.raises(J.JudgeError, match="holdout らしいパス"):
        J.run_dev(dirs["cand"], dirs["base"], m0, m1, Path("outputs/local_eval/baseline_1206"))


def test_seed_set_enforced(tmp_path):
    dirs, m0, m1 = _mk_dev(tmp_path)
    (dirs["cand"][2] / "meta.json").write_text(json.dumps({"seed": 0, "sets": {"b8_f1_a_t10": "a"}}))
    with pytest.raises(J.JudgeError, match="seed の組"):
        J.run_dev(dirs["cand"], dirs["base"], m0, m1, None)


def test_reads_log_fixed_to_clone(tmp_path, monkeypatch):
    """既定の reads.jsonl はスクリプトの置き場所ではなくクローンの実ディレクトリ。クローンが無ければ何も読まずに止まる。"""
    assert J.MAIN_CLONE == Path("/Users/taichi/work/molecule-id-dev")
    assert J.READS_REL == "outputs/local_eval/judge/reads.jsonl"
    h = _mk_holdout(tmp_path, {s: 0.01 for s in SETS})
    monkeypatch.setattr(J, "MAIN_CLONE", tmp_path / "no_clone")
    with pytest.raises(J.JudgeError, match="クローン"):
        J.run_holdout(h["cands"], h["bdir"], h["m0"], h["m1"], "x", tmp_path / "o", baseline_pins=h["pins"],
                      msg_pins=h["mpins"])
    assert not (tmp_path / "no_clone").exists()
    clone = tmp_path / "clone"
    (clone / ".git").mkdir(parents=True)
    monkeypatch.setattr(J, "MAIN_CLONE", clone)
    for lab in ("interim_1206", "final_1211"):
        J.run_holdout(h["cands"], h["bdir"], h["m0"], h["m1"], lab, tmp_path / lab, baseline_pins=h["pins"],
                      msg_pins=h["mpins"])
    assert J.count_reads(clone / J.READS_REL) == 2
    with pytest.raises(J.ReadLimitError):
        J.run_holdout(h["cands"], h["bdir"], h["m0"], h["m1"], "third", tmp_path / "third", baseline_pins=h["pins"],
                      msg_pins=h["mpins"])


def test_msg_lists_must_name_exactly_judge_sets(tmp_path):
    h = _mk_holdout(tmp_path, {s: 0.0 for s in SETS})
    d = pd.read_csv(h["m1"])
    d.loc[len(d)] = ["b8_f1_a_t10", "zz"]
    d.to_csv(h["m1"], index=False)
    h["mpins"]["msg1"] = J.sha256_file(h["m1"])
    with pytest.raises(J.JudgeError, match="4 集合とちょうど"):
        _run(tmp_path, h, tmp_path / "reads.jsonl")
