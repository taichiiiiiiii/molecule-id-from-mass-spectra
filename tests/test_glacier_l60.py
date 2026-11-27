"""Issue #242: scripts/bench/glacier_l60.py の shard の分割・再開・本数のテスト（モデルは使わない）。"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/bench"))
import glacier_l60 as G  # noqa: E402


def _jobs(n):
    return pd.DataFrame(dict(row=np.arange(n), smiles=[f"C{'C' * (i % 5)}O" for i in range(n)],
                             adduct="[M+H]+", instr="QTOF", ce=20.0))


def _fake(calls, fail_every=0, bad_every=0):
    def fn(keys):
        calls.append(len(keys))
        preds, fail = [], np.zeros(len(keys), np.int8)
        for n, k in enumerate(keys):
            if bad_every and n % bad_every == 0:
                preds.append(None); fail[n] = G.FAIL_SMILES
            elif fail_every and n % fail_every == 1:
                preds.append(None)
            else:
                m = 1 + len(k[0]) % 3
                preds.append((np.arange(m, dtype=np.float32) + 50, np.ones(m, np.float32)))
        return preds, fail, dict(n_model_calls=len(keys))
    return fn


def test_shard_bounds():
    assert G.shard_bounds(0, 10) == []
    assert G.shard_bounds(10, 10) == [(0, 10)]
    assert G.shard_bounds(25, 10) == [(0, 10), (10, 20), (20, 25)]
    b = G.shard_bounds(G.EXPECTED_N_JOBS, G.SHARD_SIZE)
    assert len(b) == 28 and b[-1] == (540_000, G.EXPECTED_N_JOBS)
    assert sum(y - x for x, y in b) == G.EXPECTED_N_JOBS
    assert all(b[i][1] == b[i + 1][0] for i in range(len(b) - 1))
    with pytest.raises(ValueError):
        G.shard_bounds(5, 0)


def test_run_all_and_counts(tmp_path):
    jobs, calls = _jobs(25), []
    r = G.run_shards(tmp_path, jobs, _fake(calls, fail_every=4, bad_every=7), shard_size=10, log=lambda *x: None)
    assert r == dict(n_shards=3, n_done=3, ran=[0, 1, 2], skipped=0)
    assert calls == [10, 10, 5]
    tot = dict(rows=0, ok=0, fp=0, fs=0)
    rows = []
    for i in range(3):
        npz, js = G.shard_paths(tmp_path, i)
        m = json.loads(js.read_text())
        z = np.load(npz)
        rows.extend(z["row"].tolist())
        assert m["n_rows"] == len(z["row"]) == m["n_ok"] + m["n_fail"]
        assert m["n_fail"] == m["n_fail_pred"] + m["n_fail_smiles"]
        assert len(z["offsets"]) == len(z["row"]) + 1 and z["offsets"][-1] == len(z["mz"]) == len(z["it"])
        lens = np.diff(z["offsets"])
        assert (lens[~z["ok"]] == 0).all() and (lens[z["ok"]] > 0).all()
        assert (z["fail"][z["ok"]] == 0).all() and (z["fail"][~z["ok"]] > 0).all()
        tot["rows"] += m["n_rows"]; tot["ok"] += m["n_ok"]
        tot["fp"] += m["n_fail_pred"]; tot["fs"] += m["n_fail_smiles"]
    assert rows == list(range(25)) and tot["rows"] == 25
    assert tot["fs"] > 0 and tot["fp"] > 0 and tot["ok"] + tot["fp"] + tot["fs"] == 25


def test_resume_skips_done_and_redoes_broken(tmp_path):
    jobs, calls = _jobs(25), []
    r1 = G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, max_shards=1, log=lambda *x: None)
    assert r1["ran"] == [0] and r1["n_done"] == 1 and calls == [10]
    # 壊れた shard（npz が json の sha と違う）と、tmp だけ残った shard はやり直す
    r2 = G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, max_shards=1, log=lambda *x: None)
    assert r2["ran"] == [1] and r2["skipped"] == 1
    npz1, _ = G.shard_paths(tmp_path, 1)
    npz1.write_bytes(npz1.read_bytes() + b"x")
    assert not G.shard_done(tmp_path, 1, 10, 20)
    (tmp_path / "shards" / "shard_00002.npz.tmp.npz").write_bytes(b"partial")
    r3 = G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, log=lambda *x: None)
    assert r3["ran"] == [1, 2] and r3["skipped"] == 1 and r3["n_done"] == 3
    assert calls == [10, 10, 10, 5]
    r4 = G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, log=lambda *x: None)
    assert r4["ran"] == [] and r4["skipped"] == 3 and calls == [10, 10, 10, 5]
    # 区間が変わった（shard の大きさを変えた）ら完了と見なさない
    assert not G.shard_done(tmp_path, 0, 0, 12)


def test_rows_must_be_in_order(tmp_path):
    j = _jobs(5).iloc[::-1].reset_index(drop=True)
    with pytest.raises(AssertionError):
        G.run_shards(tmp_path, j, _fake([]), shard_size=2, log=lambda *x: None)


def test_pack_marks_empty_prediction_as_failure():
    preds = [(np.array([1.0], np.float32), np.array([1.0], np.float32)), None,
             (np.zeros(0, np.float32), np.zeros(0, np.float32))]
    z = G.pack(np.arange(3), preds, np.array([0, G.FAIL_SMILES, 0], np.int8))
    assert z["ok"].tolist() == [True, False, False]
    assert z["fail"].tolist() == [0, G.FAIL_SMILES, G.FAIL_PRED]
    assert z["offsets"].tolist() == [0, 1, 1, 1]


def test_jobs_sha_change_invalidates_shards(tmp_path):
    jobs, calls = _jobs(15), []
    G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, log=lambda *x: None, jobs_sha256="aaa")
    assert calls == [10, 5]
    assert G.shard_done(tmp_path, 0, 0, 10, "aaa") and not G.shard_done(tmp_path, 0, 0, 10, "bbb")
    r = G.run_shards(tmp_path, jobs, _fake(calls), shard_size=10, log=lambda *x: None, jobs_sha256="bbb")
    assert r["ran"] == [0, 1] and calls == [10, 5, 10, 5]


# ---------------------------------------------------------------- #268: --part tier
def _dev():
    return pd.DataFrame(dict(
        set=["s1"] * 5 + ["s2"] * 3, type=["a"] * 5 + ["c"] * 3, molecule_id=["1"] * 5 + ["2"] * 3,
        cand_key=["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"],
        cand_smiles=["C", "CC", "CCC", "CCCC", "CO", "CCO", "N", "CN"],
        is_tier=np.array([0, 1, 0, 1, 1, 0, 1, 1], np.int8),
        ctrl_rank=np.array([2, 9, 1, 3, 5, 0, 7, 4], np.int16)))


def test_select_rows_pool_and_tier():
    d = _dev()
    pool = G.select_rows(d, "pool")
    tier = G.select_rows(d, "tier")
    assert (pool.is_tier == 0).all() and (tier.is_tier == 1).all()
    assert len(pool) + len(tier) == len(d)                   # 層は全行（上位の切り出しをしない）
    # 分子ごとに ctrl_rank の順、cand_rank は 0 始まり
    assert tier.cand_smiles.tolist() == ["CCCC", "CO", "CC", "CN", "N"]
    assert tier.cand_rank.tolist() == [0, 1, 2, 0, 1]
    assert pool.cand_smiles.tolist() == ["CCC", "C", "CCO"]
    with pytest.raises(ValueError):
        G.select_rows(d, "all")


def test_select_rows_rejects_duplicate_smiles():
    d = _dev()
    d.loc[3, "cand_smiles"] = "CC"                           # 分子 1 の層の行で重複
    with pytest.raises(AssertionError):
        G.select_rows(d, "tier")


def test_expected_counts_tier():
    assert G.EXPECTED["pool"] == dict(n_jobs=544_886, n_L=197_170)
    assert G.EXPECTED["tier"] == dict(n_jobs=673_214, n_L=233_809)
    b = G.shard_bounds(G.EXPECTED["tier"]["n_jobs"], G.SHARD_SIZE)
    assert len(b) == 34 and b[-1] == (660_000, 673_214)


def test_cand_msg_overlap_counts_decoys_only_by_key():
    d = _dev()
    tier = G.select_rows(d, "tier")
    mols = pd.DataFrame(dict(set=["s1", "s2"], molecule_id=["1", "2"], key=["K2", "K9"]))
    r = G.cand_msg_overlap(tier, mols, {"K2", "K4", "K7"})
    assert r["n_rows"] == 5 and r["n_hit"] == 3
    assert r["n_truth_rows"] == 1                            # 分子 1 の K2 が正解の行
    assert r["n_hit_decoy"] == 2                             # K4・K7（正解の K2 は除く）
    assert r["by_type"]["a"] == dict(n_rows=3, n_hit=2, n_hit_decoy=1)
    assert r["by_type"]["c"] == dict(n_rows=2, n_hit=1, n_hit_decoy=1)


def test_check_shards(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts/bench"))
    import glacier_check_shards as CS
    jobs = _jobs(25)
    jobs.to_parquet(tmp_path / "jobs.parquet", index=False)
    jsha = G.sha256(tmp_path / "jobs.parquet")
    man = dict(part="tier", n_jobs=25, shard_size=10, outputs={"jobs.parquet": jsha})
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    monkeypatch.setitem(G.EXPECTED, "tier", dict(n_jobs=25, n_L=5))
    G.run_shards(tmp_path, jobs, _fake([]), shard_size=10, max_shards=2, log=lambda *x: None, jobs_sha256=jsha)
    r = CS.check(tmp_path)
    assert not r["ok"] and r["n_done"] == 2 and r["n_rows_done"] == 20
    G.run_shards(tmp_path, jobs, _fake([]), shard_size=10, log=lambda *x: None, jobs_sha256=jsha)
    r = CS.check(tmp_path)
    assert r["ok"] and r["n_done"] == 3 and r["n_rows_done"] == 25
    monkeypatch.setitem(G.EXPECTED, "tier", dict(n_jobs=26, n_L=5))   # 期待の本数と違えば通さない
    assert not CS.check(tmp_path)["ok"]
