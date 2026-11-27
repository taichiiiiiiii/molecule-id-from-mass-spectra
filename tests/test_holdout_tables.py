"""Issue #290: 判定用 4 集合（holdout_v3）の分岐のテスト。合成データだけ（モデル・本物の表は使わない）。

- glacier_l60 の holdout の part: 本数を固定しない（list の段の数を manifest に書き、run・点検は manifest と比べる）
- 候補の MSG との重なりは holdout では正解の鍵を使わない（正解のラベルを付けた集計を出さない）
- 正解の鍵の重なりの印（raw 2 つと正準の鍵）、§4.2 (d) の 150 分子の規則
- add_neighbour_ref の holdout の期待の数（dump_dev の記録の n_rows の和、1,358 分子）
- parts の再開（完了済みの shard は飛ばす）
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts/bench", "scripts/features", "scripts/local_eval", "scripts/ranker", "scripts/validate"):
    sys.path.insert(0, str(ROOT / p))

import glacier_l60 as G  # noqa: E402
import add_neighbour_ref as AN  # noqa: E402
import make_msg_lists as MM  # noqa: E402
import dump_dev as DD  # noqa: E402

LABEL_WORDS = ("n_truth_rows", "n_hit_decoy", "decoy", "is_truth", "label", "mrr", "rr")


def _tab():
    return pd.DataFrame(dict(
        set=["s1"] * 5 + ["s2"] * 3, type=["c1"] * 5 + ["c2"] * 3, molecule_id=["1"] * 5 + ["2"] * 3,
        cand_key=["K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8"],
        cand_smiles=["C", "CC", "CCC", "CCCC", "CO", "CCO", "N", "CN"],
        is_tier=np.array([0, 1, 0, 1, 1, 0, 1, 1], np.int8),
        ctrl_rank=np.array([2, 9, 1, 3, 5, 0, 7, 4], np.int16)))


def test_holdout_parts_are_not_fixed_counts():
    for p in G.HOLDOUT_PARTS:
        assert G.EXPECTED[p] == dict(n_jobs=None, n_L=None)
    # dev の part は今までどおり固定
    assert G.expected_n_jobs(dict(part="pool", n_jobs=1)) == 544_886
    assert G.expected_n_jobs(dict(n_jobs=1)) == 544_886                 # part の無い #242 の manifest = pool
    assert G.expected_n_jobs(dict(part="tier", n_jobs=1)) == 673_214
    # holdout の part は manifest の本数
    assert G.expected_n_jobs(dict(part="holdout", n_jobs=123)) == 123
    assert G.expected_n_jobs(dict(part="holdout_tier", n_jobs=45)) == 45
    assert G.EXPECTED_N_MOLS_HOLDOUT == DD.N_HOLD == AN.N_HOLD == MM.EXPECT_N_HOLD == 1358


def test_select_rows_holdout_parts_match_pool_and_tier():
    d = _tab()
    assert G.select_rows(d, "holdout").equals(G.select_rows(d, "pool"))
    assert G.select_rows(d, "holdout_tier").equals(G.select_rows(d, "tier"))


def test_cand_overlap_holdout_has_no_truth_split():
    L = G.select_rows(_tab(), "holdout_tier")
    mols = pd.DataFrame(dict(set=["s1", "s2"], molecule_id=["1", "2"], key=["K2", "K9"]))
    r = G.cand_msg_overlap(L, mols, {"K2", "K4", "K7"}, by_truth=False)
    assert r == dict(n_rows=5, n_hit=3, by_type={"c1": dict(n_rows=3, n_hit=2), "c2": dict(n_rows=2, n_hit=1)})
    assert not any(w in json.dumps(r) for w in LABEL_WORDS)
    # 正解の鍵（mols.key）を変えても結果は同じ = 正解を使っていない
    mols2 = mols.assign(key=["K4", "K7"])
    assert G.cand_msg_overlap(L, mols2, {"K2", "K4", "K7"}, by_truth=False) == r


def test_holdout_list_stage_writes_no_label_columns():
    """stage_list_holdout の出力の列・manifest の鍵に、正解か偽物かで分けた数・MRR が無い（ソースで確かめる）。"""
    src = inspect.getsource(G.stage_list_holdout)
    assert "by_truth=False" in src
    for w in ("n_truth_rows", "n_hit_decoy", "decoy", "mrr", "calculate_mrr", "is_truth"):
        assert w not in src, w
    # 分子の表の列（正解の鍵は MSG との重なりの印のためだけ）
    assert '"inchikey14_rdkit", "key", "n_L", "n_cat_r", "n_jobs",' in src and '"msg_overlap", "msg_overlap_all"]]' in src


def test_mark_msg_uses_both_raw_keys_and_canonical():
    mols = pd.DataFrame(dict(inchikey14=["AAA", "BBB", "CCC", "DDD"], inchikey14_rdkit=["AAA", "XXX", "CCC", "DDD"],
                             key=["k1", "k2", "k3", "k4"]))
    ms = dict(raw_tv={"AAA"}, key_tv={"k3"}, raw_all={"AAA", "XXX"}, key_all={"k3", "k4"})
    G.mark_msg(mols, ms)
    assert mols.msg_overlap.tolist() == [1, 0, 1, 0]
    assert mols.msg_overlap_all.tolist() == [1, 1, 1, 1]      # BBB は RDKit の鍵 XXX で、DDD は正準の鍵 k4 で


def test_holdout_rule_150_by_counts_only():
    rows = []
    for s, n, n1 in (("v3_c1u_t10", 400, 10), ("enveda_v1u_t10", 380, 0), ("v3_c2_t10", 197, 48),
                     ("v3_c2nu_t10", 381, 231)):
        rows += [dict(set=s, molecule_id=f"m{i}", msg_overlap=int(i < n1)) for i in range(n)]
    r = MM.holdout_rule(pd.DataFrame(rows))
    assert r["v3_c1u_t10"] == dict(n=400, n_msg1=10, n_remain=390, judge_on="subset")
    assert r["v3_c2_t10"] == dict(n=197, n_msg1=48, n_remain=149, judge_on="full_set")     # 150 未満 → 全集合
    assert r["v3_c2nu_t10"]["n_remain"] == 150 and r["v3_c2nu_t10"]["judge_on"] == "subset"


def test_add_neighbour_ref_holdout_expect(tmp_path):
    for h, n in zip(AN.HOLDOUT_SETS, (10, 20, 30, 40)):
        (tmp_path / f"{h}.json").write_text(json.dumps(dict(holdout=h, n_rows=n)))
    e = AN.expect_for("holdout", tmp_path)
    assert e == dict(sets=AN.HOLDOUT_SETS, n_mol=1358, n_rows=100)
    assert AN.expect_for("dev")["n_rows"] == 430979 and len(AN.expect_for("dev")["sets"]) == 20
    assert tuple(AN.HOLDOUT_SETS) == tuple(DD.HOLDOUT_SETS)
    with pytest.raises(ValueError):
        AN.expect_for("test")


def test_holdout_shards_resume(tmp_path):
    """holdout の part でも run の段は完了済みの shard を飛ばし、点検は manifest の本数と比べる。"""
    import glacier_check_shards as CS
    jobs = pd.DataFrame(dict(row=np.arange(25), smiles=["CCO"] * 25, adduct="[M+H]+", instr="QTOF", ce=20.0))
    jobs.to_parquet(tmp_path / "jobs.parquet", index=False)
    jsha = G.sha256(tmp_path / "jobs.parquet")
    (tmp_path / "manifest.json").write_text(json.dumps(dict(part="holdout", n_jobs=25, shard_size=10,
                                                            outputs={"jobs.parquet": jsha})))
    calls = []

    def fn(keys):
        calls.append(len(keys))
        return [(np.array([50.0], np.float32), np.ones(1, np.float32))] * len(keys), np.zeros(len(keys), np.int8), {}
    G.run_shards(tmp_path, jobs, fn, shard_size=10, max_shards=1, log=lambda *x: None, jobs_sha256=jsha)
    r = CS.check(tmp_path)
    assert not r["ok"] and r["n_done"] == 1 and r["expected_n_jobs"] == 25
    G.run_shards(tmp_path, jobs, fn, shard_size=10, log=lambda *x: None, jobs_sha256=jsha)
    assert calls == [10, 10, 5]                                        # 再開: shard 0 は飛ばす
    assert CS.check(tmp_path)["ok"]
    m = json.loads((tmp_path / "manifest.json").read_text())
    m["n_jobs"] = 26                                                   # manifest と違えば通さない
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    assert not CS.check(tmp_path)["ok"]


def test_dump_holdout_tables_have_no_label_columns():
    """dump_features の入れない列にラベル系があり、本物の holdout の表があれば（手元だけ）その列が無い。"""
    import dump_features as DF
    for c in ("y", "label", "is_truth", "msg_overlap"):
        assert c in DF.EXCLUDED
    f = ROOT / "outputs/local_eval/features"
    for name in ("holdout_v3.parquet", "holdout_v3_cc.parquet", "holdout_v3_cc_n1.parquet"):
        p = f / name
        if not p.is_file():
            continue
        import pyarrow.parquet as pq
        cols = pq.read_schema(p).names
        assert not [c for c in cols if c in DF.EXCLUDED or c in ("rr", "rank_truth", "mrr25")], name
