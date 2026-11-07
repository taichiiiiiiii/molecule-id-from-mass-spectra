"""Issue #32: C2p 型（正解はプールに残し参照を全部隠す）の学習グループと v3_c2n の作り方。"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_holdout_v3", ROOT / "scripts/cv/build_holdout_v3.py")
hv = importlib.util.module_from_spec(_spec)
sys.modules["build_holdout_v3"] = hv
_spec.loader.exec_module(hv)


def meta():
    return pd.DataFrame(dict(inchikey14=["A", "A", "B", "C", "D", "E"],
                             ingest_lib=["gnps", "massbank", "gnps", "enveda-180", "gnps", "gnps"],
                             neutral_mass=[300.0, 300.0, 400.0, 500.0, 100.0, 600.0]))


def test_select_c2p_only_coconut_allowed_mass_and_not_excluded():
    keys, n = hv.select_c2p(meta(), allowed={"A", "B", "C", "D"}, coconut={"A", "B", "D", "E"}, n=10, seed=0, exclude={"B"})
    # D は質量範囲外、E は allowed 外、C は COCONUT 外、B は exclude
    assert keys == ["A"] and n == 1


def test_select_c2p_deterministic():
    m = pd.DataFrame(dict(inchikey14=[f"K{i:02d}" for i in range(30)], ingest_lib="gnps", neutral_mass=300.0))
    ks = set(m.inchikey14)
    a, _ = hv.select_c2p(m, ks, ks, 5, 7)
    b, _ = hv.select_c2p(m, ks, ks, 5, 7)
    assert a == b and len(a) == 5 and a == sorted(a)


def test_pick_queries_prefer_enveda():
    t = pd.DataFrame(dict(inchikey14=["A", "A", "B", "B", "C"],
                          instrument_type=["timsTOF", "QTOF", "QTOF", "Orbitrap", "QTOF"],
                          ingest_lib=["enveda-180", "gnps", "gnps", "massbank", "gnps"],
                          adduct=["[M+H]+", "[M+H]+", "[M+H]+", "[M+K]+", "[M+H]+"]))
    q = hv.pick_queries(t, "prefer_enveda", {"[M+H]+"})
    # A は enveda timsTOF だけ、B は公開ライブラリのうちテストの付加イオンだけ、C も公開
    assert sorted(zip(q.inchikey14, q.instrument_type)) == [("A", "timsTOF"), ("B", "QTOF"), ("C", "QTOF")]
    q2 = hv.pick_queries(t, "enveda", set())
    assert list(q2.inchikey14) == ["A"]


def test_write_holdout_c2p_keeps_pool_and_hides_all_refs(tmp_path):
    cols = {c: [None] * 3 for c in hv.TEST_COLS}
    cols.update(inchikey14=["A", "A", "B"], ingest_lib=["gnps", "enveda-180", "gnps"], instrument_type=["QTOF", "timsTOF", "QTOF"],
                adduct=["[M+H]+"] * 3, normalized_smiles=["CCO", "CCO", "CCN"], precursor_mz=[1.0, 1.0, 2.0])
    cols["ms2_mzs"] = [[1.0]] * 3; cols["ms2_normalized_intensities"] = [[1.0]] * 3
    train = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(cols)), train)
    m = pd.DataFrame(dict(inchikey14=["A", "A", "B", "X"], ingest_lib=["gnps", "enveda-180", "gnps", "gnps"]))
    info = hv.write_holdout("h", ["A", "B"], "all", m, np.random.default_rng(0), 0, train, out_root=tmp_path,
                            pool="keep", query="prefer_enveda", test_adducts={"[M+H]+"})
    rule = json.loads((tmp_path / "data/holdouts/h/hide_rule.json").read_text())
    assert rule["rule"] == "all" and rule["pool"] == "keep" and rule["query"] == "prefer_enveda"
    assert info["hidden_train_rows"] == 3 and info["remaining_ref_rows_of_truth"] == 0
    q = pd.read_parquet(tmp_path / "data/holdouts/h/test.parquet")
    assert sorted(q.instrument_type) == ["QTOF", "timsTOF"]  # A は timsTOF、B は公開


def test_write_holdout_default_pool_unchanged(tmp_path):
    # v3 と同じ既定: rule=all → remove、enveda_libs → keep
    cols = {c: [None] for c in hv.TEST_COLS}
    cols.update(inchikey14=["A"], ingest_lib=["enveda-180"], instrument_type=["timsTOF"], adduct=["[M+H]+"], normalized_smiles=["CCO"])
    cols["ms2_mzs"] = [[1.0]]; cols["ms2_normalized_intensities"] = [[1.0]]
    train = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(cols)), train)
    m = pd.DataFrame(dict(inchikey14=["A"], ingest_lib=["enveda-180"]))
    for name, rule, pool in (("a", "all", "remove"), ("b", "enveda_libs", "keep")):
        hv.write_holdout(name, ["A"], rule, m, np.random.default_rng(0), 0, train, out_root=tmp_path)
        assert json.loads((tmp_path / f"data/holdouts/{name}/hide_rule.json").read_text())["pool"] == pool
