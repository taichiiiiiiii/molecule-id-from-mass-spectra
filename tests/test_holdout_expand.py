"""holdout_expand（#199、A3a）の単体テスト。小さな合成データで、除外（raw・正準・別名・群）・select の exclude（既定は今と同じ）・
付加イオンの絞り（_t10）・合併の衝突検出・run.py の --logits-dir / --truth-dir の既定と、write_holdout の id 接頭を確かめる。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))
sys.path.insert(0, str(ROOT / "scripts/validate"))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V3 = _load("build_holdout_v3", "scripts/cv/build_holdout_v3.py")
HE = _load("holdout_expand", "scripts/cv/holdout_expand.py")


def _meta():
    # A: enveda＋公開（C1）、B: enveda-180 だけ（C2）、C: 公開だけ、D: 質量が範囲外、E: enveda＋公開（C1、除外される）
    rows = [("A", "enveda-180", 300.0), ("A", "gnps", 300.0), ("B", "enveda-180", 400.0), ("C", "mona", 500.0),
            ("D", "enveda-180", 100.0), ("D", "gnps", 100.0), ("E", "enveda-np-examples", 600.0), ("E", "riken", 600.0)]
    return pd.DataFrame(rows, columns=["inchikey14", "ingest_lib", "neutral_mass"])


def test_select_default_unchanged_and_exclude():
    m = _meta()
    hold = {"A", "B", "C", "D", "E"}
    c1, c2 = V3.eligible(m, hold)
    assert c1 == ["A", "E"] and c2 == ["B"]
    a = V3.select(m, hold, 10, 1)
    b = V3.select(m, hold, 10, 1, exclude=frozenset())
    assert a == b == (["A", "E"], ["B"], 2, 1)
    c1x, _, n1, _ = V3.select(m, hold, 10, 1, exclude={"E"})
    assert c1x == ["A"] and n1 == 1
    # hold に無い構造は選ばない
    assert V3.eligible(m, {"A", "B"})[0] == ["A"]


def test_exclusion_raw_canonical_alias_group():
    truth = pd.DataFrame({"inchikey14": ["AAAA"], "smiles": ["CCO"]})
    ts_key = {"AAAA": "KAAA", "ALIA": "KAAA", "OTHR": "KOTH"}   # ALIA は AAAA と同じ正準キー（別名）
    st = pd.DataFrame({"inchikey14": ["AAAA", "GRP2", "FREE"], "scorer_key": ["KAAA", "ZZZZ", "YYYY"],
                       "identity_group": [1, 1, 2]})
    e_raw, e_key, e_grp = HE.exclusion_sets([truth], [{"DROP"}], lambda s: "SKEY", ts_key, st)
    assert {"AAAA", "DROP"} <= e_raw and {"SKEY", "KAAA"} <= e_key and e_grp == {1}
    sc_key = dict(zip(st.inchikey14, st.scorer_key))
    grp_of = dict(zip(st.inchikey14, st.identity_group))
    got = HE.excluded({"AAAA", "ALIA", "GRP2", "FREE", "OTHR", "SKEY", "DROP"}, e_raw, e_key, e_grp, ts_key, sc_key, grp_of)
    assert got == {"AAAA", "ALIA", "GRP2", "SKEY", "DROP"}   # raw・別名（正準）・同じ群・scorer キー


def test_t10_filter_drops_rows_and_empty_molecules():
    te = pd.DataFrame({"molecule_id": ["m1", "m1", "m2"], "adduct": ["[M+H]+", "[2M+Na]+", "[2M+H]+"]})
    tr = pd.DataFrame({"molecule_id": ["m1", "m2"], "inchikey14": ["A", "B"]})
    sb = pd.DataFrame({"molecule_id": ["m1", "m2"], "smiles": "CCO"})
    t, tr2, sb2 = HE.t10_filter(te, tr, sb)
    assert t.adduct.tolist() == ["[M+H]+"] and tr2.molecule_id.tolist() == ["m1"] and sb2.molecule_id.tolist() == ["m1"]
    assert len(HE.TEST10) == 10 and set(HE.VISIBLE7) <= set(HE.TEST10)


def test_union_concat_and_collision():
    a = (pd.DataFrame({"molecule_id": ["m1"], "spectrum_id": ["s1"]}), pd.DataFrame({"molecule_id": ["m1"]}),
         pd.DataFrame({"molecule_id": ["m1"]}))
    b = (pd.DataFrame({"molecule_id": ["mb1"], "spectrum_id": ["sb1"]}), pd.DataFrame({"molecule_id": ["mb1"]}),
         pd.DataFrame({"molecule_id": ["mb1"]}))
    t, tr, sb = HE.union([a, b])
    assert t.molecule_id.tolist() == ["m1", "mb1"] and len(tr) == 2
    with pytest.raises(AssertionError):
        HE.union([a, a])


def test_same_hide_compares_three_keys():
    assert HE.same_hide({"rule": "all", "pool": "keep", "query": "x"}, {"rule": "all", "pool": "keep"})
    assert not HE.same_hide({"rule": "all", "pool": "keep"}, {"rule": "all", "pool": "remove"})
    assert HE.same_hide(None, None) and not HE.same_hide(None, {"rule": "all"})


def test_run_paths_default_unchanged():
    run = _load("local_eval_run_paths", "scripts/local_eval/run.py")
    assert run.logits_path("v3_c1_fold0") == ROOT / "outputs/holdout_eval/v3_c1_fold0/pc_logits.npz"
    assert run.truth_path("v3_c1") == ROOT / "outputs/holdouts/v3_c1/truth.csv"
    assert run.logits_path("x", "/tmp/a") == Path("/tmp/a/pc_logits.npz")
    assert run.truth_path("x", "/tmp/b") == Path("/tmp/b/truth.csv")


def test_read_rows_matches_read_table(tmp_path):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pa.table({"inchikey14": ["A", "B", "A", "C", "B"], "v": [1, 2, 3, 4, 5]})
    p = tmp_path / "t.parquet"
    pq.write_table(t, p, row_group_size=2)
    want = pq.read_table(p, filters=pc.field("inchikey14").isin(["A", "B"])).to_pandas()
    got = V3.read_rows(p, ["B", "A"], batch_size=2)
    pd.testing.assert_frame_equal(got, want)


def test_write_holdout_id_prefix(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = {"inchikey14": ["AAAA", "AAAA", "BBBB"], "normalized_smiles": ["CCO", "CCO", "CC"],
            "ingest_lib": ["enveda-180"] * 3, "adduct": ["[M+H]+"] * 3, "ionization_mode": ["positive"] * 3,
            "instrument_type": ["timsTOF"] * 3, "precursor_mz": [47.0, 47.0, 31.0],
            "ms2_mzs": [[10.0], [11.0], [12.0]], "ms2_normalized_intensities": [[1.0], [1.0], [1.0]],
            "base_peak_intensity": [1e4] * 3, "collision_energy_orig": ["20"] * 3, "collision_energy_ev": [[20.0]] * 3,
            "collision_energy_orig_units": ["eV"] * 3}
    pq.write_table(pa.table(rows), tmp_path / "train.parquet")
    meta = pd.read_parquet(tmp_path / "train.parquet", columns=["inchikey14", "ingest_lib", "instrument_type"])
    args = (["AAAA", "BBBB"], "enveda_libs", meta, np.random.default_rng(1), 1, tmp_path / "train.parquet")
    a = V3.write_holdout("h1", *args, out_root=tmp_path)
    assert sorted(pd.read_csv(tmp_path / "outputs/holdouts/h1/truth.csv").molecule_id) == ["m_00000", "m_00001"]
    b = V3.write_holdout("h2", *args[:3], np.random.default_rng(1), 1, args[5], out_root=tmp_path, id_prefix="b",
                         truth_root=tmp_path / "lv", stream=True)
    t2 = pd.read_parquet(tmp_path / "data/holdouts/h2/test.parquet")
    assert set(t2.molecule_id) == {"mb_00000", "mb_00001"} and all(s.startswith("sb_") for s in t2.spectrum_id)
    assert (tmp_path / "lv/h2/truth.csv").is_file()
    t1 = pd.read_parquet(tmp_path / "data/holdouts/h1/test.parquet")
    assert len(t1) == len(t2) and a["n_molecules"] == b["n_molecules"] == 2
