"""#230: A5 の外部ライブラリ（MassSpecGym）との一致の数え方。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import a5_extlib_match as E  # noqa: E402


def write_test(tmp_path):
    t = pd.DataFrame(dict(
        molecule_id=["m1", "m1", "m2", "m3"], spectrum_id=["s1", "s2", "s3", "s4"],
        precursor_mz=[300.0, 300.0, 400.0, 500.0], ionization_mode=["positive", "negative", "positive", "positive"],
        adduct=["[M+H]+", "[M-H]-", "[M+H]+", "[M+Na]+"],
        ms2_mzs=[[100.0, 150.0], [90.0], [120.0, 130.0], [200.0]],
        ms2_normalized_intensities=[[1.0, 0.5], [1.0], [1.0, 1.0], [1.0]]))
    p = tmp_path / "test.parquet"; t.to_parquet(p)
    return p


def write_msg(tmp_path):
    m = pd.DataFrame(dict(
        identifier=["a", "b", "c", "d", "e"],
        mzs=["150.0,100.0", "100.0,150.0", "120.0,999.0", "200.0", "200.0"],     # a は並びが逆（並べ直して比べる）
        intensities=["0.5,1.0", "1.0,0.5", "1.0,1.0", "1.0", "1.0"],
        inchikey=["AAAAAAAAAAAAAA-X", "BBBBBBBBBBBBBB-X", "CCCCCCCCCCCCCC-X", "DDDDDDDDDDDDDD-X", "EEEEEEEEEEEEEE-X"],
        precursor_mz=[300.001, 300.01, 400.0, 500.0, 500.0],            # b は 33 ppm で窓の外
        adduct=["[M+H]+", "[M+H]+", "[M+H]+", "[M+H]+", "[M+Na]+"]))   # d は付加イオン違い
    p = tmp_path / "msg.tsv"; m.to_csv(p, sep="\t", index=False)
    return p


def test_match_counts_adduct_ppm_and_train_split(tmp_path):
    tp, mp = write_test(tmp_path), write_msg(tmp_path)
    te, off, mz, it = E.load_flat(str(tp), ["molecule_id", "spectrum_id"])
    need = {ad: np.sort(g.precursor_mz.to_numpy(float)) for ad, g in te.groupby("adduct")}
    refs = E.read_refs([mp], need)
    assert sorted(r[1] for r in refs["[M+H]+"]) == ["AAAAAAAAAAAAAA", "CCCCCCCCCCCCCC"]   # b（ppm 外）・d（付加イオン違い）は読まない
    spec = E.match(te, off, mz, it, refs)
    s1 = spec.iloc[0]
    assert abs(s1.best_cos - 1.0) < 1e-9 and s1.best_key == "AAAAAAAAAAAAAA" and s1.n_ref == 1
    assert spec.iloc[1].n_ref == 0 and spec.iloc[1].best_cos == 0.0       # 負イオンは相手が無い
    m = E.per_molecule(spec, train_keys={"AAAAAAAAAAAAAA"}).set_index("molecule_id")
    assert bool(m.loc["m1", "copy_like_ext"]) and not bool(m.loc["m1", "copy_like_ext_not_in_train"])
    assert abs(m.loc["m2", "best_cos_ext"] - 0.5) < 1e-9 and not bool(m.loc["m2", "ge_tau_ext"])   # 片方のピークだけ一致 = 0.5
    assert bool(m.loc["m3", "copy_like_ext_not_in_train"]) and m.loc["m3", "best_ext_key"] == "EEEEEEEEEEEEEE"
    s = E.summarize(m.reset_index())
    assert s["copy_like_ext"]["k"] == 2 and s["copy_like_ext_not_in_train"]["k"] == 1 and s["has_ref"]["k"] == 3


def test_prep_peaks_sorts_and_normalises():
    mz, it = E.prep_peaks("150.0,100.0", "4.0,1.0")
    assert mz.tolist() == [100.0, 150.0]
    assert abs((it * it).sum() - 1.0) < 1e-12 and it[1] > it[0]


def test_struct_overlap(tmp_path):
    mp = write_msg(tmp_path)
    t = E.struct_tables([mp])
    assert "AAAAAAAAAAAAAA" in t["massspecgym"] and len(t["massspecgym"]) == 5
    o = E.struct_overlap(["AAAAAAAAAAAAAA", "ZZZZZZZZZZZZZZ"], t)
    assert o["massspecgym"] == dict(k=1, n=2, n_struct=5)
