"""scripts/engine2/select_fp_bits.py の単体テスト（#84）。

(a) 合成の頻度で選び方の境界（0.005 と 0.995 ちょうどは選ばれる）
(b) fp10407 の長さ 10,407 と、Engine 2 の casmi_engine.fp_and_mass（_g["bits"] = arange(10407)）との完全一致。
    fp_and_mass は packbits で返すので unpackbits して比べる（写し同士の比較にしない）。ファイルは読まない
(c) 実データの照合: FPBITS_STRUCTS（train_structs.parquet）と FPBITS_REF（配布の fp_bits.npy）があれば、
    作ったビット列が配布と array_equal で 6,930 ビット。無ければ skip（CI にはデータが無い）
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engine2"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "engine2", "eng"))
import select_fp_bits as SFB  # noqa: E402


# (a) ---------------------------------------------------------------------------------------------
def test_a_boundaries_inclusive():
    freq = np.array([0.0, 0.004999, 0.005, 0.0051, 0.5, 0.995, 0.99501, 1.0])
    assert SFB.select_fingerprints(freq).tolist() == [2, 3, 4, 5]


def test_a_defaults_and_order():
    assert (SFB.LOW, SFB.HIGH) == (0.005, 0.995)
    rng = np.random.RandomState(0)
    freq = rng.rand(1000)
    bits = SFB.select_fingerprints(freq)
    assert np.all(np.diff(bits) > 0)
    assert np.array_equal(bits, np.where((freq >= 0.005) & (freq <= 0.995))[0])


def test_a_breakdown_blocks():
    assert sum(s for _, s in SFB.BLOCKS) == SFB.NBITS == 10407
    bits = np.array([0, 4095, 4096, 8191, 8192, 10239, 10240, 10406])
    assert SFB.breakdown(bits) == {"ECFP4": 2, "ECFP6": 2, "RDKit": 2, "MACCS": 2}


# (b) ---------------------------------------------------------------------------------------------
MOLS = ["c1ccccc1", "CCO", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "C[C@@H](N)C(=O)O"]


def test_b_matches_engine2_fp_and_mass():
    pytest.importorskip("rdkit")
    import casmi_engine as CE
    from rdkit.Chem import rdFingerprintGenerator
    saved = dict(CE._g)
    try:
        CE._g.clear()
        CE._g["bits"] = np.arange(10407)
        CE._g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        CE._g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
        CE._g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)
        for smi in MOLS:
            mine = SFB.fp10407(smi)
            assert mine is not None and mine.dtype == np.uint8 and mine.shape == (10407,)
            packed, _mass = CE.fp_and_mass(smi)
            theirs = np.unpackbits(packed)[:10407]
            assert np.array_equal(mine, theirs), smi
            assert mine.sum() > 0
        assert SFB.fp10407("not_a_smiles") is None and CE.fp_and_mass("not_a_smiles") is None
    finally:
        CE._g.clear()
        CE._g.update(saved)


def test_b_counts_match_direct():
    pytest.importorskip("rdkit")
    cnt, n = SFB.bit_counts(MOLS + ["not_a_smiles"], workers=1, chunk=2)
    assert n == len(MOLS)
    direct = np.sum([SFB.fp10407(s) for s in MOLS], axis=0)
    assert np.array_equal(cnt, direct)


# (c) ---------------------------------------------------------------------------------------------
STRUCTS = os.environ.get("FPBITS_STRUCTS", "")
REF = os.environ.get("FPBITS_REF", "")


@pytest.mark.skipif(not (STRUCTS and REF and os.path.exists(STRUCTS) and os.path.exists(REF)),
                    reason="FPBITS_STRUCTS / FPBITS_REF が無い（CI）")
def test_c_real_data_matches_reference(tmp_path):
    pytest.importorskip("rdkit")
    import json
    out = tmp_path / "check.json"
    rc = SFB.main(["--structs", STRUCTS, "--ref", REF, "--out", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["array_equal"] is True
    assert rep["n_bits"] == 6930
    assert rep["breakdown"] == {"ECFP4": 1460, "ECFP6": 3283, "RDKit": 2048, "MACCS": 139}
