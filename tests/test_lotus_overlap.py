"""Issue #129: lotus_overlap の数え方（合成データ）。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lotus_overlap", ROOT / "scripts/cv/lotus_overlap.py")
lo = importlib.util.module_from_spec(_spec)
sys.modules["lotus_overlap"] = lo
_spec.loader.exec_module(lo)


def test_overlap_counts_and_new():
    lotus = [("A", "A"), ("B", "Bc"), ("C", None), ("D", "D"), ("E", "E")]
    ov, new = lo.overlap_counts(lotus, pool_ik={"A"}, pool_canon={"Bc"}, train_ik={"D"}, train_canon={"A"})
    assert ov["n"] == 5
    assert (ov["pool_ik"], ov["pool_canon"], ov["pool_any"]) == (1, 1, 2)
    assert (ov["train_ik"], ov["train_canon"], ov["train_any"]) == (1, 1, 2)
    assert ov["pool_or_train"] == 3 and ov["new"] == 2 and new == [2, 4]
    assert ov["new_frac"] == 0.4


def test_truth_containment():
    truth = [("A", "A"), ("X", "Bc"), ("Y", None), ("Z", "Z")]
    r = lo.truth_containment(truth, lotus_ik={"A", "Y"}, lotus_canon={"A", "Bc"}, pool_ik={"A"}, pool_canon=set())
    assert (r["n"], r["lotus_ik"], r["lotus_canon"], r["lotus_any"]) == (4, 2, 2, 3)
    assert r["pool_any"] == 1 and r["lotus_any_not_in_pool"] == 2


def test_wilson_known_values():
    lo_, hi = lo.wilson(50, 100)
    assert lo_ == pytest.approx(0.4038, abs=1e-4) and hi == pytest.approx(0.5962, abs=1e-4)
    assert lo.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-12) and lo.wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-12)
    assert lo.wilson(0, 10)[1] == pytest.approx(0.2775, abs=1e-4)


def test_choose_n_rule():
    assert lo.choose_n(1.0) == 1000          # 1000 s ≤ 1800 s
    assert lo.choose_n(1.8) == 1000          # ちょうど上限
    assert lo.choose_n(3.0) == 600           # floor(1800 / 3)
    assert lo.choose_n(20.0) == 200          # 最少 200


def test_sample_order_deterministic_and_independent_of_input_order():
    k = [f"K{i:03d}" for i in range(50)]
    a = lo.sample_order(k, 0); b = lo.sample_order(list(reversed(k)), 0)
    assert a == b and sorted(a) == sorted(k) and a != sorted(k)


def _tier(entries):
    entries = sorted(entries)
    mass = np.array([m for m, _ in entries]); smis = [s.encode() for _, s in entries]
    off = np.cumsum([0] + [len(s) for s in smis]).astype(np.int64)
    buf = np.frombuffer(b"".join(smis), dtype=np.uint8)
    return mass, off, buf


def test_window_and_in_tier():
    mass, off, buf = _tier([(100.0, "AA"), (200.0, "BB"), (200.0001, "CC"), (200.01, "DD"), (300.0, "EE")])
    assert lo.window_smiles(mass, off, buf, 200.0, ppm=1.0) == ["BB", "CC"]     # ±0.0002 Da
    assert lo.window_smiles(mass, off, buf, 250.0) == []
    key = {"BB": "ikB", "CC": "ikC"}.get
    assert lo.in_tier(mass, off, buf, "ikC", 200.0, key) == (True, 2, True)
    assert lo.in_tier(mass, off, buf, "ikX", 200.0, key) == (False, 2, True)
    assert lo.in_tier(mass, off, buf, "ikX", 400.0, key) == (False, 0, False)


def test_window_with_rdkit_keys():
    from rdkit import Chem   # CI に casmi のコードは無いので chem.raw_key と同じ式（MolToInchiKey の先頭 14 文字）を使う

    def raw_key(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToInchiKey(m)[:14] if m is not None else None

    smi = "OC(=O)c1ccccc1O"   # salicylic acid
    ik = raw_key("O=C(O)c1ccccc1O")   # 書き方の違う同じ構造
    mass, off, buf = _tier([(138.0317, "COc1ccccc1C"), (138.0317, smi), (150.0, "CCO")])
    assert lo.in_tier(mass, off, buf, ik, 138.0317, raw_key)[0]
    assert not lo.in_tier(mass, off, buf, raw_key("CCO"), 138.0317, raw_key)[0]


def test_out_dir_inside_repo_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["x", "counts", "--lotus", "x", "--out-dir", str(ROOT / "outputs/x")])
    with pytest.raises(SystemExit):
        lo.main()
