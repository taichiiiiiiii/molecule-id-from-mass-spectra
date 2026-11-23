"""Issue #196: scripts/research/count_train_test_shift.py の数え方を作り物の小さな parquet で確かめる。実データは読まない。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research"))
import count_train_test_shift as C  # noqa: E402


def test_is_multimer():
    for a in ["[2M+Na]+", "[2M+H]+", "[2M-H]-", "[3M+H]+", "[ 2M+K]+", "[10M+H]+"]:
        assert C.is_multimer(a), a
    for a in ["[M+H]+", "[M-H]-", "[M+2H]2+", "[M+Na]+", "[M+CH2O2-H]-", "[M-H2O+H]+", None, "", "2M"]:
        assert not C.is_multimer(a), a


def test_ce_state():
    assert C.ce_state(None) == "missing_null"
    assert C.ce_state([]) == "missing_empty"
    assert C.ce_state([float("nan")]) == "missing_allnan"
    assert C.ce_state([20.0]) == "ok_single"
    assert C.ce_state([20.0, float("nan")]) == "ok_single"
    assert C.ce_state([20.0, 40.0]) == "ok_multi"


def test_valid_peak_counts_matches_top_peaks_rule():
    mz = pa.array([[100.0, 200.0, 300.0], [], None, [1.0, float("nan"), 3.0, 4.0], [5.0, 6.0]],
                  pa.list_(pa.float64()))
    it = pa.array([[1.0, 0.0, 0.5], [], None, [1.0, 1.0, float("inf"), -1.0], [1.0]], pa.list_(pa.float64()))
    raw, val = C.valid_peak_counts(mz, it)
    assert raw.tolist() == [3, 0, 0, 4, 2]
    _, val2, above = C.valid_peak_counts(mz, it, prec=[150.0, 1.0, 1.0, 0.5, 2.0])
    assert val2.tolist() == val.tolist()
    # 行 0: 有効は 100・300、prec+2 = 152 → 300 が上で 1。行 3: 有効は m/z 1.0 だけ、2.5 以下 → 0。行 4: 5.0 > 4.0 → 1
    assert above.tolist() == [1, 0, 0, 0, 1]
    # 行 0: 強度 0 を除く → 2。行 3: nan の m/z・inf・負の強度を除く → 1。行 4: 強度が 1 本だけ → 短い方まで 1
    assert val.tolist() == [2, 0, 0, 1, 1]
    # fragnn_bench.top_peaks と同じ条件（有限・強度 > 0、短い方まで）
    sys.path.insert(0, str(ROOT / "scripts/train"))
    import fragnn_bench as B
    for m, i, v in zip(mz.to_pylist(), it.to_pylist(), val.tolist()):
        got = B.top_peaks(m or [], i or [], k=10_000)
        assert len(got[0]) == v


def test_group_sizes():
    g = C.group_sizes(["a", "a", "b", "c", "c", "c"])
    assert g["n_groups"] == 3 and g["median"] == 2.0 and g["max"] == 3.0
    assert g["hist_1_to_16plus"][:3] == [1, 1, 1]
    assert abs(g["share_groups_ge2"] - 2 / 3) < 1e-12


def _write(tmp: Path):
    tr = pa.table({
        "precursor_mz": [1.5, 10.0, 10.0, 10.0, 10.0, 1.0],
        "base_peak_intensity": pa.array([500.0, None, 2000.0, 999.0, 5000.0, 1e4], pa.float64()),
        "adduct": ["[M+H]+", "[2M+Na]+", "[M-H]-", "[2M-H]-", "[M+H]+", "[M+Na]+"],
        "instrument_type": ["timsTOF", "timsTOF", "Orbitrap", "timsTOF", "QTOF", "timsTOF"],
        "ionization_mode": ["positive", "positive", "negative", "negative", "positive", "positive"],
        "collision_energy_ev": pa.array([[20.0], None, [], [20.0, 40.0], [float("nan")], [40.0]],
                                        pa.list_(pa.float64())),
        "ms2_mzs": pa.array([[1.0, 2.0], [1.0], [1.0, 2.0, 3.0], [1.0], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0]],
                            pa.list_(pa.float64())),
        "ms2_normalized_intensities": pa.array([[1.0, 0.0], [1.0], [1.0, 1.0, 1.0], [1.0], [1.0, 1.0],
                                               [1.0, 1.0, 1.0, 1.0]], pa.list_(pa.float64())),
        "num_peaks": [2, 1, 3, 1, 2, 4],
        "inchikey14": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC",
                       "AAAAAAAAAAAAAA"],
        "ingest_lib": ["enveda-180", "enveda-180", "gnps", "enveda-180", "gnps", "enveda-180"],
    })
    pq.write_table(tr, tmp / "train.parquet", row_group_size=4)
    te = pa.table({
        "precursor_mz": [10.0, 10.0, 10.0],
        "base_peak_intensity": [2000.0, 3000.0, 4000.0],
        "molecule_id": ["m1", "m1", "m2"],
        "adduct": ["[M+H]+", "[M+Na]+", "[M-H]-"],
        "instrument_type": ["timsTOF"] * 3,
        "ionization_mode": ["positive", "positive", "negative"],
        "collision_energy_ev": pa.array([[20.0], [40.0, 60.0], [20.0]], pa.list_(pa.float64())),
        "ms2_mzs": pa.array([[1.0, 2.0, 3.0], [1.0], [1.0, 2.0]], pa.list_(pa.float64())),
        "ms2_normalized_intensities": pa.array([[1.0, 1.0, 1.0], [1.0], [1.0, 1.0]], pa.list_(pa.float64())),
    })
    pq.write_table(te, tmp / "test.parquet")
    # v1 の prep.npz の形（spec_row・is_val・spec_ik）
    np.savez(tmp / "prep.npz", spec_row=np.array([0, 1, 2, 3, 5], np.int64),
             is_val=np.array([False, False, False, True, False]),
             spec_ik=np.array([b"AAAAAAAAAAAAAA", b"AAAAAAAAAAAAAA", b"BBBBBBBBBBBBBB", b"BBBBBBBBBBBBBB",
                               b"AAAAAAAAAAAAAA"], "S14"))


def test_cli_counts(tmp_path):
    _write(tmp_path)
    out = tmp_path / "counts.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/research/count_train_test_shift.py"), "--train",
                    str(tmp_path / "train.parquet"), "--test", str(tmp_path / "test.parquet"), "--prep",
                    str(tmp_path / "prep.npz"), "--out", str(out)], check=True, capture_output=True)
    r = json.loads(out.read_text())
    ta = r["train_all"]
    assert ta["n_rows"] == 6
    assert ta["multimer"]["n"] == 2
    assert ta["timstof"]["n"] == 4 and ta["timstof"]["multimer_n"] == 2
    assert abs(ta["timstof"]["multimer_share_in_timstof"] - 0.5) < 1e-12
    assert ta["ce"]["states"] == {"ok_single": 2, "ok_multi": 1, "missing_null": 1, "missing_empty": 1,
                                  "missing_allnan": 1}
    assert ta["ce"]["missing"] == 3
    assert ta["peaks_raw"]["median"] == 2.0
    assert ta["peaks_valid"]["max"] == 4.0 and ta["peaks_valid"]["zero"] == 0
    assert ta["num_peaks_col_vs_list"] == {"equal": 6, "differ": 0}
    # 行 0: prec 1.5 → 有効なピークは m/z 1.0 だけ（2.0 は強度 0）で 3.5 以下 → 0。行 5: prec 1.0 → 3.0 より大きい 4.0 だけ → 1
    assert ta["above_prec2"]["rows_with_any"] == 1 and ta["above_prec2"]["peaks"] == 1
    assert ta["base_peak"] == {"null": 1, "below_1000": 2, "below_1000_share_of_nonnull": 0.4}
    vt = r["visible_test"]
    assert vt["n_rows"] == 3 and vt["multimer"]["n"] == 0
    assert vt["adduct_list"] == ["[M+H]+", "[M+Na]+", "[M-H]-"]
    assert vt["ce"]["missing"] == 0 and vt["ce"]["multi"] == 1
    assert vt["above_prec2"]["rows_with_any"] == 0 and vt["base_peak"]["below_1000"] == 0
    assert vt["spectra_per_molecule"]["n_groups"] == 2 and vt["spectra_per_molecule"]["max"] == 2.0
    v1 = r["v1_train"]
    assert v1["n_rows"] == 4 and v1["multimer"]["n"] == 1
    assert v1["after_p1"] == {"n_rows": 3, "n_dropped": 1, "n_struct_before": 2, "n_struct_after": 2,
                              "spectra_per_struct_mode": v1["after_p1"]["spectra_per_struct_mode"]}
    # (A, positive) は 3 本 → P1 の後は 2 本、(B, negative) は 1 本
    assert v1["spectra_per_struct_mode"]["max"] == 3.0
    assert v1["after_p1"]["spectra_per_struct_mode"]["max"] == 2.0
    assert r["v1_val"]["n_rows"] == 1 and r["v1_val"]["after_p1"]["n_rows"] == 0
