import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mrr25", ROOT / "scripts/validate/mrr25.py")
mrr25 = importlib.util.module_from_spec(_spec)
sys.modules["mrr25"] = mrr25
_spec.loader.exec_module(mrr25)

# InChIKey14 だけで比較できるダミーキー
A, B, C = "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"
FILL = [f"{chr(65 + i // 26)}{chr(65 + i % 26)}" + "Z" * 12 for i in range(30)]  # A/B/C と衝突しない


def test_rank_values_from_overview():
    assert mrr25.calculate_mrr25({"m": A}, {"m": [A]}) == 1.0
    assert mrr25.calculate_mrr25({"m": A}, {"m": [B, A]}) == 0.5
    assert mrr25.calculate_mrr25({"m": A}, {"m": FILL[:24] + [A]}) == pytest.approx(0.04)
    assert mrr25.calculate_mrr25({"m": A}, {"m": [B, C]}) == 0.0


def test_only_first_hit_counts():
    assert mrr25.calculate_mrr25({"m": A}, {"m": [B, A, A, A]}) == 0.5


def test_mean_over_truth_and_missing_prediction_is_zero():
    truth = {"m1": A, "m2": B, "m3": C}
    preds = {"m1": [A], "m2": [C, B], "extra": [A]}
    assert mrr25.calculate_mrr25(truth, preds) == pytest.approx((1 + 0.5 + 0) / 3)


def test_more_than_25_candidates_rejected():
    with pytest.raises(mrr25.SubmissionError):
        mrr25.calculate_mrr25({"m": A}, {"m": FILL[:26]})


def test_full_inchikey_compares_on_first_14():
    full = A + "-UHFFFAOYSA-N"
    other_stereo = A + "-XXXXXXXXXX-N"
    assert mrr25.calculate_mrr25({"m": full}, {"m": [other_stereo]}) == 1.0


def test_smiles_matching_uses_connectivity():
    pytest.importorskip("rdkit")
    # エタノールの別表記は同一、メタノールは別
    assert mrr25.calculate_mrr25({"m": "CCO"}, {"m": ["OCC"]}) == 1.0
    assert mrr25.calculate_mrr25({"m": "CCO"}, {"m": ["CO", "OCC"]}) == 0.5
    # 立体だけ違う（L/D-アラニン）は connectivity が同じなので正解扱い
    assert mrr25.calculate_mrr25({"m": "C[C@@H](N)C(=O)O"}, {"m": ["C[C@H](N)C(=O)O"]}) == 1.0
    # 互変異性体（2-ヒドロキシピリジン / 2-ピリドン）も同一
    assert mrr25.calculate_mrr25({"m": "Oc1ccccn1"}, {"m": ["O=c1cccc[nH]1"]}) == 1.0


def test_invalid_smiles_scores_zero():
    pytest.importorskip("rdkit")
    assert mrr25.calculate_mrr25({"m": "CCO"}, {"m": ["not_a_smiles", "OCC"]}) == 0.5


def test_validate_submission_ok():
    rows = [["molecule_id", "smiles"], ["m1", f"{A};{B}"], ["m2", ""]]
    out = mrr25.validate_submission(rows, expected_ids={"m1", "m2"})
    assert out == {"m1": [A, B], "m2": []}


@pytest.mark.parametrize(
    "rows",
    [
        [["id", "smiles"], ["m1", A]],  # ヘッダー違い
        [["molecule_id", "smiles"], ["m1", A], ["m1", B]],  # 重複
        [["molecule_id", "smiles"], ["", A]],  # null
        [["molecule_id", "smiles"], ["m1", ";".join(FILL[:26])]],  # 26 候補
        [["molecule_id", "smiles"], ["m1", A, "x"]],  # 列数
    ],
)
def test_validate_submission_rejects(rows):
    with pytest.raises(mrr25.SubmissionError):
        mrr25.validate_submission(rows)


def test_validate_submission_id_mismatch():
    with pytest.raises(mrr25.SubmissionError):
        mrr25.validate_submission([["molecule_id", "smiles"], ["m1", A]], expected_ids={"m1", "m2"})


def test_cli_roundtrip(tmp_path):
    truth = tmp_path / "truth.csv"
    truth.write_text(f"molecule_id,inchikey\nm1,{A}\nm2,{B}\n")
    sub = tmp_path / "submission.csv"
    sub.write_text(f"molecule_id,smiles\nm1,{A}\nm2,{C};{B}\n")
    out = tmp_path / "res.json"
    sys.argv = ["mrr25", "--truth", str(truth), "--pred", str(sub), "--output", str(out)]
    mrr25.main()
    import json

    res = json.loads(out.read_text())
    assert res["mrr25"] == pytest.approx(0.75)
    assert res["hit_at_25"] == 1.0
