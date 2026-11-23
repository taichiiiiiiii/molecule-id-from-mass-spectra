"""scripts/local_eval/selpred.py（#216）の純粋な関数の単体テスト。"""
import csv
import importlib.util
import io
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("selpred", ROOT / "scripts/local_eval/selpred.py")
sp = importlib.util.module_from_spec(_spec)
sys.modules["selpred"] = sp
_spec.loader.exec_module(sp)


def test_confidences_gap_and_softmax():
    g, c = sp.confidences([2.0, 1.0, 0.0])
    assert g == 1.0
    e = np.exp([2.0, 1.0, 0.0])
    assert math.isclose(c, e[0] / e.sum())


def test_confidences_negative_gap_after_reorder_is_kept():
    g, c = sp.confidences([0.5, 1.5])   # frag が点の低い候補を 1 位にした場合
    assert g == -1.0 and c < 0.5


def test_confidences_edge_sizes():
    assert sp.confidences([3.0]) == (math.inf, 1.0)
    assert sp.confidences([]) == (-math.inf, -math.inf)


def test_aurc_perfect_and_worst_order():
    loss = [0, 0, 1, 1]
    # 正解に高い確信度: k=1..4 のリスク 0,0,1/3,1/2
    assert math.isclose(sp.aurc([4, 3, 2, 1], loss), (0 + 0 + 1 / 3 + 1 / 2) / 4)
    # 逆順: 1,1,2/3,1/2
    assert math.isclose(sp.aurc([1, 2, 3, 4], loss), (1 + 1 + 2 / 3 + 1 / 2) / 4)


def test_aurc_all_tied_is_expected_value_under_random_order():
    loss = [0, 1, 0, 1]
    # 全部同点: 位置 j のリスクの期待値は j·0.5/j = 0.5
    assert math.isclose(sp.aurc([1, 1, 1, 1], loss), 0.5)
    # 総当たりの平均と一致する（3 要素の部分同点）
    import itertools
    k, l = [2, 1, 1], [1, 0, 1]
    vals = []
    for perm in itertools.permutations([1, 2]):
        order = [0] + list(perm)
        ls = [l[i] for i in order]
        vals.append(np.mean([sum(ls[:j]) / j for j in range(1, 4)]))
    assert math.isclose(sp.aurc(k, l), float(np.mean(vals)))


def test_aurc_summary_rel_bounds_and_degenerate():
    loss = [0, 1, 0, 1, 0]
    s_or = sp.aurc_summary([-x for x in loss], loss)
    assert math.isclose(s_or["rel_aurc"], 0.0, abs_tol=1e-12)
    s_all_wrong = sp.aurc_summary([1, 2, 3], [1, 1, 1])
    assert s_all_wrong["aurc"] == 1.0 and math.isnan(s_all_wrong["rel_aurc"])
    s_tie = sp.aurc_summary([0] * 5, loss)
    assert math.isclose(s_tie["rel_aurc"], 1.0)


def test_auc_with_ties_and_inf():
    assert sp.auc([2, 3], [1, 1]) == 1.0
    assert sp.auc([1], [1]) == 0.5
    assert sp.auc([math.inf, -math.inf], [0.0]) == 0.5
    assert math.isclose(sp.auc([1, 2, 3], [2]), (0 + 0.5 + 1) / 3)


def test_boot_is_seeded_and_brackets_point():
    rng = np.random.default_rng(1)
    k = rng.normal(size=60); loss = (rng.random(60) < 0.4).astype(float)
    a = sp.boot_aurc(k, loss, n_boot=200)
    b = sp.boot_aurc(k, loss, n_boot=200)
    assert a == b
    lo, hi = a["aurc_ci95"]
    assert lo <= sp.aurc(k, loss) <= hi
    lo2, hi2 = sp.boot_auc(k[:30], k[30:], n_boot=200)
    assert lo2 <= sp.auc(k[:30], k[30:]) <= hi2


def test_submission_bytes_matches_run_py_writer():
    preds = {"m1": ["CCO", "CCC"], "m2": []}
    buf = io.StringIO(newline="")
    w = csv.writer(buf); w.writerow(["molecule_id", "smiles"])
    for mid, c in preds.items():
        w.writerow([mid, ";".join(c)])
    assert sp.submission_bytes(["m1", "m2"], preds) == buf.getvalue().encode()
    assert sp.submission_bytes(["m1", "m2"], preds).endswith(b"m2,\r\n")
