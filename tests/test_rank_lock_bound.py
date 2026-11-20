import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cv"))

from rank_lock_bound import best_case, lock_bound  # noqa: E402


def test_lock2_only_ranks_below_three_count():
    # 1 位・2 位・3 位・4 位・10 位・圏外
    rr = [1.0, 0.5, 1 / 3, 0.25, 0.1, 0.0]
    r = lock_bound(rr, 2)
    assert r["n_movable"] == 2           # 4 位と 10 位だけが 3 位まで上がれる
    assert r["n_top"] == 3
    assert r["n_absent"] == 1
    assert abs(r["gain_sum"] - round((1 / 3 - 0.25) + (1 / 3 - 0.1), 4)) < 1e-9
    assert abs(r["gain_bound"] - round(r["gain_sum"] / 6, 4)) < 1e-4


def test_lock1_matches_net_mde_definition():
    rr = [1.0, 0.5, 0.25, 0.0]
    r = lock_bound(rr, 1)
    assert r["n_movable"] == 1
    assert abs(r["gain_sum"] - 0.25) < 1e-9


def test_lock0_is_unlocked():
    rr = [1.0, 0.5, 0.0]
    assert abs(lock_bound(rr, 0)["gain_sum"] - 0.5) < 1e-9


def test_best_case_moves_only_movable_and_matches_bound():
    ctrl = {"a": 1.0, "b": 0.5, "c": 1 / 3, "d": 0.25, "e": 0.1, "f": 0.0}
    new = best_case(ctrl, 2)
    assert new == {"a": 1.0, "b": 0.5, "c": 1 / 3, "d": 1 / 3, "e": 1 / 3, "f": 0.0}
    d = [new[m] - ctrl[m] for m in sorted(ctrl)]
    assert all(x >= 0 for x in d)                      # 最善の場合は悪化が 0
    r = lock_bound([ctrl[m] for m in sorted(ctrl)], 2)
    assert abs(sum(d) - r["gain_sum"]) < 1e-4          # 差の和 = 上限の和
