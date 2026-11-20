import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cv"))

from rank_lock_bound import lock_bound  # noqa: E402


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
