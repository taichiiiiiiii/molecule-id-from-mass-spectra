"""scripts/ranker/error_analysis.py の単体テスト（#257）: 20 件の抜き出しが決まった順で再現する。"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
import error_analysis as EA  # noqa: E402


def _per(seed=0, n=12):
    rng = np.random.default_rng(seed)
    rows = []
    for t in EA.TYPES:
        for i in range(n):
            rows.append(dict(set=f"s_{t}", molecule_id=f"m{t}_{i:03d}", type=t,
                             rank_exp=float(rng.integers(1, 6)),
                             rank_ctrl=float(rng.integers(1, 6)) if t in "abd" else np.inf,
                             rank_q=float(rng.integers(1, 6)) if t in "ce" else np.inf,
                             top_row=i, truth_row=i))
    return pd.DataFrame(rows)


def test_selection_is_deterministic_and_sha1_ordered():
    per = _per()
    sel = EA.select_cases(per)
    # 入力の行の順を変えても同じ件が同じ順で出る
    for s in range(5):
        shuf = per.sample(frac=1.0, random_state=s).reset_index(drop=True)
        sel2 = EA.select_cases(shuf)
        assert sel2["molecule_id"].tolist() == sel["molecule_id"].tolist()
    # 期待: 型ごとに、基準（a・b・d は ctrl、c・e は q）より下がった分子を sha1 昇順で先頭 4 件
    exp = []
    for t in EA.TYPES:
        g = per[per.type == t]
        base = g["rank_ctrl"] if t in "abd" else g["rank_q"]
        down = g[g["rank_exp"] > base]
        ids = sorted(down["molecule_id"], key=lambda m: hashlib.sha1(m.encode()).hexdigest())[:4]
        exp += ids
    assert sel["molecule_id"].tolist() == exp
    assert len(sel) <= 20 and all((sel.groupby("type").size() <= 4).values)
    # 3 つの数の合計 = 型の分子数（両方 ∞ は「変わらない」）
    c = EA.count_by_type(per, "ctrl")
    assert all(v["down"] + v["up"] + v["same"] == v["n"] == 12 for v in c.values())
    assert c["c"]["up"] == 12 and c["e"]["up"] == 12   # 対照に居ない正解が候補にあれば「上がった」


def test_truth_ranks_dedup_and_missing():
    df = pd.DataFrame(dict(
        set="s", molecule_id=["m1"] * 4 + ["m2"] * 2, type="a", row=range(6),
        score=[0.9, 0.8, 0.7, 0.1, 0.5, 0.4], cand_key=["K1", "K1", "K2", "K3", "A", "B"],
        label=[0, 0, 1, 0, 0, 0], ctrl_rank=[1, 61, 2, 3, 1, 2], q_rank=[61, 1, 61, 61, 61, 61]))
    per = EA.truth_ranks(df).set_index("molecule_id")
    assert per.loc["m1", "rank_exp"] == 2.0 and per.loc["m1", "rank_ctrl"] == 2.0 and per.loc["m1", "top_row"] == 0
    assert np.isinf(per.loc["m2", "rank_exp"]) and np.isinf(per.loc["m2", "rank_ctrl"])
    assert list(EA.classify([3, 1, np.inf], [2, 2, np.inf])) == ["down", "up", "same"]
