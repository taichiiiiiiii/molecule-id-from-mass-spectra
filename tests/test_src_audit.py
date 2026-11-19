"""Issue #162: scripts/cv/src_audit.py の純粋な関数のテスト（合成データだけ。プール・train・COCONUT は読まない）。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/cv"))

import src_audit as S  # noqa: E402


def _table(train_any_2=0, coco_any_1=0, train_sid_13=None):
    # src 1: 4 行（train）、src 2: 4 行（COCONUT だけ）、src 3: 2 行（両方）
    src = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
    train_any = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], bool)
    coco_any = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], bool)
    train_any[4:4 + train_any_2] = True
    coco_any[0:coco_any_1] = True
    sid = train_any.copy() if train_sid_13 is None else train_sid_13
    return S.crosstab(src, dict(train_any=train_any, coco_any=coco_any, train_sid_ge0=sid))


def test_crosstab_counts():
    t = _table()
    assert t["1"] == dict(n=4, train_any=4, coco_any=0, train_sid_ge0=4)
    assert t["2"] == dict(n=4, train_any=0, coco_any=4, train_sid_ge0=0)
    assert t["3"] == dict(n=2, train_any=2, coco_any=2, train_sid_ge0=2)
    assert S.rate(t, (1, 3), "train_any") == 1.0
    assert S.rate(t, (2,), "coco_any") == 1.0
    assert S.rate(t, (4,), "coco_any") is None


def test_judge_confirmed_and_each_failure():
    assert S.judge(_table(), True)["verdict"] == "確かめた"
    j = S.judge(_table(), False)
    assert j["verdict"] == "未確認" and j["failed"] == ["i_code_definition"]
    j = S.judge(_table(train_any_2=1), True)          # src 2 の 1/4 が train → 1% を超える
    assert j["failed"] == ["ii_src2_train_any_le"]
    j = S.judge(_table(coco_any_1=1), True)           # src 1 の 1/4 が COCONUT
    assert j["failed"] == ["iii_src1_coco_any_le"]
    sid = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1], bool)   # src 1 の 1 行が train_sid < 0
    j = S.judge(_table(train_sid_13=sid), True)
    assert j["failed"] == ["ii_src13_train_sid_ge0_ge"]


def test_judge_threshold_is_inclusive():
    # 100 行中 99 行 = ちょうど 99% は合格、1 行 = ちょうど 1% は合格
    src = np.array([1] * 100 + [2] * 100)
    train_any = np.array([True] * 99 + [False] + [True] + [False] * 99)
    coco_any = np.array([True] + [False] * 99 + [True] * 100)
    t = S.crosstab(src, dict(train_any=train_any, coco_any=coco_any, train_sid_ge0=train_any))
    t["3"] = dict(n=0, train_any=0, coco_any=0, train_sid_ge0=0)
    assert S.judge(t, True)["verdict"] == "確かめた"


def test_window_src_and_counts():
    keys = np.array(["A", "B", "C", "D", "E"], dtype=object)
    src = np.array([1, 3, 2, 2, 3])
    t, d = S.window_src(keys, src, np.array([1, 2, 3]), "B")
    assert t == 3 and d.tolist() == [2, 2]
    t, d = S.window_src(keys, src, np.array([2, 3, 4]), "B")
    assert t is None and d.tolist() == [2, 2, 3]
    assert S.counts([2, 2, 3]) == {"2": 2, "3": 1}
    assert S.counts([]) == {}


def test_check_code(tmp_path):
    p = tmp_path / "05_build_pool.py"
    lines = ["x"] * 40
    for n, s in S.TC_LINES.items():
        lines[n - 1] = "    " + s + " extra"
    p.write_text("\n".join(lines))
    r = S.check_code(p)
    assert r["lines_match"] is True and r["sha_match"] is False and r["ok"] is False   # sha が違えば根拠にしない
    assert S.check_code(tmp_path / "missing.py")["ok"] is False


def test_cid_present_treats_minus_one_as_absent():
    assert S.cid_present([123, -1, np.nan, 0, 5]).tolist() == [True, False, False, False, True]
