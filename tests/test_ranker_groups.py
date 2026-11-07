import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ranker_build_groups", ROOT / "scripts/ranker/build_groups.py")
bg = importlib.util.module_from_spec(_spec)
sys.modules["ranker_build_groups"] = bg
_spec.loader.exec_module(bg)


def make_st():
    # identity_group 0 は np 起点 A と同じ群の B を含む。C は fold 0、D/E/F は fold 1
    return pd.DataFrame(dict(inchikey14=list("ABCDEFG"), identity_group=[0, 0, 1, 2, 3, 4, 5],
                             fold_identity=[1, 0, 0, 1, 1, 1, 2]))


def test_np_is_whole_identity_group():
    assert bg.np_keys({"A"}, make_st()) == {"A", "B"}


def test_reproduce_split_np_first_then_fold():
    st = make_st()
    rep = bg.reproduce_split(set("ABCDEFG"), st, bg.np_keys({"A"}, st), 0)
    assert rep == dict(A="np", B="np", C="hold", D="train", E="train", F="train", G="train")


def test_training_pool_excludes_np_split_and_v3():
    st = make_st()
    npk = bg.np_keys({"A"}, st)
    pool = bg.training_pool(set("ABCDEFG"), st, npk, split_keys={"E"}, v3_keys={"F"}, fold=1)
    assert pool == {"D"}            # A は np、E は split、F は v3、G は別 fold
    assert pool.isdisjoint(npk | {"E", "F"})


def test_training_pool_only_train_keys():
    st = make_st()
    assert bg.training_pool({"D"}, st, set(), set(), set(), 1) == {"D"}


def test_balance_keeps_one_to_one():
    a, b = bg.balance(list("abc"), list("xy"))
    assert (a, b) == (list("ab"), list("xy"))
