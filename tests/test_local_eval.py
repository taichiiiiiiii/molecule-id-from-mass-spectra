import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("local_eval_run", ROOT / "scripts/local_eval/run.py")
le = importlib.util.module_from_spec(_spec)
sys.modules["local_eval_run"] = le
_spec.loader.exec_module(le)


def make_pool():
    meta = pd.DataFrame(dict(key=["K1", "K2", "K2", "K3", "K4"], smiles=["a", "b", "b2", "c", "d"],
                             mass=[300.0, 100.0, 100.0005, 100.0009, 100.01], pid=[0, 1, 2, 3, 4]))
    pop = np.array([0.0, 1.0, 1.0, 20.0, 0.0])
    return le.Pool(meta, pop)


def test_window_is_ppm_and_mass_sorted():
    p = make_pool()
    idx = p.window(100.0, ppm=10.0)  # ±0.001
    assert sorted(p.key[idx].tolist()) == ["K2", "K2", "K3"]
    assert p.window(500.0).size == 0


def test_dedup_keeps_first_of_each_key():
    p = make_pool()
    ki = {k: i for i, k in reversed(list(enumerate(p.key)))}
    rows = [(5.0, ki["K2"]), (4.0, ki["K2"] + 1), (3.0, ki["K3"])]
    out = le.dedup(rows, p, 25)
    assert [p.key[i] for _, i in out] == ["K2", "K3"]
    assert le.dedup(rows, p, 1) == rows[:1]


def test_prior_reorders_by_standardised_score_plus_pop():
    p = make_pool()
    i2 = int(np.where(p.key == "K2")[0][0]); i3 = int(np.where(p.key == "K3")[0][0])
    rows = [(10.0, i2), (9.0, i3)]
    # z: K2 +1, K3 -1; pop: K2 1, K3 20 -> mu=0.15: K2 1.15, K3 2.0
    out = le.rerank_with_prior(rows, p, mu=0.15)
    assert [p.key[i] for _, i in out] == ["K3", "K2"]
    assert [p.key[i] for _, i in le.rerank_with_prior(rows, p, mu=0.0)] == ["K2", "K3"]
