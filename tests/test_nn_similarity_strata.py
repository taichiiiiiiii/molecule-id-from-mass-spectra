import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/research"))

import nn_similarity_strata as ns  # noqa: E402


def test_stratum_edges():
    assert [ns.stratum(x) for x in (0.0, 0.4999, 0.5, 0.7999, 0.8, 1.0)] == [
        "<0.5", "<0.5", "0.5-0.8", "0.5-0.8", ">=0.8", ">=0.8"]
    with pytest.raises(ValueError):
        ns.stratum(float("nan"))


def test_boot_ci_deterministic_and_bounds():
    x = [0.0, 1.0, 0.5, 1.0, 0.2]
    a, b = ns.boot_ci(x), ns.boot_ci(x)
    assert a == b
    assert a[0] <= np.mean(x) <= a[1]
    assert ns.boot_ci([]) == (None, None)
    assert ns.boot_ci([1.0, 1.0]) == (1.0, 1.0)


def test_summarize_counts_and_mrr():
    sim = dict(a=0.3, b=0.6, c=0.9, d=0.85, e=0.1)
    rr = dict(a=0.0, b=0.5, c=1.0, d=0.5, e=1.0)
    s = ns.summarize(sim, rr)
    assert s["<0.5"]["n"] == 2 and s["<0.5"]["mrr"] == 0.5
    assert s["0.5-0.8"]["n"] == 1 and s["0.5-0.8"]["mrr"] == 0.5
    assert s[">=0.8"]["n"] == 2 and s[">=0.8"]["mrr"] == 0.75
    assert s["all"]["n"] == 5 and s["all"]["mrr"] == 0.6
    assert s[">=0.8"]["n_top1"] == 1 and s["<0.5"]["n_rr0"] == 1


def test_summarize_empty_stratum_and_mismatch():
    s = ns.summarize(dict(a=0.1), dict(a=1.0))
    assert s[">=0.8"] == dict(n=0, mrr=None, ci95=None, n_top1=0, n_rr0=0)
    with pytest.raises(ValueError):
        ns.summarize(dict(a=0.1), dict(b=1.0))


def test_morgan_max_tanimoto():
    ref = ["CCO", "c1ccccc1O", "not_a_smiles", "CCCCCCCC"]
    fps, ok = ns.morgan_fps(ref)
    assert ok.tolist() == [0, 1, 3]
    q, _ = ns.morgan_fps(["CCO", "c1ccccc1"])
    best, arg = ns.max_tanimoto(q, fps)
    assert best[0] == 1.0 and arg[0] == 0
    assert 0.0 < best[1] < 1.0 and arg[1] == 1
