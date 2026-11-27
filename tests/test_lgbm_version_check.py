"""Issue #284: lightgbm の版の点検（scripts/kaggle/lgbm_version_check.py）の比べる段。"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


VC = _load("lgbm_version_check", "scripts/kaggle/lgbm_version_check.py")


def _keys(n_mol=3, n_cand=30, dup_every=0):
    rows = []
    for m in range(n_mol):
        for c in range(n_cand):
            key = f"K{m}_{c // 2}" if dup_every and c % dup_every == 1 else f"K{m}_{c}"
            rows.append(dict(set="s1", molecule_id=f"m{m}", cand_key=key, cand_smiles=f"S{m}_{c}"))
    return pd.DataFrame(rows)


def test_identical_scores_are_same():
    k = _keys()
    s = np.random.default_rng(0).normal(size=len(k))
    r = VC.compare_scores(k, s, s.copy())
    assert r["verdict"] == "同じ"
    assert r["max_abs_diff"] == 0.0
    assert r["frac_rows_exact_equal"] == 1.0
    assert r["n_molecules"] == 3 and r["n_molecules_topk_identical"] == 3
    assert r["frac_molecules_topk_identical"] == 1.0


def test_small_diff_without_order_change_is_same():
    k = _keys()
    s = np.arange(len(k), dtype=float)
    r = VC.compare_scores(k, s, s + 1e-9)
    assert r["verdict"] == "同じ"
    assert r["max_abs_diff"] > 0
    assert r["n_rows_exact_equal"] == 0


def test_swap_in_one_molecule_is_different():
    k = _keys()
    s = -np.arange(len(k), dtype=float)   # 各分子で行の順が上位
    t = s.copy()
    t[[31, 32]] = t[[32, 31]]   # 分子 m1 の 2 位と 3 位を入れ替える
    r = VC.compare_scores(k, s, t)
    assert r["verdict"] == "違う"
    assert r["n_molecules_topk_identical"] == 2
    assert r["differing_molecules"] == [["s1", "m1"]]
    assert r["n_molecules_top1_differs"] == 0
    assert r["frac_molecules_topk_identical"] == pytest.approx(2 / 3)


def test_change_outside_top25_is_same():
    k = _keys(n_cand=30)
    s = -np.arange(len(k), dtype=float)
    t = s.copy()
    t[[27, 28]] = t[[28, 27]]   # 分子 m0 の 28 位と 29 位だけ入れ替え（上位 25 の外）
    assert VC.compare_scores(k, s, t)["verdict"] == "同じ"


def test_tie_broken_by_row_order_in_both():
    k = _keys(n_mol=1, n_cand=5)
    s = np.zeros(5)
    r = VC.compare_scores(k, s, s.copy())
    assert r["verdict"] == "同じ"
    t = s.copy()
    t[4] = 1e-12   # 同点が崩れると順が変わる
    assert VC.compare_scores(k, s, t)["verdict"] == "違う"


def test_dedup_by_cand_key_same_as_submission():
    from evaluate import ranked_lists
    k = _keys(n_mol=1, n_cand=30, dup_every=2)
    s = -np.arange(len(k), dtype=float)
    base = k.copy()
    base.insert(0, "row", np.arange(len(k)))
    lst = ranked_lists(base.assign(score=s), 25)[("s1", "m0")]
    assert len(lst) == len(set(lst))
    assert VC.compare_scores(k, s, s.copy())["verdict"] == "同じ"


def test_rejects_length_mismatch_and_nonfinite():
    k = _keys()
    s = np.zeros(len(k))
    with pytest.raises(ValueError):
        VC.compare_scores(k, s, s[:-1])
    bad = s.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        VC.compare_scores(k, s, bad)
    with pytest.raises(ValueError):
        VC.compare_scores(k.drop(columns=["cand_key"]), s, s)


def test_predict_and_compare_cli(tmp_path):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    k = _keys(n_mol=4, n_cand=30)
    feats = ["f0", "f1", "f2"]
    for f in feats:
        k[f] = rng.normal(size=len(k))
    y = (rng.random(len(k)) < 0.05).astype(int)
    ds = lgb.Dataset(k[feats].to_numpy(np.float64), label=y, group=[30] * 4, feature_name=feats)
    bst = lgb.train(dict(objective="lambdarank", verbose=-1, seed=0, min_data_in_leaf=5), ds, num_boost_round=5)
    model = tmp_path / "m.txt"
    bst.save_model(str(model))
    (tmp_path / "f.json").write_text(json.dumps(feats))
    tab = tmp_path / "t.parquet"
    k.to_parquet(tab, index=False)
    for name in ("a", "b"):
        VC.main(["predict", "--expect-version", lgb.__version__, "--model", str(model), "--features",
                 str(tmp_path / "f.json"), "--table", str(tab), "--out", str(tmp_path / f"{name}.npz")])
    VC.main(["compare", "--a", str(tmp_path / "a.npz"), "--b", str(tmp_path / "b.npz"), "--table", str(tab),
             "--out", str(tmp_path / "r.json")])
    r = json.loads((tmp_path / "r.json").read_text())
    assert r["verdict"] == "同じ" and r["n_molecules"] == 4 and r["max_abs_diff"] == 0.0
    with pytest.raises(SystemExit):
        VC.main(["predict", "--expect-version", "0.0.0", "--model", str(model), "--features",
                 str(tmp_path / "f.json"), "--table", str(tab), "--out", str(tmp_path / "c.npz")])
    (tmp_path / "g.json").write_text(json.dumps(["f1", "f0", "f2"]))
    with pytest.raises(ValueError):
        VC.main(["predict", "--expect-version", lgb.__version__, "--model", str(model), "--features",
                 str(tmp_path / "g.json"), "--table", str(tab), "--out", str(tmp_path / "d.npz")])
