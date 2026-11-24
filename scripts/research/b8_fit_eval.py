#!/usr/bin/env python3
"""
B8 振り分け器（Issue #216）: 向きを制約したロジスティック回帰の当てはめと、fold 1〜4 の交差での 1 回だけの判定。

事前登録: 下書き issuecomment-7121558819 §3〜§6 ＋ 確定 issuecomment-7134581003。
  §3 モデル: 特徴 8 個（b8_features.FEATURES）を学習側の平均・標準偏差で標準化、目的 = Σ logloss + ‖w‖²/(2C)、C = 1、
     切片は罰なし・向きなし、scipy L-BFGS-B（既定の許容誤差）、+ の係数 ≥ 0・− の係数 ≤ 0・0 は制約なし。重みなし
     欠け・無限: |Q|=0 → conf_Q・gap_Q は 0、d は学習側の d の有限の最大。+inf → 学習側の有限の最大、−inf → 有限の最小
     参考（採否に使わない）: conf_L だけの 1 変数モデル（向き +）
  §4 合成: L の r 番目に p/r、Q の s 番目に (1 − p)/s、重みの降順（同点は L 側が先、同じ側なら順位順）、キーで重複除去、上位 25
  §5 評価: fold j を評価、残り 3 fold の (a)〜(d) の 12 集合で当てはめ、fold j の 5 集合（(e) を含む）に p を出す。
     型ごとに 4 回分を合わせ、分子ごとの rr の差（新 − 対照）を分子の bootstrap 2000 回（seed 0、パーセンタイル 95%）
  規則（--rule で固定して渡す。log に記録）: r1_and_s6 = (a) 下限 > 0 ∧ (b)(c)(d)(e) 平均 ≥ −0.02 ∧ 5 型とも下限 ≥ −0.02。
     5 型のどれかが欠ければ採用を拒む。参考: s6_only = 5 型とも下限 ≥ −0.02 ∧ 1 つ以上で下限 > 0

    .venv/bin/python scripts/research/b8_fit_eval.py --rule r1_and_s6 --out outputs/local_eval/b8_fit_eval.json
20 集合の b8_features.json がそろっていなければ何もせず止める（部分の集合で読まない）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/research", "scripts/validate", "data/ext/v4g/casmi26-v3-models/code"):
    sys.path.insert(0, str(ROOT / p))

from b8_features import FEATURES, SIGNS  # noqa: E402

FOLDS = (1, 2, 3, 4)
TRAIN_TYPES = ("a", "b", "c", "d")
ALL_TYPES = ("a", "b", "c", "d", "e")
C_REG, N_BOOT, BOOT_SEED, K = 1.0, 2000, 0, 25
MARGIN = -0.02
RULES = ("r1_and_s6",)


# ---------------------------------------------------------------- 欠け・無限の置き換え（§2）
def _num(v):
    if v is None:
        return math.nan
    if v == "inf":
        return math.inf
    if v == "-inf":
        return -math.inf
    return float(v)


def raw_matrix(mols: list[dict], feats=FEATURES) -> np.ndarray:
    return np.array([[_num(m[f]) for f in feats] for m in mols], np.float64).reshape(len(mols), len(feats))


def impute_stats(Xr: np.ndarray, feats=FEATURES) -> dict:
    """学習側の生の行列から、列ごとの有限の最大・最小（d は |Q|>0 の有限の d の最大を含む）。"""
    st = {}
    for j, f in enumerate(feats):
        col = Xr[:, j]
        fin = col[np.isfinite(col)]
        st[f] = (float(fin.max()) if fin.size else 0.0, float(fin.min()) if fin.size else 0.0)
    return st


def impute(Xr: np.ndarray, st: dict, feats=FEATURES) -> np.ndarray:
    X = Xr.copy()
    for j, f in enumerate(feats):
        col = X[:, j]
        hi, lo = st[f]
        if f in ("conf_Q", "gap_Q"):
            col[np.isnan(col)] = 0.0        # |Q| = 0 → 0（先に当てる）
        elif f == "d":
            col[np.isnan(col)] = hi         # |Q| = 0 → 学習側の d の最大
        col[col == math.inf] = hi
        col[col == -math.inf] = lo
        assert np.isfinite(col).all(), f"{f}: 置き換えの後に有限でない値"
        X[:, j] = col
    return X


# ---------------------------------------------------------------- §3 モデル
def fit_lr(X: np.ndarray, y: np.ndarray, signs, C: float = C_REG) -> dict:
    """向きを制約したロジスティック回帰。X は置き換え済みの生の特徴。標準化の値と係数を返す。"""
    from scipy.optimize import minimize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y = y.astype(np.float64)

    def fg(th):
        b, w = th[0], th[1:]
        s = b + Z @ w
        loss = float(np.sum(np.logaddexp(0.0, s) - y * s) + w @ w / (2 * C))
        r = 1.0 / (1.0 + np.exp(-s)) - y
        g = np.concatenate([[r.sum()], Z.T @ r + w / C])
        return loss, g

    bounds = [(None, None)] + [((0.0, None) if sg > 0 else (None, 0.0) if sg < 0 else (None, None)) for sg in signs]
    res = minimize(fg, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", bounds=bounds)
    return dict(mu=mu, sd=sd, b=float(res.x[0]), w=res.x[1:].copy(), success=bool(res.success), nit=int(res.nit),
                fun=float(res.fun))


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    s = model["b"] + ((X - model["mu"]) / model["sd"]) @ model["w"]
    return 1.0 / (1.0 + np.exp(-s))


# ---------------------------------------------------------------- §4 合成
def compose(L: list, Q: list, p: float, k: int = K) -> list:
    """L・Q は [key, smiles] の順位順。L の r 番目に p/r、Q の s 番目に (1−p)/s。重みの降順、同点は L 側が先・順位順、
    キーで重複除去、上位 k の SMILES。"""
    items = [(p / r, 0, r, key, smi) for r, (key, smi) in enumerate(L, 1)]
    items += [((1.0 - p) / s, 1, s, key, smi) for s, (key, smi) in enumerate(Q, 1)]
    items.sort(key=lambda t: (-t[0], t[1], t[2]))
    out, seen = [], set()
    for _w, _side, _r, key, smi in items:
        if key in seen:
            continue
        seen.add(key); out.append(smi)
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- §5・規則
def boot_ci(delta: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    n = len(delta)
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def decide(per_type: dict, rule: str) -> dict:
    """per_type[t] = dict(n, mean, lo, hi)。5 型のどれかが欠ける（無い・n=0）なら採用を拒む。"""
    assert rule in RULES, f"規則 {rule} は事前登録に無い"
    missing = [t for t in ALL_TYPES if t not in per_type or not per_type[t].get("n")]
    s6_ref = None
    if missing:
        return dict(rule=rule, adopt=False, refused=True, missing=missing, conditions={}, s6_only_reference=s6_ref)
    cond = {"a_lower_gt_0": per_type["a"]["lo"] > 0}
    for t in ("b", "c", "d", "e"):
        cond[f"{t}_mean_ge_-0.02"] = per_type[t]["mean"] >= MARGIN
    for t in ALL_TYPES:
        cond[f"{t}_lower_ge_-0.02"] = per_type[t]["lo"] >= MARGIN
    s6_ref = bool(all(per_type[t]["lo"] >= MARGIN for t in ALL_TYPES) and any(per_type[t]["lo"] > 0 for t in ALL_TYPES))
    return dict(rule=rule, adopt=bool(all(cond.values())), refused=False, missing=[], conditions=cond,
                s6_only_reference=s6_ref)


def set_dir(k: int, t: str) -> Path:
    return ROOT / f"outputs/local_eval/b8_f{k}_{t}_t10"


def load_all() -> dict:
    sets, missing = {}, []
    for k in FOLDS:
        for t in ALL_TYPES:
            p = set_dir(k, t) / "b8_features.json"
            if not p.is_file():
                missing.append(str(p))
                continue
            sets[(k, t)] = json.loads(p.read_text())
    if missing:
        raise SystemExit(f"rc4: b8_features.json がそろっていない（{len(missing)} 集合）。部分の集合では読まない: {missing[:3]}")
    return sets


def rr_lists(sets: dict) -> dict:
    """集合ごとの truth キー（mrr25）。"""
    import pandas as pd  # noqa: F401
    from mrr25 import read_truth, truth_keys
    return {kt: {m: truth_keys(s) for m, s in read_truth(set_dir(*kt) / "truth.csv").items()} for kt in sets}


def run(rule: str) -> dict:
    from mrr25 import reciprocal_rank
    sets = load_all()
    tk = rr_lists(sets)
    rows = {}
    for kt, s in sets.items():
        ms = [m for m in s["molecules"] if not m.get("missing_logits")]
        assert len(ms) == len(s["molecules"]), f"{kt}: ロジットの無い分子がある"
        rows[kt] = ms
    out = dict(rule=rule, C=C_REG, features=list(FEATURES), signs=SIGNS, rounds={}, per_type={}, per_type_baseline={})
    deltas = {t: [] for t in ALL_TYPES}
    deltas_b = {t: [] for t in ALL_TYPES}
    ctrl_rr = {t: [] for t in ALL_TYPES}
    new_rr = {t: [] for t in ALL_TYPES}
    for j in FOLDS:
        tr = [m for (k, t), ms in rows.items() if k != j and t in TRAIN_TYPES for m in ms]
        Xr = raw_matrix(tr)
        st = impute_stats(Xr)
        X = impute(Xr, st)
        y = np.array([m["y"] for m in tr])
        model = fit_lr(X, y, [SIGNS[f] for f in FEATURES])
        jc = FEATURES.index("conf_L")
        base = fit_lr(X[:, [jc]], y, [+1])
        rd = dict(n_train=len(tr), y_mean=float(y.mean()), success=model["success"], nit=model["nit"],
                  w=dict(zip(FEATURES, model["w"].round(6).tolist())), b=round(model["b"], 6),
                  baseline_w=float(base["w"][0]), baseline_b=float(base["b"]),
                  n_train_by_type={t: sum(len(rows[(k, t)]) for k in FOLDS if k != j) for t in TRAIN_TYPES},
                  impute_stats=st, eval={})
        for t in ALL_TYPES:
            ms = rows[(j, t)]
            Xe = impute(raw_matrix(ms), st)
            p = predict(model, Xe)
            pb = predict(base, Xe[:, [jc]])
            for m, pi, pbi in zip(ms, p, pb):
                keys = tk[(j, t)][m["molecule_id"]]
                c = reciprocal_rank(keys, [s_ for _, s_ in m["L"][:K]])
                n_ = reciprocal_rank(keys, compose(m["L"], m["Q"], float(pi)))
                nb = reciprocal_rank(keys, compose(m["L"], m["Q"], float(pbi)))
                deltas[t].append(n_ - c); deltas_b[t].append(nb - c); ctrl_rr[t].append(c); new_rr[t].append(n_)
            rd["eval"][t] = dict(n=len(ms), p_mean=float(p.mean()) if len(p) else None,
                                 y_mean=float(np.mean([m["y"] for m in ms])) if ms else None)
        out["rounds"][j] = rd
    for t in ALL_TYPES:
        dlt = np.array(deltas[t]); db = np.array(deltas_b[t])
        lo, hi = boot_ci(dlt)
        out["per_type"][t] = dict(n=len(dlt), control_mrr=round(float(np.mean(ctrl_rr[t])), 4),
                                  new_mrr=round(float(np.mean(new_rr[t])), 4), mean=float(dlt.mean()), lo=lo, hi=hi)
        blo, bhi = boot_ci(db)
        out["per_type_baseline"][t] = dict(n=len(db), mean=float(db.mean()), lo=blo, hi=bhi)
    out["decision"] = decide(out["per_type"], rule)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", required=True, choices=RULES, help="確定の事前登録で固定した規則（issuecomment-7134581003）")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/b8_fit_eval.json"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"rule = {args.rule}（確定の事前登録 issuecomment-7134581003 で固定）", flush=True)
    res = run(args.rule)
    res["seconds"] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str) + "\n")
    print(json.dumps(dict(per_type=res["per_type"], decision=res["decision"]), ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
