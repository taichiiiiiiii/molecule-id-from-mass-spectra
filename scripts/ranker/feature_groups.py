#!/usr/bin/env python3
"""特徴群を 1 つずつ外す台帳の行（Issue #257、計画 2026-11-26 昼の項目 2）の群の定義と、当てはめの引数。

群は値を見る前に固定（#257 の本文）。各列はちょうど 1 つの群に属する。

    python scripts/ranker/feature_groups.py --check          # 重なり・exp003 の列への包含・sha256 を表示
    python scripts/ranker/feature_groups.py --args exp004     # lambdarank_oof.py に渡す引数（1 行）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from lambdarank_oof import _MF, _REL, FEATURES, resolve_features  # noqa: E402

BASE_EXCLUDE = ("prior_unverified_pop",)   # exp001〜exp003 と同じ（群の比べの外）
BASE_EXTRA = ("cc", "cc_z")                # exp003 = exp002 + cc,cc_z（#253）
GROUPS = {
    "R": tuple(f"{v}_{r}" for v in ("fz", "tani_pred", "lib_max_sim", "metfrag_lite", "np_score") for r in _REL)
    + ("cc_z",),
    "M": tuple(f"mf_{m}" for m in _MF) + tuple(f"mf_{m}_dmean" for m in _MF),
    "N": ("np_score", "has_np"),
}
EXPS = {"exp004": "R", "exp005": "M", "exp006": "N"}


def list_sha256(cols) -> str:
    return hashlib.sha256("\n".join(cols).encode()).hexdigest()


def exp003_features() -> tuple:
    return resolve_features(BASE_EXCLUDE, BASE_EXTRA)


def exp_args(exp_id: str) -> dict:
    """群を外した行の --exclude-features / --extra-features。追加の列（cc, cc_z）に入る群の列は extra から抜く。"""
    g = GROUPS[EXPS[exp_id]]
    exclude = BASE_EXCLUDE + tuple(c for c in g if c in FEATURES)
    extra = tuple(c for c in BASE_EXTRA if c not in g)
    feats = resolve_features(exclude, extra)
    base = exp003_features()
    assert set(base) - set(feats) == set(g) and set(feats) <= set(base), exp_id
    return dict(exclude=list(exclude), extra=list(extra), removed=list(g), removed_sha256=list_sha256(g),
                n_features=len(feats))


def check() -> dict:
    base = set(exp003_features())
    allc = [c for g in GROUPS.values() for c in g]
    assert len(allc) == len(set(allc)), "群が重なる"
    assert set(allc) <= base, "exp003 に無い列が群にある"
    return {k: dict(n=len(v), sha256=list_sha256(v)) for k, v in GROUPS.items()} | {"n_exp003": len(base)}


def compare(base_dir: Path, exp_dirs) -> dict:
    """型ごとの (exp − base) の MRR@25 の差と 95% 区間（evaluate.py と同じ対応のある bootstrap、2,000 回、seed 0）。
    per_molecule.csv の分子の一覧が同じでなければ止める。記述だけ（採否・群の組み替えには使わない）。"""
    import pandas as pd
    from evaluate import TYPES, paired_bootstrap
    b = pd.read_csv(base_dir / "per_molecule.csv", dtype={"molecule_id": str}).set_index(["set", "molecule_id"])
    out = {}
    for d in exp_dirs:
        e = pd.read_csv(Path(d) / "per_molecule.csv", dtype={"molecule_id": str}).set_index(["set", "molecule_id"])
        if set(e.index) != set(b.index):
            raise RuntimeError(f"{d}: 分子の一覧が土台と違う")
        e = e.loc[b.index]
        r = {}
        for t in TYPES:
            m = (b["type"] == t).to_numpy()
            lo, hi = paired_bootstrap(e["rr"].to_numpy()[m], b["rr"].to_numpy()[m])
            r[t] = dict(n=int(m.sum()), delta=float((e["rr"].to_numpy()[m] - b["rr"].to_numpy()[m]).mean()), lo=lo, hi=hi)
        out[Path(d).name] = r
    return out


SEEDS = (0, 1, 2)          # #272 の改定（#282）: 版 A の dev の関門は seed 0・1・2 の 3 本の平均で読む
GATE = -0.02               # 5 型すべてで（候補の平均 − 対照の平均）≥ GATE（丸める前の値で比べる）
# seed 以外は同じでなければならない meta の鍵（候補・対照それぞれの 3 本の中で）
SAME_KEYS = ("table_sha256", "features_sha256", "rows", "num_boost_round", "lightgbm", "exclude_features",
             "extra_features", "train_types", "n_rows")
SAME_LISTS = ("exclude_train_molecules", "eval_molecules")


def _load_run(d: Path):
    import pandas as pd
    meta = json.loads((Path(d) / "meta.json").read_text())
    pm = pd.read_csv(Path(d) / "per_molecule.csv", dtype={"molecule_id": str}).set_index(["set", "molecule_id"])
    return meta, pm


def _check_triplet(name: str, dirs, seeds=SEEDS) -> list:
    """3 本の seed がちょうど seeds で、seed 以外の設定が同じことを確かめる。欠け・重複・設定の違いは止める。"""
    if len(dirs) != len(seeds):
        raise RuntimeError(f"{name}: 当てはめが {len(dirs)} 本（{len(seeds)} 本が要る）")
    runs = [_load_run(d) for d in dirs]
    got = sorted(int(m["seed"]) for m, _ in runs)
    if got != sorted(seeds):
        raise RuntimeError(f"{name}: seed の組が {got}（{sorted(seeds)} が要る）")
    m0 = runs[0][0]
    for (m, _), d in zip(runs, dirs):
        if int(m["params"]["seed"]) != int(m["seed"]):
            raise RuntimeError(f"{d}: params の seed と meta の seed が違う")
        for k in SAME_KEYS:
            if m.get(k) != m0.get(k):
                raise RuntimeError(f"{name}: {d} の {k} が他の seed と違う")
        for k in SAME_LISTS:
            a, b = m.get(k) or {}, m0.get(k) or {}
            if a.get("sha256") != b.get("sha256"):
                raise RuntimeError(f"{name}: {d} の {k} の sha256 が他の seed と違う")
    return runs


def seed_mean_compare(cand_dirs, ctrl_dirs, seeds=SEEDS, gate=GATE) -> dict:
    """5 型ごとに 3 seed の MRR@25 の平均の差（候補 − 対照）、関門（すべて ≥ gate、丸める前）と、記述の 95% 区間
    （分子ごとの rr の 3 seed 平均どうしの対応のある bootstrap、evaluate.py と同じ 2,000 回・seed 0）。
    6 本の per_molecule.csv の分子の集合・型が同じでなければ止める。候補と対照で評価・学習の除外の一覧が違っても止める。"""
    import numpy as np
    from evaluate import TYPES, paired_bootstrap
    cand = _check_triplet("候補", cand_dirs, seeds)
    ctrl = _check_triplet("対照", ctrl_dirs, seeds)
    for k in SAME_LISTS:
        if (cand[0][0].get(k) or {}).get("sha256") != (ctrl[0][0].get(k) or {}).get("sha256"):
            raise RuntimeError(f"候補と対照で {k} の sha256 が違う")
    idx = ctrl[0][1].index
    types = ctrl[0][1]["type"]
    for _, pm in cand + ctrl:
        if set(pm.index) != set(idx) or len(pm) != len(idx):
            raise RuntimeError("per_molecule.csv の分子の集合が 6 本で同じでない")
        if not (pm.loc[idx, "type"] == types).all():
            raise RuntimeError("per_molecule.csv の型が 6 本で同じでない")
    rr_c = np.stack([pm.loc[idx, "rr"].to_numpy(float) for _, pm in cand])   # (3, n)
    rr_k = np.stack([pm.loc[idx, "rr"].to_numpy(float) for _, pm in ctrl])
    out, ok = {}, True
    for t in TYPES:
        m = (types == t).to_numpy()
        if not m.any():
            raise RuntimeError(f"型 {t} の分子が無い")
        mc = [float(r[m].mean()) for r in rr_c]
        mk = [float(r[m].mean()) for r in rr_k]
        delta = float(np.mean(mc) - np.mean(mk))
        lo, hi = paired_bootstrap(rr_c[:, m].mean(axis=0), rr_k[:, m].mean(axis=0))
        passed = bool(delta >= gate)
        ok &= passed
        out[t] = dict(n=int(m.sum()), cand_mrr_by_seed=mc, ctrl_mrr_by_seed=mk, cand_mean=float(np.mean(mc)),
                      ctrl_mean=float(np.mean(mk)), delta=delta, lo=lo, hi=hi, gate_pass=passed)
    return dict(seeds=list(seeds), gate=gate, cand=[str(d) for d in cand_dirs], ctrl=[str(d) for d in ctrl_dirs],
                types=out, gate_pass_all=bool(ok))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--args", choices=sorted(EXPS))
    ap.add_argument("--compare", nargs="+", help="土台の exp dir に続けて比べる exp dir（型ごとの exp − 土台）")
    ap.add_argument("--seed-mean", action="store_true", help="#282: --cand 3 本と --ctrl 3 本の 3 seed 平均の差と関門")
    ap.add_argument("--cand", nargs="+")
    ap.add_argument("--ctrl", nargs="+")
    a = ap.parse_args(argv)
    if a.seed_mean:
        print(json.dumps(seed_mean_compare([Path(x) for x in a.cand or []], [Path(x) for x in a.ctrl or []]),
                         indent=1))
    if a.compare:
        print(json.dumps(compare(Path(a.compare[0]), a.compare[1:]), indent=1))
    if a.check:
        print(json.dumps(check(), indent=1))
    if a.args:
        r = exp_args(a.args)
        print(f"--exclude-features {','.join(r['exclude'])} --extra-features {','.join(r['extra'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
