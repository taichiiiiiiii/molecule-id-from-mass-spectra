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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--args", choices=sorted(EXPS))
    ap.add_argument("--compare", nargs="+", help="土台の exp dir に続けて比べる exp dir（型ごとの exp − 土台）")
    a = ap.parse_args(argv)
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
