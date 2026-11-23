#!/usr/bin/env python3
"""
Issue #208（#195 A8）: いまの v3 の採否の規則
  「4 集合すべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、かつ v3_c1u_t10 の下限 > 0」
が、本当の効果 δ のときにどれだけの確率で採用になるか（検出力）を計算する。採点はしない。

計算の形は #208 の本文（03:21:22Z）で固定したもの:
- sd_k: outputs/local_eval/net_mde.json の集合ごと・変種ごとの sd_delta から、n_changed = 0 の変種と
  stale_excluded の変種を除き、型（集合の名前）ごとの中央値（主）・最小・最大。値が無い型は「未計算」で外す。
  json の diff（過去の変種の平均の差）は読まない。
- 下限は正規近似 L_k = mean_k − 1.96·sd_k/√n_k（sd_k は既知として扱う）。型の間は独立と仮定。
- P(採用) = Φ(δ_c1/se_c1 − 1.96) × Π_{k≠c1} Φ((δ_k + 0.02)/se_k − 1.96)。モンテカルロ（seed 0・10 万回）も並べる。

    python3 scripts/cv/adoption_power.py [--mde-json outputs/local_eval/net_mde.json] [--out outputs/local_eval/adoption_power/adoption_power.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

Z = 1.96          # 95% CI の片側
FLOOR = -0.02     # 「害なし」の下限
# 型 → (net_mde.json の集合の名前, A3a 拡大後の集合, n)
TYPES = {
    "C1": ("v3_c1", "v3_c1u_t10", 394),
    "enveda": ("enveda_v1", "enveda_v1u_t10", 373),
    "v3_c2n": ("v3_c2n", "v3_c2nu_t10", 394),
    "C2": ("v3_c2", "v3_c2_t10", 197),
}
PRIMARY = "C1"    # 下限 > 0 を求める型
DELTAS = (0.0, 0.005, 0.01, 0.02, 0.05)
SCENARIOS = {
    # 本当の効果を持つ型（他は 0。C2 型は両方で 0）
    "S1": ("C1",),
    "S2": ("C1", "enveda", "v3_c2n"),
}
STATS = ("median", "min", "max")
# lower = 今の規則（C1 以外も 95% CI 下限 ≥ −0.02）。point = 提案 R1（C1 以外は平均の差 ≥ −0.02、C1 は変えない）
GUARDS = ("lower", "point")
HARM = -0.02


def phi(x: float) -> float:
    """標準正規の累積分布関数。"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sd_by_type(mde: dict) -> dict:
    """型ごとに {median,min,max,values,n_variants} を返す。値が無い型は None。"""
    out = {}
    for t, (name, _, _) in TYPES.items():
        h = mde.get(name)
        if h is None:
            out[t] = None
            continue
        stale = set(h.get("stale_excluded", []))
        vals = [v["sd_delta"] for tag, v in sorted(h.get("variants", {}).items())
                if v.get("n_changed", 0) > 0 and tag not in stale]
        out[t] = (dict(median=statistics.median(vals), min=min(vals), max=max(vals),
                       values=vals, n_variants=len(vals)) if vals else None)
    return out


def type_prob(delta: float, sd: float, n: int, primary: bool, guard: str = "lower") -> float:
    """型 1 つが条件を満たす確率。primary は L > 0、それ以外は guard="lower" なら L ≥ FLOOR（今の規則）、
    guard="point" なら 平均の差 ≥ FLOOR（提案 R1）。sd = 0 なら L = 平均 = δ で決まる。"""
    thr = 0.0 if primary else FLOOR
    if sd == 0:
        return 1.0 if (delta > thr if primary else delta >= thr) else 0.0
    se = sd / math.sqrt(n)
    z = Z if (primary or guard == "lower") else 0.0
    return phi((delta - thr) / se - z)


def adopt_prob(deltas: dict, sds: dict, ns: dict, guard: str = "lower") -> tuple[float, dict]:
    """型ごとの確率と、その積（採用の確率）。deltas/sds/ns は型 → 値（計算に入れる型だけ）。"""
    per = {t: type_prob(deltas[t], sds[t], ns[t], t == PRIMARY, guard) for t in sds}
    p = 1.0
    for v in per.values():
        p *= v
    return p, per


def adopt_prob_mc(deltas: dict, sds: dict, ns: dict, n_sim: int = 100_000, seed: int = 0,
                  guard: str = "lower") -> float:
    """同じ確率をモンテカルロで（平均の差を N(δ, se²) から引き、下限を作って規則を当てる）。"""
    rng = random.Random(seed)
    types = list(sds)
    hit = 0
    for _ in range(n_sim):
        ok = True
        for t in types:
            se = sds[t] / math.sqrt(ns[t])
            m = rng.gauss(deltas[t], se) if se > 0 else deltas[t]
            low = m - Z * se
            g = low if (t == PRIMARY or guard == "lower") else m
            if (t == PRIMARY and not g > 0) or (t != PRIMARY and not g >= FLOOR):
                ok = False
                break
        hit += ok
    return hit / n_sim


def scenario_deltas(scn: str, delta: float, types) -> dict:
    eff = SCENARIOS[scn]
    return {t: (delta if t in eff else 0.0) for t in types}


def build(mde: dict, n_sim: int = 100_000, seed: int = 0) -> dict:
    sdt = sd_by_type(mde)
    used = [t for t in TYPES if sdt[t] is not None]
    if PRIMARY not in used:
        raise SystemExit("C1 型の sd が無い: 計算できない")
    ns = {t: TYPES[t][2] for t in used}
    res: dict = dict(types={t: dict(source=TYPES[t][0], set=TYPES[t][1], n=TYPES[t][2],
                              sd=sdt[t], status=("計算" if sdt[t] else "未計算")) for t in TYPES},
               rule=dict(z=Z, floor=FLOOR, primary=PRIMARY), deltas=list(DELTAS),
               scenarios={k: list(v) for k, v in SCENARIOS.items()}, n_sim=n_sim, seed=seed, table={})
    for st in STATS:
        sds = {t: sdt[t][st] for t in used}
        se = {t: sds[t] / math.sqrt(ns[t]) for t in used}
        res["table"][st] = dict(se=se, mde80={t: 2.8 * se[t] for t in used}, rows=[], harm=[])
        for guard in GUARDS:
            for scn in SCENARIOS:
                for d in DELTAS:
                    dl = scenario_deltas(scn, d, used)
                    p, per = adopt_prob(dl, sds, ns, guard)
                    pmc = adopt_prob_mc(dl, sds, ns, n_sim=n_sim, seed=seed, guard=guard)
                    res["table"][st]["rows"].append(dict(
                        rule=guard, scenario=scn, delta=d, p_adopt=p, p_adopt_mc=pmc, per_type=per,
                        lower_mean={t: dl[t] - Z * se[t] for t in used}, lower_sd=se))
            # 害の筋書き（提案と今の規則で何を手放すかを示すためだけ）: C1 に +0.02、他の 1 型に −0.02
            for bad in [t for t in used if t != PRIMARY and t in SCENARIOS["S2"]]:
                dl = {t: (0.02 if t == PRIMARY else (HARM if t == bad else 0.0)) for t in used}
                p, per = adopt_prob(dl, sds, ns, guard)
                res["table"][st]["harm"].append(dict(rule=guard, harmed=bad, delta_c1=0.02, delta_harm=HARM,
                                                     p_adopt=p, per_type=per))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mde-json", default=str(ROOT / "outputs/local_eval/net_mde.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/adoption_power/adoption_power.json"))
    ap.add_argument("--n-sim", type=int, default=100_000)
    a = ap.parse_args()
    p = Path(a.mde_json)
    raw = p.read_bytes()
    res = build(json.loads(raw), n_sim=a.n_sim, seed=0)
    res["mde_json"] = dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for st in STATS:
        print(f"[sd={st}] se=" + ", ".join(f"{t}:{v:.4f}" for t, v in res["table"][st]["se"].items()))
        for r in res["table"][st]["rows"]:
            print(f"  {r['rule']:5s} {r['scenario']} δ={r['delta']:+.3f} P={r['p_adopt']:.4f} MC={r['p_adopt_mc']:.4f} "
                  + " ".join(f"{t}:{v:.3f}" for t, v in r["per_type"].items()))
        for r in res["table"][st]["harm"]:
            print(f"  harm {r['rule']:5s} C1=+0.02 {r['harmed']}={r['delta_harm']:+.2f} P={r['p_adopt']:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
