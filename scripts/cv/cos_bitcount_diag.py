#!/usr/bin/env python3
"""
Issue #140 の記録用の診断（採否には使わない。結果を見た後に足した事後の診断で、事前登録の外）:
cos 段が大きく負けた理由を見るため、各ホールドアウトで次を数える。

- ロジット z の要約（全問い合わせ × 全ビット）: 平均・標準偏差・z > 0 の割合、p = σ(z) の平均
- 分子ごと（正解が窓にある分子だけ）の on のビット数（FPCache の選択ビット）の中央値:
  正解、内積 fp @ z の 1 位、cos の 1 位
- 正解の順位（窓の候補を pool の key で重複除去した並び。prior・frag なし）が 1 位の割合: 内積と cos

窓・hidden・z・指紋は run.py と同じ（visible_window・stage_z(z, None)・FPCache・score_window・cos_window）。

    python scripts/cv/cos_bitcount_diag.py [--holdouts v3_c1,v3_c2,enveda_v1,v3_c2n] [--out outputs/local_eval/cos_bitcount_diag.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/validate"))


def key_rank(rows, keys, tk) -> int | None:
    """rows=(点, 行) の並びを key で重複除去したときの正解キーの順位（1 始まり）。無ければ None。"""
    seen = set()
    for _, i in rows:
        k = keys[i]
        if k in seen:
            continue
        seen.add(k)
        if k == tk:
            return len(seen)
    return None


def run_holdout(h: str) -> dict:
    import library as L
    import run as R
    from mrr25 import read_truth, to_key
    zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    mids = [m for m in zf.files if m != "__targets__"]
    Z = np.stack([np.asarray(zf[m], np.float64) for m in mids])
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    truth = read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
    hidden = L.drop_keys(h) if L.hide_rule(h).get("pool") == "remove" else set()
    nb_t, nb_d, nb_c, r_d, r_c = [], [], [], [], []
    for mid in truth:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        z = R.stage_z(zf[mid], None)
        d, c = R.score_window(idx, pool, z, fpc), R.cos_window(idx, pool, z, fpc)
        if not d:
            continue
        tk = to_key(truth[mid]) if truth[mid] else None
        ti = [i for _, i in d if pool.key[i] == tk]
        if not ti:
            continue
        nb = lambda i: int(fpc(pool.smiles[i]).sum())  # noqa: E731
        nb_t.append(nb(ti[0])); nb_d.append(nb(d[0][1])); nb_c.append(nb(c[0][1]))
        r_d.append(key_rank(d, pool.key, tk)); r_c.append(key_rank(c, pool.key, tk))
    out = dict(holdout=h, z_mean=round(float(Z.mean()), 3), z_std=round(float(Z.std()), 3),
               z_pos_frac=round(float((Z > 0).mean()), 4), p_mean=round(float(R.sigmoid(Z).mean()), 4),
               n_truth_in_window=len(nb_t))
    if nb_t:
        out.update(onbits_median=dict(truth=float(np.median(nb_t)), top1_dot=float(np.median(nb_d)), top1_cos=float(np.median(nb_c))),
                   top1_cos_more_bits_than_truth=round(float(np.mean(np.array(nb_c) > np.array(nb_t))), 3),
                   scorer_only_rank1=dict(dot=round(float(np.mean([r == 1 for r in r_d])), 4),
                                          cos=round(float(np.mean([r == 1 for r in r_c])), 4)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", default="v3_c1,v3_c2,enveda_v1,v3_c2n")
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/cos_bitcount_diag.json"))
    a = ap.parse_args()
    res = {h: run_holdout(h) for h in a.holdouts.split(",")}
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
