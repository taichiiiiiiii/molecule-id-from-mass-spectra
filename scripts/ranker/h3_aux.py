#!/usr/bin/env python3
"""H3 の補助の数と誤り分析（Issue #296）。採否に使わない（この結果で関門・列・定数を変えない）。定義は当てはめの前に
#296 のコメント（issuecomment-7232274306）の §6・§7 に固定したもの:

- 対象 = 評価の分子（meta の eval_molecules）。ラベル = lambdarank_oof.make_labels。分子式 = fragment_stats.formula_of
- 1 位 = evaluate.ranked_lists と同じ並び（score の降順、同点は row の順）の先頭の行
- 正解の行 = その分子の正解の行のうち、新しい版（--new の seed 0）で点の最も高い行（同点は row の順）
- (i) seed 0 の組で: 土台の 1 位 ≠ 正解かつ同じ式 → 新しい版で正解が 1 位（直り）／土台の 1 位 = 正解 → 新しい版の 1 位が
  正解でなく同じ式（壊れ）。型ごとと合計
- (ii) data/holdouts/<set>/test.parquet の分子ごとの行数 ≤ 3 の分子だけで、型ごとの 3 seed 平均の差と n
- (iii) 型ごと、正解の行・偽物の行の pfs_seen_frac の 10・50・90% 分位と行数
- 誤り分析: seed 0 の組。正解の行の無い分子は除く。悪化 / 改善 / 不変 = rr(新) < / > / = rr(土台)。各群で型 a〜e の順、
  型の中は "<set>\\t<molecule_id>" の UTF-8 の sha256 の昇順で先頭 4 件ずつ（足りない型はある分だけ）。数えること:
  新しい版で点の最も高い正解でない候補について (a) 正解と分子式が同じか (b) pfs_ll(その候補) − pfs_ll(正解) の符号。
  20 件ずつと群の全部の両方

    python scripts/ranker/h3_aux.py --new exp022 exp023 exp024 --base exp016 exp017 exp018 \\
        --out docs/experiments/h3_296/h3_aux.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("scripts/validate", "scripts/ranker", "scripts/features"):
    sys.path.insert(0, str(ROOT / p))
from lambdarank_oof import group_ids, load_truth, make_labels, read_molecule_list  # noqa: E402

TYPES = ("a", "b", "c", "d", "e")
QS = (0.1, 0.5, 0.9)
N_PICK = 4
LR_DIR = ROOT / "outputs/local_eval/ranker/lambdarank"


def mol_sha(gid: str) -> str:
    return hashlib.sha256(gid.encode("utf-8")).hexdigest()


def sign(x: float) -> str:
    return "+" if x > 0 else ("-" if x < 0 else "0")


def pick(gids_by_type: dict, n: int = N_PICK) -> list:
    """型 a〜e の順、型の中は sha256 の昇順で先頭 n 件ずつ（足りない型はある分だけ、他の型で埋めない）。"""
    out = []
    for t in TYPES:
        out += sorted(gids_by_type.get(t, []), key=mol_sha)[:n]
    return out


def per_molecule_rows(df: pd.DataFrame, score: np.ndarray, y: np.ndarray) -> dict:
    """分子ごとに (1 位の行, 正解の行（点の最も高い）| None, 正解でない最上位の行 | None)。行は df の位置。"""
    order = np.lexsort((np.arange(len(df)), -score, group_ids(df)))
    g = group_ids(df)[order]
    yy = y[order]
    out = {}
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or g[i] != g[start]:
            idx = order[start:i]
            ys = yy[start:i]
            t = idx[ys == 1]
            f = idx[ys == 0]
            out[g[start]] = (int(idx[0]), int(t[0]) if len(t) else None, int(f[0]) if len(f) else None)
            start = i
    return out


def count_group(gids, rows, df, formula) -> dict:
    c = dict(n=0, n_no_decoy=0, same_formula=0, diff_formula=0, sign={"+": 0, "0": 0, "-": 0})
    for gid in gids:
        _top, t, f = rows[gid]
        if f is None:
            c["n_no_decoy"] += 1
            continue
        c["n"] += 1
        same = formula(df.cand_smiles.iat[f]) == formula(df.cand_smiles.iat[t])
        c["same_formula" if same else "diff_formula"] += 1
        c["sign"][sign(float(df.pfs_ll.iat[f] - df.pfs_ll.iat[t]))] += 1
    return c


def main(argv=None) -> int:
    import fragment_stats as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", nargs=3, required=True, help="seed 0・1・2 の exp_id（先頭が seed 0）")
    ap.add_argument("--base", nargs=3, required=True)
    ap.add_argument("--eval-root", default=str(ROOT / "outputs/local_eval"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    metas = {e: json.loads((LR_DIR / e / "meta.json").read_text()) for e in a.new + a.base}
    for e in a.new + a.base:
        assert metas[e]["seed"] == (a.new + a.base).index(e) % 3, f"{e}: seed の順が 0・1・2 でない"
    m_new = metas[a.new[0]]
    df = pd.read_parquet(m_new["table"], columns=["set", "molecule_id", "cand_key", "cand_smiles", "type", "pfs_ll",
                                                  "pfs_seen_frac"])
    oof_n = pd.read_parquet(LR_DIR / a.new[0] / "oof.parquet", columns=["row", "set", "molecule_id", "cand_key", "score"])
    oof_b = pd.read_parquet(LR_DIR / a.base[0] / "oof.parquet", columns=["row", "set", "molecule_id", "cand_key", "score"])
    for o in (oof_n, oof_b):
        if not (len(o) == len(df) and (o[["set", "molecule_id", "cand_key"]].to_numpy() ==
                                       df[["set", "molecule_id", "cand_key"]].to_numpy()).all()
                and (o["row"].to_numpy() == np.arange(len(df))).all()):
            raise RuntimeError("oof.parquet の行の鍵・順が表と違う")
    eval_ids, rec = read_molecule_list(Path(m_new["eval_molecules"]["path"]))
    assert rec["sha256"] == m_new["eval_molecules"]["sha256"]
    truth = load_truth(df, Path(a.eval_root))
    y = make_labels(df, truth)
    g = group_ids(df)
    ev = pd.Series(g).isin(eval_ids).to_numpy()
    dfe = df[ev].reset_index(drop=True)
    ye = y[ev]
    rn = per_molecule_rows(dfe, oof_n["score"].to_numpy()[ev], ye)
    rb = per_molecule_rows(dfe, oof_b["score"].to_numpy()[ev], ye)
    fcache: dict = {}

    def formula(s):
        if s not in fcache:
            fcache[s] = P.formula_of(s)
        return fcache[s]

    tmap = dict(zip(group_ids(dfe), dfe["type"]))
    # (i)
    aux_i = {t: dict(base_samef_err=0, fixed=0, base_top1_truth=0, broken_samef=0) for t in TYPES + ("all",)}
    for gid, (top_n, t_row, _f) in rn.items():
        if t_row is None:
            continue
        top_b = rb[gid][0]
        tf = formula(dfe.cand_smiles.iat[t_row])
        for k in (tmap[gid], "all"):
            r = aux_i[k]
            if ye[top_b] == 0 and formula(dfe.cand_smiles.iat[top_b]) == tf:
                r["base_samef_err"] += 1
                r["fixed"] += int(ye[top_n] == 1)
            if ye[top_b] == 1:
                r["base_top1_truth"] += 1
                r["broken_samef"] += int(ye[top_n] == 0 and formula(dfe.cand_smiles.iat[top_n]) == tf)
    # (ii)
    pms = {e: pd.read_csv(LR_DIR / e / "per_molecule.csv", dtype={"molecule_id": str}).set_index(["set", "molecule_id"])
           for e in a.new + a.base}
    idx = pms[a.base[0]].index
    nspec = {}
    for s in sorted(set(idx.get_level_values(0))):
        te = pd.read_parquet(ROOT / f"data/holdouts/{s}/test.parquet", columns=["molecule_id"])
        for mid, n in te.molecule_id.astype(str).value_counts().items():
            nspec[(s, mid)] = int(n)
    few = np.array([nspec[m] <= 3 for m in idx])
    types = pms[a.base[0]].loc[idx, "type"].to_numpy()
    rr_n = np.stack([pms[e].loc[idx, "rr"].to_numpy(float) for e in a.new])
    rr_b = np.stack([pms[e].loc[idx, "rr"].to_numpy(float) for e in a.base])
    aux_ii = {}
    for t in TYPES:
        m = few & (types == t)
        aux_ii[t] = dict(n=int(m.sum()), new_mean=float(rr_n[:, m].mean(axis=1).mean()) if m.any() else None,
                         base_mean=float(rr_b[:, m].mean(axis=1).mean()) if m.any() else None,
                         delta=float(rr_n[:, m].mean(axis=1).mean() - rr_b[:, m].mean(axis=1).mean()) if m.any() else None)
    # (iii)
    aux_iii = {}
    for t in TYPES:
        m = dfe["type"].to_numpy() == t
        r = {}
        for name, sel in (("truth", m & (ye == 1)), ("decoy", m & (ye == 0))):
            v = dfe["pfs_seen_frac"].to_numpy(np.float64)[sel]
            r[name] = dict(n_rows=int(sel.sum()), **{f"q{int(q * 100)}": (float(np.quantile(v, q)) if len(v) else None)
                                                     for q in QS})
        aux_iii[t] = r
    # 誤り分析
    rr0n = pms[a.new[0]]["rr"]
    rr0b = pms[a.base[0]]["rr"]
    groups = {"worse": {}, "better": {}, "same": {}}
    for gid, (_top, t_row, _f) in rn.items():
        if t_row is None:
            continue
        s, mid = gid.split("\t")
        dn, db = float(rr0n.loc[(s, mid)]), float(rr0b.loc[(s, mid)])
        k = "worse" if dn < db else ("better" if dn > db else "same")
        groups[k].setdefault(tmap[gid], []).append(gid)
    err = {}
    for k, by_t in groups.items():
        picked = pick(by_t)
        allg = [x for t in TYPES for x in by_t.get(t, [])]
        err[k] = dict(n_group=len(allg), n_group_by_type={t: len(by_t.get(t, [])) for t in TYPES},
                      picked=count_group(picked, rn, dfe, formula), all=count_group(allg, rn, dfe, formula),
                      picked_list=[dict(set=x.split("\t")[0], molecule_id=x.split("\t")[1], type=tmap[x],
                                        rr_new=float(rr0n.loc[tuple(x.split("\t"))]),
                                        rr_base=float(rr0b.loc[tuple(x.split("\t"))]),
                                        truth_formula=formula(dfe.cand_smiles.iat[rn[x][1]]),
                                        top_decoy_formula=(formula(dfe.cand_smiles.iat[rn[x][2]]) if rn[x][2] is not None
                                                           else None),
                                        pfs_ll_diff_sign=(sign(float(dfe.pfs_ll.iat[rn[x][2]] - dfe.pfs_ll.iat[rn[x][1]]))
                                                          if rn[x][2] is not None else None))
                                   for x in picked])
    res = dict(issue=296, definition="https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/296#issuecomment-7232274306",
               new=a.new, base=a.base, n_eval_molecules=int(len(rn)),
               n_eval_molecules_without_truth_row=int(sum(1 for v in rn.values() if v[1] is None)),
               aux_i_same_formula=aux_i, aux_ii_nspec_le3=aux_ii, aux_iii_pfs_seen_frac=aux_iii, error_analysis=err,
               note="採否に使わない（#296 §6・§7）")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("n_eval_molecules", "aux_i_same_formula")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
