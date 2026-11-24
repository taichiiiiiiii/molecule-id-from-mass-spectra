#!/usr/bin/env python3
"""
Issue #218: fragnn v3（同じ分子式の異性体を負例の主にする）の前に、仕組みを保存済みのデータで確かめる。

定義は #218 の事前登録のコメント（7120082449、2026-11-24T02:23:00Z）と、そのレビューの補い（同じ日、数える前）のとおり。
数えるのは 1 回だけ。読み方（作る / 作らない）は数える前に固定してある:
  作る = p_change >= 0.10 かつ r >= 1.25 かつ median_B(v3_c2nu_t10) < median_B(v3_c1u_t10)（等しければ成り立たない）

- (A) v2 の学習側の行 tr_i（1,658,625）ごとに、窓 = off_tr[cid[i]]:off_tr[cid[i]+1]。n = 窓の負例数、
  s = 正解と同じ分子式（RDKit CalcMolFormula、末尾の電荷を外す）の負例数。
  p_change = mean[n > 127 かつ s >= 1]、r = Σ min(s,127) / Σ s·min(127,n)/n。
  照合: B.negative_csr を pool_meta から作り直し、保存済みの off_tr/flat_tr と完全一致を assert（不一致なら止める）
- (B) v3_c1u_t10・v3_c2nu_t10 の対照（score+prior+frag）の上位 25 件で、正解でない候補（score_key が正解と違う。
  None は正解でない）のうち正解と同じ分子式の割合。分子ごとの値と集合ごとの中央値
- (C) 記述のみ: v2 の判定の悪化分子（rr 新 < rr 対照）で、fragnn2 の点 n（pool の key ごとの最大）の「正解 − 新 1 位」の中央値

    PYTHONPYCACHEPREFIX=<scratch> .venv/bin/python scripts/research/fragnn3_mechanism.py \
        [--data-root .] [--out outputs/local_eval/fragnn3_mechanism.json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train"))
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))

KMAX_NEG = 127                    # K_CAND − 1（fragnn_bench.K_CAND = 128 から作る。新しい定数ではない）
P_CHANGE_MIN, R_MIN = 0.10, 1.25  # #218 の事前登録で固定（値を見て変えない）
SETS_B = ("v3_c1u_t10", "v3_c2nu_t10")
CTRL_TAG, NEW_TAG = "score+prior+frag", "score+fragnn2+prior+frag"
TOPK = 25
_CHARGE = re.compile(r"[+-]\d*$")


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def formula_of(smiles) -> str | None:
    """RDKit CalcMolFormula、末尾の電荷（[+-]\\d*$）を外す。読めなければ None。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    if smiles is None:
        return None
    try:
        m = Chem.MolFromSmiles(str(smiles))
        if m is None:
            return None
        return _CHARGE.sub("", rdMolDescriptors.CalcMolFormula(m))
    except Exception:  # noqa: BLE001
        return None


def window_stats(off: np.ndarray, flat: np.ndarray, pos_code: np.ndarray, code: np.ndarray) -> tuple:
    """CSR の行 c ごとの n（窓の負例数）と s（正解の式と同じ式の負例数。式なし = コード −1 は一致しない）。
    flat・pos_code は同じ添字空間（code の添字）。"""
    n = np.diff(off).astype(np.int64)
    owner = np.repeat(np.arange(len(n)), n)
    pc = pos_code[owner]
    eq = (code[flat] == pc) & (pc >= 0)
    s = np.bincount(owner, weights=eq, minlength=len(n)).astype(np.int64)
    return n, s


def change_metrics(n: np.ndarray, s: np.ndarray, w: np.ndarray, kmax: int = KMAX_NEG) -> dict:
    """w = CSR の行ごとの学習の行（スペクトル）の数。p_change・r と記述の値。"""
    n = n.astype(np.float64); s = s.astype(np.float64); w = w.astype(np.float64)
    tot = w.sum()
    big = n > kmax
    ch = big & (s >= 1)
    v3 = np.minimum(s, kmax)
    v2 = np.where(n > 0, s * np.minimum(kmax, n) / np.where(n > 0, n, 1), 0.0)
    sv3, sv2 = float((w * v3).sum()), float((w * v2).sum())
    return dict(p_change=float((w * ch).sum() / tot), r=(sv3 / sv2) if sv2 > 0 else math.nan,
                sum_v3=sv3, sum_v2=sv2, n_rows=float(tot), share_window_gt_kmax=float((w * big).sum() / tot),
                share_s_ge1=float((w * (s >= 1)).sum() / tot), share_n_zero=float((w * (n == 0)).sum() / tot))


def wmedian(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    return float(x[o][np.searchsorted(cw, 0.5 * cw[-1])])


def part_a(data_root: Path) -> dict:
    import pandas as pd
    import pyarrow.parquet as pq
    import fragnn_bench as B
    import fragnn_train as T
    import sanpier_fold0 as sf
    t0 = time.time()
    P = T.data_paths(data_root)
    prep_path = data_root / T.V1_PREP_REL
    keys_path = T.CODE_ROOT / T.KEYS_REL
    D, info = T.prepare_v2(data_root, keys_path, prep_path)    # sha256 の照合・漏れの assert を含む
    z = np.load(prep_path)
    srows = z["struct_rows"]
    tr_i = D["tr_i"]
    assert len(tr_i) == T.EXPECTED_TRAIN_SPECTRA_V2, len(tr_i)
    # ---- 照合: negative_csr を作り直して保存済みと完全一致（prepare と同じ allowed & ~val_rows & is_trainstruct）
    meta = pd.read_parquet(P["pool"] / "pool_meta.parquet", columns=["key", "mass"])
    mass = meta.mass.values.astype(np.float64)
    assert bool(np.all(np.diff(mass) >= 0))
    ts = pd.read_parquet(P["pool"] / "train_structs.parquet", columns=["inchikey14", "key"])
    raw_to_key = dict(zip(ts.inchikey14, ts.key))
    split = pd.read_parquet(P["split"], columns=["inchikey14", "key", "fold"])
    hk = json.loads(Path(keys_path).read_text())
    drop_raw = sf.drop_inchikey14(split.inchikey14, hk)
    drop_key = B.drop_key_space(drop_raw, raw_to_key, hk, split.key)
    assert (len(drop_raw), len(drop_key)) == (T.EXPECTED_DROP, T.EXPECTED_DROP_KEY)
    allowed = ~meta.key.isin(drop_key).values
    is_val_all = z["is_val"]
    up_tr = srows[np.unique(z["pos_struct"][~is_val_all])]
    val_keys = set(meta.key.values[srows[z["pos_struct"][is_val_all]]])
    val_rows = meta.key.isin(val_keys).values
    is_trainstruct = meta.key.isin(set(ts.key.values)).values
    off_re, flat_re = B.negative_csr(up_tr, mass, allowed & ~val_rows & is_trainstruct)
    off_tr, flat_tr = z["off_tr"], z["flat_tr"]
    assert np.array_equal(off_re, off_tr), "off_tr mismatch"
    assert np.array_equal(flat_re, srows[flat_tr]), "flat_tr mismatch"
    assert np.array_equal(D["off_tr"], off_tr) and np.array_equal(D["flat_tr"], flat_tr)
    del meta, ts, split, off_re, flat_re
    # ---- 分子式（struct の添字空間）
    tf = time.time()
    smi = pq.read_table(P["pool"] / "pool_meta.parquet", columns=["smiles"]).column(0)
    forms = [formula_of(smi[int(r)].as_py()) for r in srows]
    del smi
    uniq = {f: j for j, f in enumerate(sorted({f for f in forms if f is not None}))}
    code = np.array([uniq[f] if f is not None else -1 for f in forms], np.int64)
    n_form_none = int((code < 0).sum())
    formula_s = time.time() - tf
    # CSR の行 c の正解 = up_tr[c]（struct の添字に直す）
    ps_c = np.searchsorted(srows, up_tr)
    assert np.array_equal(srows[ps_c], up_tr)
    cid_tr = D["cid"][tr_i]
    assert np.array_equal(D["pos_struct"][tr_i], ps_c[cid_tr]), "pos_struct != up_tr[cid]"
    n, s = window_stats(off_tr, flat_tr, code[ps_c], code)
    w = np.bincount(cid_tr, minlength=len(n))
    main = change_metrics(n, s, w)
    used = w > 0
    per_struct = change_metrics(n[used], s[used], np.ones(int(used.sum())))   # 記述のみ（読みに使わない）
    big = used & (n > KMAX_NEG)
    desc = dict(
        n_csr_rows=int(len(n)), n_csr_rows_used=int(used.sum()), n_structs=int(len(srows)), n_formula_none=n_form_none,
        pos_formula_none_rows=int((w * (code[ps_c] < 0)).sum()),
        window_n_median_rows=wmedian(n.astype(float), w.astype(float)),
        window_s_median_rows=wmedian(s.astype(float), w.astype(float)),
        window_n_median_structs=float(np.median(n[used])), window_s_median_structs=float(np.median(s[used])),
        s_median_given_n_gt_kmax_structs=float(np.median(s[big])) if big.any() else None,
        s_gt_kmax_structs=int((used & (s > KMAX_NEG)).sum()),
        same_formula_neg_share_rows=float((w * s).sum() / max(1, (w * n).sum())))
    return dict(main=main, per_struct_descriptive=per_struct, descriptive=desc,
                csr_rebuild_identical=True, v1_prep_sha256=info["v1_prep_sha256"], leak=info["leak"],
                judge_sets_overlap=info["judge_sets"], formula_seconds=round(formula_s, 1),
                seconds=round(time.time() - t0, 1))


def read_csv_lists(path: Path) -> dict[str, list[str]]:
    with open(path, newline="") as f:
        return {r["molecule_id"]: [c for c in r["smiles"].split(";") if c][:TOPK] for r in csv.DictReader(f)}


def read_truth(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["molecule_id"]: r["smiles"] for r in csv.DictReader(f)}


def part_b(data_root: Path) -> dict:
    from casmi import chem
    out = {}
    for h in SETS_B:
        d = data_root / "outputs/local_eval" / h
        cp, tp = d / f"submission_{CTRL_TAG}.csv", d / "truth.csv"
        ctrl, truth = read_csv_lists(cp), read_truth(tp)
        per, excl = {}, {"truth_key_none": 0, "truth_formula_none": 0, "no_nontruth": 0}
        for mid, tsmi in truth.items():
            tk, tf = chem.score_key(tsmi), formula_of(tsmi)
            if tk is None:
                excl["truth_key_none"] += 1; continue
            if tf is None:
                excl["truth_formula_none"] += 1; continue
            non = [c for c in ctrl.get(mid, []) if chem.score_key(c) != tk]
            if not non:
                excl["no_nontruth"] += 1; continue
            same = sum(1 for c in non if formula_of(c) == tf)
            per[mid] = dict(share=same / len(non), n_nontruth=len(non), n_same=same)
        vals = np.array([v["share"] for v in per.values()])
        out[h] = dict(n=len(per), n_truth=len(truth), excluded=excl, median=float(np.median(vals)),
                      mean=float(vals.mean()), share_zero=float((vals == 0).mean()),
                      files={cp.name: sha256_file(cp), tp.name: sha256_file(tp)}, per_molecule=per)
    return out


def part_c(data_root: Path, judge: dict) -> dict:
    import pandas as pd
    from casmi import chem
    import mrr25
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import sanpier as SP
    key = pd.read_parquet(data_root / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "mass"])
    assert bool(np.all(np.diff(key.mass.values) >= 0))   # Pool の並び（mass の安定ソート）= pool_meta の行
    keys = key.key.values
    out = {}
    for h in SETS_B:
        d = data_root / "outputs/local_eval" / h
        ctrl = read_csv_lists(d / f"submission_{CTRL_TAG}.csv")
        new = read_csv_lists(d / f"submission_{NEW_TAG}.csv")
        truth = read_truth(d / "truth.csv")
        sc = SP.load_scores(d / "fragnn2_scores.npz")
        worse, diffs, no_score = [], [], []
        for mid, tsmi in truth.items():
            tks = mrr25.truth_keys(tsmi)
            if mrr25.reciprocal_rank(tks, new.get(mid, [])) >= mrr25.reciprocal_rank(tks, ctrl.get(mid, [])):
                continue
            worse.append(mid)
            agg: dict[str, float] = {}
            for row, v in sc.get(mid, {}).items():
                if v is None or not math.isfinite(v):
                    continue
                k = str(keys[row])
                agg[k] = max(agg.get(k, -math.inf), v)
            tk = chem.score_key(tsmi)
            t1 = new[mid][0] if new.get(mid) else None
            t1k = chem.score_key(t1) if t1 else None
            if tk in agg and t1k in agg:
                diffs.append(agg[tk] - agg[t1k])
            else:
                no_score.append(mid)
        exp = judge["sets"][h]["new_minus_ctrl"]["worse"]
        assert len(worse) == exp, (h, len(worse), exp)
        out[h] = dict(n_worse=len(worse), expected_worse=exp, n_with_scores=len(diffs), n_no_score=len(no_score),
                      median_truth_minus_top1=float(np.median(diffs)) if diffs else None,
                      share_truth_below_top1=float(np.mean(np.array(diffs) < 0)) if diffs else None,
                      diffs=[round(float(x), 4) for x in diffs], no_score=no_score,
                      files={f: sha256_file(d / f) for f in (f"submission_{NEW_TAG}.csv", "fragnn2_scores.npz")})
    return out


def reading(a: dict, b: dict) -> dict:
    m = a["main"]
    c1 = dict(p_change_ge_0_10=bool(m["p_change"] >= P_CHANGE_MIN), r_ge_1_25=bool(m["r"] >= R_MIN),
              median_c2nu_lt_c1u=bool(b["v3_c2nu_t10"]["median"] < b["v3_c1u_t10"]["median"]))
    return dict(conditions=c1, verdict="作る" if all(c1.values()) else "作らない",
                rule="p_change >= 0.10 and r >= 1.25 and median_B(v3_c2nu_t10) < median_B(v3_c1u_t10)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT))
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/fragnn3_mechanism.json"))
    a = ap.parse_args()
    dr = Path(a.data_root).resolve()
    t0 = time.time()
    judge_p = dr / "outputs/local_eval/fragnn2_v3_judge.json"
    judge = json.loads(judge_p.read_text())
    A = part_a(dr)
    Bq = part_b(dr)
    C = part_c(dr, judge)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rep = dict(issue=218, definition="#218 prereg comment 7120082449 + review supplement", git_head=head,
               script_sha256=sha256_file(Path(__file__)), judge_json_sha256=sha256_file(judge_p),
               A=A, B=Bq, C_descriptive_only=C, reading=reading(A, Bq), seconds=round(time.time() - t0, 1))
    out = Path(a.out)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    tmp.replace(out)
    brief = dict(A=A["main"], B={h: {k: Bq[h][k] for k in ("n", "median", "mean", "share_zero", "excluded")} for h in SETS_B},
                 C={h: {k: C[h][k] for k in ("n_worse", "n_with_scores", "median_truth_minus_top1")} for h in SETS_B},
                 reading=rep["reading"], seconds=rep["seconds"])
    print(json.dumps(brief, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
