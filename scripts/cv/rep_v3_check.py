#!/usr/bin/env python3
"""
Issue #171 §3・補足 1-5: rep 段（家族に特有の断片・中性損失の偏り）を score+prior+frag に足した効果を、検証 v3 の
4 集合で判定する（事前登録どおり。lfus_v3_check.py と同じ型。規則だけ #171 のもの）。

前提: scripts/local_eval/run_rep_once.sh が 5 つの集合（v3_c1・v3_c2・enveda_v1・v3_c2n・e180_c2k）の
outputs/local_eval/<h>/rep_once.done を書き終えていること（e180_c2k は §5 用。ここでは commit sha の照合だけに使う）。

採否（#171 §3 で事前固定）: 4 集合 v3_c1・v3_c2・enveda_v1・v3_c2n のすべてで「新 − 対照」の 95% 下限 ≥ −0.02、
かつ少なくとも 1 集合で下限 > 0（下限は train_eval.bootstrap_diff の 4 桁に丸めた lo、2,000 回・seed 0）。必須の集合が
1 つでも欠けたら採用しない。

assert（補足 1-5。どれかが通らなければ採否を出さない）:
  (1) drop_keys.txt の sha256 = drop_keys_manifest.json の after.sha256 = 新の result json の rep_drop_keys_sha256
  (2) 新の result json の混入の検査が全部 0 件、行の一致が all_ok、全分子で作った（rep の記録の分子数 = ロジットのある分子数）
  (3) 5 つの rep_once.done の commit sha が同じ
  (4) 対照の CSV の sha256 が main の基準（下の CTRL_SHA256）と同じ
  (5) 対照の MRR = CTRL_EXPECTED、v3_c2 は新・対照の rr がすべて 0
MDE は記述のためだけで、採否には使わない。

    .venv/bin/python scripts/cv/rep_v3_check.py [--out outputs/local_eval/rep_v3_check.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/validate", "scripts/ranker", "scripts/cv", "scripts/local_eval"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

NONINF = -0.02
REQUIRED = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
DONE_SETS = REQUIRED + ("e180_c2k",)
NEW_TAG = "score+prior+rep+frag"
CTRL_TAG = "score+prior+frag"
CTRL_EXPECTED = {"v3_c1": 0.8535, "v3_c2": 0.0, "enveda_v1": 0.8376, "v3_c2n": 0.6284}
CTRL_N = {"v3_c1": 200, "v3_c2": 200, "enveda_v1": 200, "v3_c2n": 195}
CTRL_SHA256 = {   # 補足 1-5(4): main の基準（2026-11-19 23:55〜23:57 に作られたもの）
    "v3_c1": "821c05b56b4284c30a63a951a3989c4a6c8b815faecdf163238c2ed561d3ad18",
    "v3_c2": "1b561bc0dfc26adddeef3d001dc454738c9525752f2e112f2ed9b4c4fee175be",
    "enveda_v1": "bd5cc28c24ba946c58678de8edaf46d2a7ea7522abfff881d0109048a9ac28a9",
    "v3_c2n": "3a02683a3c51ce91cfe3bb653159b74db095b733beb72447f1ebea8bc114cea4",
}
# #171 §3: net_mde.json の最小・中央値・最大（記述だけ）
MDE = {"v3_c1": (0.0176, 0.0324, 0.0434), "v3_c2": None, "enveda_v1": (0.0175, 0.0321, 0.0401),
       "v3_c2n": (0.0171, 0.0428, 0.0538)}
LEAK_KEYS = ("leak_rep_drop_keys", "leak_rep_source_keys", "leak_rep_meta_lib")


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 純粋な関数（CI）
def judge(diffs: dict[str, dict]) -> dict:
    """diffs[h] = bootstrap_diff の結果（4 桁に丸めた lo を持つ）。#171 §3 の事前固定の規則だけで採否を出す。"""
    missing = [h for h in REQUIRED if h not in diffs]
    failing = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] < NONINF]
    win = [h for h in REQUIRED if h in diffs and diffs[h]["lo"] > 0]
    adopt = not missing and not failing and bool(win)
    return dict(adopt=bool(adopt), missing=missing, failing=failing, lo_gt_0=win, noninf=NONINF,
                rule="4 集合すべてで lo ≥ −0.02、かつ少なくとも 1 集合で lo > 0（集合が欠けたら不採用）")


def check_done(dones: dict[str, dict]) -> str:
    """補足 1-5(3): 5 つの rep_once.done が揃い、commit sha が同じ。その sha を返す。"""
    missing = [h for h in DONE_SETS if h not in dones]
    assert not missing, f"rep_once.done が無い集合: {missing}"
    shas = {dones[h]["commit"] for h in DONE_SETS}
    assert len(shas) == 1, f"rep_once.done の commit sha が違う: { {h: dones[h]['commit'] for h in DONE_SETS} }"
    for h in DONE_SETS:
        assert dones[h].get("holdout") == h and dones[h].get("issue") == 171, f"{h}: rep_once.done の中身が違う"
    return shas.pop()


def check_inputs(h: str, rj: dict, manifest_after_sha: str, dk_sha: str, ctrl_sha: str, ctrl_mrr: float,
                 n_logit_molecules: int) -> dict:
    """補足 1-5(1)(2)(4)(5 の MRR) の assert。記録を返す。"""
    assert ctrl_sha == CTRL_SHA256[h], f"{h}: 対照の CSV の sha256 {ctrl_sha} が main の基準と違う"
    assert round(ctrl_mrr, 4) == CTRL_EXPECTED[h], f"{h}: 対照 {ctrl_mrr} が事前登録の {CTRL_EXPECTED[h]} と違う"
    assert dk_sha == manifest_after_sha == rj["rep_drop_keys_sha256"], f"{h}: drop_keys の sha256 が違う"
    assert rj["rep_align_all_ok"] is True, f"{h}: 行の一致の確認が通っていない"
    assert all(rj["rep_audit"][k] == 0 for k in LEAK_KEYS), f"{h}: 混入 {rj['rep_audit']}"
    c = rj["rep_counts"]
    assert c["molecules"] == n_logit_molecules, f"{h}: rep を付けた分子 {c['molecules']} != ロジットのある分子 {n_logit_molecules}"
    assert rj["stages"] == NEW_TAG.split("+"), f"{h}: 新の段が違う {rj['stages']}"
    return dict(drop_keys_sha256=dk_sha, ctrl_sha256=ctrl_sha, audit=rj["rep_audit"], align_all_ok=True,
                molecules=c["molecules"])


def rr_counts(a: dict, b: dict) -> dict:
    d = [a[m] - b[m] for m in a]
    return dict(better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d), same=sum(x == 0 for x in d))


def split_by_coverage(a: dict, b: dict, cov: dict) -> dict:
    """記録だけ: 正解の家族に train の代表が 1 個以上（has）・0 個（none）・骨格が読めない（unreadable）ごとの差。
    cov[m] = reporter_reach.py と同じ数（None = 読めない）。"""
    out = {}
    groups = dict(has=[m for m in a if cov.get(m) is not None and cov[m] >= 1],
                  none=[m for m in a if cov.get(m) is not None and cov[m] == 0],
                  unreadable=[m for m in a if cov.get(m) is None])
    for name, ms in groups.items():
        d = [a[m] - b[m] for m in ms]
        out[name] = dict(n=len(ms), diff=round(sum(d) / len(d), 4) if d else None,
                         better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d))
    return out


# ---------------------------------------------------------------- 実データ
def coverage_maps(holdouts) -> dict[str, dict]:
    """reporter_reach.py と同じ数え方: 正解の骨格を持つ train の構造の数（drop_keys・集合の全正解を除いた後）。"""
    import mrr25
    import pandas as pd
    import library as L
    from reporter_reach import scaffold
    st = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet",
                         columns=["inchikey14", "key", "rawkey", "smiles"])
    st_fam = [scaffold(s) for s in st.smiles.values]
    out = {}
    for h in holdouts:
        truth = mrr25.read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
        tk = pd.read_csv(ROOT / f"outputs/holdouts/{h}/truth.csv", dtype=str)
        excl = set(L.drop_keys(h)) | set(tk.inchikey14)
        keep = ~(st.inchikey14.isin(excl) | st.key.isin(excl) | st.rawkey.isin(excl))
        fam_count: dict = {}
        for f, k in zip(st_fam, keep.values):
            if k and f is not None:
                fam_count[f] = fam_count.get(f, 0) + 1
        cov = {}
        for mid, smi in truth.items():
            tf = scaffold(smi)
            cov[mid] = None if tf is None else fam_count.get(tf, 0)
        out[h] = cov
    return out


def main() -> None:
    import numpy as np
    import mrr25
    from frag_v3_check import per_molecule_rr
    from train_eval import bootstrap_diff

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/rep_v3_check.json"))
    args = ap.parse_args()

    dones = {}
    for h in DONE_SETS:
        p = ROOT / f"outputs/local_eval/{h}/rep_once.done"
        if p.exists():
            dones[h] = json.loads(p.read_text())
    commit = check_done(dones)
    cov = coverage_maps(REQUIRED)

    res: dict[str, dict] = {}
    inputs: dict[str, dict] = {}
    for h in REQUIRED:
        d_ = ROOT / f"outputs/local_eval/{h}"
        truth_p = ROOT / f"outputs/holdouts/{h}/truth.csv"
        new_p, ctl_p, rj_p = d_ / f"submission_{NEW_TAG}.csv", d_ / f"submission_{CTRL_TAG}.csv", d_ / f"result_{NEW_TAG}.json"
        need = [truth_p, new_p, ctl_p, rj_p]
        if not all(p.exists() for p in need):
            print(f"[warn] {h}: 入力が無い {[str(p) for p in need if not p.exists()]}")
            continue
        truth = mrr25.read_truth(truth_p)
        a = per_molecule_rr(truth, mrr25.read_submission(new_p))
        b = per_molecule_rr(truth, mrr25.read_submission(ctl_p))
        assert len(b) == CTRL_N[h], f"{h}: 分子数 {len(b)} != {CTRL_N[h]}"
        rj = json.loads(rj_p.read_text())
        man = json.loads((ROOT / f"data/holdouts/{h}/drop_keys_manifest.json").read_text())
        zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
        n_logit = len({str(m) for m, _ in zf["__targets__"]} & set(truth))
        inputs[h] = check_inputs(h, rj, man["after"]["sha256"], sha256(ROOT / f"data/holdouts/{h}/drop_keys.txt"),
                                 sha256(ctl_p), sum(b.values()) / len(b), n_logit)
        inputs[h].update(new_csv_sha256=sha256(new_p), result_json_sha256=sha256(rj_p),
                         rep_scores_sha256=rj.get("rep_scores_sha256"))
        if h == "v3_c2":   # 補足 1-5(5)
            assert all(v == 0 for v in a.values()) and all(v == 0 for v in b.values()), "v3_c2 で rr が 0 でない分子がある"
            inputs[h]["v3_c2_all_rr_zero"] = True
        d = bootstrap_diff(a, b)
        c = rj["rep_counts"]
        d.update(n=len(a), new=round(sum(a.values()) / len(a), 4), ctrl=round(sum(b.values()) / len(b), 4),
                 mde_planning=MDE[h], counts=rr_counts(a, b),
                 rep_top_no_score=c["top_no_score"], rep_top_rows=c["top_rows"],
                 rep_top_no_score_frac=rj["rep_top_no_score_frac"], rep_window_no_score=c["window_no_score"],
                 rep_window_rows=c["window_rows"], rep_window_no_score_frac=rj["rep_window_no_score_frac"],
                 molecules_unchanged=c["molecules_unchanged"], same_scaffold_diff_rep=c["same_scaffold_diff_rep"],
                 decoy_overtakes_truth=c["decoy_overtakes_truth"], truth_sid_known=c["truth_sid_known"],
                 truth_sid_in_leaveout=c["truth_sid_in_leaveout"],
                 split_truth_family_coverage=split_by_coverage(a, b, cov[h]),
                 max_rss_bytes=dones[h].get("max_rss_bytes"), secs=rj.get("secs"))
        res[h] = d
        print(h, {k: v for k, v in d.items() if k not in ("split_truth_family_coverage",)}, flush=True)
    out = dict(issue=171, new=NEW_TAG, ctrl=CTRL_TAG, commit=commit, diffs=res, judge=judge(res), inputs=inputs,
               C1=res.get("v3_c1"), C2=res.get("v3_c2"),
               e180_c2k_done=dones["e180_c2k"], script_sha256=sha256(Path(__file__)))
    print(json.dumps(out["judge"], ensure_ascii=False))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
