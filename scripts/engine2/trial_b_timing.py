"""#165 の push 前の時間の関門: (B) のクエリ 50 分子で、#74 の経路（fold-safe FPNet の fold k + blk_base_fs）と
新しい経路（自前 S2 + blk_base）を同じプロセスで 10 分子ずつ交互に作り、秒を比べる（ローカル CPU）。

    OMP_NUM_THREADS=4 /usr/bin/time -l .venv/bin/python scripts/engine2/trial_b_timing.py \
        --train data/train.parquet --assets <scratch>/assets_local.json --out <scratch>/trial --threads 4 --workers 4

- 固定費（prepare: 資産・除外・ライブラリ・クエリ選び・プール・索引）は 1 回。クエリは --fp-source s2 の条件
  （#165 §2 の 1）で --n-mol 50 を選んだもの。旧の経路にも同じ 50 分子を渡す（fold k は各分子の fold_identity）
- 交互: QUERY_ALLOC の順に並べた 50 分子を 10 ずつ 5 ブロックに分け、各ブロックで 旧 → 新 の順に build()
- 換算（#165 §4、変えない）: r = 新の合計秒 ÷ 旧の合計秒、Kaggle の見込み = r × 22,636 s、関門 ≤ 36,000 s
- ついでに記録（判定には使わない）: 2 経路の sim 表が一致するか、rank_train の FP 以外の 19 列が一致するか、
  M=0 の正解の FP 1 位の割合（列 16 = 0）
"""
import argparse
import builtins
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import base_fs  # noqa: E402
import build_tables as BT  # noqa: E402

KAGGLE_74_RUN2_SECS = 22636.0
GATE_SECS = 36000.0
NON_FP_COLS = [c for c in range(31) if c not in base_fs.FP_COLS]


def main(argv=None):
    ap = argparse.ArgumentParser()
    BT.add_common_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    args.fp_source = "s2"
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    import torch
    torch.set_num_threads(args.threads)
    BT.install_guard()
    A = json.load(builtins.open(args.assets))
    t_start = time.time()
    C = BT.prepare(A, args)
    E, L, meta, I, fold = C["E"], C["L"], C["meta"], C["I"], C["fold"]
    log = E.log
    t_prep = time.time() - t_start
    single, merged, _ = BT.load_s2(A["s2_model"], "cpu")
    s2_models = (single, merged, "cpu")
    banks = {k: base_fs.load_bank(os.path.join(A["fs_models_dir"], f"fpnet_fold{k}.pt")) for k in (1, 2, 3, 4)}
    bits = np.load(A["fs_bits"])
    t_models = time.time() - t_start - t_prep
    flat = [(s, ik) for s in BT.QUERY_ALLOC for ik in C["queries"].get(s, [])]
    assert len(flat) == args.n_mol, (len(flat), args.n_mol)
    import pandas as pd
    t1 = time.time()
    qset = {ik for _, ik in flat}
    qrows = np.flatnonzero(pd.Series(meta["ik"]).isin(qset).values)
    raw = BT.read_rows_f64(args.train, qrows)
    t_raw = time.time() - t1
    blocks = [flat[i:i + args.block] for i in range(0, len(flat), args.block)]
    res = dict(old=[], new=[])
    tabs = dict(old=[], new=[])
    for bi, blk in enumerate(blocks):
        q = {}
        for s, ik in blk:
            q.setdefault(s, []).append(ik)
        for path in ("old", "new"):
            t = time.time()
            if path == "old":
                sim, rt, rep = BT.build(E, L, meta, I, q, fold, banks, bits, args.workers, log, raw=raw, fp_source="foldsafe")
            else:
                sim, rt, rep = BT.build(E, L, meta, I, q, fold, None, None, args.workers, log, fp_source="s2", s2_models=s2_models)
            dt = time.time() - t
            res[path].append(dict(block=bi, n=len(blk), molecules=rep["molecules"], secs_total=round(dt, 2), secs=rep["secs"],
                                  dropped=dict((k, rep[k]) for k in rep if k.startswith("dropped"))))
            tabs[path].append((sim, rt))
            log(f"block {bi} {path}: {dt:.1f}s {rep['secs']} molecules {rep['molecules']}")
    tot = {p: sum(x["secs_total"] for x in res[p]) for p in res}
    r = tot["new"] / tot["old"]
    est = r * KAGGLE_74_RUN2_SECS
    # 記録だけ: 2 経路の表の比較
    sim_eq, nonfp_eq, fp_diff_rows = [], [], 0
    top1 = dict(old=[], new=[])
    for (so, ro), (sn, rn) in zip(tabs["old"], tabs["new"]):
        sim_eq.append(all(np.array_equal(so[k], sn[k]) for k in so))
        nonfp_eq.append(bool(np.array_equal(ro["X"][:, NON_FP_COLS], rn["X"][:, NON_FP_COLS]) and
                             np.array_equal(ro["Y"], rn["Y"]) and np.array_equal(ro["M"], rn["M"]) and np.array_equal(ro["G"], rn["G"])))
        fp_diff_rows += int((np.abs(ro["X"][:, base_fs.FP_COLS] - rn["X"][:, base_fs.FP_COLS]).max(1) > 0).sum())
        for p, t in (("old", ro), ("new", rn)):
            m = (t["Y"] == 1) & (t["M"] == 0)
            top1[p].extend((t["X"][m, 16] == 0).tolist())
    out = dict(n_mol=args.n_mol, block=args.block, threads=args.threads, workers=args.workers,
               torch_num_threads=int(torch.get_num_threads()), omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
               fixed_secs_prepare=round(t_prep, 1), fixed_secs_models=round(t_models, 1), fixed_secs_raw_rows_old=round(t_raw, 1),
               t_fixed_engine=round(C["t_fixed"], 1), blocks=res, total_secs=tot,
               per_query_secs={p: round(tot[p] / max(1, sum(x["molecules"] for x in res[p])), 3) for p in tot},
               logits_secs={p: round(sum(x["secs"]["logits_in_channels"] for x in res[p]), 2) for p in res},
               ratio_new_over_old=round(r, 4), kaggle_74_run2_secs=KAGGLE_74_RUN2_SECS, kaggle_estimate_secs=round(est, 1),
               kaggle_estimate_hours=round(est / 3600, 3), gate_secs=GATE_SECS, gate_pass=bool(est <= GATE_SECS),
               record_only=dict(sim_equal_per_block=sim_eq, rank_train_non_fp_equal_per_block=nonfp_eq,
                                rank_train_rows_fp_cols_differ=fp_diff_rows,
                                m0_truth_fp_rank0={p: (round(float(np.mean(v)), 4) if v else None, len(v)) for p, v in top1.items()}),
               query_info=C["qinfo"], s2=C["s2_info"], queries=flat, cpu_info=BT.cpu_info())
    os.makedirs(args.out, exist_ok=True)
    json.dump(out, builtins.open(os.path.join(args.out, "trial_b_timing.json"), "w"), indent=1, default=str)
    log(f"ratio {r:.4f} estimate {est:.0f}s ({est / 3600:.2f} h) gate {'PASS' if est <= GATE_SECS else 'FAIL'}")
    return out


if __name__ == "__main__":
    main()
