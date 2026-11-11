"""自前 2 表の取得後の照合（#74）。Kaggle の出力（own/、check.json、build.log）をローカルで確かめる。

    .venv/bin/python scripts/engine2/check_tables.py --runs outputs/engine2_build/run1 outputs/engine2_build/run2 \
        --ref-dir data/ext/engine2_ref --out outputs/engine2_build/check.json

確かめること（受け入れ条件 #74）:
- sim: X 51 列・blocks = base, libx, anapol, fragx・各グループの正例 1。rank_train: X 31 列・M ∈ {0,1}・各 (G, M) の正例 1
- 恒等式（engine2_table_schema の describe）が全式 1.0（FP 列は sim で全 0）
- 分子数 = n_mol − 捨てた数、除外とクエリの積 0（report の値）、読んだファイルに元の 2 表の名前が 0 件
- 複数の run があれば 2 表の中身の sha256 が一致
- 表の統計（正例率・候補数）を元の 2 表と並べる（見て条件を変えない）
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "research"))
sys.path.insert(0, HERE)
import engine2_table_schema as SCH  # noqa: E402
from build_tables import content_sha256, table_stats  # noqa: E402

FP_COLS = list(range(15, 21)) + list(range(25, 31))


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def check_run(d):
    rep = json.load(open(os.path.join(d, "own", "build_report.json")))
    sim = load(os.path.join(d, "own", "own_sim_rank_rows.npz")); rt = load(os.path.join(d, "own", "own_rank_train.npz"))
    out = dict(dir=d, n_mol=rep["n_mol"], opened_forbidden=rep["opened_forbidden"], query_exclusion_overlap=rep["query_exclusion_overlap"],
               exclusions=rep["exclusions"], deterministic_in_kernel=rep.get("deterministic"))
    r0 = rep["runs"][0]
    dropped = {k: r0[k] for k in r0 if k.startswith("dropped_")}
    out.update(molecules=r0["molecules"], dropped=dropped, molecules_ok=r0["molecules"] == rep["n_mol"] - sum(dropped.values()),
               v3_truth_negative_rows=r0.get("v3_truth_negative_rows"), fs_fp_failed=r0.get("fs_fp_failed"), secs=r0.get("secs"),
               fixed_secs=rep.get("fixed_secs"), pool=rep.get("pool", {}).get("pool_size"))
    out["sim_shape_ok"] = sim["X"].shape[1] == 51 and [str(b) for b in sim["blocks"]] == ["base", "libx", "anapol", "fragx"]
    out["sim_fp_zero"] = bool(np.all(sim["X"][:, FP_COLS] == 0))
    out["rt_shape_ok"] = rt["X"].shape[1] == 31 and set(np.unique(rt["M"]).tolist()) <= {0.0, 1.0}
    st = table_stats(sim, rt)
    out["stats"] = st
    out["pos_one_per_group"] = all(st[k]["pos_per_group_min"] == 1 and st[k]["pos_per_group_max"] == 1 for k in st)
    idn = {}
    for name, path, blocks in (("sim", os.path.join(d, "own", "own_sim_rank_rows.npz"), None),
                               ("rank_train", os.path.join(d, "own", "own_rank_train.npz"), ["base"])):
        desc = SCH.describe(path, blocks)
        fr = {k: v.get("frac", v.get("status")) for k, v in desc["identities"].items()}
        idn[name] = dict(n_one=sum(1 for v in fr.values() if v == 1.0), n_all_zero=sum(1 for v in fr.values() if v == "all_zero"),
                         not_one=[k for k, v in fr.items() if v not in (1.0, "all_zero")])
    out["identities"] = idn
    out["identities_ok"] = all(not v["not_one"] for v in idn.values())
    out["content_sha256"] = dict(sim=content_sha256(sim), rank_train=content_sha256(rt))
    out["file_sha256"] = {f: hashlib.sha256(open(os.path.join(d, "own", f), "rb").read()).hexdigest()
                          for f in ("own_sim_rank_rows.npz", "own_rank_train.npz")}
    if len(rep["runs"]) > 1:
        out["in_kernel_repeat_sha"] = [r["sha256"] for r in rep["runs"]]
    return out


def ref_stats(ref_dir):
    sim = load(os.path.join(ref_dir, "sim_rank_rows_nofp.npz")); rt = load(os.path.join(ref_dir, "rank_train.npz"))
    return table_stats(sim, rt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref-dir", default="data/ext/engine2_ref")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(runs=[check_run(d) for d in a.runs])
    if len(res["runs"]) > 1:
        c = [r["content_sha256"] for r in res["runs"]]
        res["runs_content_equal"] = all(x == c[0] for x in c)
    if os.path.isdir(a.ref_dir):
        res["original_tables_stats"] = ref_stats(a.ref_dir)
    res["all_ok"] = all(r["sim_shape_ok"] and r["rt_shape_ok"] and r["pos_one_per_group"] and r["identities_ok"] and r["molecules_ok"]
                        and not r["opened_forbidden"] and r["query_exclusion_overlap"] == 0 and r["sim_fp_zero"]
                        for r in res["runs"]) and res.get("runs_content_equal", True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


if __name__ == "__main__":
    main()
