#!/usr/bin/env python3
"""
Issue #178: d_dom(MetFrag-lite) の集合ごとの平均と 95% 区間を、既存の分子ごとの表から出す（採否には使わない）。

定義（docs/research/train-independent-evidence.md §2 の事前登録のとおり。ここで替えない）:
- 入力は #171 の rep_dseen.py（origin/feat/171-rep-stage、commit c5bf237）が書いた
  outputs/local_eval/rep_dseen_per_molecule.csv の d_dom_metfrag 列だけ。d_dom は rep_dseen.py と同じもの
  （rep の点と MetFrag-lite の点の両方が有限の候補に絞り、sd0 の判定も rep と MetFrag-lite の組、
  d_dom = mean(z|U) − mean(z|O)）。「frag だけの定義」には替えない
- 先に、集合ごとの平均を小数 4 桁に丸めた値と n が outputs/local_eval/rep_dseen.json の d_dom.metfrag と
  一致することを確かめる。一致しなければ区間を出さずに終了コード 2
- 区間は main の scripts/local_eval/dseen.py の bootstrap_ci（2,000 回・seed 0・molecule_id の昇順、
  丸める前の lo・hi）。帯は rep_dseen.band と同じしきい値: lo > 0 → higher_train、hi < 0 → lower_train、
  それ以外 → none
- 集合: v3_c1・v3_c2・enveda_v1・v3_c2n（4 集合）と参考の e180_c2k

    /usr/local/bin/python3 scripts/research/ddom_metfrag_ci.py

出力: outputs/local_eval/ddom_metfrag_ci.json（gitignore。数値は文書に写す）。rdkit・torch は import しない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts/local_eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))

SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
REFERENCE_SETS = ("e180_c2k",)
COLUMN = "d_dom_metfrag"
MISMATCH_EXIT = 2


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def band(lo: float, hi: float) -> str:
    """rep_dseen.band（c5bf237）と同じしきい値（丸める前の lo > 0 / hi < 0）。"""
    if lo > 0:
        return "higher_train"
    if hi < 0:
        return "lower_train"
    return "none"


def per_set_values(rows, column: str = COLUMN) -> dict[str, dict[str, float]]:
    """{holdout: {molecule_id: 値}}。同じ (holdout, molecule_id) が 2 回あれば止める。"""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        h, m = r["holdout"], r["molecule_id"]
        assert m not in out[h], f"重複: {h} {m}"
        out[h][m] = float(r[column])
    return dict(out)


def check_means(vals: dict[str, dict[str, float]], ref: dict, sets) -> list[str]:
    """平均（4 桁に丸め）と n が rep_dseen.json の d_dom.metfrag と食い違う集合の一覧。"""
    bad = []
    for h in sets:
        j = ref[h]["d_dom"]["metfrag"]
        v = vals.get(h, {})
        if not v or len(v) != j["n"] or round(sum(v.values()) / len(v), 4) != j["mean"]:
            bad.append(h)
    return bad


def summarize(vals: dict[str, dict[str, float]], sets) -> dict:
    from dseen import bootstrap_ci
    out = {}
    for h in sets:
        b = bootstrap_ci(vals[h])
        out[h] = dict(n=b["n"], mean=round(b["mean"], 4), lo=round(b["lo"], 4), hi=round(b["hi"], 4),
                      mean_raw=b["mean"], lo_raw=b["lo"], hi_raw=b["hi"], band=band(b["lo"], b["hi"]),
                      positive=int(sum(x > 0 for x in vals[h].values())),
                      negative=int(sum(x < 0 for x in vals[h].values())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs/local_eval/rep_dseen_per_molecule.csv"))
    ap.add_argument("--ref-json", default=str(ROOT / "outputs/local_eval/rep_dseen.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/ddom_metfrag_ci.json"))
    a = ap.parse_args()
    with open(a.csv, newline="") as f:
        rows = list(csv.DictReader(f))
    ref = json.loads(Path(a.ref_json).read_text())
    vals = per_set_values(rows)
    all_sets = SETS + REFERENCE_SETS
    bad = check_means(vals, ref, all_sets)
    if bad:
        print(f"平均か n が rep_dseen.json と食い違う: {bad}（区間は出さない）", file=sys.stderr)
        sys.exit(MISMATCH_EXIT)
    import dseen
    res = dict(issue=178, column=COLUMN, sets=list(SETS), reference_sets=list(REFERENCE_SETS),
               n_boot=dseen.N_BOOT, seed=dseen.SEED,
               inputs=dict(csv_sha256=sha256_file(Path(a.csv)), ref_json_sha256=sha256_file(Path(a.ref_json)),
                           dseen_py_sha256=sha256_file(ROOT / "scripts/local_eval/dseen.py"),
                           rep_dseen_commit=ref.get("rep_once_commit"), rep_dseen_script_sha256=ref.get("script_sha256")),
               git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip(),
               means_match_ref_json=True)
    res.update(summarize(vals, all_sets))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for h in all_sets:
        r = res[h]
        print(f"{h}: n {r['n']} mean {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] {r['band']}")


if __name__ == "__main__":
    main()
