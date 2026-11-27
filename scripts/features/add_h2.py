#!/usr/bin/env python3
"""
H2（Issue #287）: dev 表の lib 系の 7 列（lib_max_sim・has_ref・lib_max_sim の相対の 5 列）を外し、
scripts/features/lib_xinst.py の parts の 3 列と lib_max_sim_xi の相対の 5 列（relativize.add_relative、今と同じ規則）を
入れた表を別名で書く。正解のラベル・MRR は使わない（記録は行数・sha256・NaN/inf・件数だけ）。

    PY=".../.venv/bin/python"
    $PY scripts/features/add_h2.py \
        --in    outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet \
        --parts outputs/local_eval/features/parts_h2_dev \
        --out   outputs/local_eval/features/dev_b8_cc_gl2_n1_h2.parquet

点検（どれかが通らなければ書かずに止める）:
- 各集合の part の (set, molecule_id, cand_key) が、入力の表のその集合の行と同じ順で一致する
- part の json: rule = その集合の hide_rule、table_sha256 = 入力の表、out_sha256 = part、作り直しの点検の不一致 0
- 入力の lib_max_sim から今の規則で作り直した相対の 5 列が、入力の表の lib_max_sim_* と完全に一致する（同じ規則の確認）
- 外さない列は値も順も同じ、行数 430,979・分子 3,901、足した列の NaN 0・inf 0、has_ref_xi = 0 の行は 3 列とも 0.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/features"))

import relativize as RZ  # noqa: E402

KEY = ["set", "molecule_id", "cand_key"]
MOL = ["set", "molecule_id"]
EXPECT = dict(n_mol=3901, n_rows=430979)
DEV_SETS = tuple(f"b8_f{f}_{t}_t10" for f in (1, 2, 3, 4) for t in "abcde")
DROP = ("lib_max_sim", "has_ref") + tuple(f"lib_max_sim_{s}" for s in RZ.SUFFIXES)
NEW = ("lib_max_sim_xi", "has_ref_xi", "lib_max_sim_xi_sa")
NEW_REL = tuple(f"lib_max_sim_xi_{s}" for s in RZ.SUFFIXES)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_part(g: pd.DataFrame, part: pd.DataFrame, h: str) -> dict:
    if len(part) != len(g):
        raise ValueError(f"{h}: part の行数 {len(part)} != 表 {len(g)}")
    for c in KEY:
        if not np.array_equal(g[c].astype(str).to_numpy(), part[c].astype(str).to_numpy()):
            raise ValueError(f"{h}: part の {c} が表の順と一致しない")
    v = part[list(NEW)].to_numpy(np.float64)
    if not np.isfinite(v).all():
        raise ValueError(f"{h}: part に NaN か inf がある")
    z = part["has_ref_xi"].to_numpy() == 0
    if (v[z][:, [0, 2]] != 0).any():
        raise ValueError(f"{h}: has_ref_xi = 0 の行に値がある")
    if not set(np.unique(part["has_ref_xi"])) <= {0, 1}:
        raise ValueError(f"{h}: has_ref_xi が 0/1 でない")
    return dict(set=h, n_rows=int(len(part)), n_has_ref_xi=int((~z).sum()))


def replace_columns(df: pd.DataFrame, parts: dict) -> tuple[pd.DataFrame, list]:
    """df の DROP の列を外し、parts {set: DataFrame} の NEW の列と lib_max_sim_xi の相対の列を入れる（行の順は不変）。"""
    miss = [c for c in DROP if c not in df.columns]
    if miss:
        raise ValueError(f"外す列が表に無い: {miss}")
    bad = [c for c in NEW + NEW_REL if c in df.columns]
    if bad:
        raise ValueError(f"既に列がある: {bad}")
    sets = list(pd.unique(df["set"].astype(str)))
    if sorted(sets) != sorted(parts):
        raise ValueError(f"表の集合と parts の集合が違う: {sorted(set(sets) ^ set(parts))}")
    out = df.drop(columns=list(DROP))
    cols = {c: np.zeros(len(df), np.float64) for c in NEW}
    checks = []
    for h in sets:
        m = (df["set"].astype(str) == h).to_numpy()
        checks.append(check_part(df.loc[m], parts[h], h))
        for c in NEW:
            cols[c][m] = parts[h][c].to_numpy(np.float64)
    for c in NEW:
        out[c] = cols[c].astype(np.int8) if c == "has_ref_xi" else cols[c]
    out = RZ.add_relative(out, MOL, ["lib_max_sim_xi"])
    return out, checks


def same_rule_check(df: pd.DataFrame) -> bool:
    """入力の lib_max_sim から add_relative で作り直した 5 列が、入力の表の lib_max_sim_* と完全に一致するか。"""
    re_ = RZ.add_relative(df[MOL + ["lib_max_sim"]], MOL, ["lib_max_sim"])
    return all(np.array_equal(re_[f"lib_max_sim_{s}"].to_numpy(), df[f"lib_max_sim_{s}"].to_numpy())
               for s in RZ.SUFFIXES)


def main(argv=None) -> int:
    import library as L
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    inp, out, pdir = Path(a.inp), Path(a.out), Path(a.parts)
    if out.exists():
        raise ValueError(f"出力が既にある: {out}")
    if out.parent.is_symlink() or any(p.is_symlink() for p in out.parent.parents if p != Path("/")):
        raise ValueError(f"出力先がシンボリックリンクの中: {out.parent}")
    in_sha = sha256_file(inp)
    df = pd.read_parquet(inp)
    if sorted(pd.unique(df["set"].astype(str))) != sorted(DEV_SETS):
        raise ValueError("表の集合が dev の 20 集合と違う")
    if not same_rule_check(df):
        raise ValueError("入力の lib_max_sim の相対の列が relativize の今の規則と一致しない")
    parts, metas = {}, {}
    for h in DEV_SETS:
        pp, pj = pdir / f"{h}.parquet", pdir / f"{h}.json"
        meta = json.loads(pj.read_text())
        if meta["rule"] != L.hide_rule(h)["rule"]:
            raise ValueError(f"{h}: rule が違う {meta['rule']}")
        if meta["table_sha256"] != in_sha:
            raise ValueError(f"{h}: part を作った表が入力の表と違う")
        if meta["out_sha256"] != sha256_file(pp):
            raise ValueError(f"{h}: part の sha256 が json と違う")
        c = meta["check"]
        if c["n_mismatch_lib_max_sim"] or c["n_mismatch_has_ref"]:
            raise ValueError(f"{h}: 作り直しの点検の不一致がある")
        parts[h] = pd.read_parquet(pp)
        metas[h] = meta
    new, checks = replace_columns(df, parts)
    kept = [c for c in df.columns if c not in DROP]
    same = all(new[c].equals(df[c]) for c in kept)
    keys_same = new[KEY].equals(df[KEY])
    num = new[list(NEW + NEW_REL)].to_numpy(np.float64)
    n_mol = int(len(new[MOL].drop_duplicates()))
    tmp = out.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(out)
    rep = dict(issue=287, input=str(inp), input_sha256=in_sha, parts=str(pdir), output=str(out),
               output_sha256=sha256_file(out), n_rows=int(len(new)), n_molecules=n_mol,
               n_rows_ok=len(new) == EXPECT["n_rows"], n_molecules_ok=n_mol == EXPECT["n_mol"],
               dropped=list(DROP), added=list(NEW + NEW_REL), n_columns_in=int(df.shape[1]), n_columns_out=int(new.shape[1]),
               kept_columns_identical=bool(same), keys_order_identical=bool(keys_same), same_relative_rule=True,
               n_inf=int(np.isinf(num).sum()), n_nan=int(np.isnan(num).sum()),
               n_has_ref_in=int(df["has_ref"].sum()), n_has_ref_xi=int(new["has_ref_xi"].sum()),
               part_checks=checks, part_secs={h: m["secs_total"] for h, m in metas.items()},
               secs=round(time.time() - t0, 1))
    rep["all_ok"] = bool(rep["n_rows_ok"] and rep["n_molecules_ok"] and same and keys_same and rep["n_inf"] == 0
                         and rep["n_nan"] == 0)
    rp = out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("part_checks", "part_secs")}, ensure_ascii=False),
          flush=True)
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
