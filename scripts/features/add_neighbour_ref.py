#!/usr/bin/env python3
"""
dev 表に群 N1（近傍の参照の照合、Issue #261）の 2 列 nbr_ms_mean・nbr_tani_mean を足して別名で書く。
列の値は scripts/features/neighbour_ref.py が集合ごとに書いた parts（<set>.parquet と <set>.json）をそのまま使う。
nbr_n は件数の点検にだけ使い、表には入れない（#261 の台帳の行は 2 列、11/27 夜の計画 項目 3）。
正解のラベル・MRR は使わない（記録は行数・sha256・NaN/inf・件数だけ）。

    PY=".../.venv/bin/python"
    $PY scripts/features/add_neighbour_ref.py \
        --in    /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc_n2.parquet \
        --parts /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/parts_nbr_dev \
        --out   /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/dev_b8_cc_n2_n1.parquet

点検（どれかが通らなければ書かずに止める）:
- 各集合の part の (set, molecule_id, cand_key) が、入力の表のその集合の行と同じ順で一致する
- part の json: mode = hide_rule、rule = その集合の hide_rule（library.hide_rule）、table_sha256 = 入力の表の sha256、
  out_sha256 = part の parquet の sha256、NaN の数が part の値と一致する
- NaN の決まり: nbr_n = 0 の行は両方 NaN、nbr_tani_mean はそれ以外で NaN なし（nbr_ms_mean は問い合わせの無い分子でも NaN）
- 元の列は値も順も同じ、inf は 0、行数 430,979・分子 3,901

Issue #290: `--which holdout` で判定用 4 集合（holdout_v3、分子 1,358。行数は dump_dev の記録の n_rows の和）に同じ点検で足す。
holdout では隠す規則の点検（part の rule = 集合の hide_rule、隠した正準の鍵の数 = drop_keys から作った数 > 0）も report に書く。
値は開かない（記録は行数・sha256・NaN/inf・件数だけ）。

    $PY scripts/features/add_neighbour_ref.py --which holdout \
        --in    /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/holdout_v3_cc.parquet \
        --parts /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/parts_nbr_holdout \
        --out   /Users/taichi/work/molecule-id-dev/outputs/local_eval/features/holdout_v3_cc_n1.parquet
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

from neighbour_ref import K_NEIGHBOURS  # noqa: E402

NEW_COLS = ("nbr_ms_mean", "nbr_tani_mean")
KEY = ["set", "molecule_id", "cand_key"]
MOL = ["set", "molecule_id"]
EXPECT = dict(n_mol=3901, n_rows=430979)
DEV_SETS = tuple(f"b8_f{f}_{t}_t10" for f in (1, 2, 3, 4) for t in "abcde")
HOLDOUT_SETS = ("v3_c1u_t10", "enveda_v1u_t10", "v3_c2_t10", "v3_c2nu_t10")   # dump_dev.HOLDOUT_SETS と同じ
N_HOLD = 1358
HOLD_PARTS = ROOT / "outputs/local_eval/features/parts_holdout"              # dump_dev.py --which holdout の記録


def expect_for(which: str, parts_dir: Path = HOLD_PARTS) -> dict:
    """集合の並びと期待の数。holdout の行数は dump_dev の集合ごとの記録（parts_holdout/<set>.json の n_rows）の和。"""
    if which == "dev":
        return dict(sets=DEV_SETS, **EXPECT)
    if which != "holdout":
        raise ValueError(which)
    n_rows = sum(int(json.loads((parts_dir / f"{h}.json").read_text())["n_rows"]) for h in HOLDOUT_SETS)
    return dict(sets=HOLDOUT_SETS, n_mol=N_HOLD, n_rows=n_rows)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_part(g: pd.DataFrame, part: pd.DataFrame, h: str) -> dict:
    """1 集合の part を点検し、件数を返す。g は入力の表のその集合の行（表の順）。"""
    if len(part) != len(g):
        raise ValueError(f"{h}: part の行数 {len(part)} != 表 {len(g)}")
    for c in KEY:
        a = g[c].astype(str).to_numpy()
        b = part[c].astype(str).to_numpy()
        if not np.array_equal(a, b):
            raise ValueError(f"{h}: part の {c} が表の順と一致しない")
    nn = part["nbr_n"].to_numpy()
    ms, tm = part["nbr_ms_mean"].to_numpy(np.float64), part["nbr_tani_mean"].to_numpy(np.float64)
    zero = nn == 0
    if not (np.isnan(ms[zero]).all() and np.isnan(tm[zero]).all()):
        raise ValueError(f"{h}: 近傍 0 件の行に値がある")
    if np.isnan(tm[~zero]).any():
        raise ValueError(f"{h}: 近傍がある行の nbr_tani_mean が NaN")
    if np.isinf(ms).any() or np.isinf(tm).any():
        raise ValueError(f"{h}: inf がある")
    return dict(set=h, n_rows=int(len(part)), n_nbr0=int(zero.sum()), n_lt_k=int((nn < K_NEIGHBOURS).sum()),
                n_nan=dict(nbr_ms_mean=int(np.isnan(ms).sum()), nbr_tani_mean=int(np.isnan(tm).sum())),
                n_ms_nan_with_nbr=int((np.isnan(ms) & ~zero).sum()))


def add_columns(df: pd.DataFrame, parts: dict) -> tuple[pd.DataFrame, list]:
    """df（集合ごとの行は neighbour_ref.py が読んだ順）に parts {set: DataFrame} の 2 列を足す。"""
    bad = [c for c in NEW_COLS if c in df.columns]
    if bad:
        raise ValueError(f"既に列がある: {bad}")
    sets = list(pd.unique(df["set"].astype(str)))
    if sorted(sets) != sorted(parts):
        raise ValueError(f"表の集合と parts の集合が違う: {sorted(set(sets) ^ set(parts))}")
    out = df.copy()
    for c in NEW_COLS:
        out[c] = np.nan
    checks = []
    for h in sets:
        m = (df["set"].astype(str) == h).to_numpy()
        checks.append(check_part(df.loc[m], parts[h], h))
        for c in NEW_COLS:
            out.loc[m, c] = parts[h][c].to_numpy(np.float64)
    return out, checks


def hide_rule_check(metas: dict) -> dict:
    """#290: holdout の N1 の part が集合の隠す規則どおりに作られたか。mode = hide_rule、rule = library.hide_rule(h)、
    隠した正準の鍵の数 n_hidden_canon が drop_keys（raw と正準の両方を受ける neighbour_ref.drop_canon）から作り直した数と同じで 0 より大きい、
    隠した train の行がある（rule = all のとき。enveda_libs は enveda 系の行が無い構造もあるので数だけ記録）。
    neighbour_ref.py 自身も「隠した構造が近傍に入った」なら止まる（assert）。"""
    import library as L
    from neighbour_ref import drop_canon, load_library
    cd = Path("/Users/taichi/.hermes/profiles/massspec/cache/scratch/nbr_cache")
    _sk, _si, raw2canon, _rp, _rm = load_library(cd)
    out = {}
    for h, m in metas.items():
        rule = L.hide_rule(h)["rule"]
        n_hid = len(drop_canon(L.drop_keys(h), raw2canon))
        r = dict(mode=m["mode"], rule=m["rule"], expected_rule=rule, n_hidden_canon=m["n_hidden_canon"],
                 expected_n_hidden_canon=n_hid, n_rows_hidden=m["n_rows_hidden"], n_avail=m["n_avail"])
        r["ok"] = bool(m["mode"] == "hide_rule" and m["rule"] == rule and m["n_hidden_canon"] == n_hid > 0
                       and (rule != "all" or m["n_rows_hidden"] > 0))
        out[h] = r
    out["all_ok"] = all(v["ok"] for v in out.values())
    return out


def main(argv=None) -> int:
    import library as L
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--parts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--dump-parts", default=str(HOLD_PARTS), help="holdout の行数の記録（dump_dev の parts_holdout）")
    a = ap.parse_args(argv)
    exp = expect_for(a.which, Path(a.dump_parts))
    SETS = exp["sets"]
    t0 = time.time()
    inp, out, pdir = Path(a.inp), Path(a.out), Path(a.parts)
    in_sha = sha256_file(inp)
    df = pd.read_parquet(inp)
    if sorted(pd.unique(df["set"].astype(str))) != sorted(SETS):
        raise ValueError(f"表の集合が {a.which} の {len(SETS)} 集合と違う")
    parts, metas = {}, {}
    for h in SETS:
        pp, pj = pdir / f"{h}.parquet", pdir / f"{h}.json"
        meta = json.loads(pj.read_text())
        exp_rule = L.hide_rule(h)["rule"]
        if meta["mode"] != "hide_rule" or meta["rule"] != exp_rule:
            raise ValueError(f"{h}: mode/rule が違う {meta['mode']}/{meta['rule']} (期待 hide_rule/{exp_rule})")
        if meta["table_sha256"] != in_sha:
            raise ValueError(f"{h}: part を作った表が入力の表と違う")
        if meta["out_sha256"] != sha256_file(pp):
            raise ValueError(f"{h}: part の sha256 が json と違う")
        parts[h] = pd.read_parquet(pp)
        metas[h] = meta
    new, checks = add_columns(df, parts)
    for c in checks:
        if c["n_nan"] != metas[c["set"]]["n_nan"]:
            raise ValueError(f"{c['set']}: NaN の数が json と違う")
    same = all(new[c].equals(df[c]) for c in df.columns)
    num = new[list(NEW_COLS)].to_numpy(np.float64)
    n_mol = int(len(new[MOL].drop_duplicates()))
    tmp = out.with_suffix(".tmp.parquet")
    new.to_parquet(tmp, index=False)
    tmp.replace(out)
    rep = dict(issue=261, input=str(inp), input_sha256=in_sha, parts=str(pdir), output=str(out),
               output_sha256=sha256_file(out), output_is_symlink=out.is_symlink(),
               output_realpath=str(out.resolve()), parent_is_symlink=out.parent.is_symlink(),
               which=a.which, expect=dict(n_mol=exp["n_mol"], n_rows=exp["n_rows"]),
               n_rows=int(len(new)), n_molecules=n_mol, n_rows_ok=len(new) == exp["n_rows"],
               n_molecules_ok=n_mol == exp["n_mol"], added=list(NEW_COLS), original_columns_identical=bool(same),
               n_inf=int(np.isinf(num).sum()), nan={c: int(np.isnan(num[:, j]).sum()) for j, c in enumerate(NEW_COLS)},
               n_nbr0_rows=sum(c["n_nbr0"] for c in checks),
               part_checks=checks, part_meta={h: dict(rule=m["rule"], secs_total=m["secs_total"], n_lt_k=m["n_lt_k"],
                                                      out_sha256=m["out_sha256"]) for h, m in metas.items()},
               secs=round(time.time() - t0, 1))
    rep["all_ok"] = bool(rep["n_rows_ok"] and rep["n_molecules_ok"] and same and rep["n_inf"] == 0
                         and not rep["output_is_symlink"] and not rep["parent_is_symlink"])
    if a.which == "holdout":   # #290: 隠す規則の点検（leak_check）
        rep["leak_check_hide_rule"] = hide_rule_check(metas)
        rep["all_ok"] = rep["all_ok"] and rep["leak_check_hide_rule"]["all_ok"]
    rp = out.with_name(out.stem + "_report.json")
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("part_checks", "part_meta")}, ensure_ascii=False),
          flush=True)
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
