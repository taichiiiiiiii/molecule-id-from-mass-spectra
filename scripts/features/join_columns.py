#!/usr/bin/env python3
"""別の特徴表から列を行の鍵で結合して新しい表を作る（Issue #282: dev_b8_cc_gl2 + N1 の 2 列）。

行の鍵 = (set, molecule_id, cand_key)。次のどれかなら止める（出力を作らない）:
- 土台・出典のどちらかで鍵が重複する / 土台の鍵が出典に無い（結合の欠け）
- 足す列が土台に既にある / 足す列が出典に無い
- 結果の行数・行の順・元の列の値が土台と違う / 足した列に inf がある（NaN は数えて記録、既定で止める）
- 出力が既にある、出力先がシンボリックリンクの中

    python scripts/features/join_columns.py --base outputs/local_eval/features/dev_b8_cc_gl2.parquet \\
        --src outputs/local_eval/features/dev_b8_cc_n2_n1.parquet --cols nbr_ms_mean,nbr_tani_mean \\
        --out outputs/local_eval/features/dev_b8_cc_gl2_n1.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ("set", "molecule_id", "cand_key")


class JoinError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def join_columns(base: pd.DataFrame, src: pd.DataFrame, cols, keys=KEYS, allow_nan: bool = False):
    """base の行の順のまま src の cols を足す。戻り値: (結果, 記録)。"""
    cols, keys = list(cols), list(keys)
    if not cols:
        raise JoinError("足す列が無い")
    dup_cols = [c for c in cols if c in base.columns]
    if dup_cols:
        raise JoinError(f"足す列が土台に既にある: {dup_cols}")
    miss_cols = [c for c in cols + keys if c not in src.columns]
    if miss_cols:
        raise JoinError(f"出典に無い列: {miss_cols}")
    miss_keys = [c for c in keys if c not in base.columns]
    if miss_keys:
        raise JoinError(f"土台に鍵の列が無い: {miss_keys}")
    nb, ns = int(base.duplicated(keys).sum()), int(src.duplicated(keys).sum())
    if nb or ns:
        raise JoinError(f"鍵が重複する: 土台 {nb}・出典 {ns}")
    bk = pd.MultiIndex.from_frame(base[keys].astype(str))
    sk = pd.MultiIndex.from_frame(src[keys].astype(str))
    pos = sk.get_indexer(bk)
    n_missing = int((pos < 0).sum())
    if n_missing:
        raise JoinError(f"結合の欠け: 土台の {n_missing} 行の鍵が出典に無い")
    out = base.copy()
    for c in cols:
        out[c] = src[c].to_numpy()[pos]
    if len(out) != len(base):
        raise JoinError("行数が変わった")
    for c in base.columns:   # 元の列は値も順も同じ
        if not base[c].equals(out[c]):
            raise JoinError(f"元の列 {c} の値が変わった")
    if not (out[keys].astype(str).to_numpy() == src[keys].astype(str).to_numpy()[pos]).all():
        raise JoinError("鍵の対応が合わない")
    vals = out[cols].to_numpy(dtype=np.float64)
    n_inf = int(np.isinf(vals).sum())
    n_nan = {c: int(out[c].isna().sum()) for c in cols}
    if n_inf:
        raise JoinError(f"足した列に inf が {n_inf}")
    if not allow_nan and any(n_nan.values()):
        raise JoinError(f"足した列に NaN: {n_nan}")
    rec = dict(n_rows=int(len(out)), n_rows_src=int(len(src)), n_src_unused=int(len(src) - len(np.unique(pos))),
               added=cols, keys=keys, n_missing=n_missing, n_dup_base=nb, n_dup_src=ns, n_inf=n_inf, nan=n_nan,
               original_columns_identical=True, n_cols_in=int(base.shape[1]), n_cols_out=int(out.shape[1]))
    return out, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--cols", required=True, help="カンマ区切り")
    ap.add_argument("--out", required=True)
    ap.add_argument("--issue", type=int, default=282)
    ap.add_argument("--expect-rows", type=int, default=None)
    a = ap.parse_args(argv)
    base_p, src_p, out_p = Path(a.base), Path(a.src), Path(a.out)
    if out_p.exists():
        raise JoinError(f"出力が既にある: {out_p}")
    if out_p.parent.is_symlink() or any(p.is_symlink() for p in out_p.parent.parents if p != Path("/")):
        raise JoinError(f"出力先がシンボリックリンクの中: {out_p.parent}")
    cols = [c.strip() for c in a.cols.split(",") if c.strip()]
    base = pd.read_parquet(base_p)
    src = pd.read_parquet(src_p, columns=list(KEYS) + cols)
    out, rec = join_columns(base, src, cols)
    if a.expect_rows is not None and rec["n_rows"] != a.expect_rows:
        raise JoinError(f"行数 {rec['n_rows']} が期待 {a.expect_rows} と違う")
    out.to_parquet(out_p, index=False)
    back = pd.read_parquet(out_p, columns=list(KEYS))
    if not back.equals(base[list(KEYS)].reset_index(drop=True)):
        raise JoinError("書いた表の鍵の順が土台と違う")
    rep = dict(issue=a.issue, base=str(base_p), base_sha256=sha256_file(base_p), src=str(src_p),
               src_sha256=sha256_file(src_p), output=str(out_p), output_sha256=sha256_file(out_p),
               n_molecules=int(base.groupby(["set", "molecule_id"]).ngroups), **rec)
    rep_p = out_p.with_name(out_p.stem + "_report.json")
    rep_p.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
