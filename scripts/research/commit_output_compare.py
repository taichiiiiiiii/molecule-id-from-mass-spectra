"""Issue #212 (A2(a)): 同じノートの 2 つの commit 出力の submission.csv を比べる。

比べ方は Issue #212 に結果を見る前に固定したもの:
  1. ファイルの sha256
  2. 行数・molecule_id の集合
  3. 候補の列（';' 区切り）が完全に同じ分子の数 / 全分子数
  4. 違う分子: 1 位が同じ割合、候補の集合が同じ数、共通候補の順位の差（平均・最大）
  5. 置き換えの行（commit で処理しない分子の決まった列）の分子数と、それを除いた一致数

使い方: python scripts/research/commit_output_compare.py A.csv B.csv [--out report.json]
標準ライブラリだけを使う。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

# commit の実行で処理しない分子に入る決まった列（CO;CCO;…;C25 のアルコール）
PLACEHOLDER = ";".join("C" * k + "O" for k in range(1, 26))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_submission(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """(行の順の molecule_id, molecule_id -> 候補の列)。id の重複は例外。"""
    order: list[str] = []
    rows: dict[str, list[str]] = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None or "molecule_id" not in r.fieldnames or "smiles" not in r.fieldnames:
            raise ValueError(f"{path}: molecule_id,smiles の列が無い")
        for row in r:
            mid = row["molecule_id"]
            if mid in rows:
                raise ValueError(f"{path}: molecule_id {mid} が重複")
            order.append(mid)
            s = row["smiles"] or ""
            rows[mid] = s.split(";") if s else []
    return order, rows


def is_placeholder(cands: list[str]) -> bool:
    return ";".join(cands) == PLACEHOLDER


def compare_rows(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict:
    ids_a, ids_b = set(a), set(b)
    common = sorted(ids_a & ids_b)
    identical = [m for m in common if a[m] == b[m]]
    differ = [m for m in common if a[m] != b[m]]
    ph_a = {m for m in a if is_placeholder(a[m])}
    ph_b = {m for m in b if is_placeholder(b[m])}
    real = [m for m in common if m not in ph_a and m not in ph_b]
    real_identical = [m for m in real if a[m] == b[m]]

    top1_same = 0
    set_same = 0
    rank_diffs: list[int] = []
    for m in differ:
        ca, cb = a[m], b[m]
        if ca and cb and ca[0] == cb[0]:
            top1_same += 1
        if set(ca) == set(cb):
            set_same += 1
        pos_b = {s: i for i, s in enumerate(cb)}
        for i, s in enumerate(ca):
            if s in pos_b:
                rank_diffs.append(abs(i - pos_b[s]))
    return {
        "n_a": len(a),
        "n_b": len(b),
        "ids_equal": ids_a == ids_b,
        "only_a": sorted(ids_a - ids_b),
        "only_b": sorted(ids_b - ids_a),
        "n_common": len(common),
        "n_identical": len(identical),
        "n_differ": len(differ),
        "differ_ids": differ,
        "n_placeholder_a": len(ph_a),
        "n_placeholder_b": len(ph_b),
        "placeholder_sets_equal": ph_a == ph_b,
        "n_real": len(real),
        "n_real_identical": len(real_identical),
        "differ_top1_same": top1_same,
        "differ_top1_same_rate": (top1_same / len(differ)) if differ else None,
        "differ_set_same": set_same,
        "differ_common_rank_absdiff_mean": (sum(rank_diffs) / len(rank_diffs)) if rank_diffs else None,
        "differ_common_rank_absdiff_max": max(rank_diffs) if rank_diffs else None,
    }


def compare_files(pa: Path, pb: Path) -> dict:
    oa, a = read_submission(pa)
    ob, b = read_submission(pb)
    rep = compare_rows(a, b)
    rep.update({
        "file_a": str(pa),
        "file_b": str(pb),
        "sha256_a": sha256_file(pa),
        "sha256_b": sha256_file(pb),
        "row_order_equal": oa == ob,
    })
    rep["sha256_equal"] = rep["sha256_a"] == rep["sha256_b"]
    return rep


# 追記（Issue #212 のコメント、v2 を見る前）: 時間の項目は比べない
TIME_KEYS = frozenset({"wall_s", "timings", "seconds", "total_s", "elapsed_s"})


def strip_time(obj):
    """dict の中の時間の項目を再帰的に取り除く。"""
    if isinstance(obj, dict):
        return {k: strip_time(v) for k, v in obj.items() if k not in TIME_KEYS}
    if isinstance(obj, list):
        return [strip_time(v) for v in obj]
    return obj


def diff_paths(a, b, prefix: str = "") -> list[str]:
    """2 つの JSON の違う場所（a.b.c の形）の一覧。"""
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            p = f"{prefix}.{k}" if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out.extend(diff_paths(a[k], b[k], p))
        return out
    return [] if a == b else [prefix or "<root>"]


def compare_json_no_time(pa: Path, pb: Path) -> dict:
    a = strip_time(json.loads(pa.read_text()))
    b = strip_time(json.loads(pb.read_text()))
    d = diff_paths(a, b)
    return {"equal_without_time": not d, "diff_paths": d}


def compare_p5tail(pa: Path, pb: Path) -> dict:
    """p5tail_e1.json の分子ごとの記録を比べる。"""
    a = json.loads(pa.read_text())["molecules_data"]
    b = json.loads(pb.read_text())["molecules_data"]
    common = sorted(set(a) & set(b))
    identical = [m for m in common if a[m] == b[m]]
    per_diff = {}
    for m in common:
        if a[m] == b[m]:
            continue
        fields = [f for f in sorted(set(a[m]) | set(b[m])) if a[m].get(f) != b[m].get(f)]
        base_a = [r[0] for r in a[m].get("base", [])]
        base_b = [r[0] for r in b[m].get("base", [])]
        gen_a = {r[0] for r in a[m].get("gen", [])}
        gen_b = {r[0] for r in b[m].get("gen", [])}
        per_diff[m] = {
            "fields": fields,
            "base_order_equal": base_a == base_b,
            "base_set_equal": set(base_a) == set(base_b),
            "gen_set_equal": gen_a == gen_b,
        }
    return {
        "ids_equal": set(a) == set(b),
        "n_a": len(a),
        "n_b": len(b),
        "n_identical": len(identical),
        "differ": per_diff,
    }


def file_table(da: Path, db: Path, exts=(".csv", ".json", ".log")) -> list[dict]:
    """2 つの出力ディレクトリの同じ相対パスのファイルの sha256 の表。"""
    def files(d: Path) -> dict[str, Path]:
        return {str(p.relative_to(d)): p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix in exts}
    fa, fb = files(da), files(db)
    rows = []
    for rel in sorted(set(fa) | set(fb)):
        ha = sha256_file(fa[rel]) if rel in fa else None
        hb = sha256_file(fb[rel]) if rel in fb else None
        rows.append({"file": rel, "sha256_a": ha, "sha256_b": hb, "equal": ha is not None and ha == hb})
    return rows


def compare_dirs(da: Path, db: Path) -> dict:
    rep: dict = {"dir_a": str(da), "dir_b": str(db), "files": file_table(da, db), "submissions": {}, "json": {}}
    for row in rep["files"]:
        rel = row["file"]
        if row["sha256_a"] is None or row["sha256_b"] is None:
            continue
        if rel.endswith(".csv") and Path(rel).name.startswith("submission"):
            rep["submissions"][rel] = compare_files(da / rel, db / rel)
    for name in ("p5tail_audit.json", "fr3to25_audit.json"):
        if (da / name).exists() and (db / name).exists():
            rep["json"][name] = compare_json_no_time(da / name, db / name)
    if (da / "p5tail_e1.json").exists() and (db / "p5tail_e1.json").exists():
        rep["p5tail_e1"] = compare_p5tail(da / "p5tail_e1.json", db / "p5tail_e1.json")
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path, help="csv（または --dirs のときは出力ディレクトリ）")
    ap.add_argument("b", type=Path)
    ap.add_argument("--dirs", action="store_true", help="2 つの出力ディレクトリを比べる")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    rep = compare_dirs(args.a, args.b) if args.dirs else compare_files(args.a, args.b)
    txt = json.dumps(rep, ensure_ascii=False, indent=1)
    if args.out:
        args.out.write_text(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
