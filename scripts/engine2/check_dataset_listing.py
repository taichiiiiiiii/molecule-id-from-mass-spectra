"""push 前の照合（#165）: Kaggle の dataset の一覧が code dir の manifest.json をすべて含むか。

    python scripts/engine2/check_dataset_listing.py --manifest <code-dir>/manifest.json \
        [--dataset taichiiiii/casmi26-engine2-build-code | --listing-file <saved csv>]

- 一覧は `kaggle datasets files <id> --page-size 100 --csv` の出力（`name,size,creationDate`）。
- manifest の各ファイル（相対パス）が一覧にあり、大きさが一致すれば rc 0。欠け・大きさ違いがあれば rc 1。
- 一覧が読めない（ヘッダが無い・認証切れの文言・ページが溢れた・次ページの印）は rc 2（「欠け 0」とは言わない）。
#165 の前回は版に下位フォルダ（drop_keys/・eng/）が無く、起動 17.5 s で落ちた。その見落としを止めるためのもの。
"""
import argparse
import csv
import io
import json
import subprocess
import sys

PAGE_SIZE = 100
HEADER = ["name", "size", "creationDate"]


class ListingUnreadable(Exception):
    pass


def parse_listing(text, page_size=PAGE_SIZE):
    """CSV の一覧 → {相対パス: 大きさ}。読めなければ ListingUnreadable。"""
    lines = [ln for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    if not lines:
        raise ListingUnreadable("empty listing")
    low = text.lower()
    for bad in ("authentication", "unauthorized", "401", "403", "404", "not found", "error"):
        if bad in low and not lines[0].strip().startswith("name,"):
            raise ListingUnreadable(f"looks like an error message: {lines[0][:200]!r}")
    if "next page token" in low:
        raise ListingUnreadable("listing is paginated (next page token); raise --page-size")
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if [c.strip() for c in rows[0]] != HEADER:
        raise ListingUnreadable(f"unexpected header: {rows[0]!r}")
    out = {}
    for r in rows[1:]:
        if len(r) != 3:
            raise ListingUnreadable(f"bad row: {r!r}")
        name = r[0].strip().replace("\\", "/")
        try:
            out[name] = int(r[1])
        except ValueError:
            raise ListingUnreadable(f"bad size in row: {r!r}")
    if len(rows) - 1 >= page_size:
        raise ListingUnreadable(f"{len(rows) - 1} rows >= page size {page_size}: may be truncated")
    return out


def compare(manifest, listing):
    """manifest（make_build_kernel.write_manifest の dict）と一覧 → 結果の dict。ok は欠け 0・大きさ違い 0。"""
    want = {f["path"]: f["size"] for f in manifest["files"]}
    missing = sorted(p for p in want if p not in listing)
    size_mismatch = [dict(path=p, manifest=want[p], listing=listing[p]) for p in sorted(want)
                     if p in listing and listing[p] != want[p]]
    extra = sorted(p for p in listing if p not in want)
    return dict(n_manifest=len(want), n_listing=len(listing), n_present=len(want) - len(missing),
                missing=missing, size_mismatch=size_mismatch, extra_in_listing=extra,
                ok=(not missing and not size_mismatch and len(want) > 0))


def fetch_listing(dataset, page_size=PAGE_SIZE):
    r = subprocess.run(["kaggle", "datasets", "files", dataset, "--page-size", str(page_size), "--csv"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ListingUnreadable(f"kaggle rc {r.returncode}: {(r.stderr or r.stdout)[-300:]}")
    return r.stdout


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset")
    g.add_argument("--listing-file")
    ap.add_argument("--save-listing", help="取った一覧の生の CSV を保存する")
    a = ap.parse_args(argv)
    manifest = json.load(open(a.manifest))
    try:
        text = open(a.listing_file).read() if a.listing_file else fetch_listing(a.dataset)
        if a.save_listing:
            open(a.save_listing, "w").write(text)
        listing = parse_listing(text)
    except ListingUnreadable as e:
        print(json.dumps(dict(ok=False, unreadable=str(e)), ensure_ascii=False))
        return 2
    res = compare(manifest, listing)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if res["ok"]:
        print(f"欠け 0: manifest の {res['n_manifest']} ファイルすべてが一覧にある（大きさも一致）")
        return 0
    print(f"NG: 欠け {len(res['missing'])}、大きさ違い {len(res['size_mismatch'])}（manifest {res['n_manifest']}）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
