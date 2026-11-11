"""MassSpecGym（MSG）の構造と、検証 v3 の正解・fold-safe の split hold/np との重なりを数える（読むだけ）。

- MSG の tsv（1.0 = MassSpecGym.tsv、1.5 = MassSpecGym1.5.tsv）から inchikey（先頭 14 文字）と fold 列だけを
  chunksize で読む（`inchikey` が 14 文字でなければ先頭 14 文字に切る。列が無ければ SMILES から RDKit で作る）
- 集合: v3_c1 / v3_c2 / enveda_v1 / v3_c2n の正解（`data/holdouts/<集合>/drop_keys.txt`）、split の hold と np
  （`data/ext/foldsafe/split.parquet` の inchikey14）
- 出力: 版ごと・MSG の fold ごとの重なりの件数と割合、1.0 と 1.5 の和集合での重なり、そして事前に決めた規則
  （MIST の学習側 = MSG の train ∪ val。学習側と重なる構造を除いた残りが 100 分子未満なら「測定不能」）での
  各集合の残りの分子数。保守的な側（MSG 全行を学習側とみなす）の残りも並べて出す

使い方:
  .venv/bin/python scripts/research/msg_overlap.py --msg data/ext/msg/MassSpecGym.tsv data/ext/msg/MassSpecGym1.5.tsv \
      --out outputs/research/msg_overlap.json
"""
import argparse
import hashlib
import json
import os

import pandas as pd

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
TRAIN_SIDE = ("train", "val")   # MIST の README: 学習は train、checkpoint の選択は val、test は読まない
MIN_MOLS = 100


def read_msg(path, chunksize=50_000):
    """{fold: set(inchikey14)} と、使った列の名前を返す。"""
    head = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    use_ik = "inchikey" in head
    cols = (["inchikey"] if use_ik else ["smiles"]) + ["fold"]
    out, rows = {}, 0
    for ch in pd.read_csv(path, sep="\t", usecols=cols, chunksize=chunksize, dtype=str):
        rows += len(ch)
        if use_ik:
            ks = ch["inchikey"].astype(str).str[:14]
        else:  # pragma: no cover - 今の 2 版は inchikey 列を持つ
            from rdkit import Chem
            ks = ch["smiles"].map(lambda s: (Chem.MolToInchiKey(Chem.MolFromSmiles(s)) or "")[:14]
                                  if Chem.MolFromSmiles(s) is not None else "")
        for f, k in zip(ch["fold"].astype(str), ks):
            if len(k) == 14:
                out.setdefault(f, set()).add(k)
    return out, ("inchikey[:14]" if use_ik else "smiles->RDKit inchikey[:14]"), rows


def load_sets(holdout_dir, split_path):
    sets = {}
    for h in HOLDOUTS:
        p = os.path.join(holdout_dir, h, "drop_keys.txt")
        sets[h] = {l.strip()[:14] for l in open(p) if l.strip()}
    sp = pd.read_parquet(split_path, columns=["inchikey14", "fold"])
    for f in ("hold", "np"):
        sets[f"split_{f}"] = set(sp.loc[sp.fold == f, "inchikey14"].astype(str))
    return sets


def overlap_table(msg_folds, sets):
    res = {}
    allk = set().union(*msg_folds.values()) if msg_folds else set()
    for name, ks in sets.items():
        r = {"n": len(ks)}
        for f, mk in sorted(msg_folds.items()):
            n = len(ks & mk); r[f] = dict(n=n, frac=round(n / len(ks), 4) if ks else 0.0)
        n = len(ks & allk); r["any_fold"] = dict(n=n, frac=round(n / len(ks), 4) if ks else 0.0)
        res[name] = r
    return res


def remaining(sets, excl):
    return {h: dict(n=len(sets[h]), overlap=len(sets[h] & excl), remain=len(sets[h] - excl),
                    measurable=len(sets[h] - excl) >= MIN_MOLS) for h in HOLDOUTS}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--msg", nargs="+", required=True)
    ap.add_argument("--holdout-dir", default="data/holdouts")
    ap.add_argument("--split", default="data/ext/foldsafe/split.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sets = load_sets(a.holdout_dir, a.split)
    out = dict(versions={}, rule=dict(train_side=list(TRAIN_SIDE), min_mols=MIN_MOLS,
               note="MSG 1.0 と 1.5 の和集合。重なりを見る前に決めた規則（Issue #66）"))
    union = {}
    for p in a.msg:
        folds, how, rows = read_msg(p)
        name = os.path.basename(p)
        out["versions"][name] = dict(bytes=os.path.getsize(p), rows=rows, key_from=how,
                                     sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
                                     n_struct_by_fold={f: len(v) for f, v in sorted(folds.items())},
                                     overlap=overlap_table(folds, sets))
        for f, v in folds.items():
            union.setdefault(f, set()).update(v)
    out["union"] = dict(n_struct_by_fold={f: len(v) for f, v in sorted(union.items())},
                        overlap=overlap_table(union, sets))
    train_side = set().union(*(union.get(f, set()) for f in TRAIN_SIDE))
    all_rows = set().union(*union.values()) if union else set()
    out["remaining_rule"] = remaining(sets, train_side)            # 事前登録の規則
    out["remaining_conservative"] = remaining(sets, all_rows)      # MSG 全行を学習側とみなす
    out["all_measurable_rule"] = all(v["measurable"] for v in out["remaining_rule"].values())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(dict(union=out["union"]["n_struct_by_fold"], rule=out["remaining_rule"],
                          conservative=out["remaining_conservative"]), ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()
