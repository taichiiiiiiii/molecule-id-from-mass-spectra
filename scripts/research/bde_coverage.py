#!/usr/bin/env python3
"""
Issue #184: 候補 B（BDE で重みをつけた断片の点、#178 §4.2）に使う BDE 表の候補ごとに、v3 の 4 集合の窓の候補の構造の
結合のうち「表に値がある結合」の割合を数える（採点〈MRR〉はしない。表の選定の材料）。

この スクリプトは表の **キー（元素の組 × 結合次数）だけ** を持ち、BDE の値は持たない（値の転載の条件は
docs/research/bde-table-source.md を参照。被覆の測定に値は要らない）。

定義（測る前にこのコミットで固定。v3 を見て替えない）:
- 集合: v3_c1・v3_c2・enveda_v1・v3_c2n（共通の決まりの 4 集合）。参考に e180_c2k は測らない
- 窓の候補: scripts/local_eval/run.py と同じ ±10 ppm の窓（`run.Pool.window`）から、pool=remove の集合では
  `library.drop_keys` のキーを除いたもの（`run.visible_window` をそのまま呼ぶ）。目標の質量は
  outputs/holdout_eval/<h>_fold0/pc_logits.npz の `__targets__`、分子は truth.csv にあるものだけ（run.py と同じ）
- 構造の単位: 集合ごとに、全分子の窓に出た pool の行を pool の key でまとめた **一意の構造**（同じ key は最初の行の
  SMILES）。窓に出た延べの行数も記録する
- 結合: RDKit の `Chem.MolFromSmiles`（既定の sanitize、H は暗黙）で作った分子の全結合 ＝ MetFrag-lite
  （data/ext/frag/frag_rescore.py の `_graph`）が切る重原子のグラフの辺と同じ。SMILES に明示の H 原子が残れば
  その結合も数える
- 結合のキー: (元素 a, 元素 b〈記号の昇順〉, 次数)。次数は 2 つの表し方で数える
  - kekule（主）: `Chem.Kekulize(m, clearAromaticFlags=True)` の後の SINGLE=1・DOUBLE=2・TRIPLE=3
  - aromatic（副）: 既定の芳香族の表し方。AROMATIC は 'ar'（どの表にも 'ar' は無いので被覆されない）
  - それ以外の結合の型（DATIVE など）は 'other'（被覆されない）
- 表に無い結合の扱いの規則の案（値はこの規則で機械的に決まり、v3 を見て選ばない）:
  - R0（厳密）: (a, b, 次数) が表にあるときだけ値がある
  - R1（元素の組だけで引く）: R0 で無ければ、同じ元素の組の単結合 (a, b, 1) の値を使う
  - R2（既定値）: R1 でも無ければ、その表自身の C–C 単結合の値を使う（MetFrag の既定値と同じ考え。R2 の被覆は
    定義から 100% なので数えない）
- 数える量（表 × 集合 × 表し方 × 規則 R0・R1）: 結合の割合 = 値のある結合 / 全結合（一意の構造の結合の和）、
  全結合が表にある構造の割合、被覆されない結合のキーの上位 10。SMILES が読めない構造は別に数え、分母に入れない
- 参考（採否に使わない）: 正解の構造（truth.csv の smiles）について同じ量

    .venv/bin/python scripts/research/bde_coverage.py
    /usr/bin/time -l .venv/bin/python scripts/research/bde_coverage.py      # ピークの RSS を記録

出力: outputs/local_eval/bde_coverage.json（gitignore。数値は文書に写す）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
ORDER_SYM = {"-": 1, "=": 2, "#": 3}


def parse_keys(spec: str) -> frozenset:
    """'C-C C=O C#N ...' -> {(a, b, order)}。a, b は元素記号の昇順。"""
    out = set()
    for tok in spec.split():
        sym = next(c for c in tok if c in ORDER_SYM)
        a, b = tok.split(sym)
        a, b = sorted((a, b))
        out.add((a, b, ORDER_SYM[sym]))
    return frozenset(out)


# 表のキーだけ（値は持たない）。出典・版・読んだ日は docs/research/bde-table-source.md §1
TABLES = {
    # T1: OpenStax Chemistry（初版 2015、CC BY 4.0）Table 7.2 "Bond Energies (kJ/mol)"。68 項目（2e と同じキー）
    "openstax_t7_2": parse_keys(
        "H-H H-C H-N H-O H-F H-Si H-P H-S H-Cl H-Br H-I "
        "C-C C=C C#C C-N C=N C#N C-O C=O C#O C-F C-Si C-P C-S C-Cl C-Br C-I "
        "N-N N=N N#N N-O N-F N-P N-Cl N-Br "
        "O-O O=O O-F O-Si O-P O-Cl O-I "
        "F-F F-Si F-P F-S F-Cl F-Br "
        "Si-Si Si-P Si-S Si-Cl Si-Br Si-I "
        "P-P P-S P-Cl P-Br P-I "
        "S-S S-Cl S-Br Cl-Cl Cl-Br Cl-I Br-Br Br-I I-I"),
    # T2: Wikipedia "Bond dissociation energy"（revid 1368968041、CC BY-SA 4.0）§Representative bond enthalpies の 2 表
    "wikipedia_bde": parse_keys(
        "C-C C-F H-H H-F O-H C-O C#O C=O O=O N#N C-H C=C C#C"),
    # T3: Huheey, Keiter & Keiter, Inorganic Chemistry 4th ed. (1993) の値を載せた LibreTexts（Tro の Map）9.10 Table 1
    "huheey_libretexts": parse_keys(
        "H-H H-C H-Si H-N H-P H-O H-S H-F H-Cl H-Br H-I "
        "C-C C-Si C-N C-O C-S C-F C-Cl C-Br C-I Si-Si Si-O "
        "N-N N-O N-F N-Cl N-Br P-P "
        "O-O O-F O-Cl O-Br O-I S-S S-F S-Cl S-Br "
        "F-F F-Cl F-Br F-I Cl-Cl Cl-Br Cl-I Br-Br Br-I I-I "
        "C=C C#C C=N C#N C=O C#O N=N N#N N=O O=O S=O"),
    # T4: MetFrag（MetFragRelaunched）BondEnergies.java の既定の表のキーだけ（値は写さない。'~' は三重結合）
    "metfrag_default_keys": parse_keys(
        "H-H H-C H-Si H-N H-P H-O H-S H-F H-Cl H-Br H-I H-B H-Ge H-Sn H-As H-Se H-T "
        "C-C C=C C#C C-O C=O C-N C=N C#N C-F C-Cl C-Br C-I C-S C=S C-Si C-Ge C-Sn C-Pb C-P C-B "
        "P-P P-O P=O P=S P-F P-Cl P-Br P-I F-Cl Si-Si N-N N=N N#N O-O O=O F-F Cl-Cl Br-Br I-I At-At "
        "Se-Se I-O I-F I-Cl I-Br B-B B-O B-F B-Cl B-Br S-Cl S=S S=O N=O N-O S-S F-N F-O F-S"),
    # T5: BDE-db2（figshare 10.6084/m9.figshare.19367051、CC BY 4.0）の bond_type 列に出る元素の組（すべて単結合・
    # 環に入らない結合）。ここから組ごとの代表値を自分で作る案（§4）。キーは --bdedb2 で csv から作り直して照合できる
    "bdedb2_single": parse_keys(
        "C-H C-C C-N H-N C-O H-O C-S C-F N-O N-N N-S C-Cl H-S C-I O-P C-P O-O O-S H-P N-P Br-C Cl-N F-N F-S"),
}
# T1+T5（1 回目の測定〈commit d5d3592〉の後に足した組み合わせ。docs §3.2）: 値は T1 を先に引き、T1 に無いキーだけ
# T5 の組ごとの代表値を使う。キーは和集合（T1 に足されるのは N-S・O-S の 2 つ）
TABLES["openstax_plus_bdedb2"] = TABLES["openstax_t7_2"] | TABLES["bdedb2_single"]


def bond_key(bond, mode: str):
    a, b = sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
    bt = str(bond.GetBondType())
    order = {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3}.get(bt)
    if order is None:
        order = "ar" if (bt == "AROMATIC" and mode == "aromatic") else "other"
    return (a, b, order)


def mol_bond_keys(smi: str):
    """{'kekule': [keys], 'aromatic': [keys]}。読めなければ None。"""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    if m is None:
        return None
    aro = [bond_key(b, "aromatic") for b in m.GetBonds()]
    k = Chem.Mol(m)
    try:
        Chem.Kekulize(k, clearAromaticFlags=True)
        kek = [bond_key(b, "kekule") for b in k.GetBonds()]
    except Exception:   # Kekulé にできない分子は芳香族の結合を 'other' として残す（被覆されない側）
        kek = [bond_key(b, "kekule") for b in m.GetBonds()]
    return {"kekule": kek, "aromatic": aro}


def covered(key, table: frozenset, rule: str) -> bool:
    if key in table:
        return True
    if rule == "R1":
        a, b, order = key
        return order in (1, 2, 3) and (a, b, 1) in table
    return False


def coverage(structs: list, table: frozenset, mode: str, rule: str) -> dict:
    """structs: mol_bond_keys の結果（None を除いたもの）。"""
    nb = nc = nall = 0
    miss = Counter()
    for s in structs:
        keys = s[mode]
        c = [covered(k, table, rule) for k in keys]
        nb += len(keys)
        nc += sum(c)
        nall += all(c)
        miss.update(k for k, ok in zip(keys, c) if not ok)
    fmt = lambda k: f"{k[0]}{ {1: '-', 2: '=', 3: '#'}.get(k[2], ':' + str(k[2])) }{k[1]}".replace(" ", "")
    return dict(n_struct=len(structs), n_bonds=nb, n_bonds_covered=nc,
                bond_frac=round(nc / nb, 4) if nb else None, bond_frac_raw=nc / nb if nb else None,
                n_struct_all_covered=nall, struct_frac=round(nall / len(structs), 4) if structs else None,
                top_missing=[[fmt(k), n] for k, n in miss.most_common(10)])


def window_structures(h: str, pool, R, L) -> tuple[list, int]:
    from mrr25 import read_truth
    zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    truth = read_truth(ROOT / f"outputs/holdouts/{h}/truth.csv")
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    seen, smis, n_rows = set(), [], 0
    for mid in truth:
        if mid not in targets:
            continue
        idx = R.visible_window(pool, targets[mid], hidden)
        n_rows += len(idx)
        for i in idx:
            k = pool.key[i]
            if k not in seen:
                seen.add(k)
                smis.append(pool.smiles[i])
    return smis, n_rows


def truth_smiles(h: str) -> list:
    import csv
    with open(ROOT / f"outputs/holdouts/{h}/truth.csv", newline="") as f:
        return [r["smiles"] for r in csv.DictReader(f) if r.get("smiles")]


def bdedb2_pairs(path: Path) -> frozenset:
    import pandas as pd
    bt = pd.read_csv(path, usecols=["bond_type"]).bond_type.dropna().unique()
    return frozenset(tuple(sorted(t.split("-"))) + (1,) for t in bt)


def summarize(structs: list) -> dict:
    out = {}
    for name, tab in TABLES.items():
        out[name] = {f"{mode}_{rule}": coverage(structs, tab, mode, rule)
                     for mode in ("kekule", "aromatic") for rule in ("R0", "R1")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/bde_coverage.json"))
    ap.add_argument("--bdedb2", default=None, help="bde-db2.csv.gz（あれば T5 のキーを作り直して照合する）")
    a = ap.parse_args()
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import pandas as pd
    import rdkit
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    import run as R
    import library as L
    res: dict = dict(issue=184, rdkit=rdkit.__version__,
                     git_head=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(),
                     script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     tables={k: len(v) for k, v in TABLES.items()}, sets={}, truth={})
    if a.bdedb2:
        regen = bdedb2_pairs(Path(a.bdedb2))
        res["bdedb2_keys_match"] = regen == TABLES["bdedb2_single"]
        if not res["bdedb2_keys_match"]:
            print("BDE-db2 のキーが表 T5 と食い違う", sorted(regen ^ TABLES["bdedb2_single"]), file=sys.stderr)
            sys.exit(2)
    meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "smiles", "mass", "pid"])
    pool = R.Pool(meta, None)
    del meta
    cache: dict = {}
    for h in SETS:
        smis, n_rows = window_structures(h, pool, R, L)
        structs, n_bad = [], 0
        for s in smis:
            if s not in cache:
                cache[s] = mol_bond_keys(s)
            if cache[s] is None:
                n_bad += 1
            else:
                structs.append(cache[s])
        res["sets"][h] = dict(n_window_rows=n_rows, n_unique=len(smis), n_unparsed=n_bad,
                              n_aromatic_bonds=sum(1 for s in structs for k in s["aromatic"] if k[2] == "ar"),
                              n_bonds=sum(len(s["kekule"]) for s in structs), tables=summarize(structs))
        tstructs = [x for x in (mol_bond_keys(s) for s in truth_smiles(h)) if x is not None]
        res["truth"][h] = summarize(tstructs)
        print(f"{h}: rows {n_rows} unique {len(smis)} unparsed {n_bad} ({time.time() - t0:.0f}s)", flush=True)
    res["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    for h in SETS:
        line = [f"{n} {res['sets'][h]['tables'][n]['kekule_R0']['bond_frac']:.4f}/"
                f"{res['sets'][h]['tables'][n]['kekule_R0']['struct_frac']:.4f}" for n in TABLES]
        print(h, " | ".join(line))


if __name__ == "__main__":
    main()
