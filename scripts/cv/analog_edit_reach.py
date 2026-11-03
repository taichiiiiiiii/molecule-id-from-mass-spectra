#!/usr/bin/env python3
"""
クラス 3 の模擬: 正解をプールから外し、FPNet 上位の類似体を「組成式を保つ編集」で展開して候補を生成し、
同じ z で採点して正解の順位を測る。

編集（組成式は変わらない）:
  1. 非芳香 C=C の 1 つ隣への移動
  2. 非環の単結合で切れる枝（OH, OMe, 糖, アルキルなど小さい側）を、核の別の位置（H を持つ原子）へ付け替え

    python scripts/cv/analog_edit_reach.py --run np_v2_pcopen_z --holdout np_v2 [--seeds 10]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi

RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, "data/ext/v4g/casmi26-v3-models/code")
from casmi import chem  # noqa: E402

PPM = 10.0
K = 25


def ik14(mol) -> str | None:
    try:
        return inchi.MolToInchiKey(mol)[:14]
    except Exception:
        return None


def sanitize(rw):
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def shift_double_bonds(mol):
    out = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE or b.GetIsAromatic():
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetSymbol() != "C" or a2.GetSymbol() != "C":
            continue
        for keep, move in ((a1, a2), (a2, a1)):
            for nb in move.GetNeighbors():
                if nb.GetIdx() == keep.GetIdx():
                    continue
                nb_bond = mol.GetBondBetweenAtoms(move.GetIdx(), nb.GetIdx())
                if nb_bond.GetBondType() != Chem.BondType.SINGLE or nb.GetTotalNumHs() == 0 or nb.GetIsAromatic():
                    continue
                rw = Chem.RWMol(mol)
                rw.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx()).SetBondType(Chem.BondType.SINGLE)
                rw.GetBondBetweenAtoms(move.GetIdx(), nb.GetIdx()).SetBondType(Chem.BondType.DOUBLE)
                # H 数の帳尻: keep 側は H+1、nb 側は H-1
                rw.GetAtomWithIdx(keep.GetIdx()).SetNumExplicitHs(keep.GetTotalNumHs() + 1)
                rw.GetAtomWithIdx(nb.GetIdx()).SetNumExplicitHs(nb.GetTotalNumHs() - 1)
                rw.GetAtomWithIdx(move.GetIdx()).SetNumExplicitHs(move.GetTotalNumHs())
                for i in (keep.GetIdx(), nb.GetIdx(), move.GetIdx()):
                    rw.GetAtomWithIdx(i).SetNoImplicit(True)
                m = sanitize(rw)
                if m is not None:
                    out.append(m)
    return out


def move_branches(mol, max_branch=40):
    out = []
    n = mol.GetNumAtoms()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        frags = Chem.FragmentOnBonds(mol, [b.GetIdx()], addDummies=False)
        parts = Chem.GetMolFrags(frags, asMols=False, sanitizeFrags=False)
        if len(parts) != 2:
            continue
        side = {idx: k for k, p in enumerate(parts) for idx in p}
        branch_set = set(parts[0]) if len(parts[0]) <= len(parts[1]) else set(parts[1])
        if len(branch_set) > max_branch:
            continue
        core_set = set(range(n)) - branch_set
        branch_atom, core_atom = (i, j) if i in branch_set else (j, i)
        for tgt in core_set:
            if tgt == core_atom:
                continue
            ta = mol.GetAtomWithIdx(tgt)
            if ta.GetTotalNumHs() == 0 or ta.GetSymbol() not in ("C", "N", "O"):
                continue
            rw = Chem.RWMol(mol)
            rw.RemoveBond(branch_atom, core_atom)
            rw.AddBond(branch_atom, tgt, Chem.BondType.SINGLE)
            ca = rw.GetAtomWithIdx(core_atom)
            ca.SetNumExplicitHs(mol.GetAtomWithIdx(core_atom).GetTotalNumHs() + 1); ca.SetNoImplicit(True)
            na = rw.GetAtomWithIdx(tgt)
            na.SetNumExplicitHs(ta.GetTotalNumHs() - 1); na.SetNoImplicit(True)
            m = sanitize(rw)
            if m is not None and len(Chem.GetMolFrags(m)) == 1:
                out.append(m)
    return out


def expand(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    gen = {}
    for m in shift_double_bonds(mol) + move_branches(mol):
        k = ik14(m)
        if k and k not in gen:
            gen[k] = Chem.MolToSmiles(m)
    return gen


def one(mid, k14, tsmi, targets, z, pool, mass, bits, args):
    if True:
        zz = z[mid].astype(np.float32); t = targets[mid]; tol = t * PPM * 1e-6
        a, b = np.searchsorted(mass, t - tol, "left"), np.searchsorted(mass, t + tol, "right")
        win = pool.iloc[a:b]
        if not args.keep_truth:
            win = win[win.key != k14]  # クラス 3: 正解はどこにも無い
        tform = chem.mol_props(tsmi)[0]
        cand = {}
        for key, smi, form in zip(win.key, win.smiles, win.formula):
            fp = chem.raw_fingerprint(smi)
            if fp is None or key in cand:
                continue
            cand[key] = (float(fp[bits].astype(np.float32) @ zz), smi, form, "pool")
        order = sorted(cand.items(), key=lambda kv: -kv[1][0])
        seeds = [(k, v) for k, v in order if (not args.same_formula_only or v[2] == tform)][: args.seeds]
        n_gen, reached_by = 0, None
        for sk, sv in seeds:
            for gk, gs in expand(sv[1]).items():
                if gk in cand:
                    continue
                fp = chem.raw_fingerprint(gs)
                if fp is None:
                    continue
                n_gen += 1
                cand[gk] = (float(fp[bits].astype(np.float32) @ zz), gs, chem.mol_props(gs)[0], f"edit:{sk}")
                if gk == k14 and reached_by is None:
                    reached_by = sk
        order = sorted(cand.items(), key=lambda kv: -kv[1][0])
        rank = next((r for r, (k, _) in enumerate(order, 1) if k == k14), 0)
        return dict(molecule_id=mid, n_pool=int(len(win)), n_gen=n_gen, reached=reached_by is not None,
                    rank=int(rank), fz_truth=cand[k14][0] if k14 in cand else None,
                    fz_top=order[0][1][0] if order else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--seeds", type=int, default=10, help="編集の出発点にする FPNet 上位の数")
    ap.add_argument("--same-formula-only", action="store_true")
    ap.add_argument("--keep-truth", action="store_true", help="クラス 2 の副作用測定: 正解をプールに残したまま編集候補を足す")
    args = ap.parse_args()
    bits = np.load("data/ext/v4g/casmi26-v2-pool/fp_bits.npy")
    z = np.load(f"outputs/holdout_eval/{args.run}/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in z["__targets__"]}
    truth = pd.read_csv(f"outputs/holdouts/{args.holdout}/truth.csv", dtype=str)
    pool = pd.read_parquet("data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "smiles", "formula", "mass"]).sort_values("mass").reset_index(drop=True)
    mass = pool.mass.values
    tag = f"{args.run}_s{args.seeds}{'_sf' if args.same_formula_only else ''}{'_keep' if args.keep_truth else ''}"
    Path("results").mkdir(exist_ok=True)
    part = Path(f"results/analog_edit_reach_{tag}.partial.jsonl")
    done = {json.loads(l)["molecule_id"] for l in part.read_text().splitlines()} if part.is_file() else set()
    rows = [json.loads(l) for l in part.read_text().splitlines()] if part.is_file() else []
    for mid, k14, tsmi in zip(truth.molecule_id, truth.inchikey14, truth.smiles):
        if mid not in targets or mid in done:
            continue
        try:
            row = one(mid, k14, tsmi, targets, z, pool, mass, bits, args)
        except Exception as exc:  # 1 分子の失敗で全体を止めない
            print(f"{mid} ERROR {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            row = dict(molecule_id=mid, n_pool=-1, n_gen=0, reached=False, rank=0, fz_truth=np.nan, fz_top=np.nan, error=type(exc).__name__)
        rows.append(row)
        with part.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{mid} pool={row['n_pool']} gen={row['n_gen']} reached={row['reached']} rank={row['rank']}", flush=True)
    df = pd.DataFrame(rows)
    summarize(df, args)
def summarize(df, args):
    rr = np.where((df["rank"] > 0) & (df["rank"] <= K), 1 / df["rank"].clip(lower=1), 0.0)
    out = dict(run=args.run, seeds=args.seeds, same_formula_only=args.same_formula_only, keep_truth=args.keep_truth, n=len(df),
               gen_per_molecule_median=int(df.n_gen.median()), reached=round(float(df.reached.mean()), 3),
               in_top25=round(float(((df["rank"] > 0) & (df["rank"] <= K)).mean()), 3),
               top1=round(float((df["rank"] == 1).mean()), 3), class3_mrr25=round(float(rr.mean()), 4),
               rank_dist={int(k): int(v) for k, v in df["rank"].clip(upper=26).value_counts().sort_index().items()})
    print(json.dumps(out, indent=2))
    tag = f"{args.run}_s{args.seeds}{'_sf' if args.same_formula_only else ''}{'_keep' if args.keep_truth else ''}"
    Path(f"results/analog_edit_reach_{tag}.json").write_text(json.dumps(out, indent=2))
    df.to_csv(f"results/analog_edit_reach_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
