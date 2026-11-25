"""#196 の保存済み出力から fragnn2 の悪化・改善の分子を数え、悪化の truth と押し上げられた候補を比べる（F1 分析）。

読むもの（書かない）: outputs/local_eval/<h>/{truth.csv, submission_score+prior+frag.csv, submission_score+fragnn2+prior+frag.csv}
と data/holdouts/<h>/test.parquet、data/ext/frag/frag_rescore.py。新しい特徴の検証点は計算しない（覗き見の禁止）。
現在の 17 特徴を truth と decoy で計算し「見分けられるか」を記述するだけ。

使い方: python scripts/research/fragnn_feature_gaps.py --data-root /Users/taichi/work/molecule-id-dev --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HOLDOUTS = ("v3_c1u_t10", "enveda_v1u_t10")
CTRL, NEW = "submission_score+prior+frag.csv", "submission_score+fragnn2+prior+frag.csv"


def ik14(smi, cache={}):
    from rdkit import Chem
    if smi not in cache:
        m = Chem.MolFromSmiles(smi)
        cache[smi] = Chem.MolToInchiKey(m)[:14] if m is not None else None
    return cache[smi]


def rank_of(cands, key):
    for i, s in enumerate(cands):
        if ik14(s) == key:
            return i + 1
    return None


def struct_compare(a, b):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdFMCS, Descriptors, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    fa = AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048)
    mcs = rdFMCS.FindMCS([ma, mb], timeout=5, ringMatchesRingOnly=True, completeRingsOnly=True)
    sa = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(ma))
    sb = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mb))
    gsa = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(ma)))
    gsb = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mb)))
    fr = [d for d in Descriptors._descList if d[0].startswith("fr_")]
    fg_a = {n: f(ma) for n, f in fr}; fg_b = {n: f(mb) for n, f in fr}
    fg_diff = sorted(n for n in fg_a if fg_a[n] != fg_b[n])
    sugar = Chem.MolFromSmarts("[OX2;R][CX4;R]([OX2])")  # 環内 O–C–O（アノマー炭素）の粗い印
    na = rdMolDescriptors.CalcNumAtoms(ma)
    return dict(
        same_formula=rdMolDescriptors.CalcMolFormula(ma) == rdMolDescriptors.CalcMolFormula(mb),
        tanimoto=round(DataStructs.TanimotoSimilarity(fa, fb), 3),
        mcs_atoms=mcs.numAtoms, heavy_atoms=ma.GetNumHeavyAtoms(), mcs_frac=round(mcs.numAtoms / ma.GetNumHeavyAtoms(), 3),
        scaffold_eq=sa == sb, generic_scaffold_eq=gsa == gsb,
        rings=(rdMolDescriptors.CalcNumRings(ma), rdMolDescriptors.CalcNumRings(mb)),
        arom_rings=(rdMolDescriptors.CalcNumAromaticRings(ma), rdMolDescriptors.CalcNumAromaticRings(mb)),
        fg_diff=fg_diff, anomeric=(len(ma.GetSubstructMatches(sugar)), len(mb.GetSubstructMatches(sugar))),
        n_atoms=na,
    )


def classify(c):
    """粗い型（先に当たったもの）。"""
    if not c["same_formula"]:
        return "diff_formula"
    if c["anomeric"][0] > 0 and c["anomeric"][1] > 0 and c["generic_scaffold_eq"]:
        return "glycoside_or_sugar_position"
    if c["rings"][0] != c["rings"][1]:
        return "ring_vs_chain"
    if c["scaffold_eq"] and not c["fg_diff"]:
        return "positional_isomer_same_scaffold"
    if c["scaffold_eq"]:
        return "substituent_fg_swap_same_scaffold"
    if c["generic_scaffold_eq"]:
        return "heteroatom_or_ring_position_same_generic_scaffold"
    return "different_scaffold"


def feat_compare(B, S, fr, cfg, rows, smi_t, smi_d):
    """両候補の 17 特徴（全塊・全ピーク）を比べる。"""
    spec, _ = S.chunk_spectra(rows)
    use = S.usable_chunks(spec)
    spec = S.take_rows(spec, use)
    lv = [fr.fragment_levels(s, cfg["max_breaks"], cfg["max_bonds_2"], cfg["max_bonds_3"]) for s in (smi_t, smi_d)]
    masses, off = B.levels_to_csr(lv)
    ns = len(use)
    cs = np.repeat(np.arange(2), ns); sp = np.tile(np.arange(ns), 2)
    feats, fs = B.batch_features(masses, off, cs, sp, spec)
    nn = np.asarray(spec["n"])
    ft, fd = feats[:ns], feats[ns:]
    valid = np.arange(feats.shape[1])[None, :] < nn[:, None]
    cat = slice(4, 16)  # 段・水素・担い手（カテゴリ）
    cat_eq = np.all(ft[..., cat] == fd[..., cat], -1)[valid]
    expl_t = (ft[..., 4] == 0)[valid]; expl_d = (fd[..., 4] == 0)[valid]
    w = ft[..., 0][valid]
    dmax = float(np.abs(ft - fd)[valid].max()) if valid.any() else 0.0
    return dict(
        n_chunks=int(ns), n_peaks=int(valid.sum()),
        frac_peaks_cat_identical=round(float(cat_eq.mean()), 3),
        max_abs_feat_diff=round(dmax, 3),
        pair_t=round(float(fs[:ns].mean()), 4), pair_d=round(float(fs[ns:].mean()), 4),
        int_expl_t=round(float((w * expl_t).sum() / max(ns, 1)), 4), int_expl_d=round(float((w * expl_d).sum() / max(ns, 1)), 4),
        n_only_t=int((expl_t & ~expl_d).sum()), n_only_d=int((expl_d & ~expl_t).sum()),
        n_frag_t=int(off[0, 3] - off[0, 0]), n_frag_d=int(off[1, 3] - off[1, 0]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.data_root)
    for p in (root / "scripts/train", root / "scripts/local_eval", root / "scripts", root / "data/ext/frag"):
        sys.path.insert(0, str(p))
    import fragnn_bench as B
    import fragnn2_score as S
    import frag_rescore as fr
    cfg = fr.resolve_cfg(None)
    res = {}
    for h in HOLDOUTS:
        d = root / "outputs/local_eval" / h
        truth = pd.read_csv(d / "truth.csv")
        ctrl = pd.read_csv(d / CTRL).set_index("molecule_id")["smiles"].str.split(";")
        new = pd.read_csv(d / NEW).set_index("molecule_id")["smiles"].str.split(";")
        te = pd.read_parquet(root / "data/holdouts" / h / "test.parquet")
        mrows = S.molecule_rows(te)
        recs = []
        for t in truth.itertuples():
            cc = [s for s in ctrl[t.molecule_id] if s]; nc = [s for s in new[t.molecule_id] if s]
            rc, rn = rank_of(cc, t.inchikey14), rank_of(nc, t.inchikey14)
            rr = lambda r: 0.0 if r is None or r > 25 else 1.0 / r  # noqa: E731
            r = dict(h=h, mid=t.molecule_id, rank_ctrl=rc, rank_new=rn, d_rr=round(rr(rn) - rr(rc), 4))
            if r["d_rr"] < 0:
                above_c = {ik14(s) for s in cc[: (rc or 26) - 1]}
                lim = (rn - 1) if rn else len(nc)
                dec = next((s for s in nc[:lim] if ik14(s) not in above_c), None)
                r["decoy"], r["truth"] = dec, t.smiles
                r["decoy_rank_ctrl"] = rank_of(cc, ik14(dec)) if dec else None
                if dec:
                    r["struct"] = struct_compare(t.smiles, dec)
                    r["type"] = classify(r["struct"])
                    r["feat"] = feat_compare(B, S, fr, cfg, mrows[t.molecule_id], t.smiles, dec)
                    r["adducts"] = sorted({x["adduct"] for x in mrows[t.molecule_id]}) if isinstance(mrows[t.molecule_id][0], dict) and "adduct" in mrows[t.molecule_id][0] else None
            recs.append(r)
        res[h] = recs
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    for h, recs in res.items():
        w = [r for r in recs if r["d_rr"] < 0]; b = [r for r in recs if r["d_rr"] > 0]
        print(h, "n", len(recs), "worse", len(w), "better", len(b), "sum_d_rr", round(sum(r["d_rr"] for r in recs), 3))


if __name__ == "__main__":
    main()
