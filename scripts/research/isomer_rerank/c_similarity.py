#!/usr/bin/env python3
"""(c) の分子について、1 位と正解の近さ（Morgan r2 2048 Tanimoto、Murcko 骨格一致）を数える（読むだけ）。"""
import csv, json, sys
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")
ROOT, H, TAG, MISS = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
truth = {r["molecule_id"]: r["smiles"] for r in csv.DictReader(open(ROOT / f"outputs/holdouts/{H}/truth.csv"))}
preds = {r["molecule_id"]: r["smiles"].split(";") for r in csv.DictReader(open(ROOT / f"outputs/local_eval/{H}/submission_{TAG}.csv", newline=""))}
per = json.loads(MISS.read_text())["per"]
def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
def scaf(m):
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception: return None
tans, same_scaf = [], 0
for p in per:
    if p["cls"] != "c": continue
    mt, m1 = Chem.MolFromSmiles(truth[p["molecule_id"]]), Chem.MolFromSmiles(preds[p["molecule_id"]][0])
    tans.append(DataStructs.TanimotoSimilarity(fp(mt), fp(m1)))
    same_scaf += scaf(mt) == scaf(m1)
tans.sort(); n = len(tans)
print(json.dumps(dict(holdout=H, n_c=n, tanimoto_median=round(tans[n // 2], 3) if n else None,
                      tan_ge_0_85=sum(t >= 0.85 for t in tans), tan_ge_0_7=sum(t >= 0.7 for t in tans),
                      tan_lt_0_5=sum(t < 0.5 for t in tans), same_murcko=same_scaf)))
