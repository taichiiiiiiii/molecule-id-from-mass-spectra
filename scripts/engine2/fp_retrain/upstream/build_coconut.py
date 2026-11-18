import pandas as pd, numpy as np, pickle, time
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')

def norm(args):
    smi, npl = args
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    # keep largest fragment (drop salts/counterions)
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    try:
        m = Chem.RemoveHs(m); Chem.SanitizeMol(m)
        n = m.GetNumHeavyAtoms()
        if n < 5 or n > 120: return None
        f = CalcMolFormula(m)
        if '+' in f or '-' in f: return None      # drop charged species
        mw = ExactMolWt(m)
        if mw < 100 or mw > 1300: return None
        ik = Chem.MolToInchiKey(m)
        if not ik: return None
        return (ik[:14], Chem.MolToSmiles(m), f, mw, npl)
    except Exception:
        return None

if __name__ == '__main__':
    t0=time.time()
    df = pd.read_csv('db/coconut/coconut_csv-09-2026.csv',
                     usecols=['canonical_smiles','molecular_formula','np_likeness'], low_memory=False)
    print('coconut rows', len(df), '%.0fs'%(time.time()-t0), flush=True)
    args = list(zip(df.canonical_smiles.astype(str), df.np_likeness.fillna(0).astype(float)))
    with Pool(10) as p:
        res = p.map(norm, args, chunksize=400)
    res = [r for r in res if r]
    print('normalized', len(res), '%.0fs'%(time.time()-t0), flush=True)
    out = pd.DataFrame(res, columns=['ik14','smiles','formula','mass','np_likeness'])
    out = out.sort_values('np_likeness', ascending=False).drop_duplicates('ik14')
    print('unique ik14', len(out))
    out.to_parquet('db/coconut_clean.parquet', index=False)
    print(out.head())
    print('formula counts:', out.formula.nunique(), 'median isomers/formula', out.groupby('formula').size().median())
