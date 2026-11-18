"""Candidate pool = train structures U COCONUT, selected-bit fingerprints + exact mass."""
import numpy as np, pandas as pd, pickle, time, re
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')

_g = {}
def _init():
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not _g: _init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])
        return fp, float(ExactMolWt(m))
    except Exception: return None

def just_mass(smi):
    m = Chem.MolFromSmiles(smi)
    return float(ExactMolWt(m)) if m is not None else float('nan')

if __name__=='__main__':
    t0=time.time()
    bits = np.load('up_train/fp_bits.npy')
    tm = pickle.load(open('data/struct_fp_meta.pkl','rb'))
    tr_keys, tr_smi = list(tm['keys']), list(tm['smiles'])
    coco = pd.read_parquet('db/coconut_clean.parquet')
    add = coco[~coco.ik14.isin(set(tr_keys))].reset_index(drop=True)
    print('train structs %d, new coconut %d'%(len(tr_keys), len(add)), flush=True)

    with Pool(10) as p:
        res = p.map(fp_and_mass, list(add.smiles), chunksize=500)
    keep=[i for i,r in enumerate(res) if r is not None]
    print('coconut fps %d (%.0fs)'%(len(keep), time.time()-t0), flush=True)
    co_fp   = np.stack([res[i][0] for i in keep])[:, bits]
    co_mass = np.array([res[i][1] for i in keep]); del res
    co_keys = add.ik14.values[keep]; co_smi = add.smiles.values[keep]; co_npl = add.np_likeness.values[keep]

    with Pool(10) as p:
        tr_mass = np.array(p.map(just_mass, tr_smi, chunksize=800))
    print('train masses (%.0fs)'%(time.time()-t0), flush=True)
    tr_fp = np.unpackbits(np.load('up_train/fp_sel.npy'), axis=1)[:, :len(bits)]

    FP  = np.concatenate([tr_fp, co_fp]).astype(np.uint8); del tr_fp, co_fp
    MASS= np.concatenate([tr_mass, co_mass])
    KEYS= np.array(tr_keys+list(co_keys), dtype=object)
    SMI = np.array(tr_smi+list(co_smi), dtype=object)
    SRC = np.concatenate([np.zeros(len(tr_keys),np.int8), np.ones(len(co_keys),np.int8)])
    NPL = np.concatenate([np.full(len(tr_keys), np.nan), co_npl])
    good= np.isfinite(MASS) & (MASS>=80) & (MASS<=1400)
    FP,MASS,KEYS,SMI,SRC,NPL = FP[good],MASS[good],KEYS[good],SMI[good],SRC[good],NPL[good]
    o=np.argsort(MASS)
    FP,MASS,KEYS,SMI,SRC,NPL = FP[o],MASS[o],KEYS[o],SMI[o],SRC[o],NPL[o]
    np.save('pool_fp.npy', np.packbits(FP, axis=1))
    np.save('pool_mass.npy', MASS)
    pickle.dump({'keys':KEYS,'smiles':SMI,'src':SRC,'npl':NPL,'nbits':int(FP.shape[1])},
                open('pool_meta.pkl','wb'), protocol=4)
    print('POOL', FP.shape, 'mass %.1f-%.1f'%(MASS.min(),MASS.max()), '%.0fs'%(time.time()-t0), flush=True)
