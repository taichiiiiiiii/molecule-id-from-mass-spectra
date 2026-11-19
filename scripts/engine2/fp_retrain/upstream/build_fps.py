"""Compute structure-level fingerprints for every unique training structure."""
import numpy as np, pandas as pd, pickle, time, os
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

NBITS_ECFP = 4096
mgen2 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS_ECFP)
mgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=NBITS_ECFP)
rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def one(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        e2 = np.frombuffer(mgen2.GetFingerprintAsNumPy(m).tobytes(), dtype=np.uint8)
        e3 = mgen3.GetFingerprintAsNumPy(m).astype(np.uint8)
        rk = rdkgen.GetFingerprintAsNumPy(m).astype(np.uint8)
        mc = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)
        e2 = mgen2.GetFingerprintAsNumPy(m).astype(np.uint8)
        return np.concatenate([e2, e3, rk, mc])   # 4096+4096+2048+167 = 10407
    except Exception:
        return None

if __name__ == '__main__':
    meta = pickle.load(open('data/lib_meta.pkl','rb'))
    ik = meta['inchikey14']; smi = meta['normalized_smiles']
    # unique structures
    seen = {}
    for k, s in zip(ik, smi):
        if k and s and k not in seen: seen[k] = s
    keys = sorted(seen); smis = [seen[k] for k in keys]
    print('unique structures:', len(keys), flush=True)
    t0=time.time()
    with Pool(10) as p:
        res = p.map(one, smis, chunksize=500)
    ok = [i for i,r in enumerate(res) if r is not None]
    print('valid: %d  (%.1fs)'%(len(ok), time.time()-t0), flush=True)
    FP = np.stack([res[i] for i in ok]).astype(np.uint8)
    K  = [keys[i] for i in ok]; S=[smis[i] for i in ok]
    np.save('data/struct_fp.npy', np.packbits(FP, axis=1))
    pickle.dump({'keys':K,'smiles':S,'nbits':FP.shape[1]}, open('data/struct_fp_meta.pkl','wb'))
    print('FP matrix', FP.shape, 'packed', np.packbits(FP,axis=1).shape)
    freq = FP.mean(0)
    print('bits with freq in [0.005,0.995]:', int(((freq>=0.005)&(freq<=0.995)).sum()))
    np.save('data/struct_fp_freq.npy', freq)
