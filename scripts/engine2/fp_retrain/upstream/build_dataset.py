"""Turn the 2.5M-spectrum library into model-ready tensors + a structure-level split."""
import numpy as np, pandas as pd, pickle, time, os, sys
import pyarrow.parquet as pq
from multiprocessing import Pool
import fpmodel

NP_LIBS = {'enveda-np-examples','riken','gnps','massbank','mona','spectraverse','msdial','drug_plus','masaryk'}

def _prep_chunk(args):
    mzs, its, precs = args
    out=[]
    for mz,it,p in zip(mzs,its,precs):
        a,b = fpmodel.prep_peaks(mz,it,p)
        out.append((a,b))
    return out

if __name__ == '__main__':
    t0=time.time()
    meta = pickle.load(open('data/lib_meta.pkl','rb'))
    off = np.load('data/lib_off.npy'); AMZ=np.load('data/lib_mz.npy'); AIN=np.load('data/lib_in.npy')
    err = pq.read_table('data/train.parquet', columns=['precursor_error_ppm','collision_energy_ev']).to_pandas()
    ppm = err.precursor_error_ppm.values
    ce  = np.array([float(np.mean(v)) if v is not None and len(v)>0 else -1.0 for v in err.collision_energy_ev.values], dtype=np.float32)
    print('meta loaded %.0fs'%(time.time()-t0), flush=True)

    n = len(off)-1
    npk = np.diff(off)
    ok = np.isfinite(meta['precursor_mz']) & (npk>=3)
    ok &= ~(np.isfinite(ppm) & (np.abs(ppm)>20))          # drop mislabelled spectra
    print('usable spectra: %d / %d'%(ok.sum(), n), flush=True)

    # structure index
    fpmeta = pickle.load(open('data/struct_fp_meta.pkl','rb'))
    key2ix = {k:i for i,k in enumerate(fpmeta['keys'])}
    six = np.array([key2ix.get(k,-1) for k in meta['inchikey14']], dtype=np.int32)
    ok &= (six>=0)
    idx = np.where(ok)[0]
    print('final spectra: %d  (%.0fs)'%(len(idx), time.time()-t0), flush=True)

    # ---- structure-level split ----
    coco = set(pd.read_parquet('db/coconut_clean.parquet').ik14)
    ik = meta['inchikey14']; lib = meta['ingest_lib']
    np_keys = set()
    for i in idx:
        if lib[i] in NP_LIBS and ik[i] in coco: np_keys.add(ik[i])
    np_keys = np.array(sorted(np_keys))
    forced = set(ik[i] for i in idx if lib[i]=='enveda-np-examples')
    rng = np.random.default_rng(0)
    pool = np.array([k for k in np_keys if k not in forced])
    val_extra = set(rng.choice(pool, size=min(2500,len(pool)), replace=False).tolist())
    VAL = forced | val_extra
    print('val structures: %d (forced np-examples %d)'%(len(VAL), len(forced)), flush=True)
    is_val = np.array([ik[i] in VAL for i in idx])
    print('val spectra %d / train spectra %d'%(is_val.sum(), (~is_val).sum()), flush=True)

    # ---- peak preprocessing (parallel) ----
    CH=20000
    chunks=[]
    for s in range(0,len(idx),CH):
        sel = idx[s:s+CH]
        chunks.append(([AMZ[off[i]:off[i+1]] for i in sel],
                       [AIN[off[i]:off[i+1]] for i in sel],
                       [meta['precursor_mz'][i] for i in sel]))
    print('chunks', len(chunks), flush=True)
    res=[]
    with Pool(10) as p:
        for j,r in enumerate(p.imap(_prep_chunk, chunks)):
            res.extend(r)
            if j%20==0: print('  chunk %d/%d  %.0fs'%(j,len(chunks),time.time()-t0), flush=True)
    lens = np.array([len(a) for a,_ in res], dtype=np.int32)
    noff = np.zeros(len(res)+1, dtype=np.int64); np.cumsum(lens, out=noff[1:])
    FMZ = np.concatenate([a for a,_ in res]).astype(np.float32)
    FIT = np.concatenate([b for _,b in res]).astype(np.float32)
    print('flat peaks', FMZ.shape, '%.0fs'%(time.time()-t0), flush=True)

    ad = np.array([fpmodel.ADDUCT_IX.get(a, fpmodel.ADDUCT_IX['<unk>']) for a in meta['adduct'][idx]], dtype=np.int16)
    ins= np.array([fpmodel.instr_family(s) for s in meta['instrument_type'][idx]], dtype=np.int8)
    mode=np.array([1.0 if str(m)=='positive' else -1.0 for m in meta['ionization_mode'][idx]], dtype=np.float32)
    os.makedirs('ds', exist_ok=True)
    np.savez('ds/spec.npz', off=noff, mz=FMZ, it=FIT,
             prec=meta['precursor_mz'][idx].astype(np.float32), ad=ad, ins=ins,
             ce=ce[idx], mode=mode, six=six[idx], is_val=is_val, src=idx.astype(np.int64))
    pickle.dump({'VAL':sorted(VAL),'forced':sorted(forced)}, open('ds/split.pkl','wb'))
    print('saved. total %.0fs'%(time.time()-t0), flush=True)
