"""Fast binary spectral library: flatten 2.5M spectra into contiguous float32 arrays."""
import numpy as np, pyarrow.parquet as pq, pyarrow as pa, os, time, pickle

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2*MASS['H'] + MASS['O']
NH3 = MASS['N'] + 3*MASS['H']
NH4 = MASS['N'] + 4*MASS['H']
FORMATE = MASS['C'] + 2*MASS['H'] + 2*MASS['O']     # HCOOH
ACETATE = 2*MASS['C'] + 4*MASS['H'] + 2*MASS['O']   # CH3COOH

# adduct -> (n_mer, charge, delta)  such that  mz = (n*M + delta)/charge
ADDUCTS = {
 "[M+H]+":(1,1,PROTON), "[M+NH4]+":(1,1,NH4-E), "[M+Na]+":(1,1,MASS['Na']-E),
 "[M+K]+":(1,1,MASS['K']-E), "[M-H2O+H]+":(1,1,PROTON-H2O), "[M-2H2O+H]+":(1,1,PROTON-2*H2O),
 "[M+2H]2+":(1,2,2*PROTON), "[M]+":(1,1,-E), "[M-H2O]+":(1,1,-E-H2O),
 "[M+CH3OH+H]+":(1,1,PROTON+MASS['C']+4*MASS['H']+MASS['O']),
 "[M+CH3CN+H]+":(1,1,PROTON+2*MASS['C']+3*MASS['H']+MASS['N']),
 "[M-H]-":(1,1,-PROTON), "[M-H2O-H]-":(1,1,-PROTON-H2O), "[M+CH2O2-H]-":(1,1,FORMATE-PROTON),
 "[M+C2H4O2-H]-":(1,1,ACETATE-PROTON), "[M+Cl]-":(1,1,MASS['Cl']+E), "[M]-":(1,1,E),
 "[M-2H]-":(1,2,-2*PROTON), "[M+Na-2H]-":(1,1,MASS['Na']-2*PROTON),
 "[2M+H]+":(2,1,PROTON), "[2M+Na]+":(2,1,MASS['Na']-E), "[2M+NH4]+":(2,1,NH4-E),
 "[2M+K]+":(2,1,MASS['K']-E), "[2M-H]-":(2,1,-PROTON), "[2M+CH2O2-H]-":(2,1,FORMATE-PROTON),
 "[2M+C2H4O2-H]-":(2,1,ACETATE-PROTON), "[2M+Na-2H]-":(2,1,MASS['Na']-2*PROTON),
 "[3M+H]+":(3,1,PROTON), "[3M-H]-":(3,1,-PROTON),
}

def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a,(n,z,d) in ADDUCTS.items():
        m = (ad == a)
        if m.any(): out[m] = (mz[m]*z - d)/n
    return out

def load_flat(path, columns=None, meta_cols=None):
    meta_cols = meta_cols or ['ingest_lib','inchikey14','normalized_smiles','molecular_formula',
                              'adduct','precursor_mz','instrument_type','ionization_mode','collision_energy_orig']
    t0=time.time()
    t = pq.read_table(path, columns=meta_cols+['ms2_mzs','ms2_normalized_intensities'])
    mzc = t.column('ms2_mzs').combine_chunks()
    itc = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    d = {'off':off,'allmz':allmz,'allin':allin}
    for c in meta_cols:
        col = t.column(c)
        if pa.types.is_floating(col.type): d[c]=col.to_numpy(zero_copy_only=False)
        else: d[c]=np.asarray(col.cast(pa.string()).to_pylist(), dtype=object)
    print(f'loaded {len(off)-1} spectra, {len(allmz)} peaks in {time.time()-t0:.1f}s')
    return d
