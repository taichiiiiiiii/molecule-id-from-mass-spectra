"""Engine 2 の prvsiyan ランカーの base 31 列を、FP の 12 列だけ fold-safe FPNet で作り直す（#72、設計 §4.4 案 A）。

- `blk_base_fs(r, cfp, cfp_fs, zlog_fs)`: FP 以外の 19 列は `casmi_engine.blk_base`（= `pv.rank_features(..., model_logits=None)`、
  6,930 ビットの `cfp`）のまま。FP の 12 列（15–20、25–30）だけを `pv.py` 227–243 行と同じ式で、候補の 10,226 ビット指紋
  `cfp_fs` と fold-safe FPNet のロジット `zlog_fs` から計算して差し替える
- `fs_fp(smiles, bits)`: `casmi.chem.raw_fingerprint(smiles)[bits]`（0/1 の 10,226 ビット、v2-pool の `fp_bits.npy`）
- `fs_logits(bank, spectra)`: `scripts/cv/local_fold0_logits.py` の `molecule_logits` と同じ前処理（単スペクトル平均と
  極性ごとの合成スペクトル平均の平均、先頭 16 本）
- `score_base_fs(...)`: 推論側。`casmi_engine.score_molecules(recs, rankers, ["base"], use_fp=True)` の base を
  `blk_base_fs` に替えたもの

`casmi_engine` は呼び出し側が sys.path に置いたもの（Kaggle では eng_runner の ENG_DIR、ローカルでは scripts/engine2/eng）を使う。
`casmi` パッケージ（fold-safe FPNet のコード）は `CASMI_CODE_DIR` 環境変数か `set_casmi_code_dir()` で場所を渡す。
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG_DIR = os.path.join(HERE, "eng")
FP_COLS = list(range(15, 21)) + list(range(25, 31))
RDKIT_VERSION = "2026.03.3"
_CASMI_DIR = [os.environ.get("CASMI_CODE_DIR")]


def engine():
    """呼び出し側が import 済みの casmi_engine（POOL などの状態を共有するため同じモジュールを使う）。"""
    if "casmi_engine" not in sys.modules and ENG_DIR not in sys.path:
        sys.path.insert(0, ENG_DIR)
    import casmi_engine
    return casmi_engine


def set_casmi_code_dir(path):
    _CASMI_DIR[0] = path


def _casmi():
    d = _CASMI_DIR[0]
    if d and d not in sys.path:
        sys.path.insert(0, d)
    from casmi import chem
    return chem


def check_rdkit(expected=RDKIT_VERSION):
    import rdkit
    assert rdkit.__version__ == expected, f"RDKit {rdkit.__version__} != {expected}"
    return rdkit.__version__


# ------------------------------------------------------------------------------------------------
# FP の 12 列（pv.py rank_features 227–243 行の写し。cand_fp を cfp_fs、model_logits を zlog_fs に読み替える）
# ------------------------------------------------------------------------------------------------
def fp_columns(lv, ap, cfp_fs, zlog_fs):
    pv = engine().pv
    nc = cfp_fs.shape[0]
    cf = cfp_fs.astype(np.float32); cs = cf.sum(1)
    lv = np.asarray(lv, np.float32)
    ap = np.asarray(ap, np.float32)
    lvmax = float(lv.max()) if nc else 0.0
    apmax = float(ap.max()) if nc else 0.0
    model_logits = zlog_fs
    if model_logits is not None:
        raw = cf @ np.asarray(model_logits, np.float32)
        nrm = raw / np.sqrt(np.maximum(cs, 1.0))
        mfeat = [pv._z(raw), pv._rank_norm(raw), raw - raw.max(), pv._z(nrm), pv._rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6
    if model_logits is not None and nc:
        mr = pv._rank_norm(cf @ np.asarray(model_logits, np.float32))
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]
    else:
        xfeat = [np.zeros(nc, np.float32)] * 6
    return np.column_stack([*mfeat, *xfeat]).astype(np.float32)


def blk_base_fs(r, cfp, cfp_fs, zlog_fs):
    """base 31 列。FP 以外は blk_base（zlog なし、6,930 ビット）、FP 12 列は cfp_fs・zlog_fs から。"""
    E = engine()
    X = E.blk_base(dict(r, zlog=None), cfp)
    if zlog_fs is not None and len(X):
        X[:, FP_COLS] = fp_columns(X[:, 0], X[:, 5], cfp_fs, zlog_fs)
    return X


# ------------------------------------------------------------------------------------------------
# 10,226 ビットの候補指紋（v2-pool の空間）
# ------------------------------------------------------------------------------------------------
def fs_fp(smiles, bits):
    """0/1 uint8 の (len(bits),)。RDKit が読めない SMILES は None。"""
    chem = _casmi()
    fp = chem.raw_fingerprint(smiles)
    if fp is None:
        return None
    return fp[np.asarray(bits)].astype(np.uint8)


class FsFpCache:
    """プールの行番号 -> 10,226 ビット指紋（読めない構造は全 0、数を記録）。"""

    def __init__(self, smiles_of, bits):
        self.smiles_of = smiles_of; self.bits = np.asarray(bits); self.c = {}; self.failed = 0

    def get(self, idx):
        nb = len(self.bits)
        out = np.zeros((len(idx), nb), np.uint8)
        for j, i in enumerate(np.asarray(idx, np.int64).tolist()):
            v = self.c.get(i)
            if v is None:
                f = fs_fp(self.smiles_of(i), self.bits)
                if f is None:
                    self.failed += 1; f = np.zeros(nb, np.uint8)
                v = np.packbits(f); self.c[i] = v          # packbits で持つ（メモリ 1/8）
            out[j] = np.unpackbits(v)[:nb]
        return out


# ------------------------------------------------------------------------------------------------
# fold-safe FPNet のロジット（local_fold0_logits.molecule_logits と同じ前処理）
# ------------------------------------------------------------------------------------------------
def load_bank(model_path, device="cpu"):
    _casmi()
    from casmi import fpnet
    return fpnet.ModelBank([model_path], device=device)


def spectra_from_rows(rows):
    """rows: itertuples の行（ms2_mzs, ms2_normalized_intensities, ionization_mode, adduct, precursor_mz,
    collision_energy_ev）。local_fold0_logits.main と同じ dict を作る。"""
    out = []
    for r in rows:
        ce = r.collision_energy_ev
        ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
        out.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                        mode=1 if r.ionization_mode == "positive" else -1, adduct=r.adduct,
                        prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                        ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
    return out


def fs_logits(bank, spectra):
    """local_fold0_logits.molecule_logits の写し（spectra は先頭 16 本だけ使う）。"""
    _casmi()
    from casmi import chem, fpnet
    from casmi.spectra import merge_spectra
    spectra = spectra[:16]
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s["mz"], s["it"], s["prec"])
        ce_ok = s["ce"] is not None and not np.isnan(s["ce"])
        items_s.append(dict(mz=pm, it=pi, prec=float(s["prec"]), adduct_ix=chem.adduct_index(s["adduct"]),
                            ce=float(s["ce"]) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get("ce_n", 1)), 8), mode=float(s["mode"])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s["mode"] == md]
        if not grp:
            continue
        mm, ii = merge_spectra([(s["mz"], s["it"]) for s in grp])
        prec = float(np.median([s["prec"] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s["ce"] for s in grp if s["ce"] is not None and not np.isnan(s["ce"])]
        adducts = [s["adduct"] for s in grp]
        ad = max(adducts, key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get("ce_n", 1))) for s in grp), 8), mode=float(md)))
    if not items_s:
        return None
    zs, _ = bank.logits_el(items_s)
    zm, _ = bank.logits_el(items_m)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)


def zlog_fs_for_test(te, banks):
    """推論: test の各分子に fold-safe 5 本の fs_logits の平均。{molecule_id: z} (float32)。"""
    out = {}
    for mid, sub in te.groupby("molecule_id", sort=True):
        sp = spectra_from_rows(sub.itertuples())
        zs = [fs_logits(b, sp) for b in banks]
        if any(z is None for z in zs):
            continue
        out[mid] = np.mean(np.stack(zs), axis=0).astype(np.float32)
    return out


def score_base_fs(recs, rankers, zlog_fs, cfp_fs_of, report=None):
    """casmi_engine.score_molecules(recs, rankers, ["base"], use_fp=True) の base を blk_base_fs に替えたもの。
    zlog_fs: {mid: z}、cfp_fs_of(cand) -> (n, 10226) uint8。report があれば通った分子数を数える。"""
    E = engine()
    out = {}
    for r in recs:
        if not len(r["cand"]):
            continue
        cfp = E.pool_fps(r["cand"])
        z = zlog_fs.get(r["mid"])
        Xm = blk_base_fs(r, cfp, cfp_fs_of(r["cand"]) if z is not None else None, z).astype(np.float32)
        out[r["mid"]] = np.mean([m.predict_proba(Xm)[:, 1] for m in rankers], axis=0)
        if report is not None:
            report["base_fs_molecules"] = report.get("base_fs_molecules", 0) + 1
            if z is None:
                report.setdefault("no_zlog_fs", []).append(str(r["mid"]))
    return out
