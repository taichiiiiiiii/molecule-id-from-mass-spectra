"""
ローカルの PubChem チャネル（Issue #25）。0.479 版の probe_core と同じ 2 段: ECFP4 部分で粗く上位 PC_N1 を選び、
全選択ビットの f·z で並べる。プール（学習構造＋COCONUT）のキーは除く（そちらは Engine 側の担当）。

PubChem tier（ahmedberatozer/casmi26-pubchem-tier: pc_mass.npy, pc_off.npy, pc_smiles.npy）を data/ext/pubchem_tier に置く。
結果は分子ごとに outputs/local_eval/<holdout>/pubchem_fold0.json にキャッシュする。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PPM, PC_N1, K_KEEP = 10.0, 5000, 60   # PPM, PC_N1 は 0.479 版と同じ。K_KEEP は統合前に持つ候補数（ローカル評価の TOPN）


class Tier:
    def __init__(self, d: Path = ROOT / "data/ext/pubchem_tier"):
        self.mass = np.load(d / "pc_mass.npy", mmap_mode="r")
        self.off = np.load(d / "pc_off.npy", mmap_mode="r")
        self.buf = np.load(d / "pc_smiles.npy", mmap_mode="r")

    def window(self, target: float):
        tol = target * PPM * 1e-6
        a = int(np.searchsorted(self.mass, target - tol, "left")); b = int(np.searchsorted(self.mass, target + tol, "right"))
        if b <= a:
            return []
        off = np.asarray(self.off[a:b + 1]).astype(np.int64)
        raw = bytes(self.buf[off[0]:off[-1]]); base = off[0]
        return [raw[off[i] - base:off[i + 1] - base].decode("ascii") for i in range(b - a)]


def probe(tier: Tier, target: float, z: np.ndarray, bits: np.ndarray, pool_keys: set, chem, gen, Chem):
    """(smiles, f·z, score_key) を f·z 降順で最大 K_KEEP 件。プールのキーは除く。"""
    smis = tier.window(target)
    if not smis:
        return []
    sel_e = np.where(bits < 4096)[0]; raw_e = bits[sel_e]; z_e = z[sel_e]
    E = np.zeros((len(smis), len(raw_e)), np.float32); ok = np.zeros(len(smis), bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e; sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    if n1 == 0:
        return []
    top = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < len(smis) else np.where(ok)[0]
    scored = []
    for i in top:
        fp = chem.raw_fingerprint(smis[i])
        if fp is not None:
            scored.append((float(fp[bits].astype(np.float32) @ z), smis[i]))
    scored.sort(key=lambda r: -r[0])
    out, seen = [], set()
    for fz, s in scored:
        k = chem.score_key(s)
        if k is None or k in seen or k in pool_keys:
            continue
        seen.add(k); out.append((s, fz, k))
        if len(out) >= K_KEEP:
            break
    return out


def run_holdout(holdout: str, targets: dict, zf, bits: np.ndarray, pool_keys: set) -> dict:
    cache = ROOT / f"outputs/local_eval/{holdout}/pubchem_fold0.json"
    res = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = [m for m in targets if m not in res]
    if todo:
        import sys
        sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
        from casmi import chem
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
        tier = Tier()
        for j, m in enumerate(todo, 1):
            res[m] = probe(tier, targets[m], zf[m].astype(np.float32), bits, pool_keys, chem, gen, Chem)
            if j % 20 == 0 or j == len(todo):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(res))
                print(f"  pubchem {j}/{len(todo)}", flush=True)
    return res
