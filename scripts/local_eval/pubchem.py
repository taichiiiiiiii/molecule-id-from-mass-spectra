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


def _load_ctx():
    """probe に要る (chem, gen, Chem)。fork した worker は親が読み込んだものを引き継ぐ。"""
    import sys
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    return chem, gen, Chem


def _write_json(p: Path, obj) -> None:
    """tmp に書いてから os.replace（途中で止まっても壊れたファイルを残さない）。"""
    import os
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, p)


def worker_cache_paths(cache: Path, workers: int) -> list[Path]:
    """#278: worker i（0 始まり）のキャッシュ。pubchem_fold0.json の横に pubchem_fold0.w<i>of<N>.json。"""
    return [cache.with_name(f"{cache.stem}.w{i}of{workers}{cache.suffix}") for i in range(workers)]


def _probe_loop(mids: list, targets: dict, zf, bits: np.ndarray, pool_keys: set, res: dict, cache: Path,
                tier_dir: Path | None, label: str = "") -> dict:
    """mids を順に採点し、res に足して 20 分子ごとと最後に cache へ書く（workers=1 のときは今までと同じ書き方）。"""
    chem, gen, Chem = _load_ctx()
    tier = Tier(tier_dir) if tier_dir is not None else Tier()
    todo = [m for m in mids if m not in res]
    for j, m in enumerate(todo, 1):
        res[m] = probe(tier, targets[m], zf[m].astype(np.float32), bits, pool_keys, chem, gen, Chem)
        if j % 20 == 0 or j == len(todo):
            if label:
                _write_json(cache, res)
            else:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(res))
            print(f"  pubchem{label} {j}/{len(todo)}", flush=True)
    return res


def _worker_main(i: int, mids: list, targets: dict, zf, bits, pool_keys, wpath: Path, tier_dir, n: int) -> None:
    import os
    import sys
    import traceback
    try:
        res = json.loads(wpath.read_text()) if wpath.is_file() else {}
        _probe_loop(mids, targets, zf, bits, pool_keys, res, wpath, tier_dir, label=f"[w{i}/{n}]")
        if not wpath.is_file():   # 割り当てがすべて済んでいた / 空のとき
            _write_json(wpath, res)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def run_holdout(holdout: str, targets: dict, zf, bits: np.ndarray, pool_keys: set, workers: int = 1,
                cache: Path | None = None, tier_dir: Path | None = None) -> dict:
    """分子ごとに PubChem 層の候補を採点する（結果は cache に分子ごとにキャッシュ）。

    #278: workers > 1 なら、まだ無い分子（todo）を todo[i::workers] で fork した workers 個のプロセスに分け、
    worker ごとに別のキャッシュ（worker_cache_paths）へ書き、最後に親が分子の順（既存のキャッシュ → todo の順）で
    cache にまとめる。worker が 1 つでも失敗すれば残りを止めて例外。workers=1 は今までと同じ動き。
    macOS の numpy（Accelerate）は fork の先で大きな行列の積が落ちる（SIGSEGV）ので、手元で workers > 1 を使うときは
    VECLIB_MAXIMUM_THREADS=1 を付ける（Kaggle の Linux の OpenBLAS には要らない）。
    """
    cache = Path(cache) if cache is not None else ROOT / f"outputs/local_eval/{holdout}/pubchem_fold0.json"
    workers = int(workers)
    assert workers >= 1
    res = json.loads(cache.read_text()) if cache.is_file() else {}
    todo = [m for m in targets if m not in res]
    if not todo:
        return res
    if workers == 1:
        return _probe_loop(todo, targets, zf, bits, pool_keys, res, cache, tier_dir)
    import multiprocessing as mp
    import time
    _load_ctx()   # fork の前に読み込む（worker が同じモジュールを引き継ぐ）
    # npz（zf）は開いたファイルを fork 先と共有し、同時に読むと位置がぶつかるので、要る分子の配列を先に読んでおく
    zs = {m: np.asarray(zf[m]) for m in todo}
    wpaths = worker_cache_paths(cache, workers)
    parts = [todo[i::workers] for i in range(workers)]
    ctx = mp.get_context("fork")
    procs = []
    for i, (mids, wp) in enumerate(zip(parts, wpaths)):
        pr = ctx.Process(target=_worker_main, args=(i, mids, targets, zs, bits, pool_keys, wp, tier_dir, workers),
                         daemon=True)
        pr.start()
        procs.append(pr)
    try:
        while True:
            codes = [pr.exitcode for pr in procs]
            bad = [i for i, c in enumerate(codes) if c not in (None, 0)]
            if bad:
                raise RuntimeError(f"pubchem worker {bad} が失敗した（exitcode {[codes[i] for i in bad]}。"
                                   "macOS で -11 なら VECLIB_MAXIMUM_THREADS=1 を付ける）")
            if all(c == 0 for c in codes):
                break
            time.sleep(0.2)
    finally:
        for pr in procs:
            if pr.is_alive():
                pr.terminate()
        for pr in procs:
            pr.join()
    got = {}
    for mids, wp in zip(parts, wpaths):
        wr = json.loads(wp.read_text())
        miss = [m for m in mids if m not in wr]
        if miss:
            raise RuntimeError(f"{wp.name} に {len(miss)} 分子が無い")
        got.update({m: wr[m] for m in mids})
    for m in todo:
        res[m] = got[m]
    _write_json(cache, res)
    for wp in wpaths:
        wp.unlink()
    print(f"  pubchem {len(todo)}/{len(todo)} (workers={workers})", flush=True)
    return res
