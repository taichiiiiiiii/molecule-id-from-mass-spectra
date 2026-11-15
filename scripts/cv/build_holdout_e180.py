#!/usr/bin/env python3
"""
Enveda-180 の train 外 timsTOF から class 2 型の検証集合 e180_c2 を作る（Issue #122 の事前登録どおり。定数はそこで固定）。

入力（sha256 を照合し、違えば止める）:
  - e180_not_in_train.parquet（83_e180_probe_lib.py の出力。正イオンの m/z は電子の質量を引き済み → そのまま使う）
  - e180_structs.parquet（#108 の数え上げの構造表。in_pc・pool_src だけを使う）
  - 83_e180_probe_lib.py（電子の質量を引く行があることを確かめて manifest に書く）
照合（読むだけ）: train.parquet の inchikey14、train_structs（inchikey14・key・rawkey）、pool_meta の key、
  v3 の 4 集合の正解、PubChem 層（pubchem_tier、±1 ppm の窓、ElementGraph の前段の絞り込み → score_key）。

作り方（#122）:
  1. 対象 = in_pc かつ pool_src == 0、SMILES が読める、[M+H]+ か [M-H]- のスペクトルを 1 本以上持つ構造
  2. 選ぶ前に全対象へ確認（プール・train・別名・v3 の正解・PubChem 層の再確認）をかけ、当たった構造を外す（補わない）
  3. ik14 の昇順 → default_rng(0).permutation の先頭 min(200, n)。選んだ順に m_00000…、s_000000…
  4. 1 構造 1 スペクトル: (adduct, precursor_mz, ピーク数, 元の行番号) で並べ、同じ rng の integers(len) で 1 本
出力（gitignore 先）: data/holdouts/e180_c2/ に test.parquet・sample_submission.csv・truth.csv・drop_keys.txt・
  hide_rule.json・HOLDOUT.json・manifest.json。

    .venv/bin/python scripts/cv/build_holdout_e180.py --src <scratch>/n1115n_e180
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NAME = "e180_c2"
SEED = 0
N_MAX = 200
ADDUCTS = ("[M+H]+", "[M-H]-")
PPM_PUBCHEM = 1.0
SHA = {
    "e180/e180_not_in_train.parquet": "2d55c117a5a8caf90e2cb660881a9dd361ba6ee378c3fae50e43ed339277c50a",
    "e180_structs.parquet": "344eb8c0ce7bd16a8dc33df08b8094be29fd6db4802385830301ef45e27eda30",
    "e180/83_e180_probe_lib.py": "dd9d7e386a3eeb1ca800aa6e0b7db3e4dbfb2e30e313513d42b9eea9fe92875b",
}
ELECTRON_LINE = "sh = 0.000548 if pos else 0.0"
V3_SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
TEST_COLS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
             "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_orig", "collision_energy_ev",
             "collision_energy_orig_units"]


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_electron_line(src_text: str) -> dict:
    """83 のスクリプトが正イオンから電子の質量を引いていること（= ここでは引かない）を確かめる。"""
    lines = [i + 1 for i, l in enumerate(src_text.splitlines()) if l.strip().startswith(ELECTRON_LINE)]
    if len(lines) != 1:
        raise ValueError(f"83 のスクリプトに電子の質量の行が 1 行だけあるはず: {lines}")
    used = ("precursor_mz=round(float(r['pepmass']) - sh" in src_text) and ("mzs=np.round(mz[o] - sh" in src_text)
    if not used:
        raise ValueError("83 のスクリプトで sh が precursor_mz と断片の m/z の両方に効いていない")
    return dict(line_no=lines[0], line=ELECTRON_LINE, applied_to=["precursor_mz", "mzs"], subtract_again=False)


def eligible_structures(structs: pd.DataFrame, spectra: pd.DataFrame, parse_ok) -> list[str]:
    """手順 1: in_pc かつ pool_src == 0、SMILES が読める、[M+H]+/[M-H]- のスペクトルを持つ構造（ik14 の昇順）。"""
    has = set(spectra.loc[spectra.adduct.isin(ADDUCTS), "ik14"])
    s = structs[structs.in_pc.astype(bool) & (structs.pool_src == 0) & structs.ik14.isin(has)]
    smi = spectra.drop_duplicates("ik14").set_index("ik14").smiles
    return sorted(ik for ik in s.ik14 if parse_ok(smi.get(ik)))


def exclusion_reasons(keys: dict[str, set[str]], pool_keys: set[str], train_ik14: set[str],
                      ts_pairs: list[tuple[str, str, str | None]], v3_keys: set[str],
                      pubchem_found: set[str]) -> dict[str, list[str]]:
    """手順 2: 構造ごとに外す理由。keys = {ik14: {ik14, to_key, score_key}}、ts_pairs = [(inchikey14, key, rawkey)]。
    別名 = train_structs で key が正解のキーのどれかと同じ inchikey14 の行。"""
    ts_ik = {a for a, _, _ in ts_pairs}
    ts_key = {b for _, b, _ in ts_pairs if b}
    ts_raw = {c for _, _, c in ts_pairs if c}
    out: dict[str, list[str]] = {}
    for ik, ks in keys.items():
        ks = {k for k in ks if k}
        r = []
        if ks & pool_keys:
            r.append("pool")
        if ks & train_ik14:
            r.append("train_ik14")
        if ks & ts_ik:
            r.append("train_structs_ik14")
        if ks & ts_key:
            r.append("train_structs_key")      # 別名（key が同じ）もここで当たる
        if ks & ts_raw:
            r.append("train_structs_rawkey")
        if ks & v3_keys:
            r.append("v3_truth")
        if ik not in pubchem_found:
            r.append("pubchem_not_found")
        if r:
            out[ik] = r
    return out


def select(eligible: list[str], spectra: pd.DataFrame, n_max: int = N_MAX, seed: int = SEED) -> list[tuple[str, int]]:
    """手順 3・4。eligible は ik14 の昇順。spectra は元の行番号の列 row を持つ。返り値 [(ik14, row)]（選んだ順）。"""
    eligible = sorted(eligible)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))[:min(n_max, len(eligible))]
    sp = spectra[spectra.adduct.isin(ADDUCTS)]
    by = {ik: g for ik, g in sp.groupby("ik14", sort=False)}
    out = []
    for j in order:
        ik = eligible[int(j)]
        g = by[ik]
        g = g.assign(_np=g.mzs.map(len)).sort_values(["adduct", "precursor_mz", "_np", "row"], kind="stable")
        k = int(rng.integers(len(g)))
        out.append((ik, int(g.row.iloc[k])))
    return out


def build_frames(chosen: list[tuple[str, int]], spectra: pd.DataFrame, smiles_of: dict[str, str]):
    """test.parquet（data/test.parquet と同じ 12 列）・truth・sample_submission。m/z はそのまま。"""
    sp = spectra.set_index("row")
    rows, truth = [], []
    for i, (ik, r) in enumerate(chosen):
        x = sp.loc[r]
        assert x.ik14 == ik
        mid = f"m_{i:05d}"
        rows.append(dict(molecule_id=mid, spectrum_id=f"s_{i:06d}", ms2_mzs=np.asarray(x.mzs, np.float64),
                         ms2_normalized_intensities=np.asarray(x.intensities, np.float64), base_peak_intensity=np.nan,
                         adduct=x.adduct, ionization_mode=x["mode"], instrument_type="timsTOF",
                         precursor_mz=float(x.precursor_mz), collision_energy_orig="", collision_energy_ev=None,
                         collision_energy_orig_units=""))
        truth.append(dict(molecule_id=mid, inchikey14=ik, smiles=smiles_of[ik]))
    te = pd.DataFrame(rows, columns=TEST_COLS)
    tr = pd.DataFrame(truth, columns=["molecule_id", "inchikey14", "smiles"])
    ss = pd.DataFrame(dict(molecule_id=tr.molecule_id, smiles="CCO"))
    return te, tr, ss


def ppm_summary(nm: np.ndarray, exact: np.ndarray, adducts: np.ndarray) -> dict:
    """診断 (a): neutral_mass(precursor_mz) と正解の厳密質量の差（ppm）の付加イオン別。補正には使わない。"""
    ppm = (np.asarray(nm, float) - np.asarray(exact, float)) / np.asarray(exact, float) * 1e6
    out = {}
    for a in ADDUCTS:
        v = ppm[(np.asarray(adducts) == a) & np.isfinite(ppm)]
        out[a] = dict(n=int(len(v)), median=round(float(np.median(v)), 3) if len(v) else None,
                      mean=round(float(np.mean(v)), 3) if len(v) else None)
    return out


def pos_neg_pair_ppm(mol: np.ndarray, nm: np.ndarray, adducts: np.ndarray) -> dict:
    """診断 (b): 同じ分子の [M+H]+ と [M-H]- から出した中性質量の差（正 − 負、ppm、分子ごとの中央値どうし）。"""
    d = pd.DataFrame(dict(m=mol, nm=nm, a=adducts))
    d = d[d.a.isin(ADDUCTS) & np.isfinite(d.nm)]
    g = d.groupby(["m", "a"]).nm.median().unstack()
    if not set(ADDUCTS) <= set(g.columns):
        return dict(n=0, median=None)
    g = g.dropna(subset=list(ADDUCTS))
    v = (g[ADDUCTS[0]] - g[ADDUCTS[1]]) / g[ADDUCTS[1]] * 1e6
    return dict(n=int(len(v)), median=round(float(v.median()), 3) if len(v) else None)


# ---------------------------------------------------------------- 重いところ（ローカルだけ）

def _casmi():
    sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
    from casmi import chem
    return chem


def _eg(smi):
    from rdkit import Chem
    from rdkit.Chem import rdMolHash as H
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    Chem.RemoveStereochemistry(m)
    return H.MolHash(m, H.HashFunction.ElementGraph)


def _pc_job(args):
    """PubChem 層の 1 窓: ElementGraph が一致する候補だけ score_key を計算（#108 と同じ前段の絞り込み）。"""
    (a, b), targets, d = args
    chem = _casmi()
    off = np.load(d / "pc_off.npy", mmap_mode="r")
    buf = np.load(d / "pc_smiles.npy", mmap_mode="r")
    o = np.asarray(off[a:b + 1]).astype(np.int64)
    raw = bytes(buf[o[0]:o[-1]])
    want = {t[1] for t in targets}
    found = set()
    for i in range(b - a):
        s = raw[o[i] - o[0]:o[i + 1] - o[0]].decode("ascii")
        if _eg(s) not in want:
            continue
        k = chem.score_key(s)
        if k is None:
            continue
        for ik, _sig, ks in targets:
            if k in ks:
                found.add(ik)
    return found, b - a


def pubchem_recheck(rows: list[tuple[str, float, str, set]], workers: int = 4) -> tuple[set, dict]:
    from multiprocessing import Pool
    d = ROOT / "data/ext/pubchem_tier"
    m = np.load(d / "pc_mass.npy", mmap_mode="r")
    groups: dict[float, list] = {}
    for ik, mass, sig, ks in rows:
        groups.setdefault(round(mass, 6), []).append((ik, sig, ks))
    jobs = []
    for M, tg in groups.items():
        tol = M * PPM_PUBCHEM * 1e-6
        a, b = int(np.searchsorted(m, M - tol)), int(np.searchsorted(m, M + tol, "right"))
        if b > a:
            jobs.append(((a, b), tg, d))
    found, ncand = set(), 0
    with Pool(workers) as p:
        for f, n in p.imap_unordered(_pc_job, jobs, chunksize=4):
            found |= f; ncand += n
    return found, dict(windows=len(jobs), candidates_in_window=ncand, ppm=PPM_PUBCHEM, prefilter="RDKit MolHash ElementGraph（立体を消す）")


def train_ik14(path: Path) -> set:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    s = set()
    for b in pf.iter_batches(batch_size=65536, columns=["inchikey14"]):
        s.update(b.column(0).to_pylist())
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="n1115n_e180 のフォルダ（e180/ と e180_structs.parquet がある）")
    ap.add_argument("--out", default=ROOT / f"data/holdouts/{NAME}", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    out = args.out
    if out.is_symlink():
        raise SystemExit(f"{out} はリンク。クローンの中の実体のフォルダに書く")
    out.mkdir(parents=True, exist_ok=False)   # 作り直さない（事前登録どおり 1 回）

    files = {}
    for rel, want in SHA.items():
        got = sha256_file(args.src / rel)
        if got != want:
            raise SystemExit(f"sha256 が違う: {rel} {got}")
        files[rel] = dict(sha256=got, bytes=(args.src / rel).stat().st_size)
    electron = check_electron_line((args.src / "e180/83_e180_probe_lib.py").read_text())

    sys.path.insert(0, str(ROOT / "scripts/validate"))
    from mrr25 import to_key
    chem = _casmi()
    from rdkit import Chem, RDLogger, rdBase
    from rdkit.Chem import Descriptors
    RDLogger.DisableLog("rdApp.*")

    sp = pd.read_parquet(args.src / "e180/e180_not_in_train.parquet")
    sp["row"] = np.arange(len(sp))
    st = pd.read_parquet(args.src / "e180_structs.parquet", columns=["ik14", "in_pc", "pool_src"])
    smiles_of = sp.drop_duplicates("ik14").set_index("ik14").smiles.to_dict()
    elig = eligible_structures(st, sp, lambda s: isinstance(s, str) and Chem.MolFromSmiles(s) is not None)

    keys, info = {}, {}
    for ik in elig:
        smi = smiles_of[ik]
        k1, k2 = to_key(smi), chem.score_key(smi)
        keys[ik] = {ik, k1, k2}
        mol = Chem.MolFromSmiles(smi)
        info[ik] = dict(to_key=k1, score_key=k2, mass=Descriptors.ExactMolWt(mol), sig=_eg(smi))

    pm = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key"])
    pool_keys = set(pm.key.dropna()); n_pool = len(pm); del pm
    tik = train_ik14(ROOT / "data/train.parquet")
    ts = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet", columns=["inchikey14", "key", "rawkey"])
    ts_pairs = list(zip(ts.inchikey14, ts.key, ts.rawkey)); n_ts = len(ts); del ts
    v3_keys = set()
    for s in V3_SETS:
        t = pd.read_csv(ROOT / f"outputs/holdouts/{s}/truth.csv", dtype=str)
        v3_keys |= set(t.inchikey14.dropna()) | set((ROOT / f"data/holdouts/{s}/drop_keys.txt").read_text().split())
    tp = time.time()
    found, pc_info = pubchem_recheck([(ik, info[ik]["mass"], info[ik]["sig"], {k for k in keys[ik] if k}) for ik in elig],
                                     workers=args.workers)
    pc_info["secs"] = round(time.time() - tp, 1)

    excl = exclusion_reasons(keys, pool_keys, tik, ts_pairs, v3_keys, found)
    kept = [ik for ik in elig if ik not in excl]
    chosen = select(kept, sp)
    te, tr, ss = build_frames(chosen, sp, smiles_of)

    # 数え上げ（選んだ集合で、ik14・正準キー 2 種のそれぞれ）
    sel = [ik for ik, _ in chosen]
    def cnt(fn, S):
        return sum(1 for ik in sel if fn(ik) in S)
    ts_ik = {a for a, _, _ in ts_pairs}; ts_key = {b for _, b, _ in ts_pairs if b}; ts_raw = {c for _, _, c in ts_pairs if c}
    checks = {}
    for nm_, fn in (("ik14", lambda ik: ik), ("to_key", lambda ik: info[ik]["to_key"]), ("score_key", lambda ik: info[ik]["score_key"])):
        checks[nm_] = dict(pool=cnt(fn, pool_keys), train_ik14=cnt(fn, tik), train_structs_ik14=cnt(fn, ts_ik),
                           train_structs_key=cnt(fn, ts_key), train_structs_rawkey=cnt(fn, ts_raw), v3_truth=cnt(fn, v3_keys))
    canon = {k for ik in sel for k in (info[ik]["to_key"], info[ik]["score_key"], ik) if k}
    aliases = sorted({a for a, b, _ in ts_pairs if b in canon})
    checks["alias_train_structs_rows"] = len(aliases)
    checks["pubchem_found"] = sum(1 for ik in sel if ik in found)

    # drop_keys（#116 と同じ定義）
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    from rebuild_drop_keys import rebuild_keys, format_keys
    drop, _ = rebuild_keys(list(zip(tr.molecule_id, tr.inchikey14, tr.smiles)), set(),
                           [(a, b) for a, b, _ in ts_pairs if b], to_key)

    # 診断（補正には使わない）
    exact = np.array([info[ik]["mass"] for ik in tr.inchikey14])
    nm_sel = chem.neutral_mass(te.precursor_mz.values, te.adduct.values)
    spe = sp[sp.ik14.isin(set(kept)) & sp.adduct.isin(ADDUCTS)]
    nm_all = chem.neutral_mass(spe.precursor_mz.values, spe.adduct.values)
    ex_all = spe.ik14.map({ik: info[ik]["mass"] for ik in kept}).values
    diag = dict(ppm_vs_truth=dict(e180_c2=ppm_summary(nm_sel, exact, te.adduct.values),
                                  e180_eligible_all_spectra=ppm_summary(nm_all, ex_all, spe.adduct.values)),
                pos_minus_neg_ppm=dict(e180_eligible_all_spectra=pos_neg_pair_ppm(spe.ik14.values, nm_all, spe.adduct.values)))
    for s in ("v3_c2", "v3_c1"):
        h = pd.read_parquet(ROOT / f"data/holdouts/{s}/test.parquet", columns=["molecule_id", "adduct", "precursor_mz"])
        t = pd.read_csv(ROOT / f"outputs/holdouts/{s}/truth.csv", dtype=str).set_index("molecule_id").smiles
        exm = {m: Descriptors.ExactMolWt(Chem.MolFromSmiles(t[m])) for m in h.molecule_id.unique()}
        nmh = chem.neutral_mass(h.precursor_mz.values, h.adduct.values)
        diag["ppm_vs_truth"][s] = ppm_summary(nmh, h.molecule_id.map(exm).values, h.adduct.values)
        diag["pos_minus_neg_ppm"][s] = pos_neg_pair_ppm(h.molecule_id.values, nmh, h.adduct.values)
    vt = pd.read_parquet(ROOT / "data/test.parquet", columns=["molecule_id", "adduct", "precursor_mz"])
    diag["pos_minus_neg_ppm"]["visible_test"] = pos_neg_pair_ppm(vt.molecule_id.values, chem.neutral_mass(vt.precursor_mz.values, vt.adduct.values), vt.adduct.values)

    te.to_parquet(out / "test.parquet", index=False)
    tr.to_csv(out / "truth.csv", index=False)
    ss.to_csv(out / "sample_submission.csv", index=False)
    (out / "drop_keys.txt").write_text(format_keys(drop))
    (out / "hide_rule.json").write_text(json.dumps(dict(rule="all", pool="remove"), indent=2) + "\n")
    ad = te.adduct.value_counts().to_dict()
    holdout = dict(name=NAME, issue=122, seed=SEED, n_max=N_MAX, n_molecules=len(te), n_spectra=len(te),
                   adducts={a: int(ad.get(a, 0)) for a in ADDUCTS},
                   ionization_mode=te.ionization_mode.value_counts().to_dict(),
                   n_eligible=len(elig), n_excluded=len(excl), excluded=excl,
                   excluded_by_reason={r: sum(1 for v in excl.values() if r in v) for r in sorted({x for v in excl.values() for x in v})},
                   n_kept=len(kept), checks=checks, n_drop_keys=len(drop), pubchem_recheck=pc_info, diagnostics=diag,
                   reference_sizes=dict(pool_rows=n_pool, train_ik14=len(tik), train_structs_rows=n_ts, v3_keys=len(v3_keys)),
                   truth_location="data/holdouts/e180_c2/truth.csv（run.py は outputs/holdouts/<h>/truth.csv を読むので、採点の回に読み先を決める）")
    (out / "HOLDOUT.json").write_text(json.dumps(holdout, ensure_ascii=False, indent=1) + "\n")
    manifest = dict(issue=122, inputs=files, electron_mass=electron, script_sha256=sha256_file(Path(__file__)),
                    rdkit=rdBase.rdkitVersion, numpy=np.__version__, pandas=pd.__version__,
                    outputs={f.name: sha256_file(f) for f in sorted(out.iterdir()) if f.name != "manifest.json"},
                    created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), secs=round(time.time() - t0, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(dict(holdout=holdout, manifest=manifest), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
