#!/usr/bin/env python3
"""
Issue #154: 「MassSpecGym in the Wild」（arXiv:2606.19624）§5.1 の近道（正解と候補の表記・正準化の不一致）が
検証 v3 の 4 集合に当たっていないかを数える、読むだけの確認（採点はしない。v3 の対照は変わらない）。

読むもの: outputs/holdouts/<h>/truth.csv、data/holdouts/<h>/hide_rule.json・drop_keys.txt、
outputs/holdout_eval/<h>_fold0/pc_logits.npz の __targets__（質量窓の中心）、v2-pool の pool_meta.parquet、
v3_c2 だけ outputs/local_eval/v3_c2/pubchem_fold0.json（PubChem チャネルのキャッシュ。プールに無い正解の行の表記を見る）。
train.parquet は読まない。

定義（数える前に固定）:
  キー: canon = casmi.chem.score_key（互変異性体を正準化した InChIKey14。指標 mrr25.to_key・pool_meta.key と同じ作り方）、
        raw = casmi.chem.raw_key（正準化しない InChIKey14）
  1. キーの一致（正解ごと）
     - col_eq_canon: truth.csv の inchikey14 列 == canon(truth.smiles)
     - raw_ne_canon: raw(truth.smiles) != canon(truth.smiles)（互変異性体の別名を持つ正解）
     - in_pool: canon が pool_meta.key にある / in_window: run.visible_window（±10 ppm、pool=remove なら drop_keys を除く）にある
     - remove 集合だけ: drop_has_canon（drop_keys に canon がある）、pool_has_canon（窓を除く前のプールにある）
  2. 表記（正解がプールの窓にある分子だけ）: 窓の中の正解の行（key == canon の最初の行）と、窓の中のおとり（key != canon）から
     seed 0 で N_DECOY 行を抜いた行で、次の 4 つの割合を比べる
     - rdkit_canon_str: Chem.MolToSmiles(Chem.MolFromSmiles(s)) == s（RDKit の正準 SMILES の文字列そのものか）
     - taut_canon: Chem.MolToSmiles(TautomerEnumerator().Canonicalize(m)) == Chem.MolToSmiles(m)
       （書かれた互変異性体が RDKit の正準の互変異性体か。InChI は アミド ↔ イミド酸 などの移動する H を同じキーにするので、
       raw と canon のキーの比較では表記の違いが見えない）
     - imidic: SMARTS [CX3]([OX2H1])=[NX2]（イミド酸の形。アミドの別の書き方）を含む
     - stereo: SMILES に '@' か '/' か '\\' がある
     加えて src（pool_meta の由来の符号。意味は未確認の推定: 1 = train だけ、2 = COCONUT、3 = 両方）の内訳
  3. v3_c2（正解はプールに無い）: PubChem のキャッシュの上位 60 に正解の canon があれば、その行の表記の 4 つと、
     同じ分子の他の提案の表記の 4 つ（全行）を比べる
  判定の目安（記録用。採否には使わない）: 正解の行とおとりで割合の差が 0.2 以上の特徴を「表記の差あり」と書く

    python scripts/cv/msg_wild_repr_check.py [--out outputs/local_eval/msg_wild_repr_check.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

HOLDOUTS = ("v3_c1", "v3_c2", "v3_c2n", "enveda_v1")
N_DECOY = 10
SEED = 0
IMIDIC = "[CX3]([OX2H1])=[NX2]"
FEATS = ("rdkit_canon_str", "taut_canon", "imidic", "stereo")


_TE = None


def repr_features(smi: str) -> dict | None:
    """SMILES 1 本の表記の特徴。解析できなければ None。"""
    global _TE
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if _TE is None:
        _TE = rdMolStandardize.TautomerEnumerator()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    cs = Chem.MolToSmiles(m)
    try:
        tc = Chem.MolToSmiles(_TE.Canonicalize(m)) == cs
    except Exception:
        tc = None
    return dict(rdkit_canon_str=cs == smi,
                taut_canon=tc,
                imidic=m.HasSubstructMatch(Chem.MolFromSmarts(IMIDIC)),
                stereo=any(ch in smi for ch in "@/\\"))


def rates(rows: list[dict]) -> dict:
    rows = [r for r in rows if r is not None]
    out = {}
    for f in FEATS:
        v = [r[f] for r in rows if r[f] is not None]
        out[f] = round(float(np.mean(v)), 4) if v else None
    out["n"] = len(rows)
    return out


def flag_diffs(truth_rates: dict, decoy_rates: dict, th: float = 0.2) -> list[str]:
    return [f for f in FEATS if truth_rates.get(f) is not None and decoy_rates.get(f) is not None
            and abs(truth_rates[f] - decoy_rates[f]) >= th]


def read_truth_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def check_set(h: str, pool, src, chem, rng) -> dict:
    import library as L
    from run import visible_window
    rows = read_truth_rows(ROOT / f"outputs/holdouts/{h}/truth.csv")
    zf = np.load(ROOT / f"outputs/holdout_eval/{h}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    rule = L.hide_rule(h)
    remove = rule.get("pool") == "remove"
    drop = L.drop_keys(h)
    hidden = drop if remove else set()
    pool_keys = set(pool.key)
    pc = json.loads((ROOT / f"outputs/local_eval/{h}/pubchem_fold0.json").read_text()) if h == "v3_c2" else {}
    c = dict(n=len(rows), n_target=0, col_eq_canon=0, raw_ne_canon=0, in_pool=0, in_window=0,
             drop_has_canon=0, pool_has_canon=0, truth_smiles_eq_pool_smiles=0, pubchem_top60_has_truth=0)
    t_feat, d_feat, t_src, d_src, pc_t, pc_d, listed = [], [], [], [], [], [], {}
    truth_csv_feat = []
    for r in rows:
        mid, smi = r["molecule_id"], r["smiles"]
        ck, rk = chem.score_key(smi), chem.raw_key(smi)
        truth_csv_feat.append(repr_features(smi))
        c["col_eq_canon"] += int(r.get("inchikey14") == ck)
        c["raw_ne_canon"] += int(rk != ck)
        c["in_pool"] += int(ck in pool_keys)
        if remove:
            c["drop_has_canon"] += int(ck in drop)
            c["pool_has_canon"] += int(ck in pool_keys)
        if mid not in targets:
            continue
        c["n_target"] += 1
        idx = visible_window(pool, targets[mid], hidden)
        keys = pool.key[idx]
        hit = idx[keys == ck]
        if len(hit):
            c["in_window"] += 1
            if remove:
                listed.setdefault("in_window", []).append(mid)
            ti = int(hit[0])
            c["truth_smiles_eq_pool_smiles"] += int(pool.smiles[ti] == smi)
            t_feat.append(repr_features(pool.smiles[ti]))
            t_src.append(int(src[ti]))
            dec = idx[keys != ck]
            if len(dec):
                pick = rng.choice(dec, size=min(N_DECOY, len(dec)), replace=False)
                for j in pick:
                    d_feat.append(repr_features(pool.smiles[j]))
                    d_src.append(int(src[j]))
        if pc:
            props = pc.get(mid) or []
            tk = [p for p in props if p[2] == ck]
            if tk:
                c["pubchem_top60_has_truth"] += 1
                pc_t.append(repr_features(tk[0][0]))
                pc_d += [repr_features(p[0]) for p in props if p[2] != ck]
    res = dict(holdout=h, hide_rule=rule, counts=c, truth_csv_smiles=rates(truth_csv_feat))
    if t_feat:
        tr, dr = rates(t_feat), rates(d_feat)
        res["pool_rows"] = dict(truth=tr, decoy=dr, diff_ge_0_2=flag_diffs(tr, dr),
                                truth_src={str(k): int(v) for k, v in zip(*np.unique(t_src, return_counts=True))},
                                decoy_src={str(k): int(v) for k, v in zip(*np.unique(d_src, return_counts=True))})
    if pc_t:
        tr, dr = rates(pc_t), rates(pc_d)
        res["pubchem_rows"] = dict(truth=tr, decoy=dr, diff_ge_0_2=flag_diffs(tr, dr))
    if listed:
        res["remove_set_truth_in_window"] = listed.get("in_window", [])
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/msg_wild_repr_check.json"))
    a = ap.parse_args()
    import pandas as pd
    from casmi import chem
    from run import load_pool
    t0 = time.time()
    pool = load_pool()
    meta = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["mass", "src"])
    src = meta["src"].values[np.argsort(meta["mass"].values, kind="stable")]   # Pool と同じ並び
    rng = np.random.default_rng(SEED)
    import rdkit
    out = dict(issue=154, definition=__doc__.split("定義（数える前に固定）:")[1].split("    python")[0].strip(),
               rdkit=rdkit.__version__, n_decoy=N_DECOY, seed=SEED, sets={})
    for h in HOLDOUTS:
        out["sets"][h] = check_set(h, pool, src, chem, rng)
        print(h, json.dumps(out["sets"][h]["counts"], ensure_ascii=False), flush=True)
    out["secs"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", a.out, out["secs"], "s")


if __name__ == "__main__":
    main()
