#!/usr/bin/env python3
"""
Issue #232（2026-11-25 夜 計画 項目 4、記録だけ）: v3 の 4 集合の正解を「fold-0 の FPNet の学習側の構造への
最大 Tanimoto」で 3 区分に分け、対照 score,prior,frag の MRR・n・95% CI を出す。採否・定数・集合の選択には使わない。

定義（#232 の本文で値を見る前に固定）:
- 集合: v3_c1u_t10・v3_c2_t10・enveda_v1u_t10・v3_c2nu_t10。正解 = outputs/local_eval/<h>/truth.csv の smiles
- 学習側: data/ext/v4g/casmi26-v2-pool/train_structs.parquet の行のうち sid が data/ext/foldsafe/split.parquet に
  無い行（make_split.py の fold == 'train'）。SMILES が読めない行は除いて数を記録する
- 近さ: RDKit Morgan 半径 2・2048 bit のビット指紋の Tanimoto の、学習側全体での最大値
- 区分: < 0.5 / [0.5, 0.8) / ≥ 0.8
- rr: frag_v3_check.per_molecule_rr（mrr25.to_key で照合、予測の無い分子は 0）、対照 = submission_score+prior+frag.csv
- CI: 区分の中で分子を復元抽出する percentile bootstrap 2000 回、各セルで numpy.random.default_rng(0)
- 関門: 区分の n の和 = 集合の分子数。リークの検査に Tanimoto == 1.0 の数と inchikey14 / key の一致数も出す

    .venv/bin/python scripts/research/nn_similarity_strata.py \\
        --out outputs/local_eval/nn_similarity_strata/nn_similarity_strata.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

ISSUE = 232
ROOT = Path(__file__).resolve().parents[2]
HOLDOUTS = ("v3_c1u_t10", "v3_c2_t10", "enveda_v1u_t10", "v3_c2nu_t10")
CTRL_TAG = "score+prior+frag"
TSTRUCT = ROOT / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
SPLIT = ROOT / "data/ext/foldsafe/split.parquet"
RADIUS, NBITS = 2, 2048
EDGES = (0.5, 0.8)
STRATA = ("<0.5", "0.5-0.8", ">=0.8")
N_BOOT, SEED = 2000, 0


# ---------------------------------------------------------------- 小さな道具（単体テストの対象）

def stratum(sim: float) -> str:
    """最大 Tanimoto → 区分の名前（< 0.5 / [0.5, 0.8) / ≥ 0.8）。"""
    if not np.isfinite(sim):
        raise ValueError(f"近さが有限でない: {sim}")
    return STRATA[0] if sim < EDGES[0] else (STRATA[1] if sim < EDGES[1] else STRATA[2])


def boot_ci(x: Sequence[float], n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float | None, float | None]:
    """平均の percentile bootstrap 95% CI。n = 0 なら (None, None)。"""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    m = a[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(sim: Mapping[str, float], rr: Mapping[str, float]) -> dict:
    """分子ごとの近さと rr から区分ごとの n・MRR・CI。sim と rr の分子は同じでなければならない。"""
    if set(sim) != set(rr):
        raise ValueError(f"分子の集合が違う: sim {len(sim)} / rr {len(rr)}")
    groups: dict[str, list[float]] = {s: [] for s in STRATA}
    for m in sorted(rr):
        groups[stratum(sim[m])].append(rr[m])
    out = {}
    for s in STRATA:
        v = groups[s]
        lo, hi = boot_ci(v)
        out[s] = dict(n=len(v), mrr=(round(float(np.mean(v)), 4) if v else None),
                      ci95=([round(lo, 4), round(hi, 4)] if v else None),
                      n_top1=int(sum(x == 1.0 for x in v)), n_rr0=int(sum(x == 0.0 for x in v)))
    allv = [rr[m] for m in sorted(rr)]
    lo, hi = boot_ci(allv)
    out["all"] = dict(n=len(allv), mrr=round(float(np.mean(allv)), 4), ci95=[round(lo, 4), round(hi, 4)])
    if sum(out[s]["n"] for s in STRATA) != len(rr):
        raise AssertionError("区分の n の和が分子数と合わない")
    return out


def morgan_fps(smiles: Sequence[str]):
    """SMILES → (ビット指紋のリスト, 読めた位置の配列)。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=RADIUS, fpSize=NBITS)
    fps, ok = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) and s else None
        if mol is None:
            continue
        fps.append(gen.GetFingerprint(mol))
        ok.append(i)
    return fps, np.asarray(ok, dtype=np.int64)


def max_tanimoto(query_fps, ref_fps) -> tuple[np.ndarray, np.ndarray]:
    """各問い合わせの参照全体での最大 Tanimoto と、その参照の位置。"""
    from rdkit import DataStructs
    best = np.empty(len(query_fps), dtype=np.float64)
    arg = np.empty(len(query_fps), dtype=np.int64)
    for i, q in enumerate(query_fps):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(q, ref_fps))
        arg[i] = int(np.argmax(s))
        best[i] = float(s[arg[i]])
    return best, arg


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs/local_eval/nn_similarity_strata/nn_similarity_strata.json"))
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    sys.path.insert(0, str(ROOT / "scripts/cv"))
    import pandas as pd
    import rdkit
    import mrr25
    from frag_v3_check import per_molecule_rr

    t0 = time.time()
    ts = pd.read_parquet(TSTRUCT, columns=["sid", "inchikey14", "key", "smiles"])
    sp = pd.read_parquet(SPLIT, columns=["sid", "fold"])
    assert set(sp.fold) <= {"hold", "np"}, set(sp.fold)
    tr = ts[~ts.sid.isin(set(sp.sid))].reset_index(drop=True)
    ref_fps, ok = morgan_fps(tr.smiles.tolist())
    tr_ok = tr.iloc[ok].reset_index(drop=True)
    t_fp = time.time() - t0
    train_ik, train_key = set(tr.inchikey14), set(tr.key)

    out = dict(issue=ISSUE, rdkit=rdkit.__version__, fingerprint=f"Morgan r{RADIUS} {NBITS} bit (bit vector)",
               strata=list(STRATA), n_boot=N_BOOT, seed=SEED,
               train_side=dict(n_rows=int(len(tr)), n_fp_ok=int(len(ok)), n_unparsed=int(len(tr) - len(ok)),
                               n_split_rows=int(len(sp)), n_train_structs=int(len(ts))),
               files={str(p.relative_to(ROOT)): sha256(p) for p in (TSTRUCT, SPLIT)}, sets={})
    for h in HOLDOUTS:
        truth_p = ROOT / f"outputs/local_eval/{h}/truth.csv"
        ctl_p = ROOT / f"outputs/local_eval/{h}/submission_{CTRL_TAG}.csv"
        td = pd.read_csv(truth_p)
        q_fps, q_ok = morgan_fps(td.smiles.tolist())
        if len(q_ok) != len(td):
            raise AssertionError(f"{h}: 正解の SMILES が読めない {len(td) - len(q_ok)} 件")
        best, arg = max_tanimoto(q_fps, ref_fps)
        assert np.isfinite(best).all()
        sim = dict(zip(td.molecule_id, best.tolist()))
        rr = per_molecule_rr(mrr25.read_truth(truth_p), mrr25.read_submission(ctl_p))
        summ = summarize(sim, rr)
        out["sets"][h] = dict(
            n_molecules=int(len(td)), summary=summ,
            leak_check=dict(n_tanimoto_1=int((best >= 1.0).sum()),
                            n_inchikey14_in_train=int(td.inchikey14.isin(train_ik).sum()),
                            n_inchikey14_in_train_key=int(td.inchikey14.isin(train_key).sum())),
            sim_quantiles={q: round(float(np.quantile(best, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            files={str(p.relative_to(ROOT)): sha256(p) for p in (truth_p, ctl_p)},
            per_molecule=[dict(molecule_id=m, max_tanimoto=round(float(s), 4), rr=rr[m],
                               nn_inchikey14=str(tr_ok.inchikey14[j]))
                          for m, s, j in zip(td.molecule_id, best, arg)],
        )
        print(h, json.dumps(summ, ensure_ascii=False), flush=True)
    out["secs"] = dict(fingerprints=round(t_fp, 1), total=round(time.time() - t0, 1))
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1)  # macOS: bytes
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("train_side", "secs", "peak_rss_mb")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
