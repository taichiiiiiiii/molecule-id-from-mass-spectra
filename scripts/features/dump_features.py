"""
候補ごとの特徴表の書き出し（Issue #244）。`scripts/local_eval/run.py --dump-features <path>` から呼ばれる。
定義は #244 の本文（コードを書く前に固定）。正解のラベルの列は作らない（G1 未承認の間は値の比べをしない）。

行 = 分子ごとの L ∪ Q:
  L = 対照 score,prior,frag の重複除去した上位 60（run.py の lists[mid]）
  Q = pubchem.run_holdout のキャッシュ（f·z 順、最大 60、プールの見えるキーを除く）。キャッシュに無い分子があれば止める
列は COLUMNS（識別）+ FEATURES。入れない列は EXCLUDED（あれば止める）。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/features"))
sys.path.insert(0, str(ROOT / "scripts/local_eval"))

ID_COLS = ["set", "type", "molecule_id", "query_fold", "model_fold", "np_model_fold", "cand_key", "cand_smiles"]
CHANNELS = ["fz", "tani_pred", "lib_max_sim", "metfrag_lite", "np_score"]
BASE = ["fz", "tani_pred", "lib_max_sim", "has_ref", "is_tier", "prior_unverified_pop", "ctrl_rank", "q_rank",
        "metfrag_lite", "has_frag", "np_score", "has_np"]
EXCLUDED = ("src", "train_sid", "np_coconut", "p", "b8_p", "y", "label", "is_truth", "used_b8", "msg_overlap", "in_train")
# #287 H2: lib 系の列を装置の系統の規則で作り直した列に置き換える（build の h2_refs を渡したときだけ。既定は今のまま）
H2_DROP = ("lib_max_sim", "has_ref")
H2_COLS = ("lib_max_sim_xi", "has_ref_xi", "lib_max_sim_xi_sa")
CHANNELS_H2 = tuple("lib_max_sim_xi" if c == "lib_max_sim" else c for c in CHANNELS)
MISSING_RANK = 61   # TOPN + 1: その候補源の並びに居ない
SHA_NAME = "pubchem_fold0.logits_sha256"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def set_meta(holdout: str) -> tuple[str, int]:
    """(型, fold)。dev = b8_f{k}_{t}_t10（#216: fold k の集合は fpnet_fold{k}.pt）、holdout = #237 の holdout_list.csv の型と fold 0。"""
    m = re.fullmatch(r"b8_f([1-4])_([a-e])_t10", holdout)
    if m:
        return m.group(2), int(m.group(1))
    hl = pd.read_csv(ROOT / "docs/bench/holdout_list.csv")
    t = hl.loc[hl.set == holdout, "type"].unique()
    assert len(t) == 1, f"{holdout} の型が holdout_list.csv に 1 つでない: {t}"
    return str(t[0]), 0


def tanimoto_pred(fp: np.ndarray, z: np.ndarray) -> float:
    """連続 Tanimoto: p = σ(z)、Σpf / (Σp + Σf − Σpf)。分母 0 は 0。"""
    from scipy.special import expit
    p = expit(np.asarray(z, np.float64))
    f = np.asarray(fp, np.float64)
    inter = float(p @ f)
    den = float(p.sum() + f.sum() - inter)
    return inter / den if den > 0 else 0.0


def q_refs(holdout: str, qrows: pd.DataFrame, rule: dict, drop: set) -> dict:
    """Q の候補の参照（library.load_refs と同じ隠す規則を qrows（train の行）に当てる）。"""
    import library as L
    if qrows is None or len(qrows) == 0:
        return {}
    hid = qrows.inchikey14.isin(drop)
    if rule["rule"] == "enveda_libs":
        hid &= qrows.ingest_lib.isin(L.ENVEDA)
    out: dict = {}
    for k, mz, it in zip(qrows.inchikey14[~hid], qrows.ms2_mzs[~hid], qrows.ms2_normalized_intensities[~hid]):
        out.setdefault(k, []).append(L._prep(mz, it))
    return out


class NPScorer:
    """#241 の fold-k モデルで天然物らしさ（SMILES ごとにキャッシュ）。"""

    def __init__(self, fold: int):
        import lightgbm as lgb
        import np_likeness as NP
        self.NP, self.fold = NP, fold
        self.bst = lgb.Booster(model_file=str(ROOT / f"outputs/local_eval/np_likeness/model_fold{fold}.txt"))
        self.c: dict = {}

    def __call__(self, smiles: list[str]) -> np.ndarray:
        todo = [s for s in dict.fromkeys(smiles) if s not in self.c]
        if todo:
            X, ok = self.NP.featurize(todo)
            sc = np.full(len(todo), np.nan)
            if ok.any():
                sc[ok] = self.bst.predict(X[np.where(ok)[0]])
            self.c.update(zip(todo, sc.tolist()))
        return np.array([self.c[s] for s in smiles], np.float64)


def build(holdout: str, truth: dict, targets: dict, zf, pool, fpc, hidden: set, rule: dict, spectra: dict, frag, fcache,
          lists: dict, preds: dict, logits: Path, refs_cache: Path, qrows: pd.DataFrame | None = None,
          np_scorer: NPScorer | None = None, h2_refs: dict | None = None, h2_inst: dict | None = None):
    """1 集合の表（DataFrame）と記録（dict）。
    h2_refs（#287）: 隠す規則を当てた後の L ∪ Q の参照 key -> [(prep, 系統, 付加イオン)]（library.refs_with_meta）。渡すと
    lib_max_sim・has_ref（と相対の列）を外し、H2_COLS と lib_max_sim_xi の相対の列を入れる。h2_inst は分子 -> 問い合わせ
    （spectra[mid] と同じ順）の instrument_type。作り直しの点検として、系統の規則なしの値が今の lib_max_sim・has_ref と
    一致することを確かめて止める。"""
    import library as L
    import mass_feasibility as MF
    import relativize as RZ
    from selpred import submission_bytes
    t0 = time.time()
    typ, fold = set_meta(holdout)
    d = ROOT / f"outputs/local_eval/{holdout}"
    # 対照の submission とのバイト一致（既存の CSV があるときだけ）
    ctrl = d / "submission_score+prior+frag.csv"
    got = submission_bytes(list(truth), preds)
    ctrl_identical = (ctrl.read_bytes() == got) if ctrl.is_file() else None
    # Q のキャッシュ（作らない。無い分子があれば止める）
    cache, mark = d / "pubchem_fold0.json", d / SHA_NAME
    lsha = sha256_file(Path(logits))
    assert mark.is_file() and cache.is_file(), f"{holdout}: PubChem のキャッシュか sha の印が無い"
    assert mark.read_text().strip() == lsha, f"{holdout}: キャッシュの sha256 の印がロジットと違う"
    pcres = json.loads(cache.read_text())
    miss = [m for m in lists if m not in pcres]
    assert not miss, f"{holdout}: PubChem のキャッシュに無い分子 {len(miss)}"
    # 参照: L は load_refs（集合の規則）、Q は qrows に同じ規則
    lkeys = {pool.key[i] for rows in lists.values() for _, i in rows}
    refs = L.load_refs(holdout, lkeys, refs_cache)
    refs_q = q_refs(holdout, qrows, rule, L.drop_keys(holdout))
    np_scorer = np_scorer or NPScorer(fold)
    assert np_scorer.fold == fold
    parts = []
    for mid, rows in lists.items():
        z = np.asarray(zf[mid], np.float32)
        q = pcres[mid] or []
        lk = [pool.key[i] for _, i in rows]
        qk = [k for _s, _fz, k in q]
        assert not (set(lk) & set(qk)), f"{holdout}/{mid}: L と Q の鍵が重なる"
        smis = [pool.smiles[i] for _, i in rows] + [s for s, _fz, _k in q]
        keys = lk + qk
        n_l, n = len(rows), len(rows) + len(q)
        fz, tani = np.zeros(n), np.zeros(n)
        for j, s in enumerate(smis):
            fp = fpc(s)
            if fp is None:
                raise AssertionError(f"{holdout}/{mid}: 指紋が作れない候補 {s}")
            fz[j] = float(fp @ z)
            tani[j] = tanimoto_pred(fp, z)
        for j, (_s, qfz, _k) in enumerate(q):   # キャッシュの f·z と作り直しの一致
            assert abs(fz[n_l + j] - qfz) <= 1e-4 * max(1.0, abs(qfz)), f"{holdout}/{mid}: Q の f·z が作り直しと違う"
        sp = spectra.get(mid, [])
        lib = L.lib_scores(sp, lk, refs) if sp else {}
        if sp and refs_q:
            lib.update(L.lib_scores(sp, qk, refs_q))
        lms = np.array([lib.get(k, 0.0) for k in keys])
        has_ref = np.array([bool(refs.get(k)) or bool(refs_q.get(k)) for k in keys], np.int8)
        h2 = None
        if h2_refs is not None:
            ins = (h2_inst or {}).get(mid, [])
            assert len(ins) == len(sp), f"{holdout}/{mid}: h2_inst の数が問い合わせと違う"
            qm = [dict(q, inst=v) for q, v in zip(sp, ins)]
            xs = L.lib_scores_xinst(qm, keys, h2_refs)
            arr = np.array([xs[k] for k in keys], np.float64).reshape(len(keys), 5)
            assert np.allclose(arr[:, 0], lms, rtol=0, atol=1e-12), f"{holdout}/{mid}: 作り直しの lib_max_sim が違う"
            assert (arr[:, 1].astype(np.int8) == has_ref).all(), f"{holdout}/{mid}: 作り直しの has_ref が違う"
            h2 = dict(lib_max_sim_xi=arr[:, 2], has_ref_xi=arr[:, 3].astype(np.int8), lib_max_sim_xi_sa=arr[:, 4])
        fr = frag.score_molecule(sp, smis, cache=fcache) if sp else [None] * n
        mfl = np.array([0.0 if v is None else float(v) for v in fr])
        has_frag = np.array([v is not None for v in fr], np.int8)
        mf = MF.group_features(smis, sp)
        nps = np_scorer(smis)
        has_np = np.isfinite(nps).astype(np.int8)
        nps = np.where(has_np == 1, nps, 0.0)
        df = pd.DataFrame(dict(
            set=holdout, type=typ, molecule_id=mid, query_fold=fold, model_fold=fold, np_model_fold=np_scorer.fold,
            cand_key=keys, cand_smiles=smis,
            fz=fz, tani_pred=tani, lib_max_sim=lms, has_ref=has_ref,
            is_tier=np.r_[np.zeros(n_l, np.int8), np.ones(n - n_l, np.int8)],
            prior_unverified_pop=np.r_[np.array([pool.pop[i] for _, i in rows], np.float64), np.zeros(n - n_l)],
            ctrl_rank=np.r_[np.arange(1, n_l + 1), np.full(n - n_l, MISSING_RANK)].astype(np.int16),
            q_rank=np.r_[np.full(n_l, MISSING_RANK), np.arange(1, n - n_l + 1)].astype(np.int16),
            metfrag_lite=mfl, has_frag=has_frag, np_score=nps, has_np=has_np))
        if h2 is not None:
            df = df.drop(columns=list(H2_DROP))
            for c in H2_COLS:
                df.insert(df.columns.get_loc("is_tier"), c, h2[c])
        assert all(c.startswith("mf_") for c in MF.feature_names())
        df = pd.concat([df, pd.DataFrame(mf, columns=MF.feature_names())], axis=1)
        parts.append(df)
    tab = pd.concat(parts, ignore_index=True)
    tab = RZ.add_relative(tab, ["set", "molecule_id"], CHANNELS if h2_refs is None else CHANNELS_H2)
    bad = [c for c in tab.columns if c in EXCLUDED]
    assert not bad, f"入れない列がある: {bad}"
    num = tab.select_dtypes(include=[np.number])
    rec = dict(holdout=holdout, type=typ, fold=fold, n_molecules=int(tab.molecule_id.nunique()), n_rows=int(len(tab)),
               n_L=int((tab.is_tier == 0).sum()), n_Q=int((tab.is_tier == 1).sum()),
               control_csv_identical=ctrl_identical, logits_sha256=lsha,
               n_nan=int(np.isnan(num.to_numpy()).sum()), n_inf=int(np.isinf(num.to_numpy()).sum()),
               secs=round(time.time() - t0, 1))
    return tab, rec
