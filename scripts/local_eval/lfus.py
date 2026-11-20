#!/usr/bin/env python3
"""
Issue #150: lfus 段（ライブラリの類縁体から点を伝える、NAP の Fusion 型）の部品と、類縁検索の前処理（build）。

段 1（run.py の lfus 段、prior の後・frag の前）:
    N(c) = Σ_a s_a·T(c, a) / Σ_a s_a（a は類縁検索の結果、最大 100、指紋の作れない類縁体は分子・分母から外す）
    新しい点 = z(prior 後の点) + z(N)。z は分子の中の上位 60 で (x − mean)/(std + 1e−9)。std(N) = 0 なら z(N) = 0。
    並べ替えは np.argsort(-新しい点, kind='stable')。類縁体の無い分子（空、または Σ s_a = 0）はそのまま。
    T = a·b / (|a| + |b| − a·b)（fp_bits を選んだ 0/1 指紋）、分母 0 なら 0。

build（集合ごと。重いので memguard の下で 1 回だけ。結果は outputs/local_eval/<h>/lfus_analogs.json）:
    1. train.parquet を row group ごとに必要な列だけ読み、spec_meta.parquet と全行の一致を確かめる（合わなければ止める）
    2. 隠す印 = load_refs と同じ判定（行の inchikey14 ∈ drop_keys、enveda_libs なら ingest_lib ∈ ENVEDA の行だけ）
       ∪ 行の sid の train_structs.key・rawkey ∈ drop_keys（同じ規則、#150 の違い (iii)）∪ sid < 0 の行
    3. Library.representatives(exclude=印, key=<h>) → 分子ごとに analog_search（SearchCfg の既定値）
    4. 混入の検査（印とは別の経路。代表の全行、返った sid × 問い合わせのモードの代表、元から作り直した第 2 のキー集合）
       → 0 件でなければ止める
    5. 段 0（記録だけ）: 正解との Tanimoto（T1・Tmax10・Nw）と無作為な基準線（R1・Rmax10・Rmean、seed 0）

    python scripts/local_eval/lfus.py build --holdout v3_c1 [--limit 10] [--workdir outputs/fpnet_full_mps/run/work]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("LOCAL_EVAL_ROOT", CODE))   # run.py と同じ（データ・出力の木）
ENVEDA = {"enveda-180", "enveda-np-examples"}
LFUS_ALLOWED = ("score", "prior", "frag")   # #150 で測る組み合わせ（score,prior,lfus,frag）の範囲
QS = [0, 25, 50, 75, 100]                    # #101 と同じ分位点
N_RANDOM, SEED = 100, 0                      # 段 0 の基準線（#150 で固定）
TOP_MAX = 10                                 # 段 0 の Tmax10・Rmax10


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）
def _isin(a, keys) -> np.ndarray:
    """文字列の配列の所属（pandas のハッシュ。None・NaN は False）。"""
    import pandas as pd
    return pd.Series(np.asarray(a, dtype=object)).isin(set(keys)).to_numpy(bool, copy=True)


def tanimoto_matrix(C: np.ndarray, A: np.ndarray) -> np.ndarray:
    """C (n×B)・A (k×B) の 0/1 指紋の Tanimoto（n×k）。分母 0 は 0。"""
    C = (np.asarray(C) > 0).astype(np.float64)
    A = (np.asarray(A) > 0).astype(np.float64)
    inter = C @ A.T
    den = C.sum(1)[:, None] + A.sum(1)[None, :] - inter
    return np.divide(inter, den, out=np.zeros_like(inter), where=den > 0)


def neighbour_score(cand_fps: np.ndarray, analog_fps: np.ndarray, s: np.ndarray) -> np.ndarray | None:
    """N(c) = Σ_a s_a·T(c, a) / Σ_a s_a。類縁体が無いか Σ s = 0 なら None。"""
    s = np.asarray(s, np.float64)
    if len(s) == 0 or s.sum() <= 0:
        return None
    return tanimoto_matrix(cand_fps, analog_fps) @ s / s.sum()


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def lfus_mix(rows: list, N: np.ndarray | None) -> list:
    """rows = prior 後の上位（(点, pool 行)）、N = 各行の N(c)。新しい点 = z(点) + z(N)。
    N が None（類縁体なし）・行が 2 未満・std(N) = 0（z(N) = 0、#150「点を変えない」）ならそのまま返す。"""
    if N is None or len(rows) < 2:
        return rows
    N = np.asarray(N, np.float64)
    assert len(N) == len(rows)
    if N.std() == 0:
        return rows
    f = zscore([r[0] for r in rows]) + zscore(N)
    order = np.argsort(-f, kind="stable")
    return [(float(f[o]), rows[o][1]) for o in order]


def lfus_rows(rows: list, cand_smiles: list, analogs: list | None, analog_smiles, fpc) -> tuple[list, dict]:
    """run.py から呼ぶ: analogs = [(sid, s)]、analog_smiles[sid] = SMILES、fpc(smiles) = 指紋 or None。
    返り値: (並べ替えた rows, 記録 dict(n_analog, n_fp_fail, applied))。"""
    info = dict(n_analog=0, n_fp_fail=0, applied=False)
    if not analogs or len(rows) < 2:
        return rows, info
    afp, s = [], []
    for sid, sim in analogs:
        fp = fpc(analog_smiles[int(sid)])
        if fp is None:
            info["n_fp_fail"] += 1
            continue
        afp.append(fp); s.append(float(sim))
    info["n_analog"] = len(s)
    if not s:
        return rows, info
    cfp = [fpc(x) for x in cand_smiles]
    assert all(c is not None for c in cfp), "候補の指紋は score 段で作れているはず"
    N = neighbour_score(np.stack(cfp), np.stack(afp), np.asarray(s))
    if N is None or N.std() == 0:
        return rows, info
    info["applied"] = True
    return lfus_mix(rows, N), info


def check_stages_lfus(stages: list[str]) -> None:
    """lfus と併用できるのは LFUS_ALLOWED だけで、prior が要る（#150 の範囲）。"""
    if "lfus" not in stages:
        return
    bad = [s for s in stages if s != "lfus" and s not in LFUS_ALLOWED]
    assert not bad, f"lfus は {bad} と併用できない（#150 の範囲外。使えるのは {LFUS_ALLOWED}）"
    assert "prior" in stages, "lfus は prior の後に掛ける段なので prior が要る（#150）"


def hide_mask(tr_ik14: np.ndarray, tr_lib: np.ndarray, sid: np.ndarray, st_key: np.ndarray, st_rawkey: np.ndarray,
              drop: set, rule: dict) -> tuple[np.ndarray, dict]:
    """スペクトルの行ごとの隠す印。load_refs と同じ判定 ∪ sid の key/rawkey ∈ drop（同じ規則）∪ sid < 0。"""
    sid = np.asarray(sid, np.int64)
    base = _isin(tr_ik14, drop)
    ok = sid >= 0
    ext = np.zeros(len(sid), bool)
    ext[ok] = _isin(np.asarray(st_key, object)[sid[ok]], drop) | _isin(np.asarray(st_rawkey, object)[sid[ok]], drop)
    lib_ok = _isin(tr_lib, ENVEDA) if rule.get("rule") == "enveda_libs" else np.ones(len(sid), bool)
    m_load = base & lib_ok
    m = (base | ext) & lib_ok
    m |= ~ok
    return m, dict(n_hidden_loadrefs_rule=int(m_load.sum()), n_hidden_extra_key=int((m & ~m_load & ok).sum()),
                   n_sid_negative=int((~ok).sum()), n_hidden_total=int(m.sum()))


def leaked_rows(rows: np.ndarray, tr_ik14, tr_lib, sid, st_ik14, st_key, st_rawkey, keys: set, rule: dict) -> np.ndarray:
    """rows（スペクトルの行）のうち、hide_rule の下で keys により隠すべき行（印とは別に、行から直接引く）。"""
    rows = np.asarray(rows, np.int64)
    s = np.asarray(sid, np.int64)[rows]
    hit = _isin(np.asarray(tr_ik14, object)[rows], keys)
    ok = s >= 0
    so = s[ok]
    hit[ok] |= (_isin(np.asarray(st_ik14, object)[so], keys) | _isin(np.asarray(st_key, object)[so], keys)
                | _isin(np.asarray(st_rawkey, object)[so], keys))
    if rule.get("rule") == "enveda_libs":
        hit &= _isin(np.asarray(tr_lib, object)[rows], ENVEDA)
    return rows[hit]


def source_keys(truth: list[tuple[str, str]], to_key, st_ik14, st_key, st_rawkey) -> set:
    """第 2 のキー集合（drop_keys.txt を使わず元から）: truth の inchikey14 ∪ to_key(SMILES) ∪
    train_structs で key/inchikey14 がそれらに当たる行の inchikey14・key・rawkey。truth = [(ik14, smiles)]。"""
    base = set()
    for ik, smi in truth:
        if isinstance(ik, str) and ik:
            base.add(ik)
        k = to_key(smi) if isinstance(smi, str) and smi else None
        if k:
            base.add(k)
    sel = _isin(st_ik14, base) | _isin(st_key, base)
    extra = set(np.asarray(st_ik14, object)[sel]) | set(np.asarray(st_key, object)[sel]) | set(np.asarray(st_rawkey, object)[sel])
    return base | {k for k in extra if isinstance(k, str) and k}


def random_baseline(rep_sid: np.ndarray, rep_mass: np.ndarray, rep_mode: np.ndarray, target: float, modes: set,
                    win: float, rng, n: int = N_RANDOM) -> np.ndarray:
    """±win Da・modes の代表の一意な sid（昇順）から rng.choice(size=min(n, 数), replace=False)。"""
    lo = np.searchsorted(rep_mass, target - win, "left")
    hi = np.searchsorted(rep_mass, target + win, "right")
    s = rep_sid[lo:hi][np.isin(rep_mode[lo:hi], sorted(modes))]
    u = np.unique(s)
    if len(u) == 0:
        return u
    return rng.choice(u, size=min(n, len(u)), replace=False)


def quantiles(x) -> dict | None:
    x = np.asarray([v for v in x if v is not None and not np.isnan(v)], float)
    if x.size == 0:
        return None
    return {f"q{q}": round(float(np.percentile(x, q)), 4) for q in QS} | {"mean": round(float(x.mean()), 4), "n": int(x.size)}


def stage0_molecule(tfp, analogs: list, afp_of, rand_sids, exclude_same: set | None = None) -> dict:
    """段 0（記録だけ）: 正解の指紋 tfp と、類縁体 [(sid, s)]（sim の降順）・無作為な sid の Tanimoto。
    afp_of(sid) = 指紋 or None。exclude_same = 正解と同じ化合物の sid（v3_c1 の除いた版）。"""
    out = {}
    an = [(sid, s) for sid, s in analogs if afp_of(sid) is not None]
    if exclude_same is not None:
        an = [(sid, s) for sid, s in an if sid not in exclude_same]
    if an and tfp is not None:
        T = tanimoto_matrix(tfp[None, :], np.stack([afp_of(sid) for sid, _ in an]))[0]
        s = np.array([x for _, x in an], np.float64)
        out.update(T1=float(T[0]), Tmax10=float(T[:TOP_MAX].max()), Nw=float(T @ s / s.sum()) if s.sum() > 0 else None)
    else:
        out.update(T1=None, Tmax10=None, Nw=None)
    if exclude_same is None:
        rf = [afp_of(int(x)) for x in rand_sids]
        rf = [f for f in rf if f is not None]
        if rf and tfp is not None:
            R = tanimoto_matrix(tfp[None, :], np.stack(rf))[0]
            out.update(R1=float(R[0]), Rmax10=float(R[:TOP_MAX].max()), Rmean=float(R.mean()), n_random=len(rf))
        else:
            out.update(R1=None, Rmax10=None, Rmean=None, n_random=0)
    return out


def load_analogs(path: Path) -> dict:
    """build の json から {molecule_id: [(sid, s)]}。"""
    d = json.loads(Path(path).read_text())
    return {m: [(int(a), float(b)) for a, b in v] for m, v in d["analogs"].items()}


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def lib_ix_from_build(build_py: Path) -> dict:
    """data/casmi26-fpnet-code/casmi/build.py の LIBS（spec_meta.lib の符号）を ast で読む（import しない）。"""
    for node in ast.parse(build_py.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LIBS" for t in node.targets):
            return {l: i for i, l in enumerate(ast.literal_eval(node.value))}
    raise KeyError("LIBS が build.py に無い")


# ---------------------------------------------------------------- build（重い。memguard の下で）
def read_train_cols(path: Path, cols=("inchikey14", "ingest_lib", "precursor_mz", "adduct", "num_peaks")) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    parts = {c: [] for c in cols}
    for b in pf.iter_batches(batch_size=8192, columns=list(cols)):
        for c in cols:
            parts[c].append(b.column(c))
    out = {}
    for c in cols:
        a = pa.chunked_array(parts[c]).to_numpy(zero_copy_only=False)
        out[c] = a.astype(object) if a.dtype == object else a
    return out


def check_alignment(tr: dict, meta, st_ik14: np.ndarray, lib_ix: dict) -> dict:
    n = len(tr["inchikey14"])
    sid = meta["sid"].values.astype(np.int64)
    res = dict(n_train=n, n_meta=len(meta), n_structs=len(st_ik14), sid_max=int(sid.max()), sid_min=int(sid.min()))
    res["rows_equal"] = n == len(meta)
    if res["rows_equal"]:
        res["precursor_mz_equal"] = bool(np.array_equal(np.asarray(tr["precursor_mz"], np.float64),
                                                        meta["precursor_mz"].values.astype(np.float64), equal_nan=True))
        import pandas as pd
        a1 = pd.Series(np.asarray(tr["adduct"], object)).fillna("<NA>").astype(str).to_numpy()
        a2 = pd.Series(meta["adduct"].values.astype(object)).fillna("<NA>").astype(str).to_numpy()
        res["adduct_equal"] = bool((a1 == a2).all())
        res["num_peaks_equal"] = bool(np.array_equal(np.asarray(tr["num_peaks"]), meta["num_peaks"].values))
        lib_codes = np.array([lib_ix.get(x, -99) for x in tr["ingest_lib"]], np.int64)
        res["lib_equal"] = bool(np.array_equal(lib_codes, meta["lib"].values.astype(np.int64)))
        ok = sid >= 0
        res["sid_lt_n_structs"] = bool(sid.max() < len(st_ik14))
        res["sid_inchikey14_equal"] = bool(res["sid_lt_n_structs"] and (st_ik14[sid[ok]] == tr["inchikey14"][ok]).all())
        res["n_sid_negative"] = int((~ok).sum())
    keys = ("rows_equal", "precursor_mz_equal", "adduct_equal", "num_peaks_equal", "lib_equal", "sid_lt_n_structs",
            "sid_inchikey14_equal")
    res["all_ok"] = all(res.get(k, False) for k in keys)
    return res


def build(holdout: str, workdir: Path, limit: int | None, out: Path) -> dict:
    import pandas as pd
    sys.path.insert(0, os.environ.get("CASMI_CODE", str(ROOT / "data/ext/v4g/casmi26-v3-models/code")))
    sys.path.insert(0, str(CODE / "scripts/local_eval"))
    sys.path.insert(0, str(CODE / "scripts/validate"))
    import library as L
    import run as R
    import mrr25
    from casmi.library import Library, SearchCfg

    t0 = time.time()
    timing = {}
    rule = L.hide_rule(holdout)
    drop = L.drop_keys(holdout)
    dk_path = ROOT / f"data/holdouts/{holdout}/drop_keys.txt"
    lib_ix = lib_ix_from_build(ROOT / "data/casmi26-fpnet-code/casmi/build.py")
    st = pd.read_parquet(workdir / "train_structs.parquet", columns=["inchikey14", "key", "rawkey", "smiles", "sid"])
    assert (st.sid.values == np.arange(len(st))).all(), "train_structs の sid が行の番号でない"
    st_ik14, st_key, st_rawkey = (st[c].values.astype(object) for c in ("inchikey14", "key", "rawkey"))
    tr = read_train_cols(ROOT / "data/train.parquet")
    timing["read_train"] = round(time.time() - t0, 1)
    lib = Library(str(workdir), verbose=True)
    timing["load_library"] = round(time.time() - t0, 1)
    align = check_alignment(tr, lib.meta, st_ik14, lib_ix)
    print("[align]", align, flush=True)
    assert align["all_ok"], f"spec_meta と train.parquet の行が一致しない: {align}"
    mask, mstat = hide_mask(tr["inchikey14"], tr["ingest_lib"], lib.sid, st_key, st_rawkey, drop, rule)
    rep = lib.representatives(exclude=mask, key=holdout)
    timing["representatives"] = round(time.time() - t0, 1)

    truth_df = pd.read_csv(ROOT / f"outputs/holdouts/{holdout}/truth.csv")
    skeys = source_keys(list(zip(truth_df.inchikey14, truth_df.smiles)), mrr25.to_key, st_ik14, st_key, st_rawkey)
    lk = dict(tr_ik14=tr["inchikey14"], tr_lib=tr["ingest_lib"], sid=lib.sid, st_ik14=st_ik14, st_key=st_key, st_rawkey=st_rawkey)
    leak_rep_drop = leaked_rows(rep["idx"], keys=drop, rule=rule, **lk)
    leak_rep_src = leaked_rows(rep["idx"], keys=skeys, rule=rule, **lk)
    # spec_meta の lib 符号でも ingest_lib を引き直す（train.parquet の列と別の経路）
    inv = {v: k for k, v in lib_ix.items()}
    tr_lib_meta = np.array([inv.get(int(x), "?") for x in lib.lib], dtype=object)
    leak_rep_meta = leaked_rows(rep["idx"], keys=drop, rule=rule, **(lk | dict(tr_lib=tr_lib_meta)))

    zf = np.load(ROOT / f"outputs/holdout_eval/{holdout}_fold0/pc_logits.npz", allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    spectra = R.query_spectra(holdout)
    cfg = SearchCfg()
    rep_of = {(int(s), int(m)): int(i) for s, m, i in zip(rep["sid"], rep["mode"], rep["idx"])}
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    afp_of = lambda sid: fpc(st.smiles.values[int(sid)])  # noqa: E731
    rng = np.random.default_rng(SEED)
    analogs, s0, s0x, n_leak_hits, secs = {}, {}, {}, 0, []
    mids = [m for m in truth_df.molecule_id.astype(str) if m in targets]
    if limit:
        mids = mids[:limit]
    tkey = dict(zip(truth_df.molecule_id.astype(str), zip(truth_df.inchikey14, truth_df.smiles)))
    ik2sid = {k: i for i, k in enumerate(st_ik14)}
    for j, mid in enumerate(mids):
        t1 = time.time()
        qs = [(q["mz"], q["it"], q["mode"]) for q in spectra.get(mid, [])]
        res = lib.analog_search(qs, targets[mid], cfg, rep) if qs else []
        analogs[mid] = [(int(sid), float(s)) for sid, s, _ in res]
        modes = {q[2] for q in qs}
        hit_rows = [rep_of[(sid, m)] for sid, _ in analogs[mid] for m in modes if (sid, m) in rep_of]
        n_leak_hits += len(leaked_rows(np.array(hit_rows, np.int64), keys=drop, rule=rule, **lk)) if hit_rows else 0
        n_leak_hits += len(leaked_rows(np.array(hit_rows, np.int64), keys=skeys, rule=rule, **lk)) if hit_rows else 0
        ik, smi = tkey[mid]
        tfp = fpc(smi)
        rs = random_baseline(rep["sid"], rep["mass"], rep["mode"], targets[mid], modes, cfg.analog_win, rng) if modes else np.array([])
        s0[mid] = stage0_molecule(tfp, analogs[mid], afp_of, rs)
        if rule.get("rule") == "enveda_libs":
            same_keys = {ik, mrr25.to_key(smi)} | ({st_key[ik2sid[ik]]} if ik in ik2sid else set())
            same_keys = {k for k in same_keys if isinstance(k, str) and k}
            same = {int(i) for i in np.where(_isin(st_ik14, same_keys) | _isin(st_key, same_keys))[0]}
            s0x[mid] = stage0_molecule(tfp, analogs[mid], afp_of, rs, exclude_same=same)
        secs.append(time.time() - t1)
        if j % 20 == 0:
            print(f"[{holdout}] {j+1}/{len(mids)} {mid} n_analog={len(analogs[mid])} {secs[-1]:.1f}s", flush=True)
    timing["search_total"] = round(sum(secs), 1)
    timing["search_per_molecule_mean"] = round(float(np.mean(secs)), 2) if secs else None
    audit = dict(n_rep=int(len(rep["idx"])), n_source_keys=len(skeys), n_drop_keys=len(drop),
                 leak_rep_drop_keys=int(len(leak_rep_drop)), leak_rep_source_keys=int(len(leak_rep_src)),
                 leak_rep_meta_lib=int(len(leak_rep_meta)), leak_analog_hits=int(n_leak_hits))
    print("[audit]", audit, flush=True)
    assert audit["leak_rep_drop_keys"] == 0 and audit["leak_rep_source_keys"] == 0 and audit["leak_rep_meta_lib"] == 0 \
        and audit["leak_analog_hits"] == 0, f"隠すべき行が代表・類縁体に混ざっている: {audit}"
    if rule.get("rule") == "enveda_libs":   # 記録用: 正解の sid が他の装置の参照として代表に残る数（正当）
        tsids = {ik2sid[k] for k in truth_df.inchikey14 if k in ik2sid}
        audit["truth_sid_reps_kept_other_instruments"] = int(np.isin(rep["sid"], sorted(tsids)).sum())
    agg = lambda d, k: quantiles([v[k] for v in d.values()])  # noqa: E731
    stage0 = {k: agg(s0, k) for k in ("T1", "Tmax10", "Nw", "R1", "Rmax10", "Rmean")}
    stage0["n_no_analog"] = sum(1 for m in mids if not analogs[m])
    if s0x:
        stage0["excl_same_compound"] = {k: agg(s0x, k) for k in ("T1", "Tmax10", "Nw")}
    res = dict(issue=150, holdout=holdout, rule=rule, limit=limit, n_molecules=len(mids),
               drop_keys_sha256=sha256(dk_path), workdir=str(workdir),
               train_structs_sha256=sha256(workdir / "train_structs.parquet"), search_cfg=vars(cfg),
               align=align, mask=mstat, audit=audit, stage0=stage0, timing=timing,
               secs_total=round(time.time() - t0, 1), analogs=analogs, stage0_per_molecule=s0,
               stage0_excl_same_per_molecule=s0x)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, ensure_ascii=False))
    tmp.replace(out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--holdout", required=True)
    b.add_argument("--limit", type=int, default=None, help="時間の関門用: 先頭 N 分子だけ（結果は別名に書く）")
    b.add_argument("--workdir", default=str(ROOT / "outputs/fpnet_full_mps/run/work"))
    b.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / f"outputs/local_eval/{a.holdout}/lfus_analogs{'_limit%d' % a.limit if a.limit else ''}.json"
    r = build(a.holdout, Path(a.workdir), a.limit, out)
    print(json.dumps({k: r[k] for k in ("holdout", "n_molecules", "align", "mask", "audit", "stage0", "timing", "secs_total")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
