#!/usr/bin/env python3
"""
Issue #171（事前登録、補足 1・2 とレビュー記録の「実装の決めごと」を含む）: rep 段
＝ 家族（Murcko 骨格）に特有の断片・中性損失の偏りを train の参照の代表スペクトルから数え、候補の点にする。
文書: docs/research/diagnostic-reporters.md §6.2〜6.5。

定義（#171 §1。定数はすべて事前登録で固定）:
  家族 F(s)   : reporter_reach.scaffold（MurckoScaffold の正準 SMILES、環なしは ""、読めなければ None）。
                None の構造は表に入れない（数を記録）。候補の骨格が None なら点なし
  参照         : 隠す印（lfus と同じ hide_mask）を当てた後の Library.representatives(exclude=印, key=<h>)
                （(構造 sid, モード) ごとに 1 本）。イオン化モードごとに別の表、holdout ごとに数え直す
  特徴         : 代表スペクトルの全ピークの bin = rint(mz / 0.01)（断片）と、rint((prec − mz) / 0.01)（損失、bin ≤ 0 は捨てる）。
                (種類, bin) の組。mz・prec は float64 に直してから丸める
  数え方       : 構造の単位。a(f,F)・n_F・c_f・N（モードごと）
  偏り         : r(f,F) = log((a + ½)/(n_F + 1)) − log((c_f + ½)/(N + 1))
  leave-own-structure-out:
                K(c) = {pool.key[i], chem.raw_key(smiles), mrr25.to_key(smiles)}（None は除く）。
                E = K ∪ dseen.alias_index で 1 段の別名。train_structs で inchikey14・key・rawkey のどれかが E に入る sid を
                「c 自身」とし、その寄与をモードごとに c_f・N から全部、a・n_F からは F(c) に属する分だけ引く
  候補の点     : rep(c|q) = Q(q) の (m, f) のうち、引いた後で n_F(c),m ≥ 1 かつ c_f,m ≥ 1 のものの r_m の平均。無ければ点なし
  合わせ方     : prior の後・frag の前、上位 TOPN = 60 の行で 新しい点 = z(prior 後の点) + z(rep)。
                z = (x − mean)/(std + 1e−9)（ddof 0）。z(prior) は全行、z(rep) は点のある行だけで mean・std。点なしは z(rep) = 0。
                点のある行が 2 未満か std(rep) = 0 の分子は並びを変えない。np.argsort(-新しい点, kind='stable')

段の組み合わせ: --stages score,prior,rep,frag だけ（check_stages_rep）。

隠す印・混入の検査・行の一致の関数（_isin・hide_mask・leaked_rows・source_keys・lib_ix_from_build・read_train_cols・
check_alignment）は lfus（PR #151 の head 5f9c6fc、scripts/local_eval/lfus.py、未マージ）から式を変えずに写した。

補足 2(b) の「同じ骨格」: 正解かおとりのどちらかの骨格が None なら同じ骨格としない（点なしの規則に合わせる）。

使い方:
  run.py の rep 段（v3_c1・v3_c2・enveda_v1・v3_c2n）: submission と窓全体の点 outputs/local_eval/<h>/rep_scores.npz を
    同じ 1 本のプロセスで書く
  e180_c2k（#171 §5 の Δ_dom 用だけ）: 窓全体の点の npz だけ
    .venv/bin/python scripts/local_eval/rep.py window --holdout e180_c2k
本番は scripts/local_eval/run_rep_once.sh から 1 回だけ（#171 §4）。
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENVEDA = {"enveda-180", "enveda-np-examples"}
REP_STAGES = ("score", "prior", "rep", "frag")   # #171 §2: rep を使うときはこの組み合わせだけ
BIN = 0.01             # SearchCfg.tol の既定値（#171 §1）
HALF, ONE = 0.5, 1.0   # Jeffreys 型の補正（決め打ち）
KIND_FRAG, KIND_LOSS = 0, 1
CODE_SHIFT = 32        # code = 種類 << 32 | bin
FAM_SHIFT = 34         # a の表のキー = 家族 << 34 | code
SPEC_SHIFT = 34        # 表を作るときの (行, code) の組
WORKDIR = ROOT / "outputs/fpnet_full_mps/run/work"
E180K = "e180_c2k"
E180_DROP_SHA256 = "4ddce6584b1fe3f40d11d53de8f30360332cabc71d50f84e1b36c5d707aa6022"   # #171 補足 2-1


# ---------------------------------------------------------------- 段の組み合わせ
def check_stages_rep(stages: list[str]) -> None:
    """rep を使うのは score,prior,rep,frag だけ（他の段との併用・prior 無しを止める。#171 §2）。"""
    if "rep" not in stages:
        return
    assert list(stages) == list(REP_STAGES), \
        f"rep は --stages {','.join(REP_STAGES)} だけで使える（#171 の範囲外: {stages}）"


# ---------------------------------------------------------------- 特徴（純粋な関数）
def to_bins(x) -> np.ndarray:
    return np.rint(np.asarray(x, np.float64) / BIN).astype(np.int64)


def spectrum_codes(mz, prec: float) -> np.ndarray:
    """1 本のスペクトルの特徴（断片と損失の code、昇順・重複なし）。"""
    mz = np.asarray(mz, np.float64)
    f = to_bins(mz)
    loss = to_bins(np.float64(prec) - mz)
    loss = loss[loss > 0]
    codes = np.concatenate([f + (KIND_FRAG << CODE_SHIFT), loss + (KIND_LOSS << CODE_SHIFT)])
    assert codes.size == 0 or (codes.min() >= 0 and f.max(initial=0) < (1 << CODE_SHIFT)), "bin が code の範囲外"
    return np.unique(codes)


def query_codes(spectra: list[dict]) -> dict[int, np.ndarray]:
    """問い合わせのスペクトル（run.query_spectra の形）→ {モード: 特徴の和集合}（重複は 1 つ）。"""
    by = {}
    for q in spectra:
        by.setdefault(int(q["mode"]), []).append(spectrum_codes(q["mz"], q["prec"]))
    return {m: np.unique(np.concatenate(v)) for m, v in by.items()}


def r_value(a, nF, c, N) -> np.ndarray:
    """r(f,F) = log((a + ½)/(n_F + 1)) − log((c_f + ½)/(N + 1))（自然対数）。"""
    a, c = np.asarray(a, np.float64), np.asarray(c, np.float64)
    return np.log((a + HALF) / (nF + ONE)) - np.log((c + HALF) / (N + ONE))


def _lookup(keys: np.ndarray, cnt: np.ndarray, q: np.ndarray) -> np.ndarray:
    if keys.size == 0:
        return np.zeros(len(q), np.int64)
    pos = np.searchsorted(keys, q)
    pos = np.minimum(pos, keys.size - 1)
    return np.where(keys[pos] == q, cnt[pos], 0).astype(np.int64)


# ---------------------------------------------------------------- 表（モードごと）
class ModeTable:
    """1 つのモードの表。行 j = 表に入った構造（骨格が None でない代表を持つ sid）。"""

    def __init__(self, sids: np.ndarray, fam_of: np.ndarray, ptr: np.ndarray, codes: np.ndarray, n_fam: int):
        self.sids, self.fam_of, self.ptr, self.codes = sids, fam_of, ptr, codes
        self.sid2row = {int(s): j for j, s in enumerate(sids)}
        assert len(self.sid2row) == len(sids), "同じ sid が 1 つのモードに 2 回ある"
        self.N = int(len(sids))
        row_of = np.repeat(np.arange(len(sids)), np.diff(ptr))
        self.c_keys, c_cnt = np.unique(codes, return_counts=True)
        self.c_cnt = c_cnt.astype(np.int64)
        ak = (fam_of[row_of].astype(np.int64) << FAM_SHIFT) | codes
        self.a_keys, a_cnt = np.unique(ak, return_counts=True)
        self.a_cnt = a_cnt.astype(np.int64)
        self.nF = np.bincount(fam_of, minlength=n_fam).astype(np.int64)

    def row_codes(self, j: int) -> np.ndarray:
        return self.codes[self.ptr[j]:self.ptr[j + 1]]

    def c_of(self, q: np.ndarray) -> np.ndarray:
        return _lookup(self.c_keys, self.c_cnt, q)

    def a_of(self, fam: int, q: np.ndarray) -> np.ndarray:
        return _lookup(self.a_keys, self.a_cnt, (np.int64(fam) << FAM_SHIFT) | q)


def build_tables(rep_idx, rep_sid, rep_mode, off, mz, prec, sid_fam: np.ndarray, n_fam: int,
                 chunk: int = 20000) -> dict[int, ModeTable]:
    """代表スペクトル（行 rep_idx、CSR の off/mz、prec）と sid → 家族 id（−1 = 骨格 None）からモードごとの表。"""
    rep_idx = np.asarray(rep_idx, np.int64)
    rep_sid = np.asarray(rep_sid, np.int64)
    rep_mode = np.asarray(rep_mode)
    tables = {}
    for m in sorted({int(x) for x in np.unique(rep_mode)}):
        sel = (rep_mode == m) & (sid_fam[rep_sid] >= 0)
        idx, sids = rep_idx[sel], rep_sid[sel]
        o = np.argsort(sids, kind="stable")
        idx, sids = idx[o], sids[o]
        off_a = np.asarray(off)
        parts = []
        for c0 in range(0, len(idx), chunk):   # メモリを抑えるため代表を chunk 本ずつ（結果は一度に作るのと同じ）
            ci = idx[c0:c0 + chunk]
            st, ln = off_a[ci].astype(np.int64), (off_a[ci + 1] - off_a[ci]).astype(np.int64)
            tot = int(ln.sum())
            cum = np.concatenate([[0], np.cumsum(ln)[:-1]]) if len(ln) else np.zeros(0, np.int64)
            pos = np.repeat(st - cum, ln) + np.arange(tot)
            row = np.repeat(np.arange(c0, c0 + len(ci), dtype=np.int64), ln)
            pmz = np.asarray(mz[pos], np.float64)
            pprec = np.asarray(prec, np.float64)[ci][row - c0]
            fr = to_bins(pmz)
            ls = to_bins(pprec - pmz)
            keep = ls > 0
            code = np.concatenate([fr + (KIND_FRAG << CODE_SHIFT), ls[keep] + (KIND_LOSS << CODE_SHIFT)])
            rr = np.concatenate([row, row[keep]])
            assert code.size == 0 or (code.min() >= 0 and code.max() < (1 << SPEC_SHIFT)), "code が範囲外"
            parts.append(np.unique((rr << SPEC_SHIFT) | code))   # chunk の行は前の chunk より後なので連結しても昇順
            del pos, row, pmz, pprec, fr, ls, keep, code, rr
        pair = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        del parts
        prow = pair >> SPEC_SHIFT
        pcode = pair & ((1 << SPEC_SHIFT) - 1)
        del pair
        ptr = np.searchsorted(prow, np.arange(len(idx) + 1))
        del prow
        tables[m] = ModeTable(sids, sid_fam[sids].astype(np.int64), ptr, pcode, n_fam)
    return tables


# ---------------------------------------------------------------- 候補の点（純粋な関数）
def rep_score(tables: dict[int, ModeTable], Q: dict[int, np.ndarray], fam: int | None, own: set,
              a_cache: dict | None = None, c_cache: dict | None = None) -> float:
    """rep(c|q)。fam = 候補の家族 id（None = 骨格 None か家族が表に無い）、own = 「c 自身」の sid。点なしは NaN。"""
    if fam is None:
        return math.nan
    vals = []
    for m, q in Q.items():
        t = tables.get(m)
        if t is None or len(q) == 0 or fam >= len(t.nF):
            continue
        if a_cache is not None and (m, fam) in a_cache:
            a = a_cache[(m, fam)]
        else:
            a = t.a_of(fam, q)
            if a_cache is not None:
                a_cache[(m, fam)] = a
        if c_cache is not None and m in c_cache:
            c = c_cache[m]
        else:
            c = t.c_of(q)
            if c_cache is not None:
                c_cache[m] = c
        nF, N = int(t.nF[fam]), t.N
        for s in own:
            j = t.sid2row.get(int(s))
            if j is None:
                continue
            h = np.isin(q, t.row_codes(j), assume_unique=True).astype(np.int64)
            c = c - h
            N -= 1
            if int(t.fam_of[j]) == fam:
                a = a - h
                nF -= 1
        if nF < 1:
            continue
        ok = c >= 1
        if ok.any():
            vals.append(r_value(a[ok], nF, c[ok], N))
    if not vals:
        return math.nan
    v = np.concatenate(vals)
    return float(v.mean()) if v.size else math.nan


def zscore(x) -> np.ndarray:
    x = np.asarray(x, np.float64)
    return (x - x.mean()) / (x.std() + 1e-9)


def rep_mix(rows: list, rep: np.ndarray) -> tuple[list, bool]:
    """rows = prior 後の上位（(点, pool 行)）、rep = 各行の rep（NaN = 点なし）。(並べ替えた rows, 変えたか)。"""
    rep = np.asarray(rep, np.float64)
    assert len(rep) == len(rows)
    ok = np.isfinite(rep)
    if len(rows) < 2 or ok.sum() < 2 or rep[ok].std() == 0:
        return rows, False
    zr = np.zeros(len(rows))
    zr[ok] = (rep[ok] - rep[ok].mean()) / (rep[ok].std() + 1e-9)
    f = zscore([r[0] for r in rows]) + zr
    order = np.argsort(-f, kind="stable")
    return [(float(f[o]), rows[o][1]) for o in order], True


class OwnIndex:
    """leave-own-structure-out の「c 自身」の sid（#171 補足 1）。"""

    def __init__(self, st_ik14, st_key, st_rawkey):
        sys.path.insert(0, str(ROOT / "scripts/local_eval"))
        from dseen import alias_index
        self.aidx = alias_index(zip(st_ik14, st_rawkey, st_key))
        self.k2s: dict[str, set] = {}
        for sid, ks in enumerate(zip(st_ik14, st_key, st_rawkey)):
            for k in ks:
                if isinstance(k, str) and k:
                    self.k2s.setdefault(k, set()).add(sid)

    def sids(self, K) -> set:
        K = {k for k in K if isinstance(k, str) and k}
        E = set(K)
        for k in K:
            E |= self.aidx.get(k, set())
        out = set()
        for k in E:
            out |= self.k2s.get(k, set())
        return out


def same_scaffold(a, b) -> bool:
    """補足 2(b)・レビュー記録の決めごと 4: どちらかが None なら同じ骨格としない。"""
    return a is not None and b is not None and a == b


# ---------------------------------------------------------------- 写し（lfus 5f9c6fc、式は変えない）
def _isin(a, keys) -> np.ndarray:
    """文字列の配列の所属（pandas のハッシュ。None・NaN は False）。"""
    import pandas as pd
    return pd.Series(np.asarray(a, dtype=object)).isin(set(keys)).to_numpy(bool, copy=True)


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


def lib_ix_from_build(build_py: Path) -> dict:
    """data/casmi26-fpnet-code/casmi/build.py の LIBS（spec_meta.lib の符号）を ast で読む（import しない）。"""
    for node in ast.parse(build_py.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LIBS" for t in node.targets):
            return {l: i for i, l in enumerate(ast.literal_eval(node.value))}
    raise KeyError("LIBS が build.py に無い")


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


def _maxrss_gb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1 << 30) if sys.platform == "darwin" else r / (1 << 20), 2)   # macOS はバイト、Linux は KiB


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 実データの文脈（run.py の rep 段と window から使う）
class RepContext:
    """holdout ごとに 1 回作る: 印 → 代表 → 混入の検査 → 表。候補ごとに rep を計算し、記録を集める。"""

    def __init__(self, holdout: str, pool, spectra: dict, truth_csv: Path, workdir: Path = WORKDIR):
        import pandas as pd
        sys.path.insert(0, str(ROOT / "scripts/local_eval"))
        sys.path.insert(0, str(ROOT / "scripts/validate"))
        sys.path.insert(0, str(ROOT / "scripts/cv"))
        sys.path.insert(0, str(ROOT / "data/ext/v4g/casmi26-v3-models/code"))
        import library as L
        import mrr25
        from casmi import chem
        from casmi.library import Library
        from reporter_reach import scaffold

        t0 = time.time()
        self.timing = {}
        self.holdout, self.pool, self.spectra = holdout, pool, spectra
        self.scaffold_fn, self.to_key, self.raw_key = scaffold, mrr25.to_key, chem.raw_key
        rule = L.hide_rule(holdout)
        drop = L.drop_keys(holdout)
        dk = ROOT / f"data/holdouts/{holdout}/drop_keys.txt"
        dk_sha = sha256(dk)
        man = ROOT / f"data/holdouts/{holdout}/drop_keys_manifest.json"
        if holdout == E180K:
            assert dk_sha == E180_DROP_SHA256, f"e180_c2k の drop_keys.txt が事前登録と違う: {dk_sha}"
            man_sha = None
        else:
            assert man.is_file(), f"drop_keys_manifest.json が無い: {man}"   # 補足 1-5(1)
            man_sha = json.loads(man.read_text())["after"]["sha256"]
            assert dk_sha == man_sha, "drop_keys.txt が drop_keys_manifest.json の after.sha256 と違う"
        lib_ix = lib_ix_from_build(ROOT / "data/casmi26-fpnet-code/casmi/build.py")
        st = pd.read_parquet(workdir / "train_structs.parquet", columns=["inchikey14", "key", "rawkey", "smiles", "sid"])
        assert (st.sid.values == np.arange(len(st))).all(), "train_structs の sid が行の番号でない"
        st_ik14, st_key, st_rawkey = (st[c].values.astype(object) for c in ("inchikey14", "key", "rawkey"))
        tr = read_train_cols(ROOT / "data/train.parquet")
        self.timing["read_train"] = round(time.time() - t0, 1)
        self.timing["maxrss_gb_read_train"] = _maxrss_gb()
        lib = Library(str(workdir), verbose=True)
        self.timing["maxrss_gb_load_library"] = _maxrss_gb()
        align = check_alignment(tr, lib.meta, st_ik14, lib_ix)
        print("[rep align]", align, flush=True)
        assert align["all_ok"], f"spec_meta と train.parquet の行が一致しない: {align}"
        mask, mstat = hide_mask(tr["inchikey14"], tr["ingest_lib"], lib.sid, st_key, st_rawkey, drop, rule)
        rep = lib.representatives(exclude=mask, key=holdout)
        truth_df = pd.read_csv(truth_csv)
        skeys = source_keys(list(zip(truth_df.inchikey14, truth_df.smiles)), mrr25.to_key, st_ik14, st_key, st_rawkey)
        lk = dict(tr_ik14=tr["inchikey14"], tr_lib=tr["ingest_lib"], sid=lib.sid, st_ik14=st_ik14, st_key=st_key,
                  st_rawkey=st_rawkey)
        inv = {v: k for k, v in lib_ix.items()}
        tr_lib_meta = np.array([inv.get(int(x), "?") for x in lib.lib], dtype=object)
        audit = dict(n_rep=int(len(rep["idx"])), n_source_keys=len(skeys), n_drop_keys=len(drop),
                     leak_rep_drop_keys=int(len(leaked_rows(rep["idx"], keys=drop, rule=rule, **lk))),
                     leak_rep_source_keys=int(len(leaked_rows(rep["idx"], keys=skeys, rule=rule, **lk))),
                     leak_rep_meta_lib=int(len(leaked_rows(rep["idx"], keys=drop, rule=rule, **(lk | dict(tr_lib=tr_lib_meta))))))
        print("[rep audit]", audit, flush=True)
        assert audit["leak_rep_drop_keys"] == 0 and audit["leak_rep_source_keys"] == 0 and audit["leak_rep_meta_lib"] == 0, \
            f"隠すべき行が代表に混ざっている: {audit}"
        del lk, tr_lib_meta, tr, mask   # 表を作る前に要らなくなった大きな配列を放す（メモリ）
        lib.it = None
        self.timing["representatives"] = round(time.time() - t0, 1)
        self.timing["maxrss_gb_representatives"] = _maxrss_gb()
        # 家族（骨格）: 代表を持つ sid だけ
        usid = np.unique(rep["sid"])
        smiles = st.smiles.values
        fam_str = {}
        for s in usid:
            fam_str[int(s)] = scaffold(smiles[int(s)])
        names = sorted({v for v in fam_str.values() if v is not None})
        self.fam_id = {f: i for i, f in enumerate(names)}
        sid_fam = np.full(len(st), -1, np.int64)
        for s, f in fam_str.items():
            if f is not None:
                sid_fam[s] = self.fam_id[f]
        n_none = sum(1 for v in fam_str.values() if v is None)
        self.timing["scaffolds"] = round(time.time() - t0, 1)
        self.timing["maxrss_gb_scaffolds"] = _maxrss_gb()
        self.tables = build_tables(rep["idx"], rep["sid"], rep["mode"], lib.off, lib.mz, lib.prec, sid_fam, len(names))
        self.timing["tables"] = round(time.time() - t0, 1)
        self.timing["maxrss_gb_tables"] = _maxrss_gb()
        del lib
        self.own_index = OwnIndex(st_ik14, st_key, st_rawkey)
        # pool の train_sid（Pool と同じ並び）: 補足 1 の assert
        pm = pd.read_parquet(ROOT / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "mass", "train_sid"])
        order = np.argsort(pm["mass"].values, kind="stable")
        assert (pm["key"].values[order] == pool.key).all(), "pool_meta の並びが Pool と違う"
        self.train_sid = pm["train_sid"].values[order].astype(np.int64)
        self.ik2sid = {k: i for i, k in enumerate(st_ik14)}
        self.truth_rows = {str(m): (ik, smi) for m, ik, smi in zip(truth_df.molecule_id, truth_df.inchikey14, truth_df.smiles)}
        self.info = dict(issue=171, holdout=holdout, rule=rule, workdir=str(workdir), drop_keys_sha256=dk_sha,
                         drop_keys_manifest_after_sha256=man_sha,
                         train_structs_sha256=sha256(workdir / "train_structs.parquet"), align=align, mask=mstat,
                         audit=audit, bin=BIN, half=HALF, one=ONE,
                         table={str(m): dict(N=t.N, n_families_nonzero=int((t.nF > 0).sum()), n_codes=int(t.c_keys.size))
                                for m, t in self.tables.items()},
                         n_rep_structures=int(len(usid)), n_rep_structures_scaffold_none=int(n_none),
                         n_families=len(names))
        self._scaf, self._own, self._tk = {}, {}, {}
        self.window_scores: dict[str, dict[int, float]] = {}
        self.cnt = dict(window_rows=0, window_no_score=0, top_rows=0, top_no_score=0, molecules=0,
                        molecules_unchanged=0, train_sid_checked=0, same_scaffold_diff_rep=0, decoy_overtakes_truth=0,
                        truth_in_window=0, truth_sid_known=0, truth_sid_in_leaveout=0, no_query_features=0)
        print(f"[rep] {holdout} ready {time.time() - t0:.1f}s", self.info["table"], flush=True)

    # 候補ごとのキャッシュ
    def scaffold(self, i: int):
        smi = self.pool.smiles[i]
        if smi not in self._scaf:
            self._scaf[smi] = self.scaffold_fn(smi)
        return self._scaf[smi]

    def tkey(self, smi: str):
        if smi not in self._tk:
            self._tk[smi] = self.to_key(smi) if isinstance(smi, str) and smi else None
        return self._tk[smi]

    def own(self, i: int) -> set:
        if i not in self._own:
            smi = self.pool.smiles[i]
            K = {self.pool.key[i], self.raw_key(smi), self.tkey(smi)}
            s = self.own_index.sids(K)
            ts = int(self.train_sid[i])
            if ts >= 0:
                assert ts in s, f"pool の train_sid {ts} が leave-out の sid に入らない（行 {i}）"
                self.cnt["train_sid_checked"] += 1
            self._own[i] = s
        return self._own[i]

    def score_window_rows(self, mid: str, idx: list[int]) -> dict[int, float]:
        """窓の全候補（score 段の行）の rep(c|q)。npz にも入れる。"""
        Q = query_codes(self.spectra.get(mid, []))
        if not any(len(v) for v in Q.values()):
            self.cnt["no_query_features"] += 1
        ac, cc = {}, {}
        out = {}
        for i in idx:
            i = int(i)
            f = self.scaffold(i)
            fam = self.fam_id.get(f) if f is not None else None
            own = self.own(i)   # 点なしの候補でも計算する（補足 1 の train_sid の assert を窓の全候補に掛けるため）
            out[i] = rep_score(self.tables, Q, fam, own, ac, cc) if fam is not None else math.nan
        self.window_scores[mid] = out
        self.cnt["window_rows"] += len(out)
        self.cnt["window_no_score"] += sum(1 for v in out.values() if not math.isfinite(v))
        # 補足 1 の記録: 正解の sid が正解の候補の leave-out に入ったか
        ik, smi = self.truth_rows.get(mid, (None, None))
        tk = self.tkey(smi) if isinstance(smi, str) else None
        hit = [i for i in out if tk is not None and self.tkey(self.pool.smiles[i]) == tk]
        if hit:
            self.cnt["truth_in_window"] += 1
            if ik in self.ik2sid:
                self.cnt["truth_sid_known"] += 1
                self.cnt["truth_sid_in_leaveout"] += int(self.ik2sid[ik] in self.own(hit[0]))
        return out

    def mix(self, mid: str, rows: list) -> list:
        """prior 後の上位 TOPN を z(prior) + z(rep) で並べ替える。点は score_window_rows で計算済みのもの。"""
        sc = self.window_scores[mid]
        rep = np.array([sc[int(i)] for _, i in rows], np.float64)
        new, applied = rep_mix(rows, rep)
        self.cnt["molecules"] += 1
        self.cnt["top_rows"] += len(rows)
        self.cnt["top_no_score"] += int((~np.isfinite(rep)).sum())
        self.cnt["molecules_unchanged"] += int(not applied)
        # 補足 2(a): 上位で同じ骨格（None は除く）の候補の間で rep の点が違った分子
        by = {}
        for (_, i), v in zip(rows, rep):
            f = self.scaffold(int(i))
            if f is not None:
                by.setdefault(f, set()).add("nan" if not math.isfinite(v) else round(float(v), 12))
        self.cnt["same_scaffold_diff_rep"] += int(any(len(v) > 1 for v in by.values()))
        # 補足 2(b)・補足 2-3: prior 後の順で正解より下にいた、正解と同じ骨格のおとりが、rep の後に正解より上になった分子
        _, smi = self.truth_rows.get(mid, (None, None))
        tk = self.tkey(smi) if isinstance(smi, str) else None
        pos_t = [j for j, (_, i) in enumerate(rows) if tk is not None and self.tkey(self.pool.smiles[i]) == tk]
        if pos_t:
            ti = rows[pos_t[0]][1]
            tf = self.scaffold(int(ti))
            new_pos = {int(i): j for j, (_, i) in enumerate(new)}
            below = [int(i) for _, i in rows[pos_t[0] + 1:] if same_scaffold(self.scaffold(int(i)), tf)]
            self.cnt["decoy_overtakes_truth"] += int(any(new_pos[i] < new_pos[int(ti)] for i in below))
        return new

    def write_npz(self, out_dir: Path) -> dict:
        """窓全体の点（sanpier.load_scores で読める形、点は float64、NaN = 点なし）と記録の json。"""
        mids = list(self.window_scores)
        offs = np.zeros(len(mids) + 1, np.int64)
        for j, m in enumerate(mids):
            offs[j + 1] = offs[j] + len(self.window_scores[m])
        idx = np.array([i for m in mids for i in self.window_scores[m]], np.int64)
        val = np.array([v for m in mids for v in self.window_scores[m].values()], np.float64)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "rep_scores.npz"
        tmp = out_dir / "rep_scores.tmp.npz"
        np.savez_compressed(tmp, mids=np.array(mids, dtype=object), offsets=offs, idx=idx, n=val)
        tmp.replace(p)
        res = dict(self.info, counts=dict(self.cnt), timing=self.timing, rep_scores_npz=str(p), rep_scores_sha256=sha256(p))
        (out_dir / "rep_scores.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return res

    def finish(self, out_dir: Path) -> dict:
        """run.py の result json に足す記録。"""
        r = self.write_npz(out_dir)
        c = self.cnt
        return dict(rep_counts=dict(c), rep_top_no_score_frac=round(c["top_no_score"] / max(c["top_rows"], 1), 4),
                    rep_window_no_score_frac=round(c["window_no_score"] / max(c["window_rows"], 1), 4),
                    rep_audit=self.info["audit"], rep_align_all_ok=self.info["align"]["all_ok"],
                    rep_drop_keys_sha256=self.info["drop_keys_sha256"], rep_table=self.info["table"],
                    rep_scores_npz=r["rep_scores_npz"], rep_scores_sha256=r["rep_scores_sha256"], rep_timing=self.timing)


# ---------------------------------------------------------------- e180_c2k: 窓全体の点だけ
def window_only(holdout: str) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/local_eval"))
    import dseen
    import library as L
    import run as R
    from mrr25 import read_truth
    t0 = time.time()
    zf = np.load(dseen.logits_path(holdout), allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    truth = read_truth(dseen.truth_path(holdout))
    rule = L.hide_rule(holdout)
    hidden = L.drop_keys(holdout) if rule.get("pool") == "remove" else set()
    pool = R.load_pool()
    fpc = R.FPCache(np.load(ROOT / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
    ctx = RepContext(holdout, pool, R.query_spectra(holdout), dseen.truth_path(holdout))
    for mid in truth:
        if mid not in targets:
            continue
        rows = R.score_window(R.visible_window(pool, targets[mid], hidden), pool, R.stage_z(zf[mid], None), fpc)
        ctx.score_window_rows(mid, [i for _, i in rows])
    res = ctx.write_npz(ROOT / f"outputs/local_eval/{holdout}")
    res["secs"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("window", help="窓全体の rep の点の npz だけを作る（#171 §4: e180_c2k 用）")
    w.add_argument("--holdout", required=True)
    a = ap.parse_args()
    r = window_only(a.holdout)
    print(json.dumps({k: r[k] for k in ("holdout", "audit", "table", "counts", "rep_scores_sha256", "secs")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
