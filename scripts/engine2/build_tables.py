"""Engine 2 の 2 表（sim_rank_rows_nofp.npz 型・rank_train.npz 型）を自前で生成する（#72、設計 docs/design/engine2-rebuild.md §4）。

1 つの模擬で 2 表を同時に作る:
- クエリ: 6 集合（gnps 400・riken 300・mona 300・msdial 300・massbank 300・pluskal_ms2 400 = 2,000 分子）。集合ごとに
  学習スペクトルの inchikey14 を辞書順に並べ、RandomState(0) の permutation の先頭から取る（N_MOL < 2,000 のときは
  最大剰余法で配り、同じ並びの先頭を使う）。クエリのスペクトルはその集合のその分子の行全部
- class-1 模擬（S=0 / M=0）: lib 検索の窓とアナログの代表から、クエリ自身の行を隠す（代表がクエリ行なら、その構造の
  残りの行から `_reps` と同じ順で次の代表を選ぶ）
- class-2 模擬（S=1 / M=1）: 正解構造の行をすべて隠し、アナログの代表からも正解構造を落とす（構造はプールに残す）
- 候補・MetFrag-lite・特徴は推論コード（casmi_engine の build_pool / pool_window / compute_frag / BLOCKS）をそのまま呼ぶ。
  アナログ数は推論の eng_runner.py と同じ 200。正解が窓に無い分子は捨てて数を記録する
- prvsiyan ランカー用の base は base_fs.blk_base_fs（FP 12 列を fold-safe FPNet の fold k モデルのロジットで）。
  クエリは fold_identity 1–4 だけで、fold k のクエリには fpnet_fold{k}.pt を使う
- 除外: v3 の 4 集合の drop_keys、split.parquet の hold・np、structures.parquet の fold_identity == 0 を、
  inchikey14 と key（scorer_key / split の key）の両方で。積が空を assert し、件数を report に書く
- 2 表のファイル名（rank_train.npz、sim_rank_rows*.npz）を読もうとしたら例外（open / np.load を包む）

使い方（Kaggle CPU カーネルの例）:
  python build_tables.py --train /kaggle/input/.../train.parquet --assets assets.json --out /kaggle/working/own --n-mol 2000
assets.json はパスの対応（coco_dir, bio_dir, pc_dir, fs_models_dir, fs_bits, casmi_code, split, structures, drop_keys_dir）。
"""
import argparse
import builtins
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import base_fs  # noqa: E402

SETS = ["np", "gnps", "riken", "mona", "msdial", "massbank", "pluskal_ms2"]   # megayak 版と同じ並び（Q の番号）
QUERY_ALLOC = {"gnps": 400, "riken": 300, "mona": 300, "msdial": 300, "massbank": 300, "pluskal_ms2": 400}
BLOCKS_SIM = ["base", "libx", "anapol", "fragx"]
N_ANALOG = 200
SEED = 0
FORBIDDEN = ("rank_train.npz", "sim_rank_rows")
HOLDOUTS = ["v3_c1", "v3_c2", "enveda_v1", "v3_c2n"]
OPENED = []


# ------------------------------------------------------------------------------------------------
# 2 表のファイル名ガード（読んだファイルを記録し、元の 2 表の名前なら例外）
# ------------------------------------------------------------------------------------------------
def _check_name(p):
    s = os.fspath(p) if isinstance(p, (str, bytes, os.PathLike)) else None
    if s is None:
        return
    s = s.decode() if isinstance(s, bytes) else s
    base = os.path.basename(s)
    if any(f in base for f in FORBIDDEN) and not base.startswith("own_"):
        raise PermissionError(f"original Engine 2 table must not be read: {s}")
    OPENED.append(s)


def install_guard():
    """open / io.open / np.load を包む。戻り値は元に戻す関数。"""
    if getattr(builtins, "_e2_guard", False):
        return builtins._e2_guard
    _open, _npload, _ioopen = builtins.open, np.load, io.open

    def g_open(file, *a, **k):
        mode = a[0] if a else k.get("mode", "r")
        if "r" in mode and "+" not in mode:
            _check_name(file)
        return _open(file, *a, **k)

    def g_npload(file, *a, **k):
        _check_name(file)
        return _npload(file, *a, **k)

    def uninstall():
        builtins.open = _open; np.load = _npload; io.open = _ioopen
        builtins._e2_guard = False

    builtins.open = g_open; np.load = g_npload; io.open = g_open
    builtins._e2_guard = uninstall
    return uninstall


# ------------------------------------------------------------------------------------------------
# 除外集合とクエリ選び
# ------------------------------------------------------------------------------------------------
def load_exclusions(split_path, structures_path, drop_dir):
    ex_ik, ex_key, counts = set(), set(), {}
    for h in HOLDOUTS:
        ks = [x.strip() for x in open(os.path.join(drop_dir, h, "drop_keys.txt")) if x.strip()]
        ex_ik.update(ks); ex_key.update(ks); counts[f"drop_keys_{h}"] = len(ks)
    sp = pd.read_parquet(split_path)
    for f in ("hold", "np"):
        s = sp[sp.fold == f]
        ex_ik.update(s.inchikey14); ex_key.update(s.key); counts[f"split_{f}"] = int(len(s))
    st = pd.read_parquet(structures_path, columns=["inchikey14", "scorer_key", "fold_identity"])
    f0 = st[st.fold_identity == 0]
    ex_ik.update(f0.inchikey14); ex_key.update(f0.scorer_key); counts["fold_identity_0"] = int(f0.inchikey14.nunique())
    excl = ex_ik | ex_key
    counts["union_keys"] = len(excl)
    fold = st.drop_duplicates("inchikey14").set_index("inchikey14")
    return excl, counts, fold


def allocate(n_mol):
    """2,000 分子の割り当て比で n_mol を最大剰余法で配る（剰余が同じなら集合名の辞書順）。"""
    tot = sum(QUERY_ALLOC.values())
    raw = {s: QUERY_ALLOC[s] * n_mol / tot for s in QUERY_ALLOC}
    out = {s: int(np.floor(v)) for s, v in raw.items()}
    rest = n_mol - sum(out.values())
    order = sorted(QUERY_ALLOC, key=lambda s: (-(raw[s] - out[s]), s))
    for s in order[:rest]:
        out[s] += 1
    return out


def select_queries(lib_set, lib_ik, excl, fold, n_mol):
    """{set: [inchikey14, ...]}。fold_identity 1–4 だけ、除外集合に入らないもの。"""
    alloc = allocate(n_mol)
    df = pd.DataFrame({"s": lib_set, "ik": lib_ik}).dropna().drop_duplicates()
    q, info = {}, {}
    for s in QUERY_ALLOC:
        ks = np.array(sorted(df.ik[df.s == s].unique()), dtype=object)
        f = fold.reindex(ks)
        ok = np.array(f.fold_identity.isin([1, 2, 3, 4]).values, dtype=bool)
        ok &= ~pd.Series(ks).isin(excl).values
        ok &= ~f.scorer_key.isin(excl).values
        cand = ks[ok]
        perm = np.random.RandomState(SEED).permutation(len(cand))
        q[s] = list(cand[perm[:alloc[s]]])
        info[s] = dict(available=int(len(ks)), eligible=int(len(cand)), chosen=len(q[s]))
    return q, info


def assert_disjoint(queries, excl, fold):
    keys = {k for ks in queries.values() for k in ks}
    sk = set(fold.reindex(sorted(keys)).scorer_key.dropna())
    inter = (keys | sk) & excl
    assert not inter, f"queries overlap the exclusion set: {sorted(inter)[:5]} ({len(inter)})"
    return 0


# ------------------------------------------------------------------------------------------------
# 隠した行つきのチャネル（casmi_engine.molecule_channels の写し + 隠す処理）
# ------------------------------------------------------------------------------------------------
def _rep_order_pick(L, rows):
    """_reps と同じ順（tims 優先、ピーク数の多い順、行番号の小さい順）の先頭。"""
    if len(rows) == 0:
        return None
    npk = np.diff(L["off"])[rows]
    df = pd.DataFrame({"row": rows, "tims": L["tims"][rows], "npk": npk})
    df = df.sort_values(["tims", "npk", "row"], ascending=[False, False, True])
    return int(df.row.values[0])


def analog_search_hidden(E, L, R, Q, target, hide_rows, truth_p, truth_rows_sub, mode, debug=None, tag="", pre=None):
    """E.analog_search と同じ（p, sim）を返す。mode='c1': 代表がクエリ行なら次の代表に替える、'c2': 正解構造を落とす。
    pre に E.analog_search(R, Q, target) の結果を渡すと検索を省く（同じ入力なら同じ値）。"""
    CFG, pv = E.CFG, E.pv
    lo = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")
    hi = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")
    p, s = E.analog_search(R, Q, target) if pre is None else pre
    reps = R["rep"][lo:hi]
    if mode == "c2":
        keep = p != truth_p
        p, s, reps = p[keep], s[keep], reps[keep]
    elif mode == "c1":
        bad = np.isin(reps, hide_rows)
        if bad.any():
            assert bad.sum() == 1 and (p[bad] == truth_p).all()
            j = int(np.where(bad)[0][0])
            alt = _rep_order_pick(L, np.setdiff1d(truth_rows_sub, hide_rows))
            nm = L["nm"][alt] if alt is not None else np.nan
            if alt is not None and np.isfinite(nm) and target - CFG.ANALOG_WIN <= nm <= target + CFG.ANALOG_WIN:
                v = np.float32(0)
                if Q:
                    ro, rm, ri = pv.clean_store(np.array([alt], np.int64), L["off"], L["mz"], L["it"], CFG.INT_FLOOR,
                                                CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
                    sh = np.array([target - nm], np.float32)
                    for qm, qp in Q:
                        v = np.maximum(v, pv.search_shift_pre(qm, qp, 0, 1, ro, rm, ri, CFG.MZ_TOL, sh)[0])
                s = s.copy(); s[j] = v; reps = reps.copy(); reps[j] = alt
            else:
                keep = ~bad
                p, s, reps = p[keep], s[keep], reps[keep]
    if debug is not None:
        debug.setdefault(f"reps{tag}", []).extend(reps.tolist())
    return p, s


def channels_hidden(E, L, I, specs, precs, pols, target, hide_rows, truth_p, truth_rows, mode, debug=None):
    """E.molecule_channels と同じ戻り値。hide_rows を lib 検索の窓から落とし、アナログは mode の規則で隠す。
    mode=None なら何も隠さない（E.molecule_channels と完全一致）。"""
    CFG, pv = E.CFG, E.pv
    hide_rows = np.asarray(hide_rows, np.int64)
    Q = [E.clean(a, b) for a, b in specs]
    keep = [i for i, (a, _) in enumerate(Q) if len(a)]
    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]
    tol = target * CFG.PPM_WIN / 1e6
    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")
    crow = I["lib_rows"][lo:hi]
    if mode is not None:
        crow = crow[~np.isin(crow, hide_rows)]
    if debug is not None:
        debug.setdefault("crow", []).extend(crow.tolist())
    u, inv = np.unique(L["row_p"][crow], return_inverse=True)
    M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)
    for j, (qm, qp) in enumerate(Q):
        if len(crow) == 0: break
        s = pv.search(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                      CFG.INT_POWER, CFG.ENT_WEIGHT)
        np.maximum.at(M[j], inv, s)
        sh = (precs[j] - L["prec"][crow]).astype(np.float32)
        s2 = pv.search_shift_rows(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                                  CFG.INT_POWER, CFG.ENT_WEIGHT, sh)
        np.maximum.at(MS[j], inv, s2)
    cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)
    z = np.zeros(0, np.float32)
    lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)
    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)

    def _search(R, QQ, rows_sub, tag):
        if mode is None:
            p, s = E.analog_search(R, QQ, target)
            if debug is not None:
                lo_ = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")
                hi_ = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")
                debug.setdefault(f"reps{tag}", []).extend(R["rep"][lo_:hi_].tolist())
            return p, s
        return analog_search_hidden(E, L, R, QQ, target, hide_rows, truth_p, rows_sub, mode, debug, tag)

    tr = truth_rows[ok[truth_rows]] if truth_rows is not None else np.zeros(0, np.int64)
    p, s = _search(I["all"], Q, tr, "_all")
    ana = E.top_analogs(p, s)
    ps, ss = [], []
    for sign, name in ((1, "pos"), (-1, "neg")):
        QQ = [q for q, pl in zip(Q, pols) if pl == sign]
        if QQ:
            p, s = _search(I[name], QQ, tr[L["pol"][tr] == sign], "_" + name); ps.append(p); ss.append(s)
    anapol = E.top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []
    return lib, ana, anapol


def channels_pair(E, L, I, specs, precs, pols, target, hide_c1, hide_c2, truth_p, truth_rows):
    """channels_hidden(..., hide_c1, ..., 'c1') と channels_hidden(..., hide_c2, ..., 'c2') を、検索を 1 回だけにして返す。
    lib の窓の pv.search / search_shift_rows は行ごとに独立、E.analog_search は隠す前の値なので、結果は 2 回呼ぶのと完全一致
    （tests/test_engine2_build_tables.py で確認）。"""
    CFG, pv = E.CFG, E.pv
    Q = [E.clean(a, b) for a, b in specs]
    keep = [i for i, (a, _) in enumerate(Q) if len(a)]
    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]
    tol = target * CFG.PPM_WIN / 1e6
    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")
    crow_all = I["lib_rows"][lo:hi]
    S1, S2 = [], []
    for j, (qm, qp) in enumerate(Q):
        if len(crow_all) == 0: break
        S1.append(pv.search(qm, qp, crow_all, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                            CFG.INT_POWER, CFG.ENT_WEIGHT))
        sh = (precs[j] - L["prec"][crow_all]).astype(np.float32)
        S2.append(pv.search_shift_rows(qm, qp, crow_all, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                                       CFG.INT_POWER, CFG.ENT_WEIGHT, sh))
    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)
    tr = truth_rows[ok[truth_rows]]
    pre = {"all": E.analog_search(I["all"], Q, target)}
    QQs = {}
    for sign, name in ((1, "pos"), (-1, "neg")):
        QQ = [q for q, pl in zip(Q, pols) if pl == sign]
        if QQ:
            QQs[name] = (sign, QQ); pre[name] = E.analog_search(I[name], QQ, target)
    out = []
    for hide, mode in ((np.asarray(hide_c1, np.int64), "c1"), (np.asarray(hide_c2, np.int64), "c2")):
        m = ~np.isin(crow_all, hide)
        crow = crow_all[m]
        u, inv = np.unique(L["row_p"][crow], return_inverse=True)
        M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)
        if len(crow):
            for j in range(len(Q)):
                np.maximum.at(M[j], inv, S1[j][m]); np.maximum.at(MS[j], inv, S2[j][m])
        cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)
        z = np.zeros(0, np.float32)
        lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)
        p, s = analog_search_hidden(E, L, I["all"], Q, target, hide, truth_p, tr, mode, pre=pre["all"])
        ana = E.top_analogs(p, s)
        ps, ss = [], []
        for name in ("pos", "neg"):
            if name in QQs:
                sign, QQ = QQs[name]
                p, s = analog_search_hidden(E, L, I[name], QQ, target, hide, truth_p, tr[L["pol"][tr] == sign], mode, pre=pre[name])
                ps.append(p); ss.append(s)
        anapol = E.top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []
        out.append((lib, ana, anapol))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------
# 1 分子の記録と行
# ------------------------------------------------------------------------------------------------
def query_specs(L, rows):
    """compute_channels と同じ前処理（precursor + 2 までのピーク、最大で割る）。"""
    specs = []
    for r in rows:
        a_ = L["mz"][L["off"][r]:L["off"][r + 1]].astype(np.float32); b_ = L["it"][L["off"][r]:L["off"][r + 1]].astype(np.float32)
        k_ = a_ <= float(L["prec"][r]) + 2.0
        a_, b_ = a_[k_], b_[k_]
        specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))
    return specs


def make_rec(E, L, I, rows, adducts, modes, target, cand, hide_rows, truth_p, truth_rows, mode, debug=None, channels=None):
    specs = query_specs(L, rows)
    if channels is None:
        channels = channels_hidden(E, L, I, specs, L["prec"][rows].astype(np.float64), modes, target,
                                   hide_rows, truth_p, truth_rows, mode, debug)
    lib, ana, anapol = channels
    u, lmax, lmean, lsh, lcnt = lib
    pos = {int(x): i for i, x in enumerate(u)}
    ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0
    rec = dict(target=target, cand=cand, zlog=None, specs=specs, mode=float(np.mean(modes)), modes=modes,
               adducts=list(adducts), ana=ana, anapol=anapol)
    for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):
        v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v
    return rec


def rows_for_molecule(E, rec, cfp, cfp_fs, zlog_fs):
    """(sim の X 51 列, rank_train の X 31 列)。"""
    rr = dict(rec, zlog=None)
    Xs = np.hstack([E.BLOCKS[b](rr, cfp) for b in BLOCKS_SIM]).astype(np.float32)
    Xr = base_fs.blk_base_fs(rec, cfp, cfp_fs, zlog_fs).astype(np.float32)
    return Xs, Xr


def sha256(path):
    h = hashlib.sha256()
    with builtins.open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------
# 本体
# ------------------------------------------------------------------------------------------------
def read_rows_f64(train_path, rows):
    """指定行の ms2 を parquet の float64 のまま読む（fs_logits の入力を推論の test.parquet と同じ精度にする）。"""
    import pyarrow.parquet as pq
    f = pq.ParquetFile(train_path)
    rows = np.unique(np.asarray(rows, np.int64)); out = {}
    r0 = 0
    for i in range(f.num_row_groups):
        n = f.metadata.row_group(i).num_rows
        sel = rows[(rows >= r0) & (rows < r0 + n)]
        if len(sel):
            t = f.read_row_group(i, columns=["ms2_mzs", "ms2_normalized_intensities"]).take(sel - r0)
            for r, a, b in zip(sel.tolist(), t.column(0).to_pylist(), t.column(1).to_pylist()):
                out[r] = (np.asarray(a, np.float64), np.asarray(b, np.float64))
        r0 += n
    return out


def mol_logits(meta, L, rows, add, k, banks, raw=None):
    """1 分子の fold-safe FP ロジット（fold k のモデル）。入力が同じなら build の中と同じ計算。"""
    if raw is not None:
        mzs = [raw[r][0] for r in rows.tolist()]; its = [raw[r][1] for r in rows.tolist()]
    else:
        mzs = [L["mz"][L["off"][r]:L["off"][r + 1]].astype(np.float64) for r in rows]
        its = [L["it"][L["off"][r]:L["off"][r + 1]].astype(np.float64) for r in rows]
    sub = pd.DataFrame(dict(ms2_mzs=mzs, ms2_normalized_intensities=its,
                            ionization_mode=np.asarray(meta["mode"][rows]), adduct=np.asarray(add, dtype=str),
                            precursor_mz=L["prec"][rows], collision_energy_ev=list(meta["ce"][rows])))
    return base_fs.fs_logits(banks[k], base_fs.spectra_from_rows(sub.itertuples()))


def build(E, L, meta, I, queries, fold, banks, bits, workers, log=print, raw=None, v3keys=None,
          channels_impl="pair", zcache=None, zout=None):
    """queries: {set: [ik]}。L は row_p つき、E.POOL は設定済み。戻り値 (sim dict, rt dict, report)。
    channels_impl: 'pair'（channels_pair、検索 1 回）か 'hidden'（channels_hidden を class-1・class-2 で 2 回、高速化前）。
    zcache: dict を渡すと (set, ik) ごとの z を入れ、すでにあればそれを使う（2 つの経路で同じ z を共有する試走用）。
    zout: list を渡すと採用した分子の (set, ik, fold, z) を順に足す（z の保存用）。表の中身は変えない。"""
    assert channels_impl in ("pair", "hidden"), channels_impl
    POOL = E.POOL
    rep = dict(dropped_out_of_window=0, dropped_no_pool=0, dropped_no_mass=0, dropped_no_logits=0,
               molecules=0, per_set={}, v3_truth_negative_rows=0)
    qk = {k for ks in queries.values() for k in ks}
    sel = np.flatnonzero(pd.Series(meta["ik"]).isin(qk).values)
    gidx = pd.DataFrame({"s": meta["ingest_lib"][sel], "ik": meta["ik"][sel]}).groupby(["s", "ik"], sort=False).indices
    mols = []
    for s in QUERY_ALLOC:
        for ik in queries.get(s, []):
            g = gidx.get((s, ik))
            rows = np.sort(sel[g]).astype(np.int64) if g is not None else np.zeros(0, np.int64)
            mols.append((s, ik, rows))
    recs = []
    t0 = time.time()
    for mi, (s, ik, rows) in enumerate(mols):
        if not len(rows):
            rep["dropped_no_pool"] += 1; continue
        p = int(POOL["k2i"].get(ik, -1))
        if p < 0:
            rep["dropped_no_pool"] += 1; continue
        add = meta["adduct"][rows]
        nms = E.pv.neutral_mass(L["prec"][rows].astype(np.float64), np.asarray(add, dtype=str))
        nms = nms[np.isfinite(nms)]
        if not len(nms):
            rep["dropped_no_mass"] += 1; continue
        target = float(np.median(nms))
        cand = E.pool_window(target, E.CFG.PPM_WIN)
        if len(cand) == 0: cand = E.pool_window(target, E.CFG.PPM_FALLBACK)
        if p not in set(cand.tolist()):
            rep["dropped_out_of_window"] += 1; continue
        modes = np.where(np.asarray(meta["mode"][rows]) == "positive", 1.0, -1.0)
        truth_rows = np.flatnonzero(L["row_p"] == p)
        k = int(fold.loc[ik, "fold_identity"])
        assert k in (1, 2, 3, 4)
        if zcache is not None and (s, ik) in zcache:
            z = zcache[(s, ik)]["z"]
        else:
            z = mol_logits(meta, L, rows, add, k, banks, raw)
            if zcache is not None:
                zcache[(s, ik)] = dict(z=z, rows=rows, add=add, k=k)
        if z is None:
            rep["dropped_no_logits"] += 1; continue
        if channels_impl == "pair":
            ch1, ch2 = channels_pair(E, L, I, query_specs(L, rows), L["prec"][rows].astype(np.float64), modes, target,
                                     rows, truth_rows, p, truth_rows)
        else:
            ch1 = ch2 = None                    # make_rec の中で channels_hidden（b6902e1 と同じ呼び方）
        r1 = make_rec(E, L, I, rows, add, modes, target, cand, rows, p, truth_rows, "c1", channels=ch1)
        r2 = make_rec(E, L, I, rows, add, modes, target, cand, truth_rows, p, truth_rows, "c2", channels=ch2)
        recs.append(dict(set=s, ik=ik, fold=k, p=p, cand=cand, r1=r1, r2=r2, z=z))
        if zout is not None:
            zout.append((s, ik, k, z))
        if mi % 100 == 0:
            log(f"  channels {mi}/{len(mols)} {time.time() - t0:.0f}s")
    t_ch = time.time() - t0
    # MetFrag-lite: 推論と同じ compute_frag（候補の重複を除いて 1 回）。class-2 は同じ候補・同じスペクトル
    E.compute_frag([x["r1"] for x in recs], workers)
    for x in recs:
        for f in ("frag", "frag_ad", "frag_mean", "frag_pol"):
            x["r2"][f] = x["r1"][f]
        x["r2"].pop("specs", None)
    t_fr = time.time() - t0 - t_ch
    fpc = base_fs.FsFpCache(lambda i: POOL["smiles"][i], bits)
    v3 = set(v3keys or ())
    Xs_l, Ys, Ss, Gs, Qs, Xr_l, Yr, Mr, Gr = [], [], [], [], [], [], [], [], []
    for gi, x in enumerate(recs):
        cand = x["cand"]; cfp = E.pool_fps(cand); cfs = fpc.get(cand)
        y = (cand == x["p"]).astype(np.int8)
        rep["v3_truth_negative_rows"] += 2 * int((pd.Series(POOL["keys"][cand]).isin(v3).values & (y == 0)).sum())
        rep["dup_truth_key_rows"] = rep.get("dup_truth_key_rows", 0) + 2 * int(((POOL["keys"][cand] == x["ik"]) & (y == 0)).sum())
        for S, rec in ((0, x["r1"]), (1, x["r2"])):
            Xs, Xr = rows_for_molecule(E, rec, cfp, cfs, x["z"])
            Xs_l.append(Xs); Ys.append(y); Ss.append(np.full(len(y), S, np.int8)); Gs.append(np.full(len(y), 2 * gi + S, np.int32))
            Qs.append(np.full(len(y), SETS.index(x["set"]), np.int8))
            Xr_l.append(Xr); Yr.append(y.astype(np.float64)); Mr.append(np.full(len(y), float(S))); Gr.append(np.full(len(y), gi, np.int64))
        rep["per_set"][x["set"]] = rep["per_set"].get(x["set"], 0) + 1
    rep["molecules"] = len(recs)
    rep["fs_fp_failed"] = fpc.failed
    rep["secs"] = dict(channels=round(t_ch, 1), frag=round(t_fr, 1), rows=round(time.time() - t0 - t_ch - t_fr, 1))
    sim = dict(X=np.vstack(Xs_l), Y=np.concatenate(Ys), S=np.concatenate(Ss), G=np.concatenate(Gs), Q=np.concatenate(Qs),
               blocks=np.array(BLOCKS_SIM), sets=np.array(SETS))
    rt = dict(X=np.vstack(Xr_l), Y=np.concatenate(Yr), M=np.concatenate(Mr), G=np.concatenate(Gr))
    rep["queries"] = [dict(set=x["set"], ik=x["ik"], fold=x["fold"], n_cand=int(len(x["cand"]))) for x in recs]
    rep["channels_impl"] = channels_impl
    return sim, rt, rep


def cpu_info():
    """実行した機械の記録（FP ロジットの CPU 推論の機械差の手がかり）。取れない項目はエラー文を入れる。"""
    import platform
    import subprocess
    d = dict(os_cpu_count=os.cpu_count(), platform=platform.platform(), machine=platform.machine(),
             processor=platform.processor())
    try:
        d["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # macOS など lscpu が無い
        d["lscpu"] = f"unavailable: {type(e).__name__}: {e}"
    try:
        import torch
        d["torch_version"] = torch.__version__
        d["torch_config"] = torch.__config__.show()
        d["torch_num_threads"] = int(torch.get_num_threads())
        d["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
    except Exception as e:
        d["torch_version"] = d["torch_config"] = f"unavailable: {type(e).__name__}: {e}"
        d["torch_num_threads"] = d["torch_num_interop_threads"] = None
    d["env_threads"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    return d


def save_logits(out, zl, tag=""):
    """zout の (set, ik, fold, z) を own_fp_logits{tag}.npz に保存（Z は分子 × ビットの float32、表と同じ分子順）。"""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"own_fp_logits{tag}.npz")
    Z = np.stack([np.asarray(z, np.float32) for *_, z in zl]) if zl else np.zeros((0, 0), np.float32)
    np.savez(path, Z=Z, set=np.array([x[0] for x in zl], dtype="U16"), ik=np.array([x[1] for x in zl], dtype="U32"),
             fold=np.array([x[2] for x in zl], np.int8))
    return path


def compare_logits(za, zb):
    """2 回分の zout を分子ごとに比べる。"""
    assert [x[:3] for x in za] == [x[:3] for x in zb], "z の分子の並びが違う"
    eq = [bool(np.array_equal(a[3], b[3])) for a, b in zip(za, zb)]
    mx = max((float(np.max(np.abs(np.asarray(a[3], np.float64) - np.asarray(b[3], np.float64)))) for a, b in zip(za, zb)),
             default=0.0)
    return dict(n=len(eq), n_equal=int(sum(eq)), all_equal=bool(all(eq)), max_abs_diff=mx)


def table_stats(sim, rt):
    d = {}
    for name, t, gk in (("sim", sim, "G"), ("rank_train", rt, None)):
        X = t["X"]; Y = t["Y"]
        if name == "sim":
            g = t["G"]
        else:
            g = t["G"] * 2 + t["M"].astype(np.int64)
        u, c = np.unique(g, return_counts=True)
        pos = np.bincount(np.searchsorted(u, g), weights=Y.astype(np.float64), minlength=len(u))
        d[name] = dict(shape=list(X.shape), pos_rate=round(float(Y.mean()), 6), groups=int(len(u)),
                       pos_per_group_min=float(pos.min()), pos_per_group_max=float(pos.max()),
                       cand_mean=round(float(c.mean()), 2), cand_median=float(np.median(c)))
    return d


def save_tables(out, sim, rt, tag=""):
    os.makedirs(out, exist_ok=True)
    ps = os.path.join(out, f"own_sim_rank_rows{tag}.npz"); pr = os.path.join(out, f"own_rank_train{tag}.npz")
    # np.savez は zip の日時を書くので、決定性の比較は npz の中身（配列のバイト）で行う
    np.savez(ps, **sim); np.savez(pr, **rt)
    return ps, pr


def content_sha256(d):
    h = hashlib.sha256()
    for k in sorted(d):
        a = np.ascontiguousarray(d[k]); h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--assets", required=True, help="json: coco_dir, bio_dir, pc_dir, fs_models_dir, fs_bits, casmi_code, split, structures, drop_keys_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mol", type=int, default=2000)
    ap.add_argument("--repeat", type=int, default=1, help="同じ条件で表を何回作るか（決定性の確認）")
    ap.add_argument("--ab-check", action="store_true",
                    help="同じセッションで高速化前（channels_hidden 2 回）と channels_pair の 2 表を作り比べる試走（z は共有、z をもう 1 回計算）")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--rdkit", default=base_fs.RDKIT_VERSION)
    args = ap.parse_args(argv)
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)   # 推論（Kaggle の Linux）と同じ fork。差し替えた E.find をワーカーに引き継ぐ
    install_guard()
    A = json.load(builtins.open(args.assets))
    base_fs.set_casmi_code_dir(A["casmi_code"])
    rd = base_fs.check_rdkit(args.rdkit)
    E = base_fs.engine()
    E.CFG.N_ANALOG = N_ANALOG
    explicit = {"coco_fp.npy": os.path.join(A["coco_dir"], "coco_fp.npy"),
                "fp_bits.npy": os.path.join(A["coco_dir"], "fp_bits.npy"),
                os.path.join("casmi-chebi-lmsd-structures", "bio_fp.npy"): os.path.join(A["bio_dir"], "bio_fp.npy"),
                os.path.join("casmi26-pubchemlite-cc0-candidate-fingerprints", "pubchemlite_fp.npy"): os.path.join(A["pc_dir"], "pubchemlite_fp.npy")}

    def find(n):
        if n not in explicit:
            raise FileNotFoundError(f"build_tables: no explicit path for {n}")
        return explicit[n]
    E.find = find
    log = E.log
    for n_, f_ in explicit.items():
        assert os.path.exists(f_), f"missing asset {n_}: {f_}"
    assert len(np.load(explicit["fp_bits.npy"])) == 6930, "Engine 2 の候補指紋は 6,930 ビット"
    for f_ in (args.train, A["split"], A["structures"]):
        OPENED.append(f_)                      # pyarrow で読むファイルも記録
    excl, counts, fold = load_exclusions(A["split"], A["structures"], A["drop_keys_dir"])
    log(f"exclusions {counts}")
    import pyarrow.parquet as pq
    L = E.load_library(args.train)
    mt = pq.read_table(args.train, columns=["ingest_lib", "adduct", "ionization_mode", "collision_energy_ev"])
    meta = dict(ingest_lib=np.asarray(mt.column("ingest_lib").to_pylist(), dtype=object), ik=L["ik"].copy(),
                adduct=np.asarray(mt.column("adduct").cast("string").to_pylist(), dtype=object),
                mode=np.asarray(mt.column("ionization_mode").cast("string").to_pylist(), dtype=object),
                ce=np.asarray(mt.column("collision_energy_ev").to_pylist() + [None], dtype=object)[:-1])
    del mt
    queries, qinfo = select_queries(meta["ingest_lib"], meta["ik"], excl, fold, args.n_mol)
    assert_disjoint(queries, excl, fold)
    log(f"queries {qinfo}")
    msgs, _log = [], E.log
    E.log = lambda m: (msgs.append(str(m)), _log(m))
    try:
        E.POOL = E.build_pool(L, args.workers)
    finally:
        E.log = _log
    assert any(m.startswith("+ PubChemLite CC0") for m in msgs) and not any("PubChemLite CC0 skipped" in m for m in msgs), \
        f"PubChemLite CC0 がプールに入っていない（build_pool の try/except）: {msgs}"
    pool_info = dict(pool_size=int(len(E.POOL["keys"])), build_pool_log=msgs)
    log(f"pool {pool_info}")
    L["row_p"] = E.POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)
    del L["ik"], L["smi"]
    I = E.build_index(L)
    t_fixed = time.time() - E.T0
    log(f"fixed cost {t_fixed:.0f}s")
    banks = {k: base_fs.load_bank(os.path.join(A["fs_models_dir"], f"fpnet_fold{k}.pt")) for k in (1, 2, 3, 4)}
    bits = np.load(A["fs_bits"])
    v3keys = set()
    for h in HOLDOUTS:
        v3keys.update(x.strip() for x in builtins.open(os.path.join(A["drop_keys_dir"], h, "drop_keys.txt")) if x.strip())
    report = dict(rdkit=rd, n_mol=args.n_mol, alloc=allocate(args.n_mol), exclusions=counts, query_info=qinfo,
                  query_exclusion_overlap=0, fixed_secs=round(t_fixed, 1), pool=pool_info, runs=[], cpu_info=cpu_info(),
                  ab_check_mode=bool(args.ab_check))
    qrows = [r for s_, ks in queries.items() for r in np.flatnonzero(pd.Series(meta["ingest_lib"]).eq(s_).values & pd.Series(meta["ik"]).isin(set(ks)).values)]
    raw = read_rows_f64(args.train, qrows)
    log(f"query rows {len(raw):,}")

    def one_run(tag, impl, zcache=None):
        t1 = time.time(); zl = []
        sim, rt, rep = build(E, L, meta, I, queries, fold, banks, bits, args.workers, log, raw=raw, v3keys=v3keys,
                             channels_impl=impl, zcache=zcache, zout=zl)
        ps, pr = save_tables(args.out, sim, rt, tag)
        pz = save_logits(args.out, zl, tag)
        rep.update(stats=table_stats(sim, rt), sha256=dict(sim_file=sha256(ps), rank_train_file=sha256(pr),
                                                           sim_content=content_sha256(sim), rank_train_content=content_sha256(rt)),
                   secs_total=round(time.time() - t1, 1), files=[ps, pr], logits_file=pz)
        report["runs"].append(rep)
        log(f"run {tag or '(main)'} [{impl}]: {rep['stats']} {rep['sha256']}")
        return zl

    if args.ab_check:
        # 同じセッションで (i) 高速化前（channels_hidden 2 回）→ (ii) channels_pair。z は (i) で計算して (ii) で共有。
        # report["runs"][0] が (ii)（own_*.npz）、[1] が (i)（own_*_hidden.npz）
        zc = {}
        zl_i = one_run("_hidden", "hidden", zcache=zc)
        zl_ii = one_run("", "pair", zcache=zc)
        report["runs"].insert(0, report["runs"].pop())
        assert [x[:3] for x in zl_i] == [x[:3] for x in zl_ii] and all(a[3] is b[3] for a, b in zip(zl_i, zl_ii))
        # z をもう 1 回（同じ機械での決定性）。同じ分子・同じ行・同じ fold のモデル
        t2 = time.time()
        zl_2 = [(s_, ik_, zc[(s_, ik_)]["k"], mol_logits(meta, L, zc[(s_, ik_)]["rows"], zc[(s_, ik_)]["add"],
                                                         zc[(s_, ik_)]["k"], banks, raw)) for s_, ik_, _, _ in zl_i]
        pz2 = save_logits(args.out, zl_2, "_zrep1")
        r_i, r_ii = report["runs"][1]["sha256"], report["runs"][0]["sha256"]
        report["ab_check"] = dict(
            sim_content_equal=r_i["sim_content"] == r_ii["sim_content"],
            rank_train_content_equal=r_i["rank_train_content"] == r_ii["rank_train_content"],
            hidden=r_i, pair=r_ii, z_shared=True,
            z_recompute=dict(compare_logits(zl_i, zl_2), file=pz2, secs=round(time.time() - t2, 1)))
        log(f"ab_check {report['ab_check']}")
    else:
        for it in range(args.repeat):
            one_run("" if it == 0 else f"_rep{it}", "pair")
    report["opened_files"] = OPENED
    report["opened_forbidden"] = [p for p in OPENED if any(f in os.path.basename(p) for f in FORBIDDEN) and not os.path.basename(p).startswith("own_")]
    if args.repeat > 1:
        report["deterministic"] = all(r["sha256"]["sim_content"] == report["runs"][0]["sha256"]["sim_content"] and
                                      r["sha256"]["rank_train_content"] == report["runs"][0]["sha256"]["rank_train_content"]
                                      for r in report["runs"])
    json.dump(report, builtins.open(os.path.join(args.out, "build_report.json"), "w"), indent=1, default=str)
    log("done")
    return report


if __name__ == "__main__":
    main()
