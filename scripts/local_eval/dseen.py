#!/usr/bin/env python3
"""
Issue #146（事前登録、#144 §3）: d_seen — fragnn が「学習で見た構造（T）」のおとりを、化学の分布は同じで学習で
見ていない構造（train ∖ T）のおとりより高く（低く）出すかを、正解を除いたおとりだけで測る（採否には使わない）。

定義は #146 の本文と、事前登録のレビューの後の補足のコメントのとおり。要点:
- 候補 = run.py の score 段の窓（±10 ppm、visible_window → score_window）を pool の key でまとめたもの。点はキーの全行の最大
- 正解の除き方 E = {truth の inchikey14, raw_key, to_key, score_key} ∪ train_structs の別名（inchikey14・rawkey・key の
  どれかが E0 に入る行の 3 つの key）。候補の pool の key か、どれかの行の raw_key が E に入れば除く
- fragnn・fragnnblind・MetFrag-lite の 3 つすべてで点が有限の候補だけ
- 群 S = T（canon か raw のどちらかで入る）、U = S でなく train_structs の key ∪ rawkey ∪ inchikey14 に入る、O = それ以外
- 分子ごとに点を z（ddof 0）にし d_seen = mean(z|S) − mean(z|U)、d_dom = mean(z|U) − mean(z|O)（記録だけ）
- 外す理由（この順）: no_logits・no_candidates・one_group・sd0（fragnn か MetFrag-lite の sd が 0）
- Δ_seen = d_seen(fragnn) − d_seen(MetFrag-lite)、対の bootstrap 2,000 回・seed 0（train_eval.bootstrap_diff と同じ手順）、
  帯は丸める前の lo・hi で: lo > 0 → higher_seen、hi < 0 → lower_seen、それ以外 → none
- 読む集合 v3_c1・e180_c2k（使い道の 4 つの分かれ）。記録だけ enveda_v1・v3_c2n・v3_c2

    # e180_c2k の準備（data/holdouts/e180_c2k を作る → ロジット → 窓の一致の確認 → fragnn の npz）
    python scripts/local_eval/dseen.py prepare-e180c2k
    python scripts/cv/local_fold0_logits.py --holdout e180_c2k --out-dir outputs/local_eval/e180_c2k
    python scripts/local_eval/dseen.py check-window
    python scripts/local_eval/fragnn_score.py --holdout e180_c2k --logits outputs/local_eval/e180_c2k/pc_logits.npz \\
        --truth data/holdouts/e180_c2k/truth.csv
    # 時間だけを測る（d は計算しない、点も保存しない）
    python scripts/local_eval/dseen.py probe --n 20
    # 本番（1 回だけ）
    python scripts/local_eval/dseen.py run [--sets v3_c1,e180_c2k,enveda_v1,v3_c2n,v3_c2]

純粋な関数（CI でテストする）は rdkit・torch を import しない。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("scripts/local_eval", "scripts/validate", "scripts/ranker", "data/ext/v4g/casmi26-v3-models/code"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

READ_SETS = ("v3_c1", "e180_c2k")
RECORD_SETS = ("enveda_v1", "v3_c2n", "v3_c2")
ALL_SETS = READ_SETS + RECORD_SETS
N_BOOT, SEED = 2000, 0
EXPECTED_TR = 278_555            # train_structs の key ∪ rawkey ∪ inchikey14
E180, E180K = "e180_c2", "e180_c2k"
E180_SHA = {   # #122 の manifest.json の outputs（写す前に照合する）
    "test.parquet": "45b8b2364834ee50c4f8404f8a3ff49fc999aa9c087d98b1a7b8acfd7e4efc1f",
    "truth.csv": "9e56a4d436aa58f9ffa98257e1351a8d6ed38eea7758d0d205734f6981da11c6",
    "drop_keys.txt": "4ddce6584b1fe3f40d11d53de8f30360332cabc71d50f84e1b36c5d707aa6022",
}
KEEP_RULE = {"rule": "all", "pool": "keep"}
SCORES = ("n", "nb", "f", "s")   # fragnn・fragnnblind・MetFrag-lite・FPNet
NAMES = dict(n="fragnn", nb="fragnnblind", f="metfrag", s="fpnet")
REASONS = ("no_logits", "no_candidates", "one_group", "sd0")
WINDOW_MISMATCH_EXIT = 3


# ---------------------------------------------------------------- 読み先
def logits_path(h: str, root: Path = ROOT) -> Path:
    return root / ("outputs/local_eval/e180_c2k/pc_logits.npz" if h == E180K else f"outputs/holdout_eval/{h}_fold0/pc_logits.npz")


def truth_path(h: str, root: Path = ROOT) -> Path:
    return root / ("data/holdouts/e180_c2k/truth.csv" if h == E180K else f"outputs/holdouts/{h}/truth.csv")


def npz_path(h: str, name: str, root: Path = ROOT) -> Path:
    return root / f"outputs/local_eval/{h}/{name}_scores.npz"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- 純粋な関数（CI）
def alias_index(ts_rows: Iterable[tuple]) -> dict[str, set]:
    """train_structs の行 (inchikey14, rawkey, key) → {どれかの key: その行たちの 3 つの key の和}。"""
    idx: dict[str, set] = {}
    for row in ts_rows:
        ks = {k for k in row if isinstance(k, str) and k}
        for k in ks:
            idx.setdefault(k, set()).update(ks)
    return idx


def truth_exclusion(truth_ik14: str | None, truth_smiles: str | None, fns: Mapping[str, Callable], aidx: Mapping[str, set]) -> set:
    """E = E0 ∪ 別名。E0 = {truth の inchikey14, raw_key, to_key, score_key}（None は除く）。
    別名 = train_structs で inchikey14・rawkey・key のどれかが E0 に入る行の 3 つの key（1 段だけ）。"""
    e0 = {truth_ik14} if truth_ik14 else set()
    if truth_smiles:
        for name in ("raw", "to_key", "canon"):
            k = fns[name](truth_smiles)
            if k:
                e0.add(k)
    e = set(e0)
    for k in e0:
        e |= aidx.get(k, set())
    return e


def is_truth(canon: str, raws: Iterable[str | None], E: set) -> bool:
    return canon in E or any(r is not None and r in E for r in raws)


def group_of(canon: str, raws: Iterable[str | None], T_canon: set, T_raw: set, TR: set) -> str:
    """S（T に canon か raw のどれかで入る）/ U（S でなく TR に canon か raw のどれかで入る）/ O。"""
    raws = [r for r in raws if r is not None]
    if canon in T_canon or any(r in T_raw for r in raws):
        return "S"
    if canon in TR or any(r in TR for r in raws):
        return "U"
    return "O"


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float, np.floating)) and math.isfinite(float(v))


def zscore(v: np.ndarray) -> np.ndarray | None:
    """ddof 0 の z。sd が 0（か候補が 0）なら None。"""
    v = np.asarray(v, np.float64)
    if len(v) == 0:
        return None
    sd = v.std()
    if not sd > 0:
        return None
    return (v - v.mean()) / sd


def molecule_d(cands: list[dict]) -> tuple[dict | None, str | None]:
    """cands = [{group, n, nb, f, s}]（正解は除いた後）。3 つの点（n・nb・f）がすべて有限の候補だけを使う。
    返り値 (rec, None) か (None, 理由)。rec = {d_seen: {点: 値|None}, d_dom: {...}, n_S, n_U, n_O, n_cands}。"""
    c = [x for x in cands if _finite(x.get("n")) and _finite(x.get("nb")) and _finite(x.get("f"))]
    if not c:
        return None, "no_candidates"
    g = np.array([x["group"] for x in c])
    S, U, O = g == "S", g == "U", g == "O"
    if not S.any() or not U.any():
        return None, "one_group"
    z = {}
    for k in SCORES:
        vals = [x.get(k) for x in c]
        z[k] = zscore(np.array(vals, np.float64)) if all(_finite(v) for v in vals) else None
    if z["n"] is None or z["f"] is None:
        return None, "sd0"
    rec: dict = dict(n_cands=len(c), n_S=int(S.sum()), n_U=int(U.sum()), n_O=int(O.sum()), d_seen={}, d_dom={})
    for k in SCORES:
        zk = z[k]
        rec["d_seen"][k] = None if zk is None else float(zk[S].mean() - zk[U].mean())
        rec["d_dom"][k] = None if zk is None or not O.any() else float(zk[U].mean() - zk[O].mean())
    return rec, None


def bootstrap_ci(x: Mapping[str, float], n: int = N_BOOT, seed: int = SEED) -> dict:
    """train_eval.bootstrap_diff と同じ手順（molecule_id の昇順、default_rng(seed).integers(0, len, (n, len))）。
    丸める前の mean・lo・hi を返す。"""
    mids = sorted(x)
    v = np.array([x[m] for m in mids], np.float64)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return dict(n=len(v), mean=float(v.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))


def band(lo: float, hi: float) -> str:
    if lo > 0:
        return "higher_seen"
    if hi < 0:
        return "lower_seen"
    return "none"


def usage(b1: str, b2: str) -> str:
    """読む 2 集合の帯 → 使い道（#146 A §3.4 の 4 つの分かれ）。"""
    bs = {b1, b2}
    if b1 == b2 == "higher_seen":
        return "neg_outside_T"          # 学習の負例に T の外の候補を混ぜる
    if b1 == b2 == "lower_seen" or bs == {"lower_seen", "none"}:
        return "freeze_no_gate"         # 門なし FREEZE
    if bs == {"higher_seen", "lower_seen"}:
        return "no_decision"            # 判定なし
    return "no_treatment"               # 手当てなし


def windows_match(windows_hidden: Mapping[str, np.ndarray], windows_keep: Mapping[str, np.ndarray]) -> list[str]:
    """2 つの窓（Pool の添字の配列）が違う分子の一覧（分子の集合の違いも数える）。"""
    bad = sorted(set(windows_hidden) ^ set(windows_keep))
    for m in sorted(set(windows_hidden) & set(windows_keep)):
        if not np.array_equal(np.asarray(windows_hidden[m]), np.asarray(windows_keep[m])):
            bad.append(m)
    return bad


def summarize_set(recs: Mapping[str, dict], excluded: Mapping[str, str]) -> dict:
    """分子ごとの rec → 集合の要約（Δ_seen の帯、記録だけの数）。"""
    out: dict = dict(n=len(recs), excluded={r: sum(1 for v in excluded.values() if v == r) for r in REASONS})
    if not recs:
        out["delta_seen"] = None
        out["band"] = "none"
        return out
    x = {m: r["d_seen"]["n"] - r["d_seen"]["f"] for m, r in recs.items()}
    b = bootstrap_ci(x)
    out["delta_seen"] = dict(mean=round(b["mean"], 4), lo=round(b["lo"], 4), hi=round(b["hi"], 4), n=b["n"],
                             better=int(sum(v > 0 for v in x.values())), worse=int(sum(v < 0 for v in x.values())))
    out["band"] = band(b["lo"], b["hi"])
    # 記録だけ: blind − MetFrag（blind の sd > 0 の分子）、各点の d_seen・d_dom の平均
    xb = {m: r["d_seen"]["nb"] - r["d_seen"]["f"] for m, r in recs.items() if r["d_seen"]["nb"] is not None}
    out["blind_minus_metfrag"] = None
    if xb:
        bb = bootstrap_ci(xb)
        out["blind_minus_metfrag"] = dict(mean=round(bb["mean"], 4), lo=round(bb["lo"], 4), hi=round(bb["hi"], 4), n=bb["n"])
    out["blind_sd0_excluded"] = len(recs) - len(xb)
    for kind in ("d_seen", "d_dom"):
        out[kind] = {}
        for k in SCORES:
            v = [r[kind][k] for r in recs.values() if r[kind][k] is not None]
            out[kind][NAMES[k]] = dict(mean=round(float(np.mean(v)), 4), n=len(v)) if v else None
    for g in ("n_S", "n_U", "n_O", "n_cands"):
        out[f"{g}_median"] = float(np.median([r[g] for r in recs.values()]))
    return out


# ---------------------------------------------------------------- e180_c2k の準備
def prepare_e180c2k(root: Path = ROOT) -> dict:
    """data/holdouts/e180_c2k/ に e180_c2 の 3 つのファイルを写し（sha256 を照合）、hide_rule.json を keep にする。"""
    src, dst = root / f"data/holdouts/{E180}", root / f"data/holdouts/{E180K}"
    dst.mkdir(parents=True, exist_ok=True)
    rep = {}
    for name, sha in E180_SHA.items():
        got = sha256_file(src / name)
        if got != sha:
            raise SystemExit(f"[stop] {src / name} の sha256 が #122 と違う: {got}")
        shutil.copyfile(src / name, dst / name)
        assert sha256_file(dst / name) == sha
        rep[name] = sha
    (dst / "hide_rule.json").write_text(json.dumps(KEEP_RULE, indent=2) + "\n")
    (dst / "HOLDOUT.json").write_text(json.dumps(dict(name=E180K, issue=146, source=E180, hide_rule=KEEP_RULE,
                                                       files_sha256=rep), indent=1, ensure_ascii=False) + "\n")
    return dict(dir=str(dst), files=rep, hide_rule=KEEP_RULE)


def check_window(root: Path = ROOT) -> dict:
    """e180_c2k の全分子で、hidden = e180_c2 の drop_keys の窓と hidden = ∅ の窓が同じか。fragnn の npz があれば
    その idx とも照合する。違えば exit 3。"""
    import run as R
    import library as L
    import sanpier as SP
    zf = np.load(logits_path(E180K, root), allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    pool = R.load_pool(root)
    hidden = L.drop_keys(E180)
    assert len(hidden) == 200, len(hidden)
    assert L.hide_rule(E180K) == KEEP_RULE, L.hide_rule(E180K)
    wh = {m: R.visible_window(pool, t, hidden) for m, t in targets.items()}
    wk = {m: R.visible_window(pool, t, set()) for m, t in targets.items()}
    bad = windows_match(wh, wk)
    rep = dict(molecules=len(targets), rows=int(sum(len(v) for v in wk.values())), window_mismatch=len(bad),
               mismatched=bad[:20])
    for name in ("fragnn", "fragnnblind"):
        p = npz_path(E180K, name, root)
        if p.exists():
            sc = SP.load_scores(p)
            npz_w = {m: np.array(sorted(d), np.int64) for m, d in sc.items()}
            bad2 = windows_match({m: np.sort(wk[m]) for m in npz_w if m in wk}, npz_w)
            bad2 += sorted(set(wk) - set(npz_w))
            rep[f"{name}_npz_mismatch"] = len(bad2)
            bad += bad2
    rep["ok"] = not bad
    return rep


# ---------------------------------------------------------------- 実データ
class Ctx:
    """集合をまたいで共通の入力（T・train_structs・pool・関数）。"""

    def __init__(self, root: Path = ROOT):
        import pandas as pd
        import run as R
        from casmi import chem
        import mrr25
        from trainstruct_bias import EXPECTED_T, PREP, train_struct_keys
        t0 = time.time()
        self.root, self.R = root, R
        T_canon, tinfo = train_struct_keys(PREP)
        if tinfo["n_rows"] != EXPECTED_T:
            raise SystemExit(f"[stop] T の行数 {tinfo['n_rows']} != {EXPECTED_T}")
        self.raw_cache: dict[str, str | None] = {}
        self.fns = dict(raw=self.raw_key, to_key=mrr25.to_key, canon=chem.score_key)
        self._chem_raw = chem.raw_key
        t1 = time.time()
        T_raw_l = [self.raw_key(s) for s in tinfo["smiles"]]
        self.T_canon, self.T_raw = T_canon, {k for k in T_raw_l if k is not None}
        ts = pd.read_parquet(root / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet", columns=["inchikey14", "rawkey", "key"])
        self.TR = set(ts["inchikey14"].dropna()) | set(ts["rawkey"].dropna()) | set(ts["key"].dropna())
        if len(self.TR) != EXPECTED_TR:
            raise SystemExit(f"[stop] train_structs の key の和 {len(self.TR)} != {EXPECTED_TR}")
        self.aidx = alias_index(ts.itertuples(index=False, name=None))
        self.pool = R.load_pool(root)
        self.fpc = R.FPCache(np.load(root / "data/ext/v4g/casmi26-v2-pool/fp_bits.npy"))
        sys.path.insert(0, str(root / "data/ext/frag"))   # numba のキャッシュは通常の import でないと読めない
        import frag_rescore as frag
        frag.warmup()
        self.frag = frag
        self.info = dict(T_rows=tinfo["n_rows"], T_keys=tinfo["n_keys"], T_canon=len(T_canon), T_raw=len(self.T_raw),
                         T_raw_none=int(sum(k is None for k in T_raw_l)), T_raw_secs=round(time.time() - t1, 1),
                         TR=len(self.TR), prep_sha256=sha256_file(PREP), secs=round(time.time() - t0, 1))
        self.info["T_raw_cache_rows"] = len(self.raw_cache)
        self.raw_cache.clear()   # 窓の候補の分だけを持つ（メモリ）

    def raw_key(self, smi: str):
        if smi not in self.raw_cache:
            self.raw_cache[smi] = self._chem_raw(smi)
        return self.raw_cache[smi]


def set_inputs(h: str, ctx: Ctx) -> dict:
    import library as L
    import sanpier as SP
    from mrr25 import read_truth
    root = ctx.root
    zf = np.load(logits_path(h, root), allow_pickle=True)
    targets = {m: float(t) for m, t in zf["__targets__"]}
    tp = truth_path(h, root)
    truth = read_truth(tp)
    with open(tp, newline="") as f:
        trows = {r["molecule_id"]: r for r in csv.DictReader(f)}
    rule = L.hide_rule(h)
    hidden = L.drop_keys(h) if rule.get("pool") == "remove" else set()
    maps = {c: SP.load_scores(npz_path(h, name, root)) for c, name in (("n", "fragnn"), ("nb", "fragnnblind"))}
    files = {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): sha256_file(p)
             for p in (logits_path(h, root), tp, npz_path(h, "fragnn", root), npz_path(h, "fragnnblind", root))}
    return dict(zf=zf, targets=targets, truth=truth, trows=trows, hidden=hidden, rule=rule, maps=maps, files=files,
                spectra=ctx.R.query_spectra(h))


def window_keys(h: str, mid: str, inp: dict, ctx: Ctx) -> tuple[dict, int]:
    """score 段の窓をキーでまとめた候補（s・n・nb、f はまだ）。"""
    import iso_rank_diag as IR
    R, pool = ctx.R, ctx.pool
    rows = R.score_window(R.visible_window(pool, inp["targets"][mid], inp["hidden"]), pool,
                          R.stage_z(inp["zf"][mid], None), ctx.fpc)
    recs = [dict(key=pool.key[i], row=i, smiles=pool.smiles[i], s=s,
                 **{c: inp["maps"][c].get(mid, {}).get(i) for c in ("n", "nb")}) for s, i in rows]
    return IR.aggregate_by_key(recs), len(rows)


def metfrag_max(ctx: Ctx, spectra, smiles_all, fcache) -> float:
    import iso_rank_diag as IR
    v = math.nan
    for x in ctx.frag.score_molecule(spectra, smiles_all, cache=fcache):
        v = IR._nanmax(v, IR._num(x))
    return v


def run_set(h: str, ctx: Ctx, per_rows: list) -> dict:
    t0 = time.time()
    inp = set_inputs(h, ctx)
    recs, excl, n_truth_removed, n_rows = {}, {}, {}, 0
    fcache = {}
    for mid in inp["truth"]:
        if mid not in inp["targets"]:
            excl[mid] = "no_logits"
            continue
        cd, nr = window_keys(h, mid, inp, ctx)
        n_rows += nr
        tr = inp["trows"].get(mid, {})
        E = truth_exclusion(tr.get("inchikey14") or None, tr.get("smiles") or None, ctx.fns, ctx.aidx)
        cands = []
        n_truth_removed[mid] = 0
        for k, a in cd.items():
            raws = [ctx.raw_key(s) for s in a["smiles_all"]]
            if is_truth(k, raws, E):
                n_truth_removed[mid] += 1
                continue
            a["f"] = metfrag_max(ctx, inp["spectra"].get(mid, []), a["smiles_all"], fcache)
            cands.append(dict(key=k, group=group_of(k, raws, ctx.T_canon, ctx.T_raw, ctx.TR),
                              **{c: a[c] for c in SCORES}))
        rec, why = molecule_d(cands)
        if rec is None:
            excl[mid] = why
        else:
            recs[mid] = rec
            per_rows.append(dict(holdout=h, molecule_id=mid, delta_seen=rec["d_seen"]["n"] - rec["d_seen"]["f"],
                                 **{f"d_seen_{NAMES[k]}": rec["d_seen"][k] for k in SCORES},
                                 **{f"d_dom_{NAMES[k]}": rec["d_dom"][k] for k in SCORES},
                                 n_S=rec["n_S"], n_U=rec["n_U"], n_O=rec["n_O"], n_cands=rec["n_cands"],
                                 truth_removed=n_truth_removed[mid]))
    out = summarize_set(recs, excl)
    out.update(holdout=h, read=h in READ_SETS, n_truth=len(inp["truth"]), hide_rule=inp["rule"], hidden_keys=len(inp["hidden"]),
               window_rows=n_rows, truth_removed_total=int(sum(n_truth_removed.values())),
               molecules_with_truth_removed=int(sum(v > 0 for v in n_truth_removed.values())),
               files_sha256=inp["files"], secs=round(time.time() - t0, 1))
    return out


def ram_peak_gb() -> float:
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)   # macOS は bytes


def probe(n: int, root: Path = ROOT) -> dict:
    """時間だけ: v3_c1 の最初の n 分子（truth の順で pc_logits にある分子）で、窓の採点・raw_key・MetFrag-lite の秒を測る。
    z・d_seen・Δ_seen は計算せず、点も保存しない。全 5 集合の窓の行数で外挿する。"""
    t0 = time.time()
    ctx = Ctx(root)
    fixed = time.time() - t0
    h = "v3_c1"
    inp = set_inputs(h, ctx)
    mids = [m for m in inp["truth"] if m in inp["targets"]][:n]
    t1, rows, fcache = time.time(), 0, {}
    for mid in mids:
        cd, nr = window_keys(h, mid, inp, ctx)
        rows += nr
        for k, a in cd.items():
            [ctx.raw_key(s) for s in a["smiles_all"]]
            metfrag_max(ctx, inp["spectra"].get(mid, []), a["smiles_all"], fcache)
    secs = time.time() - t1
    total_rows = {}
    import library as L
    for s in ALL_SETS:
        zf = np.load(logits_path(s, root), allow_pickle=True)
        rule = L.hide_rule(s)
        hidden = L.drop_keys(s) if rule.get("pool") == "remove" else set()
        total_rows[s] = int(sum(len(ctx.R.visible_window(ctx.pool, float(t), hidden)) for _, t in zf["__targets__"]))
    per_row = secs / max(rows, 1)
    est = {s: round(r * per_row, 1) for s, r in total_rows.items()}
    est_all = fixed + sum(est.values())
    est_read = fixed + sum(est[s] for s in READ_SETS)
    return dict(issue=146, mode="probe", molecules=len(mids), rows=rows, secs=round(secs, 1), secs_per_row=round(per_row, 5),
                fixed_secs=round(fixed, 1), window_rows=total_rows, est_secs=est, est_all_secs=round(est_all, 1),
                est_read_secs=round(est_read, 1), over_2h=est_all > 7200,
                plan="read_only" if est_all > 7200 else "all_sets", ram_peak_gb=ram_peak_gb())


def run_all(sets: list[str], root: Path = ROOT) -> dict:
    t0 = time.time()
    if E180K in sets:
        chk = check_window(root)
        # 起動前レビューの minor 1: 本番では fragnn・fragnnblind の npz の照合も済んでいることを求める
        chk["ok"] = chk["ok"] and all(f"{x}_npz_mismatch" in chk for x in ("fragnn", "fragnnblind"))
        if not chk["ok"]:
            print(json.dumps(chk, ensure_ascii=False))
            print("[stop] e180_c2k の窓が remove 版（か fragnn の npz）と一致しない。何も数えていない")
            sys.exit(WINDOW_MISMATCH_EXIT)
    else:
        chk = None
    ctx = Ctx(root)
    per_rows: list = []
    res: dict = dict(issue=146, definition="#146 本文 + 事前登録レビューの補足（2026-11-18）", sets=sets, window_check_e180_c2k=chk,
               inputs=ctx.info, script_sha256=sha256_file(Path(__file__)))
    for h in sets:
        res[h] = run_set(h, ctx, per_rows)
        print(h, json.dumps({k: res[h][k] for k in ("n", "excluded", "delta_seen", "band")}, ensure_ascii=False), flush=True)
    if all(s in res for s in READ_SETS):
        res["usage"] = usage(res[READ_SETS[0]]["band"], res[READ_SETS[1]]["band"])
        res["usage_bands"] = {s: res[s]["band"] for s in READ_SETS}
    res["secs"] = round(time.time() - t0, 1)
    res["ram_peak_gb"] = ram_peak_gb()
    out = root / "outputs/local_eval/dseen.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    with open(root / "outputs/local_eval/dseen_per_molecule.csv", "w", newline="") as f:
        if per_rows:
            w = csv.DictWriter(f, fieldnames=list(per_rows[0]))
            w.writeheader()
            w.writerows(per_rows)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="#146 d_seen")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-e180c2k")
    sub.add_parser("check-window")
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=20)
    r = sub.add_parser("run")
    r.add_argument("--sets", default=",".join(ALL_SETS))
    a = ap.parse_args()
    if a.cmd == "prepare-e180c2k":
        print(json.dumps(prepare_e180c2k(), ensure_ascii=False))
    elif a.cmd == "check-window":
        rep = check_window()
        print(json.dumps(rep, ensure_ascii=False))
        if not rep["ok"]:
            sys.exit(WINDOW_MISMATCH_EXIT)
    elif a.cmd == "probe":
        rep = probe(a.n)
        (ROOT / "outputs/local_eval/dseen_probe.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
        print(json.dumps(rep, ensure_ascii=False))
    else:
        sets = a.sets.split(",")
        assert set(READ_SETS) <= set(sets), "読む 2 集合は必ず含める（#146 B6）"
        assert set(sets) <= set(ALL_SETS), sets
        res = run_all(sets)
        print(json.dumps({k: res.get(k) for k in ("usage", "usage_bands", "secs", "ram_peak_gb")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
