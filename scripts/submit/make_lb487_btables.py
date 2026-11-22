#!/usr/bin/env python3
"""
診断（#189、設計 docs/design/engine2-own-s2-diag.md §5・§6 = #158 (B)）: 0.487 写し（kernels/lb487）の Engine 2 の
2 表だけを (B) の自前表（#180 で作り直し、#165 の関門を通った b_run1 version 3 の出力）に替えたノートブックを作る。

#77 の生成器（make_lb487_owntables.py）との違い: 推論の FP は元のまま（casmi26-fp-models-v2 の 2 本、6,930 ビット）。
base_fs・fold-safe FPNet・score_molecules の差し替えは入れない。表は dataset ではなく kernel_sources
（taichiiiii/casmi26-engine2-build-b-run1 の出力の own/ の下）から読む。

変更は次だけ:
1. 凍結コード（外側セル 12 の FROZEN_B64）のセル 32（Engine 2 の起動）
   - Engine 2 を起動する前に 2 表のパスを決めて sha256 を照合（_e2own）: kernel_sources の b_run1 のディレクトリの下にあり
     相対パスが引数の値で終わるファイルが、ちょうど 1 つ見つかり、sha256 が引数の値と同じこと
   - 元の eng_runner.py を書いた直後に、(B) の eng_runner.py で上書きする。元の runner と同じ処理で、違いは
     (a) 2 表を上で決めた明示パスで読み、もう一度パスと sha256 を照合（E.find('rank_train.npz') などを呼ばない）
     (b) レポート /kaggle/working/engine2_btables_report.json（status・読んだパスと sha256・リストの分子数）
   - Engine 2 が落ちたとき: commit では止める。再実行（IS_RERUN）では #77 と同じく（オーナー判断 #108 09:19Z の 2）
     Engine 2 の出力を使わない v4g のリストに戻る（active な全分子に空のリスト）。レポートに fallback: true と理由
   - 凍結コードの宣言 sha256（外側セル 12 の FROZEN_SHA256、notebook メタデータ）を新しい値にする
2. 採点に影響しない: 外側セル 5 の COMMIT_MODE = 'fast' を 'smoke' にする（commit で見える 12 分子が全段を通る）
3. 時間測定の版（--smoke-n、slug に timing、提出しない）: 凍結セル 30 の SMOKE_N = 12 だけを変える

kernel-metadata.json: dataset_sources は元の 10 件のまま、kernel_sources に b_run1 を足す。id_no は消す。

    python scripts/submit/make_lb487_btables.py [--out-dir kernels/lb487_btables]
    python scripts/submit/make_lb487_btables.py --slug casmi26-lb487-btables-timing400 --smoke-n 400 \\
        --out-dir kernels/lb487_btables_timing400
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("make_lb487_owntables", ROOT / "scripts" / "submit" / "make_lb487_owntables.py")
OT = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("make_lb487_owntables", OT)
_spec.loader.exec_module(OT)

SRC = OT.SRC
SRC_NB = OT.SRC_NB
SRC_SHA = OT.SRC_SHA
FROZEN_CELLS = OT.FROZEN_CELLS
CHANGED_FROZEN = OT.CHANGED_FROZEN
CHANGED_OUTER = OT.CHANGED_OUTER
RUN_MARK = OT.RUN_MARK
WRITE_MARK = OT.WRITE_MARK
EXC_OLD = OT.EXC_OLD
replace_once = OT.replace_once
split_frozen = OT.split_frozen

# #180 で作り直し、#165 の関門を通った表（b_run1 version 3、#189 §1）
KSRC = "taichiiiii/casmi26-engine2-build-b-run1"
KSRC_NAME = KSRC.split("/")[1]
SIM_REL = "own/own_sim_rank_rows.npz"
RT_REL = "own/own_rank_train.npz"
SIM_SHA = "9ee9011314e158ef410e499368137e96aec8eafc13d3999594cfa70b3650d013"
RT_SHA = "74c0a37545d8c3cc578ef14d24ac79cfc1c9cbdae7eb65fe8c76ccf4633bf269"
SIM_SIZE = 90_404_186
RT_SIZE = 63_410_966
REPORT = "/kaggle/working/engine2_btables_report.json"
SOURCE_LIMIT = 1_000_000   # kaggle kernels push の上限（ソース 1 MB 未満）

B_RUNNER = r'''"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40).
#189 (#158 (B)): same as the original runner, but the two Engine 2 tables are read from explicit paths inside the
kernel source taichiiiii/casmi26-engine2-build-b-run1 (sha256 checked). FP inference is unchanged (fp-models-v2)."""
import os, sys, json, time
import numpy as np
import hashlib
KSRC_NAME = 'casmi26-engine2-build-b-run1'
RP = '/kaggle/working/engine2_btables_report.json'


def verify_own_tables(own):
    """own: {'sim': path, 'sim_sha': hex, 'rt': path, 'rt_sha': hex} -> {'sim': sha, 'rt': sha}. Raises on any mismatch."""
    sha = {}
    for k in ('sim', 'rt'):
        p = own[k]
        if KSRC_NAME not in os.path.normpath(p).split(os.sep)[:-1] or not os.path.basename(p).startswith('own_'):
            raise ValueError(('#189 own table outside ' + KSRC_NAME, k, p))
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        if h != own[k + '_sha']:
            raise ValueError(('#189 own table sha256 mismatch', k, p, h, own[k + '_sha']))
        sha[k] = h
    return sha


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    OWN = json.loads(os.environ['E2_OWN'])
    REP = dict(status='started', own=OWN)
    json.dump(REP, open(RP, 'w'), indent=1)
    import casmi_engine as E
    _orig_find = E.find
    def _find(n):                                  # v4g ships its own fp_bits.npy (10,226 bits): take prvsiyan's 6,930
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        assert 'rank_train' not in n and 'sim_rank_rows' not in n, ('original Engine 2 table requested', n)
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    import glob
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    print('engine fp models', fp_models, flush=True)
    sha = verify_own_tables(OWN)
    REP.update(tables_read={k: OWN[k] for k in ('sim', 'rt')}, tables_sha256=sha, fp_models=fp_models)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, OWN['sim'], OWN['rt'], fp_models,
                           workers=os.cpu_count(), w_pv=0.88)
    bl = E.blend_scores(S['pv'], S['ours'], 0.88)
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:80]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= 40:
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)
    REP.update(status='ok', molecules_listed=len(res), engine_secs=round(time.time() - T0, 1))
    json.dump(REP, open(RP, 'w'), indent=1, default=str)
'''

PATCH_BEGIN = "# ── #189（#158 (B) の診断）: Engine 2 の 2 表だけを自前表（b_run1 version 3）に"
PATCH_END = "print('Engine 2 own tables (#189):', _E2_OWN, flush=True)\n"
PATCH = PATCH_BEGIN + '''。推論の FP は元のまま ──
import glob as _e2g, json as _e2j, hashlib as _e2h
_E2_KSRC = '{ksrc_name}'
def _e2own(rel, sha):
    # kernel_sources の b_run1 の下にあり rel で終わるファイルがちょうど 1 つ。sha256 が違えば Engine 2 を起動する前に止める
    _h = sorted(p for p in _e2g.glob('/kaggle/input/**/' + rel, recursive=True)
                if _E2_KSRC in p.split('/')[:-1] and p.endswith('/' + rel))
    if len(_h) != 1:
        raise RuntimeError(('#189 own table not found exactly once under ' + _E2_KSRC, rel, _h))
    _s = _e2h.sha256(open(_h[0], 'rb').read()).hexdigest()
    if _s != sha:
        raise RuntimeError(('#189 own table sha256 mismatch', _h[0], _s, sha))
    return _h[0]
# #108 09:19Z の 2: 再実行では自前表の確認の失敗も Engine 2 の失敗と同じく v4g のリストへの戻りにする。commit では止める
_E2_FAIL = None
try:
    _E2_SIM = _e2own('{sim_rel}', '{sim_sha}')
    _E2_RT = _e2own('{rt_rel}', '{rt_sha}')
except Exception as _e2x:
    if not IS_RERUN:
        raise
    _E2_FAIL = ('own tables', repr(_e2x)); _E2_SIM = _E2_RT = None
_E2_OWN = dict(sim=_E2_SIM, sim_sha='{sim_sha}', rt=_E2_RT, rt_sha='{rt_sha}')
open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write({runner!r})
os.environ['E2_OWN'] = _e2j.dumps(_E2_OWN)
_E2_EXC = None   # 下の except で Engine 2 の例外の repr を残す
rr = None        # subprocess が終わらなかった（時間切れ・起動の失敗）ときは None のまま
''' + PATCH_END

RUN_ADD = ("    if _E2_FAIL is not None:\n"
           "        raise RuntimeError(('#189 own tables not verified; Engine 2 not started', _E2_FAIL))\n")

FALLBACK_NOTE = ("fallback to v4g lists (Engine 2 output unused); do not read this run as the #189 diagnostic "
                 "(#108 09:19Z item 2)")
EXC_ADD = ("\n    _E2_EXC = repr(e)\n"
           "# ── #189: Engine 2 の結果の確認。commit では止める。再実行では v4g のリストに戻る（#108 09:19Z の 2）──\n"
           f"_E2_RP = {REPORT!r}\n"
           "_e2why = []\n"
           "if _E2_FAIL is not None:\n"
           "    _e2why.append(_E2_FAIL)\n"
           "if _E2_EXC is not None:\n"
           "    _e2why.append(('exception', _E2_EXC))\n"
           "if rr is not None and rr.returncode != 0:\n"
           "    _e2why.append(('returncode', rr.returncode))\n"
           "try:\n"
           "    _e2rep = _e2j.load(open(_E2_RP))\n"
           "    assert isinstance(_e2rep, dict), type(_e2rep)\n"
           "except Exception as _e2x:\n"
           "    _e2rep = {}\n"
           "    _e2why.append(('report unreadable', repr(_e2x)))\n"
           "if _e2rep.get('status') != 'ok':\n"
           "    _e2why.append(('report status', _e2rep.get('status')))\n"
           "_e2act = sorted(set(pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id']).molecule_id.astype(str)))\n"
           "_e2have = set(map(str, ENG)) if isinstance(ENG, dict) else set()\n"
           "_e2miss = [_m for _m in _e2act if _m not in _e2have]\n"
           "if _e2miss:\n"
           "    _e2why.append(('ENG does not cover active molecules', len(_e2miss), len(_e2act)))\n"
           "if _e2why and not IS_RERUN:\n"
           "    raise AssertionError(('#189 own-table Engine 2 did not finish', _e2why))\n"
           "if _e2why:\n"
           "    ENG = {_m: dict(smiles=[], keys=[]) for _m in _e2act}\n"
           "    _e2rep.update(fallback=True, fallback_reasons=_e2why, fallback_molecules=len(ENG),\n"
           f"                  fallback_note={FALLBACK_NOTE!r})\n"
           "    print('ENGINE 2 FALLBACK (#189, rerun) -> v4g lists kept, empty Engine 2 lists for', len(ENG), 'molecules:', _e2why, flush=True)\n"
           "else:\n"
           "    _e2rep['fallback'] = False\n"
           "_e2j.dump(_e2rep, open(_E2_RP, 'w'), indent=1, default=str)\n"
           "print('Engine 2 own tables report:', {_k: _e2rep.get(_k) for _k in ('status', 'fallback', 'tables_sha256', 'molecules_listed', 'engine_secs')}, flush=True)")


def patch_frozen(frozen: dict) -> dict:
    cells = frozen["cells"]
    hit = [i for i, c in enumerate(cells) if RUN_MARK in "".join(c["source"])]
    assert len(hit) == 1 and hit[0] == CHANGED_FROZEN, hit
    s = "".join(cells[CHANGED_FROZEN]["source"])
    assert s.count(WRITE_MARK) == 1
    patch = PATCH.format(ksrc_name=KSRC_NAME, sim_sha=SIM_SHA, rt_sha=RT_SHA, sim_rel=SIM_REL, rt_rel=RT_REL,
                         runner=B_RUNNER)
    s = replace_once(s, RUN_MARK, patch + RUN_MARK + RUN_ADD, "engine 2 run")
    s = replace_once(s, EXC_OLD, EXC_OLD + EXC_ADD, "engine 2 except")
    compile(s, "frozen_cell_32", "exec")
    compile(B_RUNNER, "eng_runner_b", "exec")
    cells[CHANGED_FROZEN]["source"] = s
    return frozen


def push_source_bytes(nb: dict) -> int:
    """kaggle kernels push が送るソースの大きさの見積もり（出力を空にし、source を連結、json.dumps の既定）。"""
    nb2 = json.loads(json.dumps(nb))
    for c in nb2["cells"]:
        if isinstance(c.get("source"), list):
            c["source"] = "".join(c["source"])
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return len(json.dumps(nb2).encode())


def build(slug: str = "casmi26-lb487-btables", smoke_n: int | None = None) -> tuple[dict, dict, dict]:
    """(notebook, kernel-metadata, build_info) を返す。ファイルには書かない。"""
    raw_nb = (SRC / SRC_NB).read_bytes()
    assert hashlib.sha256(raw_nb).hexdigest().startswith(SRC_SHA), "0.487 写しのノートではない"
    nb = json.loads(raw_nb)
    orig = json.loads(raw_nb)
    c12 = "".join(nb["cells"][12]["source"])
    i, j, raw = split_frozen(c12)
    old_sha = hashlib.sha256(raw).hexdigest()
    assert f"FROZEN_SHA256 = '{old_sha}'" in c12
    frozen0 = json.loads(raw)
    frozen = patch_frozen(json.loads(raw))
    timing_cells = set()
    if smoke_n is not None:
        assert smoke_n > 12 and "timing" in slug, "時間測定の版は slug に timing を入れる（提出用の slug と分ける）"
        c30 = "".join(frozen["cells"][30]["source"])
        frozen["cells"][30]["source"] = replace_once(c30, "SMOKE_N = 12", f"SMOKE_N = {int(smoke_n)}", "smoke n")
        timing_cells.add(30)
    else:
        assert "timing" not in slug, "提出用の版の slug に timing を入れない"
    assert len(frozen["cells"]) == len(frozen0["cells"]) == FROZEN_CELLS
    for k, (a, b) in enumerate(zip(frozen0["cells"], frozen["cells"])):
        same = "".join(a["source"]) == "".join(b["source"])
        assert same == (k != CHANGED_FROZEN and k not in timing_cells), k
        if b["cell_type"] == "code":
            compile("".join(b["source"]), f"frozen_{k}", "exec")
    new_raw = json.dumps(frozen, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    c12 = c12[:i] + base64.b64encode(new_raw).decode() + c12[j:]
    c12 = replace_once(c12, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'", "frozen sha")
    nb["cells"][12]["source"] = c12
    c5 = "".join(nb["cells"][5]["source"])
    nb["cells"][5]["source"] = replace_once(c5, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'", "commit mode")
    cm = nb["metadata"].get("casmi", {})
    meta_changed = []
    if cm.get("frozen_source_sha256") == old_sha:
        cm["frozen_source_sha256"] = new_sha; meta_changed.append("casmi.frozen_source_sha256")
    if cm.get("upstream_casmi_metadata", {}).get("frozen_source_sha256") == old_sha:
        cm["upstream_casmi_metadata"]["frozen_source_sha256"] = new_sha
        meta_changed.append("casmi.upstream_casmi_metadata.frozen_source_sha256")
    for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])):
        same = "".join(a["source"]) == "".join(b["source"])
        assert same == (k not in CHANGED_OUTER), k
        if b["cell_type"] == "code" and k != 12:
            compile("".join(b["source"]), f"outer_{k}", "exec")
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"taichiiiii/{slug}", title=slug.replace("-", " "), is_private=True, code_file="notebook.ipynb")
    meta["kernel_sources"] = list(meta["kernel_sources"]) + [KSRC]
    size = push_source_bytes(nb)
    assert size < SOURCE_LIMIT, size
    rep = dict(frozen_sha_old=old_sha, frozen_sha_new=new_sha, changed_frozen_cell=CHANGED_FROZEN,
               changed_outer_cells=list(CHANGED_OUTER), changed_metadata=meta_changed,
               kernel_source=KSRC, sim_rel=SIM_REL, sim_sha=SIM_SHA, sim_size=SIM_SIZE,
               rt_rel=RT_REL, rt_sha=RT_SHA, rt_size=RT_SIZE,
               dataset_sources=meta["dataset_sources"], kernel_sources=meta["kernel_sources"],
               push_source_bytes=size)
    if smoke_n is not None:
        rep.update(timing_only=True, smoke_n=int(smoke_n),
                   do_not_submit="時間測定の版。提出しない（凍結セル 30 が提出版と違う）")
    return nb, meta, rep


def write(out_dir: Path, nb: dict, meta: dict, rep: dict) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_dir / "build_info.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="casmi26-lb487-btables")
    ap.add_argument("--out-dir", default=str(ROOT / "kernels" / "lb487_btables"))
    ap.add_argument("--smoke-n", type=int, default=None, help="時間測定の版: commit で通す分子数（提出しない。slug に timing が要る）")
    args = ap.parse_args()
    nb, meta, rep = build(args.slug, args.smoke_n)
    out = write(Path(args.out_dir), nb, meta, rep)
    print(json.dumps(dict(rep, out_dir=str(out)), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
