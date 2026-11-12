#!/usr/bin/env python3
"""
診断（#77、設計 docs/design/engine2-rebuild.md §6）: 0.487 写し（kernels/lb487）の Engine 2 を自前表に替えた
ノートブックを作る。

変更は 2 つだけ:
1. 本体: 凍結コード（外側セル 12 の FROZEN_B64、0.465 系統のノート 59 セル）のセル 32（Engine 2 の起動）だけを書き換える
   - セル 32 の中で、Engine 2 を起動する前に 2 表の場所を決めて確かめる（_e2own）:
     自前表の dataset（casmi26-engine2-own-tables）のディレクトリの下にある、引数で渡した相対パスのファイルだけを使い、
     ファイルの sha256 が引数の値と違えば例外。dataset の外にある同名のファイルは使わない
   - 元の eng_runner.py を書いた直後に、自前の eng_runner.py で上書きする。自前の runner は元の runner と同じ処理に加えて
     (a) 2 表を上で決めた明示パスで読み、もう一度パスと sha256 を確かめる（verify_own_tables。元の名前
         rank_train.npz / sim_rank_rows_nofp.npz は find させない）
     (b) prvsiyan ランカーの base を base_fs.blk_base_fs に替える（score_molecules の ["base"], use_fp=True の呼び出しだけを
         base_fs.score_base_fs に置き換え。zlog_fs は fold-safe FPNet 5 本のロジットの平均〈CPU〉、cfp_fs は 10,226 ビット）
     (c) レポート /kaggle/working/engine2_owntables_report.json（status、読んだ表のパスと sha256、blk_base_fs を通った分子数）
   - Engine 2 が落ちたら、元のように v4g のリストへ黙って戻らず止める（レポートの status が ok でなければ例外）
   - base_fs.py は生成コードの dataset（taichiiiii/casmi26-engine2-build-code）から ENG_DIR に写す
   - 凍結コードの宣言 sha256（外側セル 12 の FROZEN_SHA256、notebook メタデータ casmi.frozen_source_sha256）を新しい値にする
2. 採点に影響しない: 外側セル 5 の COMMIT_MODE = 'fast' を 'smoke' にする（commit で見える 12 分子が全段を通る）

dataset_sources は元の 10 件を残し、自前表と生成コードの 2 件を足す。kernel-metadata.json の id_no は消す。

    python scripts/submit/make_lb487_owntables.py --sim-sha <64 桁> --rt-sha <64 桁> \\
        [--sim-rel own_sim_rank_rows.npz] [--rt-rel own_rank_train.npz] [--out-dir kernels/lb487_owntables]

仮の sha256（例: 試走の表の値）で作るときは --provisional を付ける。build_info.json に provisional: true が入り、
リポジトリの kernels/ の下には書けない（仮のノートを push しないため。scratch などに出す）。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kernels" / "lb487"
SRC_NB = "casmi26-lb487-copy.ipynb"
SRC_SHA = "380e2c820c2a47cfad26"       # ノート全体の sha256 の先頭（0.487 写し）
OWN_DS = "taichiiiii/casmi26-engine2-own-tables"
CODE_DS = "taichiiiii/casmi26-engine2-build-code"
OWN_DS_NAME = OWN_DS.split("/")[1]
FROZEN_CELLS = 59
CHANGED_FROZEN = 32
CHANGED_OUTER = (5, 12)
RUN_MARK = "ENG = {}\ntry:\n    _t0 = time.time()\n"
WRITE_MARK = "open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write("
PATCH_BEGIN = "# ── #77（診断）: Engine 2 の 2 表を自前表に"
PATCH_END = "print('Engine 2 own tables (#77):', _E2_OWN, flush=True)\n"

OWN_RUNNER = r'''"""Engine 2 with own tables (#77): same as the original eng_runner.py, but the two tables are read from explicit
paths inside taichiiiii/casmi26-engine2-own-tables (sha256 checked) and the prvsiyan ranker base is base_fs.blk_base_fs."""
import os, sys, json, time, hashlib, glob
import numpy as np
OWN_DS_NAME = 'casmi26-engine2-own-tables'


def verify_own_tables(own):
    """own: {'sim': path, 'sim_sha': hex, 'rt': path, 'rt_sha': hex} -> {'sim': sha, 'rt': sha}. Raises on any mismatch."""
    sha = {}
    for k in ('sim', 'rt'):
        p = own[k]
        if OWN_DS_NAME not in os.path.normpath(p).split(os.sep)[:-1] or not os.path.basename(p).startswith('own_'):
            raise ValueError(('#77 own table outside ' + OWN_DS_NAME, k, p))
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        if h != own[k + '_sha']:
            raise ValueError(('#77 own table sha256 mismatch', k, p, h, own[k + '_sha']))
        sha[k] = h
    return sha


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    OWN = json.loads(os.environ['E2_OWN'])
    REP = dict(status='started', own=OWN)
    RP = '/kaggle/working/engine2_owntables_report.json'
    json.dump(REP, open(RP, 'w'), indent=1)
    import casmi_engine as E
    import base_fs
    _orig_find = E.find
    def _find(n):
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        assert 'rank_train' not in n and 'sim_rank_rows' not in n, ('original Engine 2 table requested', n)
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    print('engine fp models', fp_models, flush=True)
    sha = verify_own_tables(OWN)
    REP.update(tables_read={k: OWN[k] for k in ('sim', 'rt')}, tables_sha256=sha)
    base_fs.set_casmi_code_dir(OWN['casmi_code'])
    import pandas as pd
    T0 = time.time()
    banks = [base_fs.load_bank(p, 'cpu') for p in OWN['fs_models']]
    zfs = base_fs.zlog_fs_for_test(pd.read_parquet(test), banks)
    bits = np.load(OWN['fs_bits'])
    assert len(bits) == 10226
    REP.update(zlog_fs_molecules=len(zfs), zlog_fs_secs=round(time.time() - T0, 1), fs_models=OWN['fs_models'])
    print('zlog_fs (fold-safe x5)', len(zfs), f'{time.time()-T0:.0f}s', flush=True)
    _orig_score = E.score_molecules
    CALLS = []
    def _score(recs, rankers, blocks, use_fp):
        CALLS.append([list(blocks), bool(use_fp)])
        if list(blocks) == ['base'] and use_fp:
            cache = base_fs.FsFpCache(lambda i: E.POOL['smiles'][i], bits)
            r = base_fs.score_base_fs(recs, rankers, zfs, cache.get, REP)
            REP['fs_fp_failed'] = cache.failed
            return r
        return _orig_score(recs, rankers, blocks, use_fp)
    E.score_molecules = _score
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, OWN['sim'], OWN['rt'], fp_models, workers=os.cpu_count(), w_pv=0.88)
    REP['score_calls'] = CALLS
    REP['molecules_with_candidates'] = int(sum(1 for r in recs if len(r['cand'])))
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
    assert CALLS == [[['base', 'libx', 'anapol', 'fragx'], False], [['base'], True]], CALLS
    REP['status'] = 'ok'
    json.dump(REP, open(RP, 'w'), indent=1, default=str)
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', 'base_fs molecules', REP.get('base_fs_molecules'), flush=True)
'''

PATCH = PATCH_BEGIN + '''、prvsiyan ランカーの base を blk_base_fs に替える（設計 §6）──
import glob as _e2g, shutil as _e2s, json as _e2j, hashlib as _e2h
_E2_DS = '{own_ds_name}'
def _e2one(pat, must):
    _h = sorted((p for p in _e2g.glob('/kaggle/input/**/' + pat, recursive=True) if must in p), key=len)
    assert _h, (pat, must)
    return _h[0]
def _e2own(rel, sha):
    # 自前表の dataset のディレクトリの下にある rel だけを使う。sha256 が違えば Engine 2 を起動する前に止める
    _h = sorted((p for p in _e2g.glob('/kaggle/input/**/' + rel, recursive=True)
                 if _E2_DS in p.split('/')[:-1] and p.endswith('/' + rel)), key=len)
    if not _h:
        raise RuntimeError(('#77 own table not found under ' + _E2_DS, rel))
    _s = _e2h.sha256(open(_h[0], 'rb').read()).hexdigest()
    if _s != sha:
        raise RuntimeError(('#77 own table sha256 mismatch', _h[0], _s, sha))
    return _h[0]
_E2_OWN = dict(sim=_e2own('{sim_rel}', '{sim_sha}'), sim_sha='{sim_sha}',
               rt=_e2own('{rt_rel}', '{rt_sha}'), rt_sha='{rt_sha}',
               casmi_code=os.path.dirname(os.path.dirname(_e2one('casmi/chem.py', 'casmi26-v3-models/code'))),
               fs_models=[_e2one(f'fpnet_fold{{_k}}.pt', 'casmi26-fold-safe-fpnet') for _k in range(5)],
               fs_bits=os.path.join(os.path.dirname(_e2one('fpnet_fold0.pt', 'casmi26-fold-safe-fpnet')), 'fp_bits.npy'))
_e2s.copy2(_e2one('base_fs.py', 'casmi26-engine2-build-code'), os.path.join(ENG_DIR, 'base_fs.py'))
open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write({runner!r})
os.environ['E2_OWN'] = _e2j.dumps(_E2_OWN)
''' + PATCH_END


EXC_OLD = "except Exception as e:\n    print('ENGINE FAILED -> v4g lists kept:', repr(e))"
# 自前表の診断では、Engine 2 が落ちて v4g のリストに黙って戻ることを許さない（commit でも再実行でも止める）
EXC_ADD = ("\n_e2rep = _e2j.load(open('/kaggle/working/engine2_owntables_report.json'))\n"
           "assert _e2rep.get('status') == 'ok' and ENG, ('#77 own-table Engine 2 did not finish', _e2rep.get('status'))\n"
           "print('Engine 2 own tables report:', {_k: _e2rep.get(_k) for _k in ('status', 'tables_sha256', 'base_fs_molecules', 'zlog_fs_molecules')}, flush=True)")

HEX64 = re.compile(r"[0-9a-f]{64}")
REL_OK = re.compile(r"[A-Za-z0-9_./-]+")


def replace_once(s: str, old: str, new: str, what: str) -> str:
    assert s.count(old) == 1, (what, s.count(old))
    return s.replace(old, new)


def check_args(sim_sha: str, rt_sha: str, sim_rel: str, rt_rel: str) -> None:
    for name, v in (("sim_sha", sim_sha), ("rt_sha", rt_sha)):
        if not HEX64.fullmatch(v):
            raise ValueError(f"{name} は 64 桁の小文字 16 進: {v!r}")
    for name, v in (("sim_rel", sim_rel), ("rt_rel", rt_rel)):
        parts = v.split("/")
        if (not REL_OK.fullmatch(v) or v.startswith("/") or ".." in parts or "" in parts
                or not parts[-1].startswith("own_") or not v.endswith(".npz")):
            raise ValueError(f"{name} は自前表の dataset の中の相対パス（own_*.npz）: {v!r}")
    if sim_rel == rt_rel:
        raise ValueError("sim_rel と rt_rel が同じ")


def patch_frozen(frozen: dict, sim_sha: str, rt_sha: str, sim_rel: str, rt_rel: str) -> tuple[dict, int]:
    cells = frozen["cells"]
    hit = [i for i, c in enumerate(cells) if RUN_MARK in "".join(c["source"])]
    assert len(hit) == 1 and hit[0] == CHANGED_FROZEN, hit
    s = "".join(cells[CHANGED_FROZEN]["source"])
    assert s.count(WRITE_MARK) == 1
    patch = PATCH.format(own_ds_name=OWN_DS_NAME, sim_sha=sim_sha, rt_sha=rt_sha, sim_rel=sim_rel, rt_rel=rt_rel,
                         runner=OWN_RUNNER)
    s = replace_once(s, RUN_MARK, patch + RUN_MARK, "engine 2 run")
    s = replace_once(s, EXC_OLD, EXC_OLD + EXC_ADD, "engine 2 except")
    compile(s, "frozen_cell_32", "exec")
    compile(OWN_RUNNER, "eng_runner_own", "exec")
    cells[CHANGED_FROZEN]["source"] = s
    return frozen, CHANGED_FROZEN


def split_frozen(c12: str) -> tuple[int, int, bytes]:
    """外側セル 12 の FROZEN_B64 の中身の位置 (i, j) と、復号した凍結コード（JSON のバイト列）。"""
    i = c12.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    j = c12.index("'", i)
    return i, j, base64.b64decode(c12[i:j])


def build(sim_sha: str, rt_sha: str, sim_rel: str = "own_sim_rank_rows.npz", rt_rel: str = "own_rank_train.npz",
          slug: str = "casmi26-lb487-owntables") -> tuple[dict, dict, dict]:
    """(notebook, kernel-metadata, build_info) を返す。ファイルには書かない。"""
    check_args(sim_sha, rt_sha, sim_rel, rt_rel)
    raw_nb = (SRC / SRC_NB).read_bytes()
    assert hashlib.sha256(raw_nb).hexdigest().startswith(SRC_SHA), "0.487 写しのノートではない"
    nb = json.loads(raw_nb)
    orig = json.loads(raw_nb)
    c12 = "".join(nb["cells"][12]["source"])
    i, j, raw = split_frozen(c12)
    old_sha = hashlib.sha256(raw).hexdigest()
    assert f"FROZEN_SHA256 = '{old_sha}'" in c12
    frozen0 = json.loads(raw)
    frozen, changed = patch_frozen(json.loads(raw), sim_sha, rt_sha, sim_rel, rt_rel)
    # 変更セル以外は sha256 で一致、全セル compile
    assert len(frozen["cells"]) == len(frozen0["cells"]) == FROZEN_CELLS
    for k, (a, b) in enumerate(zip(frozen0["cells"], frozen["cells"])):
        ha = hashlib.sha256("".join(a["source"]).encode()).hexdigest(); hb = hashlib.sha256("".join(b["source"]).encode()).hexdigest()
        assert (ha == hb) == (k != changed), k
        if b["cell_type"] == "code":
            compile("".join(b["source"]), f"frozen_{k}", "exec")
    new_raw = json.dumps(frozen, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    c12 = c12[:i] + base64.b64encode(new_raw).decode() + c12[j:]
    c12 = replace_once(c12, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'", "frozen sha")
    nb["cells"][12]["source"] = c12
    c5 = "".join(nb["cells"][5]["source"])
    c5 = replace_once(c5, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'", "commit mode")
    nb["cells"][5]["source"] = c5
    cm = nb["metadata"].get("casmi", {})
    meta_changed = []
    if cm.get("frozen_source_sha256") == old_sha:
        cm["frozen_source_sha256"] = new_sha; meta_changed.append("casmi.frozen_source_sha256")
    if cm.get("upstream_casmi_metadata", {}).get("frozen_source_sha256") == old_sha:
        cm["upstream_casmi_metadata"]["frozen_source_sha256"] = new_sha
        meta_changed.append("casmi.upstream_casmi_metadata.frozen_source_sha256")
    # 外側のセル: 5 と 12 以外は元と同一
    for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])):
        same = "".join(a["source"]) == "".join(b["source"])
        assert same == (k not in CHANGED_OUTER), k
        if b["cell_type"] == "code" and k != 12:
            compile("".join(b["source"]), f"outer_{k}", "exec")
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"taichiiiii/{slug}", title=slug.replace("-", " "), is_private=True, code_file="notebook.ipynb")
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [OWN_DS, CODE_DS]
    rep = dict(frozen_sha_old=old_sha, frozen_sha_new=new_sha, changed_frozen_cell=changed,
               changed_outer_cells=list(CHANGED_OUTER), changed_metadata=meta_changed,
               sim_rel=sim_rel, sim_sha=sim_sha, rt_rel=rt_rel, rt_sha=rt_sha, dataset_sources=meta["dataset_sources"])
    return nb, meta, rep


def write(out_dir: Path, nb: dict, meta: dict, rep: dict, provisional: bool) -> Path:
    out_dir = Path(out_dir).resolve()
    kdir = (ROOT / "kernels").resolve()
    if provisional and (out_dir == kdir or kdir in out_dir.parents):
        raise SystemExit("仮の sha256 のノートはリポジトリの kernels/ の下に書かない（--out-dir を scratch などにする）")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    rep = dict(rep, provisional=bool(provisional))
    (out_dir / "build_info.json").write_text(json.dumps(rep, indent=1))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-sha", required=True, help="own_sim_rank_rows.npz のファイルの sha256")
    ap.add_argument("--rt-sha", required=True, help="own_rank_train.npz のファイルの sha256")
    ap.add_argument("--sim-rel", default="own_sim_rank_rows.npz", help="自前表の dataset の中の相対パス")
    ap.add_argument("--rt-rel", default="own_rank_train.npz", help="自前表の dataset の中の相対パス")
    ap.add_argument("--slug", default="casmi26-lb487-owntables")
    ap.add_argument("--out-dir", default=str(ROOT / "kernels" / "lb487_owntables"))
    ap.add_argument("--provisional", action="store_true", help="仮の sha256 で作る（kernels/ の下には書けない）")
    args = ap.parse_args()
    nb, meta, rep = build(args.sim_sha, args.rt_sha, args.sim_rel, args.rt_rel, args.slug)
    out = write(Path(args.out_dir), nb, meta, rep, args.provisional)
    print(json.dumps(dict(rep, out_dir=str(out), provisional=args.provisional), indent=1))


if __name__ == "__main__":
    main()
