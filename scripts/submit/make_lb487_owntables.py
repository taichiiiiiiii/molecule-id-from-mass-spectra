#!/usr/bin/env python3
"""
診断（計画 2026-11-12 夜 項目 3、設計 docs/design/engine2-rebuild.md §6）: 0.487 写し（kernels/lb487）の Engine 2 を自前表に替えた
ノートブックを作る。

変更は 2 つだけ:
1. 本体: 凍結コード（外側セル 12 の FROZEN_B64、0.465 系統のノート 59 セル）のセル 32（Engine 2 の起動）だけを書き換える
   - 元の eng_runner.py を書いた直後に、自前の eng_runner.py で上書きする。自前の runner は元の runner と同じ処理に加えて
     (a) 2 表を明示パス（taichiiiii/casmi26-engine2-own-tables の own_sim_rank_rows.npz / own_rank_train.npz）で読み、
         パスと sha256 を assert（元の名前 rank_train.npz / sim_rank_rows_nofp.npz は find させない）
     (b) prvsiyan ランカーの base を base_fs.blk_base_fs に替える（score_molecules の ["base"], use_fp=True の呼び出しだけを
         base_fs.score_base_fs に置き換え。zlog_fs は fold-safe FPNet 5 本のロジットの平均〈CPU〉、cfp_fs は 10,226 ビット）
     (c) レポート /kaggle/working/engine2_owntables_report.json（status、読んだ表のパスと sha256、blk_base_fs を通った分子数）
   - base_fs.py は生成コードの dataset（taichiiiii/casmi26-engine2-build-code）から ENG_DIR に写す
   - 凍結コードの宣言 sha256（外側セル 12 の FROZEN_SHA256、notebook メタデータ casmi.frozen_source_sha256）を新しい値にする
2. 採点に影響しない: 外側セル 5 の COMMIT_MODE = 'fast' を 'smoke' にする（commit で見える 12 分子が全段を通る）

dataset_sources は元の 10 件を残し、自前表と生成コードの 2 件を足す。

    python scripts/submit/make_lb487_owntables.py --sim-sha <sha> --rt-sha <sha> [--slug casmi26-lb487-owntables]
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

SRC = Path("kernels/lb487")
SRC_NB = "casmi26-lb487-copy.ipynb"
SRC_SHA = "380e2c820c2a47cfad26"       # ノート全体の sha256 の先頭（0.487 写し）
OWN_DS = "taichiiiii/casmi26-engine2-own-tables"
CODE_DS = "taichiiiii/casmi26-engine2-build-code"
RUN_MARK = "ENG = {}\ntry:\n    _t0 = time.time()\n"
WRITE_MARK = "open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write("

OWN_RUNNER = r'''"""Engine 2 with own tables (#77): same as the original eng_runner.py, but the two tables are read from explicit
paths of taichiiiii/casmi26-engine2-own-tables (sha256 asserted) and the prvsiyan ranker base is base_fs.blk_base_fs."""
import os, sys, json, time, hashlib, glob
import numpy as np
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
    sha = {}
    for k in ('sim', 'rt'):
        p = OWN[k]
        assert 'casmi26-engine2-own-tables' in p and os.path.basename(p).startswith('own_'), p
        sha[k] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        assert sha[k] == OWN[k + '_sha'], (k, sha[k], OWN[k + '_sha'])
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

PATCH = '''# ── #77（診断）: Engine 2 の 2 表を自前表に、prvsiyan ランカーの base を blk_base_fs に替える（設計 §6）──
import glob as _e2g, shutil as _e2s, json as _e2j
def _e2one(pat, must):
    _h = sorted((p for p in _e2g.glob('/kaggle/input/**/' + pat, recursive=True) if must in p), key=len)
    assert _h, (pat, must)
    return _h[0]
_E2_OWN = dict(sim=_e2one('own_sim_rank_rows.npz', 'casmi26-engine2-own-tables'), sim_sha='{sim_sha}',
               rt=_e2one('own_rank_train.npz', 'casmi26-engine2-own-tables'), rt_sha='{rt_sha}',
               casmi_code=os.path.dirname(os.path.dirname(_e2one('casmi/chem.py', 'casmi26-v3-models/code'))),
               fs_models=[_e2one(f'fpnet_fold{{_k}}.pt', 'casmi26-fold-safe-fpnet') for _k in range(5)],
               fs_bits=os.path.join(os.path.dirname(_e2one('fpnet_fold0.pt', 'casmi26-fold-safe-fpnet')), 'fp_bits.npy'))
_e2s.copy2(_e2one('base_fs.py', 'casmi26-engine2-build-code'), os.path.join(ENG_DIR, 'base_fs.py'))
open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w').write({runner!r})
os.environ['E2_OWN'] = _e2j.dumps(_E2_OWN)
print('Engine 2 own tables (#77):', _E2_OWN, flush=True)
'''


EXC_OLD = "except Exception as e:\n    print('ENGINE FAILED -> v4g lists kept:', repr(e))"
# 自前表の診断では、Engine 2 が落ちて v4g のリストに黙って戻ることを許さない（commit でも再実行でも止める）
EXC_ADD = ("\n_e2rep = _e2j.load(open('/kaggle/working/engine2_owntables_report.json'))\n"
           "assert _e2rep.get('status') == 'ok' and ENG, ('#77 own-table Engine 2 did not finish', _e2rep.get('status'))\n"
           "print('Engine 2 own tables report:', {_k: _e2rep.get(_k) for _k in ('status', 'tables_sha256', 'base_fs_molecules', 'zlog_fs_molecules')}, flush=True)")


def replace_once(s: str, old: str, new: str, what: str) -> str:
    assert s.count(old) == 1, (what, s.count(old))
    return s.replace(old, new)


def patch_frozen(frozen: dict, sim_sha: str, rt_sha: str) -> tuple[dict, int]:
    cells = frozen["cells"]
    hit = [i for i, c in enumerate(cells) if RUN_MARK in "".join(c["source"])]
    assert len(hit) == 1 and hit[0] == 32, hit
    s = "".join(cells[32]["source"])
    assert s.count(WRITE_MARK) == 1
    patch = PATCH.format(sim_sha=sim_sha, rt_sha=rt_sha, runner=OWN_RUNNER)
    s = replace_once(s, RUN_MARK, patch + RUN_MARK, "engine 2 run")
    s = replace_once(s, EXC_OLD, EXC_OLD + EXC_ADD, "engine 2 except")
    compile(s, "frozen_cell_32", "exec")
    compile(OWN_RUNNER, "eng_runner_own", "exec")
    cells[32]["source"] = s
    return frozen, 32


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-sha", required=True)
    ap.add_argument("--rt-sha", required=True)
    ap.add_argument("--slug", default="casmi26-lb487-owntables")
    args = ap.parse_args()
    raw_nb = (SRC / SRC_NB).read_bytes()
    assert hashlib.sha256(raw_nb).hexdigest().startswith(SRC_SHA), "0.487 写しのノートではない"
    nb = json.loads(raw_nb)
    orig = json.loads(raw_nb)
    c12 = "".join(nb["cells"][12]["source"])
    i = c12.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    j = c12.index("'", i)
    raw = base64.b64decode(c12[i:j])
    old_sha = hashlib.sha256(raw).hexdigest()
    assert f"FROZEN_SHA256 = '{old_sha}'" in c12
    frozen0 = json.loads(raw)
    frozen, changed = patch_frozen(json.loads(raw), args.sim_sha, args.rt_sha)
    # 変更セル以外は sha256 で一致、全セル compile
    for k, (a, b) in enumerate(zip(frozen0["cells"], frozen["cells"])):
        ha = hashlib.sha256("".join(a["source"]).encode()).hexdigest(); hb = hashlib.sha256("".join(b["source"]).encode()).hexdigest()
        assert (ha == hb) == (k != changed), k
        if b["cell_type"] == "code":
            compile("".join(b["source"]), f"frozen_{k}", "exec")
    assert len(frozen["cells"]) == len(frozen0["cells"]) == 59
    new_raw = json.dumps(frozen, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    c12 = c12[:i] + base64.b64encode(new_raw).decode() + c12[j:]
    c12 = replace_once(c12, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'", "frozen sha")
    nb["cells"][12]["source"] = c12
    c5 = "".join(nb["cells"][5]["source"])
    c5 = replace_once(c5, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'", "commit mode")
    nb["cells"][5]["source"] = c5
    cm = nb["metadata"].get("casmi", {})
    if cm.get("frozen_source_sha256") == old_sha:
        cm["frozen_source_sha256"] = new_sha
    if cm.get("upstream_casmi_metadata", {}).get("frozen_source_sha256") == old_sha:
        cm["upstream_casmi_metadata"]["frozen_source_sha256"] = new_sha
    # 外側のセル: 5 と 12 以外は元と同一
    for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])):
        same = "".join(a["source"]) == "".join(b["source"])
        assert same == (k not in (5, 12)), k
        if b["cell_type"] == "code" and k != 12:
            compile("".join(b["source"]), f"outer_{k}", "exec")
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"taichiiiii/{args.slug}", title=args.slug.replace("-", " "), is_private=True, code_file="notebook.ipynb")
    meta["dataset_sources"] = list(meta["dataset_sources"]) + [OWN_DS, CODE_DS]
    dst = Path("kernels") / args.slug.replace("-", "_")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    rep = dict(frozen_sha_old=old_sha, frozen_sha_new=new_sha, changed_frozen_cell=changed, changed_outer_cells=[5, 12],
               sim_sha=args.sim_sha, rt_sha=args.rt_sha, dataset_sources=meta["dataset_sources"])
    (dst / "build_info.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
