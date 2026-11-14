#!/usr/bin/env python3
"""
0.487 写しから、fold-safe の部品に差し替えて v3 のホールドアウトを test として回すノートを作る生成スクリプト（骨組み）。
Issue #114、設計 docs/design/lineage487-foldsafe-eval.md §3.1・§7 の 3。

いまは骨組みだけ（2 日目に §2 の Tier 0 の差し替えを PATCHES に足す）。ここにあるもの:
  - replace_exactly_once : どの置き換えも「ちょうど 1 回当たる」ことを確かめる
  - get_frozen / set_frozen : 外側のセル 12 の FROZEN_B64（凍結ノートの JSON を base64 にしたもの）の復号と再エンコード、
                              FROZEN_SHA256 の書き換え
  - compile_all : 外側の全コードセル、MODULES の入れ子のモジュール、凍結ノートの全コードセルを compile()（構文だけ）
  - FORBIDDEN / GUARD_CELL : 読み込みの監視。torch.load・np.load・pickle.load・joblib.load・lgb.Booster を包み、禁止の名前を
                             読んだら例外、読んだファイルを全部 opened_files.json に書く（§3.1）。
                             lgb.Booster は model_file を位置引数（3 番目）でもキーワードでも見る。
                             既知の穴（2 日目に塞ぐ）: 監視のセルより前に `from torch import load` などで関数そのものを取った
                             名前は包めない（モジュールの属性を差し替えるだけなので）。セル 13 の前に入れるので凍結コード（F c*）の
                             import は包んだ後になるが、外側のセル 0〜12 で取った参照は対象外。open() も包まない（§3.1）
  - apply_patches : Patch の一覧を当て、変えたセルの一覧を返す

元のノート: kernels/lb487/casmi26-lb487-copy.ipynb（origin/feat/77-lb487-owntables の c3d9992、sha256 380e2c82…）。

    .venv/bin/python scripts/cv/make_lineage_fs_kernel.py --dry-run [--base <ipynb>]
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import fnmatch
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "c3d9992"
BASE_PATH = "kernels/lb487/casmi26-lb487-copy.ipynb"
BASE_SHA256 = "380e2c820c2a47cfad268b75cdd0dfc564350b599b9b052a1915d292987113c5"
FROZEN_CELL = 12                     # 外側のセル番号（FROZEN_B64 がある）
FROZEN_SHA256_BASE = "d1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88"
_B64_RX = re.compile(r"^FROZEN_B64 = '([A-Za-z0-9+/=]*)'$", re.M)
_SHA_RX = re.compile(r"^FROZEN_SHA256 = '([0-9a-f]{64})'$", re.M)

# §3.1 の禁止の名前。(basename のパターン, 許すパスの部分文字列)。許すパスに当たれば禁止しない（#54 の重みも final_model.pt）
FORBIDDEN: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("fpnet_full1.pt", ()),
    ("fpnet_0.pt", ()),               # v3-models・v4b の両方
    ("fpnet_1.pt", ()),
    ("fe_A.pt", ()), ("fe_B.pt", ()), ("fe_D.pt", ()),
    ("final_model.pt", ("casmi26-sanpier-fold0", "casmi_spectrum_model_fold0")),   # 公開 v1・v80 は禁止、#54 は許す
    ("fp_single_s2.pt", ()), ("fp_merged_m1.pt", ()),
    ("rank_train.npz", ()), ("sim_rank_rows*.npz", ()),
    ("fpnet_fold1.pt", ()), ("fpnet_fold2.pt", ()), ("fpnet_fold3.pt", ()), ("fpnet_fold4.pt", ()),
)


def is_forbidden(path: str, forbidden=FORBIDDEN) -> bool:
    p = str(path)
    name = p.replace("\\", "/").rsplit("/", 1)[-1]
    for pat, allow in forbidden:
        if fnmatch.fnmatchcase(name, pat) and not any(a in p for a in allow):
            return True
    return False


GUARD_CELL = r'''
# ---- Issue #114: load guard (forbidden weights / tables raise; every load is logged) ----
import builtins as _bi, json as _json, os as _os, pickle as _pickle, functools as _ft
import numpy as _np
LINEAGE_FORBIDDEN = __FORBIDDEN__
OPENED_FILES, OPENED_FORBIDDEN = [], []
def _lineage_is_forbidden(path):
    import fnmatch as _fn
    p = str(path); name = p.replace('\\', '/').rsplit('/', 1)[-1]
    return any(_fn.fnmatchcase(name, pat) and not any(a in p for a in allow) for pat, allow in LINEAGE_FORBIDDEN)
def _lineage_record(kind, path):
    if isinstance(path, (str, bytes, _os.PathLike)):
        p = _os.fsdecode(path)
    else:
        p = getattr(path, 'name', None)
        if not isinstance(p, str):
            return
    OPENED_FILES.append(dict(kind=kind, path=p))
    if _lineage_is_forbidden(p):
        OPENED_FORBIDDEN.append(p)
        _lineage_dump()
        raise RuntimeError(f'LINEAGE GUARD: forbidden load {kind} {p}')
def _lineage_dump(path='/kaggle/working/opened_files.json'):
    try:
        with open(path, 'w') as f:
            _json.dump(dict(opened=OPENED_FILES, opened_forbidden=OPENED_FORBIDDEN), f, indent=1)
    except OSError:
        pass
def _lineage_wrap(mod, attr, kind, key='file'):
    orig = getattr(mod, attr)
    if getattr(orig, '_lineage_wrapped', False):
        return
    @_ft.wraps(orig)
    def w(*a, **k):
        src = a[0] if a else k.get(key)
        _lineage_record(kind, src)
        return orig(*a, **k)
    w._lineage_wrapped = True
    setattr(mod, attr, w)
_lineage_wrap(_np, 'load', 'np.load')
_lineage_wrap(_pickle, 'load', 'pickle.load')
try:
    import joblib as _joblib
    _lineage_wrap(_joblib, 'load', 'joblib.load', key='filename')
except ImportError:
    pass
try:
    import torch as _torch
    _lineage_wrap(_torch, 'load', 'torch.load', key='f')
except ImportError:
    pass
try:
    import lightgbm as _lgb
    _OrigBooster = _lgb.Booster
    if not getattr(_OrigBooster, '_lineage_wrapped', False):
        class _GuardBooster(_OrigBooster):
            _lineage_wrapped = True
            def __init__(self, *a, **k):
                mf = k.get('model_file', a[2] if len(a) > 2 else None)   # Booster(params, train_set, model_file, ...)
                if mf is not None:
                    _lineage_record('lgb.Booster', mf)
                super().__init__(*a, **k)
        _lgb.Booster = _GuardBooster
except ImportError:
    pass
'''


def guard_cell_source(forbidden=FORBIDDEN) -> str:
    return GUARD_CELL.replace("__FORBIDDEN__", repr(tuple((p, tuple(a)) for p, a in forbidden))).strip("\n") + "\n"


@dataclass(frozen=True)
class Patch:
    where: str          # "outer" | "frozen"
    cell: int           # 外側のセル番号、または凍結ノートのセル番号（F cNN の NN）
    old: str
    new: str
    label: str          # §2 の行番号など


# 2 日目に §2 の Tier 0 の差し替えを足す（F c37 の FP バンクと model_views、F c34 の PubChem バンク、F c32 の Engine 2 の表、
# 窓のパッチ、§4 の結果で F c36 の固定表と pin・in_train_structure の行、外側のセル 15 の _swap_to_ours）。
PATCHES: tuple[Patch, ...] = ()


def replace_exactly_once(src: str, old: str, new: str, label: str = "") -> str:
    n = src.count(old)
    if n != 1:
        raise ValueError(f"置き換え {label!r}: {n} 回当たった（ちょうど 1 回のはず）")
    return src.replace(old, new, 1)


def cell_source(cell: dict) -> str:
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def set_cell_source(cell: dict, src: str) -> None:
    cell["source"] = src.splitlines(keepends=True)


def get_frozen(nb: dict, cell: int = FROZEN_CELL) -> tuple[dict, str]:
    """外側のセルから凍結ノート（dict）と、その元の sha256 を返す。FROZEN_B64 と FROZEN_SHA256 はちょうど 1 つずつのはず。"""
    src = cell_source(nb["cells"][cell])
    b, h = _B64_RX.findall(src), _SHA_RX.findall(src)
    if len(b) != 1 or len(h) != 1:
        raise ValueError(f"セル {cell}: FROZEN_B64 {len(b)} 個、FROZEN_SHA256 {len(h)} 個（1 個ずつのはず）")
    raw = base64.b64decode(b[0])
    if hashlib.sha256(raw).hexdigest() != h[0]:
        raise ValueError("FROZEN_B64 の sha256 が FROZEN_SHA256 と違う")
    return json.loads(raw), h[0]


def encode_frozen(inner: dict) -> bytes:
    """元の FROZEN_B64 と同じ書き方（ensure_ascii・区切り詰め）。差し替え無しなら元のバイト列に戻る（c3d9992 で確認）。"""
    return json.dumps(inner, ensure_ascii=True, separators=(",", ":")).encode()


def set_frozen(nb: dict, inner: dict, cell: int = FROZEN_CELL) -> str:
    """凍結ノートを再エンコードして FROZEN_B64 と FROZEN_SHA256 を書き換える。新しい sha256 を返す。"""
    raw = encode_frozen(inner)
    h = hashlib.sha256(raw).hexdigest()
    src = cell_source(nb["cells"][cell])
    src = _B64_RX.sub(lambda m: f"FROZEN_B64 = '{base64.b64encode(raw).decode()}'", src, count=1)
    src = _SHA_RX.sub(lambda m: f"FROZEN_SHA256 = '{h}'", src, count=1)
    set_cell_source(nb["cells"][cell], src)
    return h


def _strip_magics(src: str) -> str:
    return "\n".join(("pass  # " + l) if l.lstrip().startswith(("!", "%")) else l for l in src.splitlines())


def nested_modules(src: str) -> dict[str, str]:
    """`MODULES['name'] = '<source>'` の形の入れ子のモジュールの中身（ast で読むだけ、実行しない）。"""
    out = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
            t = node.targets[0]
            if isinstance(t.value, ast.Name) and t.value.id == "MODULES" and isinstance(node.value, ast.Constant) \
                    and isinstance(node.value.value, str) and isinstance(t.slice, ast.Constant):
                out[str(t.slice.value)] = node.value.value
    return out


def compile_all(nb: dict, frozen_cell: int | None = FROZEN_CELL) -> dict:
    """全部 compile()。失敗したら SyntaxError をそのまま投げる。数を返す。"""
    n_outer = n_mod = n_frozen = 0
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = _strip_magics(cell_source(c))
        compile(src, f"<outer cell {i}>", "exec"); n_outer += 1
        for name, msrc in nested_modules(src).items():
            compile(msrc, f"<module {name} in cell {i}>", "exec"); n_mod += 1
    if frozen_cell is not None:
        inner, _ = get_frozen(nb, frozen_cell)
        for j, c in enumerate(inner["cells"]):
            if c.get("cell_type") == "code":
                compile(_strip_magics(cell_source(c)), f"<frozen c{j:02d}>", "exec"); n_frozen += 1
    return dict(outer=n_outer, modules=n_mod, frozen=n_frozen)


def apply_patches(nb: dict, patches=PATCHES, frozen_cell: int = FROZEN_CELL) -> tuple[dict, dict]:
    """patches を当てた新しいノートと報告（変えたセル、凍結ノートの新旧 sha256）を返す。元の nb は変えない。"""
    nb = copy.deepcopy(nb)
    changed = {"outer": set(), "frozen": set()}
    fz = [p for p in patches if p.where == "frozen"]
    old_sha = new_sha = None
    if fz:
        inner, old_sha = get_frozen(nb, frozen_cell)
        for p in fz:
            c = inner["cells"][p.cell]
            set_cell_source(c, replace_exactly_once(cell_source(c), p.old, p.new, p.label)); changed["frozen"].add(p.cell)
        new_sha = set_frozen(nb, inner, frozen_cell)
    for p in patches:
        if p.where == "outer":
            c = nb["cells"][p.cell]
            set_cell_source(c, replace_exactly_once(cell_source(c), p.old, p.new, p.label)); changed["outer"].add(p.cell)
        elif p.where != "frozen":
            raise ValueError(p.where)
    return nb, dict(changed_outer=sorted(changed["outer"]), changed_frozen=sorted(changed["frozen"]),
                    frozen_sha256_before=old_sha, frozen_sha256_after=new_sha, labels=[p.label for p in patches])


def insert_guard(nb: dict, before: int) -> dict:
    nb = copy.deepcopy(nb)
    nb["cells"].insert(before, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                                "source": guard_cell_source().splitlines(keepends=True)})
    return nb


def load_base(path: str | None) -> tuple[dict, str]:
    raw = Path(path).read_bytes() if path else subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BASE_COMMIT}:{BASE_PATH}"], check=True, capture_output=True).stdout
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None, help=f"元のノート（既定 git show {BASE_COMMIT}:{BASE_PATH}）")
    ap.add_argument("--dry-run", action="store_true", help="ノートは書かず、報告だけ出す（骨組みの間は必須）")
    a = ap.parse_args(argv)
    nb, sha = load_base(a.base)
    if sha != BASE_SHA256:
        raise SystemExit(f"元のノートの sha256 が違う: {sha}")
    inner, fsha = get_frozen(nb)
    assert fsha == FROZEN_SHA256_BASE, fsha
    assert hashlib.sha256(encode_frozen(inner)).hexdigest() == fsha, "再エンコードが元のバイト列に戻らない"
    out, rep = apply_patches(nb)
    out = insert_guard(out, before=FROZEN_CELL + 1)    # 凍結コードを動かす前（セル 13 の前）
    rep.update(base_sha256=sha, compiled=compile_all(out, FROZEN_CELL), n_patches=len(PATCHES), forbidden=[p for p, _ in FORBIDDEN])
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if not a.dry_run:
        raise SystemExit("骨組みの間はノートを書かない（PATCHES が空）。--dry-run で確かめる")


if __name__ == "__main__":
    main()
