#!/usr/bin/env python3
"""
診断（#82、事前登録 #82 のコメント 7072907310 と 7072943936）: 0.487 写し（kernels/lb487）の Engine 1 の FP バンク
fpnet_full1 だけを、公開キットの train_fpnet.py で全 train から作り直した FPNet（taichiiiii/casmi26-fpnet-full-train-s2 の
出力 fpnet_final.pt、sha256 3b653e76…e332）に替えたノートを作る。

変更（これだけ）:
1. 凍結コード（外側セル 12 の FROZEN_B64、0.465 系統のノート 59 セル）の F セル 37 の 3 行
   （fpnet_full1 の glob・fpnet_0 へ黙って落ちる ModelBank・print）を、明示のパス 1 本の読み込みに置き換える。
   見つからない・sha256 が違う・鍵や形が合わない（load_model の strict な load_state_dict）ときは例外で止める（fallback なし）。
   ログに OWN_FPNET path <パス> route=explicit|glob と OWN_FPNET ok <sha256> を 1 行ずつ出す。
   FROZEN_B64・FROZEN_SHA256・notebook の metadata の 2 つの frozen_source_sha256 を作り直す。
2. 外側セル 5 の COMMIT_MODE = 'fast' を 'smoke' に（commit run だけに効く。採点の再実行は IS_RERUN で全体が走る）
3. kernel-metadata.json: 新しい slug・title・code_file、id_no を消す、kernel_sources に S2 のカーネルを足す

    python scripts/submit/make_lb487_ownfpnet.py [--out-dir kernels/lb487_ownfpnet]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "kernels" / "lb487"
SRC_NB = "casmi26-lb487-copy.ipynb"
SRC_NB_SHA = "380e2c820c2a47cfad268b75cdd0dfc564350b599b9b052a1915d292987113c5"
FROZEN_SHA_OLD = "d1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88"
FROZEN_CELLS = 59
CHANGED_FROZEN = 37
CHANGED_OUTER = (5, 12)
SLUG = "casmi26-lb487-ownfpnet"
S2_KERNEL = "taichiiiii/casmi26-fpnet-full-train-s2"
OWN_SHA = "3b653e76c457885af6824b4ea2f5f0a225ca608f840259807c68f45066aee332"
OWN_PATH = "/kaggle/input/notebooks/taichiiiii/casmi26-fpnet-full-train-s2/fpnet_final.pt"
OWN_GLOB = "/kaggle/input/**/casmi26-fpnet-full-train-s2/**/fpnet_final.pt"
NBITS = 10226

OLD_BLOCK = (
    "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m\n"
    "bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)\n"
    "print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)\n"
)

NEW_BLOCK = (
    "# #82（診断）: Engine 1 の FP バンクを、公開キットの train_fpnet.py で全 train から作り直した FPNet（S2 の fpnet_final.pt）に。\n"
    "# 見つからない・sha256 が違う・鍵や形が合わないときは例外で止める（fpnet_0 に落ちない）\n"
    "import hashlib as _h82\n"
    f"_OWN_FPNET_SHA = '{OWN_SHA}'\n"
    f"_OWN_FPNET = '{OWN_PATH}'\n"
    "_OWN_ROUTE = 'explicit'\n"
    "if not os.path.isfile(_OWN_FPNET):\n"
    f"    _g82 = sorted(glob.glob('{OWN_GLOB}', recursive=True))\n"
    "    if len(_g82) != 1:\n"
    "        raise RuntimeError(('#82 own FPNet must be found exactly once', _OWN_FPNET, _g82))\n"
    "    _OWN_FPNET, _OWN_ROUTE = _g82[0], 'glob'\n"
    "print('OWN_FPNET path', _OWN_FPNET, 'route=' + _OWN_ROUTE, flush=True)\n"
    "_s82 = _h82.sha256()\n"
    "with open(_OWN_FPNET, 'rb') as _f82:\n"
    "    for _b82 in iter(lambda: _f82.read(1 << 20), b''):\n"
    "        _s82.update(_b82)\n"
    "if _s82.hexdigest() != _OWN_FPNET_SHA:\n"
    "    raise RuntimeError(('#82 own FPNet sha256 mismatch', _OWN_FPNET, _s82.hexdigest(), _OWN_FPNET_SHA))\n"
    "_FULL = [_OWN_FPNET]\n"
    "bank = fpnet.ModelBank(_FULL, device=dev)   # load_model: strict load_state_dict (key/shape mismatch raises)\n"
    f"assert bank.nbits == {NBITS} and len(bank.models) == 1, ('#82 own FPNet bank shape', bank.nbits, len(bank.models))\n"
    "print('OWN_FPNET ok', _s82.hexdigest(), flush=True)\n"
    "print('engine FP bank:', _FULL[0], flush=True)\n"
)


def replace_once(s: str, old: str, new: str, what: str) -> str:
    n = s.count(old)
    assert n == 1, (what, n)
    return s.replace(old, new)


def split_frozen(c12: str) -> tuple[int, int, bytes]:
    """外側セル 12 の FROZEN_B64 の中身の位置 (i, j) と、復号した凍結コード（JSON のバイト列）。"""
    i = c12.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    j = c12.index("'", i)
    return i, j, base64.b64decode(c12[i:j])


def patch_frozen(frozen: dict) -> dict:
    cells = frozen["cells"]
    hit = [k for k, c in enumerate(cells) if OLD_BLOCK in "".join(c["source"])]
    assert hit == [CHANGED_FROZEN], hit
    s = "".join(cells[CHANGED_FROZEN]["source"])
    s = replace_once(s, OLD_BLOCK, NEW_BLOCK, "F cell 37 FP bank")
    compile(s, "frozen_cell_37", "exec")
    cells[CHANGED_FROZEN]["source"] = s
    return frozen


def build(slug: str = SLUG) -> tuple[dict, dict, dict]:
    """(notebook, kernel-metadata, build_info) を返す。ファイルには書かない。"""
    raw_nb = (SRC / SRC_NB).read_bytes()
    assert hashlib.sha256(raw_nb).hexdigest() == SRC_NB_SHA, "0.487 写しのノートではない"
    nb = json.loads(raw_nb)
    orig = json.loads(raw_nb)
    c12 = "".join(nb["cells"][12]["source"])
    i, j, raw = split_frozen(c12)
    old_sha = hashlib.sha256(raw).hexdigest()
    assert old_sha == FROZEN_SHA_OLD and c12.count(f"FROZEN_SHA256 = '{old_sha}'") == 1
    frozen0 = json.loads(raw)
    frozen = patch_frozen(json.loads(raw))
    assert len(frozen["cells"]) == len(frozen0["cells"]) == FROZEN_CELLS
    for k, (a, b) in enumerate(zip(frozen0["cells"], frozen["cells"])):
        same = "".join(a["source"]) == "".join(b["source"]) and a["cell_type"] == b["cell_type"]
        assert same == (k != CHANGED_FROZEN), k
        if b["cell_type"] == "code":
            compile("".join(b["source"]), f"frozen_{k}", "exec")
    frozen_rest0 = dict(frozen0); frozen_rest0.pop("cells")
    frozen_rest = dict(frozen); frozen_rest.pop("cells")
    assert frozen_rest0 == frozen_rest
    new_raw = json.dumps(frozen, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    c12 = c12[:i] + base64.b64encode(new_raw).decode() + c12[j:]
    c12 = replace_once(c12, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'", "frozen sha")
    nb["cells"][12]["source"] = c12
    c5 = "".join(nb["cells"][5]["source"])
    nb["cells"][5]["source"] = replace_once(c5, "COMMIT_MODE = 'fast'", "COMMIT_MODE = 'smoke'", "commit mode")
    cm = nb["metadata"]["casmi"]
    assert cm["frozen_source_sha256"] == old_sha
    assert cm["upstream_casmi_metadata"]["frozen_source_sha256"] == old_sha
    cm["frozen_source_sha256"] = new_sha
    cm["upstream_casmi_metadata"]["frozen_source_sha256"] = new_sha
    meta_changed = ["casmi.frozen_source_sha256", "casmi.upstream_casmi_metadata.frozen_source_sha256"]
    for k, (a, b) in enumerate(zip(orig["cells"], nb["cells"])):
        same = "".join(a["source"]) == "".join(b["source"])
        assert same == (k not in CHANGED_OUTER), k
        if b["cell_type"] == "code":
            compile("".join(b["source"]), f"outer_{k}", "exec")
    assert len(orig["cells"]) == len(nb["cells"])
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"taichiiiii/{slug}", title=slug.replace("-", " "), is_private=True, code_file="notebook.ipynb")
    assert S2_KERNEL not in meta["kernel_sources"]
    meta["kernel_sources"] = list(meta["kernel_sources"]) + [S2_KERNEL]
    rep = dict(issue=82, frozen_sha_old=old_sha, frozen_sha_new=new_sha, changed_frozen_cell=CHANGED_FROZEN,
               changed_outer_cells=list(CHANGED_OUTER), changed_metadata=meta_changed,
               own_fpnet_sha256=OWN_SHA, own_fpnet_path=OWN_PATH, own_fpnet_glob=OWN_GLOB,
               kernel_sources=meta["kernel_sources"], source_notebook_sha256=SRC_NB_SHA)
    return nb, meta, rep


def push_source_bytes(nb: dict) -> int:
    """kaggle kernels push が送るソースの大きさの見積もり（出力を空にし、source を連結、json.dumps の既定）。"""
    d = json.loads(json.dumps(nb))
    for c in d["cells"]:
        c["source"] = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None
    return len(json.dumps(d).encode())


def write(out_dir: Path, nb: dict, meta: dict, rep: dict) -> Path:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1))
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (out_dir / "build_info.json").write_text(json.dumps(rep, indent=1))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "kernels" / "lb487_ownfpnet"))
    args = ap.parse_args()
    nb, meta, rep = build()
    rep["push_source_bytes"] = push_source_bytes(nb)
    assert rep["push_source_bytes"] < 1_000_000, rep["push_source_bytes"]
    out = write(Path(args.out_dir), nb, meta, rep)
    print(json.dumps(dict(rep, out_dir=str(out)), indent=1))


if __name__ == "__main__":
    main()
