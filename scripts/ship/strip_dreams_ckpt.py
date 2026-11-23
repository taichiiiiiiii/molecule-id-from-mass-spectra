#!/usr/bin/env python3
"""DreaMS の `ssl_model.ckpt` から optimizer などの学習の状態を外し、前向きに要るものだけを書く（Issue #127、案 A）。

    PYTHONPYCACHEPREFIX=<scratch>/pyc .venv/bin/python scripts/ship/strip_dreams_ckpt.py \
        --ckpt data/ext/dreams/ssl_model.ckpt \
        --dreams-root data/ext/dreams/DreaMS-dbec3a0b514a99e5056cfccde4559fda8cfe8129 \
        --out <scratch>/dreams_ssl_state.pt --report <scratch>/strip_report.json

- 入力の ckpt は読むだけ（md5 と大きさを #99/#109 の値 `4deee91d…`・1,392,710,212 B と完全照合）。出力は入力と同じフォルダの中には
  書かない（元のフォルダに書き込まないため）。tmp に書いてから os.replace
- 残すキー: `state_dict`（strict に読むので全部）、`hyper_parameters`（`load_dreams` がモデルを作るのに使う）、
  `global_step`・`epoch`・`pytorch-lightning_version`（記録）。それ以外（`optimizer_states`・`lr_schedulers`・`callbacks`・
  `loops` など）は外す
- 確かめること: 出力を読み直して `state_dict` のキー・dtype・形・値が入力と同じ（torch.equal）、`hyper_parameters` の pickle の
  バイト列が同じ、`align_speed_trial.load_dreams` が出力を strict に読める
- report: 入力の md5・大きさ、出力の sha256・大きさ（B と MB）、外したキーと残したキー、`state_dict` の先頭の名前ごとの MB、検証の結果

torch はモジュールの先頭で import しない（CI は torch を入れないので、純粋な関数だけをテストする）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

KEEP_KEYS = ("state_dict", "hyper_parameters", "global_step", "epoch", "pytorch-lightning_version")
REQUIRED_KEYS = ("state_dict", "hyper_parameters")
CKPT_MD5 = "4deee91dd4486c2b6cbcde1f200b8f50"     # Zenodo 10997887（#99・#109 と同じ値）
CKPT_SIZE = 1_392_710_212
MB = 1e6


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def tensor_nbytes(t) -> int:
    """torch.Tensor（numel×element_size）と numpy 配列（nbytes）の両方を数える。どちらでもなければ 0。"""
    if hasattr(t, "element_size") and hasattr(t, "numel"):
        return int(t.numel()) * int(t.element_size())
    if hasattr(t, "nbytes"):
        return int(t.nbytes)
    return 0


def strip_checkpoint(state: dict, keep=KEEP_KEYS, required=REQUIRED_KEYS) -> tuple[dict, dict]:
    """(出力の dict, {kept, dropped}) を返す。必須のキーが無ければ KeyError。入力の dict は変えない。"""
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must be a dict, got {type(state).__name__}")
    missing = [k for k in required if k not in state]
    if missing:
        raise KeyError(f"checkpoint lacks required keys: {missing}")
    out = {k: state[k] for k in keep if k in state}
    return out, {"kept": [k for k in keep if k in state], "dropped": sorted(k for k in state if k not in out)}


def size_by_prefix(state_dict: dict) -> dict:
    """state_dict の先頭の名前（`a.b.c` の `a`）ごとの MB（小数 1 桁）と全体。"""
    per: dict[str, int] = {}
    for name, t in state_dict.items():
        top = name.split(".", 1)[0]
        per[top] = per.get(top, 0) + tensor_nbytes(t)
    total = sum(per.values())
    return {"per_prefix_mb": {k: round(v / MB, 1) for k, v in per.items()}, "total_bytes": int(total),
            "total_mb": round(total / MB, 1), "n_tensors": len(state_dict)}


def check_out_path(out: Path, ckpt: Path) -> None:
    """出力が入力の ckpt と同じフォルダ（とその下）に無いこと。元のフォルダに書かないための守り。"""
    src_dir = Path(os.path.realpath(ckpt)).parent
    dst = Path(os.path.realpath(out.parent)) / out.name
    if dst == Path(os.path.realpath(ckpt)) or src_dir == dst.parent or src_dir in dst.parents:
        raise ValueError(f"refusing to write {dst} inside the input folder {src_dir}")


def file_digest(path, algo: str, chunk: int = 1 << 22) -> tuple[str, int]:
    h, n = hashlib.new(algo), 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
            n += len(b)
    return h.hexdigest(), n


# ---------------------------------------------------------------- torch の部分

def _load(path, dreams_root: str):
    """ckpt の hyper_parameters は DreaMS のクラスを含むので、align_speed_trial と同じ stub を入れて読む。"""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
    import align_speed_trial as A
    if dreams_root not in sys.path:
        sys.path.insert(0, dreams_root)
    finder = A.install_stubs(A.STUB_TOP)
    try:
        import types
        import dreams.utils.dformats as dfm   # align_speed_trial.load_dreams と同じ別名（pickle の msml.utils.*）
        import dreams.utils.data as ddata
        msml = types.ModuleType("msml"); msml.__path__ = []
        msml_utils = types.ModuleType("msml.utils"); msml_utils.__path__ = []
        sys.modules.setdefault("msml", msml)
        sys.modules.setdefault("msml.utils", msml_utils)
        sys.modules.setdefault("msml.utils.dformats", dfm)
        sys.modules.setdefault("msml.utils.data", ddata)
        return torch.load(path, map_location="cpu", weights_only=False)
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in list(finder.made):
            sys.modules.pop(name, None)


def verify_same(orig: dict, new: dict) -> dict:
    import torch
    a, b = orig["state_dict"], new["state_dict"]
    same_keys = list(a.keys()) == list(b.keys())
    bad = [k for k in a if k not in b or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
           or not torch.equal(a[k], b[k])]
    hp_same = pickle.dumps(orig["hyper_parameters"]) == pickle.dumps(new["hyper_parameters"])
    return {"state_dict_keys_same_order": same_keys, "tensors_differing": len(bad), "first_differing": bad[:5],
            "hyper_parameters_pickle_same": hp_same,
            "ok": bool(same_keys and not bad and hp_same)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dreams-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--expect-md5", default=CKPT_MD5)
    ap.add_argument("--expect-size", type=int, default=CKPT_SIZE)
    ap.add_argument("--skip-load-dreams", action="store_true", help="テスト用: load_dreams での strict 読みを省く")
    args = ap.parse_args()
    sys.dont_write_bytecode = True
    import torch
    t0 = time.time()
    ckpt, out, rpath = Path(args.ckpt), Path(args.out), Path(args.report)
    check_out_path(out, ckpt)
    check_out_path(rpath, ckpt)
    md5, size = file_digest(ckpt, "md5")
    rep = {"issue": 127, "input": {"path": str(ckpt), "md5": md5, "bytes": size, "mb": round(size / MB, 1),
                                   "expected_md5": args.expect_md5, "expected_bytes": args.expect_size}}
    if md5 != args.expect_md5 or size != args.expect_size:
        rep["status"] = "input_mismatch"
        rpath.write_text(json.dumps(rep, indent=1))
        raise SystemExit(f"input md5/size mismatch: {md5} {size}")
    orig = _load(ckpt, args.dreams_root)
    new, keys = strip_checkpoint(orig)
    rep["keys"] = keys
    rep["state_dict"] = size_by_prefix(orig["state_dict"])
    rep["dropped_mb"] = {k: round(sum(tensor_nbytes(t) for t in _iter_tensors(orig[k])) / MB, 1) for k in keys["dropped"]}
    tmp = out.with_name(out.name + ".tmp")
    torch.save(new, tmp)
    os.replace(tmp, out)
    sha, osize = file_digest(out, "sha256")
    rep["output"] = {"path": str(out), "sha256": sha, "bytes": osize, "mb": round(osize / MB, 1)}
    re_read = _load(out, args.dreams_root)
    rep["verify"] = verify_same(orig, re_read)
    del re_read
    if not args.skip_load_dreams:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
        import align_speed_trial as A
        model, _, _, info = A.load_dreams(args.dreams_root, str(out), "cpu")
        rep["verify"]["load_dreams"] = {k: info[k] for k in ("n_layers", "d_model", "params_backbone", "load_state_dict",
                                                               "ckpt_global_step")}
        del model
    rep["torch"] = torch.__version__
    rep["seconds"] = round(time.time() - t0, 1)
    rep["status"] = "ok" if rep["verify"]["ok"] else "verify_failed"
    rpath.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep[k] for k in ("status", "output", "state_dict", "seconds")}, default=str))
    if rep["status"] != "ok":
        raise SystemExit(1)


def _iter_tensors(obj):
    """入れ子の dict / list の中のテンソル・配列を全部たどる（外したキーの大きさを数えるため）。"""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensors(v)
    elif hasattr(obj, "nbytes") or hasattr(obj, "element_size"):
        yield obj


if __name__ == "__main__":
    main()
