#!/usr/bin/env python3
"""
Issue #46（診断）: 0.479 版（kernels/lb479）の FPNet を fold-safe 5 本の平均に差し替えたノートブックを作る。

lb479 cell 10 の FROZEN_B64（0.465 ノートブック）を復号し、2 か所だけ置き換えて再エンコード、FROZEN_SHA256 を更新する。
  - F セル 37: Engine 1 の FP バンク（fpnet_full1、無ければ fpnet_0）→ fold-safe fpnet_fold0..4
  - F セル 34: PubChem チャネルの probe のバンク（A+B ＋ fold0）→ fold-safe fpnet_fold0..4
置き換え対象の文字列はそれぞれ 1 回だけ現れることを assert する。

    python scripts/submit/make_foldsafe_swap_kernel.py [--slug casmi26-lb479-foldsafe5]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

SRC = Path("kernels/lb479")

# F セル 37（ノートブックのセルのソースそのもの）
ENG_OLD = "bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)"
ENG_NEW = ("_FS5 = sorted(glob.glob('/kaggle/input/**/casmi26-fold-safe-fpnet/**/fpnet_fold[0-4].pt', recursive=True))  # Issue #46\n"
           "assert len(_FS5) == 5, _FS5\n"
           "bank = fpnet.ModelBank(_FS5, device=dev)")
ENG_PRINT_OLD = "print('engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) FALLBACK', flush=True)"
ENG_PRINT_NEW = "print('engine FP bank: fold-safe x5 (Issue #46)', _FS5, flush=True)"

# F セル 34 は probe のソースを文字列リテラルとして持つ（改行は \\n）
PC_OLD = "bank = fpnet.ModelBank(models + [fold0], device=dev)"
PC_NEW = ("models = sorted(glob.glob(os.path.join(os.path.dirname(fold0), \\'fpnet_fold[0-4].pt\\')))  # Issue #46\\n"
          "    assert len(models) == 5, models\\n"
          "    bank = fpnet.ModelBank(models, device=dev)")


def replace_once(s: str, old: str, new: str, where: str) -> str:
    n = s.count(old)
    assert n == 1, f"{where}: {n} 回見つかった: {old[:60]}"
    return s.replace(old, new)


def patch_frozen(frozen: dict) -> dict:
    cells = frozen["cells"]
    eng = [i for i, c in enumerate(cells) if ENG_OLD in "".join(c["source"])]
    pcs = [i for i, c in enumerate(cells) if PC_OLD in "".join(c["source"])]
    assert len(eng) == 1 and len(pcs) == 1, (eng, pcs)
    s = "".join(cells[eng[0]]["source"])
    s = replace_once(s, ENG_OLD, ENG_NEW, "engine bank")
    s = replace_once(s, ENG_PRINT_OLD, ENG_PRINT_NEW, "engine print")
    cells[eng[0]]["source"] = s
    s = "".join(cells[pcs[0]]["source"])
    cells[pcs[0]]["source"] = replace_once(s, PC_OLD, PC_NEW, "pubchem bank")
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="casmi26-lb479-foldsafe5")
    args = ap.parse_args()
    nb = json.loads((SRC / "notebook.ipynb").read_text())
    c10 = "".join(nb["cells"][10]["source"])
    i = c10.index("FROZEN_B64 = '") + len("FROZEN_B64 = '")
    j = c10.index("'", i)
    raw = base64.b64decode(c10[i:j])
    old_sha = hashlib.sha256(raw).hexdigest()
    assert f"FROZEN_SHA256 = '{old_sha}'" in c10, "元の SHA256 と一致しない"
    frozen = patch_frozen(json.loads(raw))
    new_raw = json.dumps(frozen, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    c10 = c10[:i] + base64.b64encode(new_raw).decode() + c10[j:]
    c10 = c10.replace(f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'")
    nb["cells"][10]["source"] = c10
    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"taichiiiii/{args.slug}", title=args.slug.replace("-", " "), is_private=True)
    dst = Path("kernels") / args.slug.replace("-", "_")
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", dst, meta["id"], "frozen sha", old_sha[:12], "->", new_sha[:12])


if __name__ == "__main__":
    main()
