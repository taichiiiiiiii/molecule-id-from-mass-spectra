#!/usr/bin/env python3
"""
Issue #62（診断）: 0.479 版（kernels/lb479）から Engine 2 を外したノートブックを作る。

lb479 cell 10 の FROZEN_B64（0.465 ノートブック）を復号し、2 か所だけ置き換えて再エンコード、FROZEN_SHA256 を更新する
（復号・再エンコードは make_foldsafe_swap_kernel.py と同じ手順）。
  - F セル 32: Engine 2 のサブプロセスを起動しない。パイプライン自身の失敗時の分岐（`ENGINE FAILED -> v4g lists kept`）に入れ、
    そのあと ENG を「test の全分子が空リスト」で埋める（後段の assert「Engine 2 が全分子を覆う」を満たすため）
  - F セル 50（postflight）: コミット実行で「Engine 2 のリストが空でない」を要求する条件だけを外す
置き換え対象の文字列はそれぞれ 1 回だけ現れることを assert する。

    python scripts/submit/make_no_engine2_kernel.py [--slug casmi26-lb479-noeng2]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from make_foldsafe_swap_kernel import SRC, replace_once

RUN_OLD = "ENG = {}\ntry:\n    _t0 = time.time()\n"
RUN_NEW = ("ENG = {}\n_SKIP_ENGINE2 = True   # Issue #62: Engine 2 off (diagnostic)\ntry:\n"
           "    if _SKIP_ENGINE2:\n        raise RuntimeError('Engine 2 disabled by Issue #62')\n    _t0 = time.time()\n")
EXC_OLD = "except Exception as e:\n    print('ENGINE FAILED -> v4g lists kept:', repr(e))"
EXC_NEW = (EXC_OLD + "\nif _SKIP_ENGINE2:   # Issue #62: every molecule gets an empty Engine 2 list (downstream fusion steps become no-ops)\n"
           "    ENG = {str(_m): dict(smiles=[], keys=[]) for _m in\n"
           "           pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id']).molecule_id.unique()}\n"
           "    print('Engine 2 disabled (Issue #62): empty lists for', len(ENG), 'molecules', flush=True)")
POST_OLD = "assert all(eng[mid]['smiles'] and pc.get(mid, {}).get('pc') for mid in processed)"
POST_NEW = "assert all(pc.get(mid, {}).get('pc') for mid in processed)   # Issue #62: Engine 2 lists are empty by design"


def patch_frozen(frozen: dict) -> dict:
    cells = frozen["cells"]
    run = [i for i, c in enumerate(cells) if RUN_OLD in "".join(c["source"])]
    post = [i for i, c in enumerate(cells) if POST_OLD in "".join(c["source"])]
    assert len(run) == 1 and len(post) == 1, (run, post)
    s = "".join(cells[run[0]]["source"])
    s = replace_once(s, RUN_OLD, RUN_NEW, "engine 2 run")
    s = replace_once(s, EXC_OLD, EXC_NEW, "engine 2 except")
    cells[run[0]]["source"] = s
    cells[post[0]]["source"] = replace_once("".join(cells[post[0]]["source"]), POST_OLD, POST_NEW, "postflight")
    return frozen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="casmi26-lb479-noeng2")
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
