#!/usr/bin/env python3
"""
提出用ノートブック: 0.479 版（kernels/lb479）に「編集候補」パッチだけを入れる。

    python scripts/submit/make_edit_submission_kernel.py --edit-seeds 10 [--slug casmi26-lb479-edit10]

ホールドアウト用の差し替え（COMP、drop_keys、hide/unhide）は入れない。変更点は PubChem チャネルの候補追加のみ。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cv"))
from make_holdout_kernel import SRC, probe_core_patch, src_of  # noqa: E402

ENV_CELL = "import os\nos.environ['CASMI_EDIT_SEEDS'] = '__EDIT_SEEDS__'\nprint('edit-candidate route: seeds =', os.environ['CASMI_EDIT_SEEDS'])\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit-seeds", type=int, required=True)
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()
    assert args.edit_seeds > 0
    slug = args.slug or f"casmi26-lb479-edit{args.edit_seeds}"
    dst = Path("kernels") / slug.replace("-", "_")

    nb = json.loads((SRC / "notebook.ipynb").read_text())
    pin = [c for c in nb["cells"] if c["cell_type"] == "code" and "FROZEN_B64 = '" in src_of(c)]
    assert len(pin) == 1
    s = src_of(pin[0])
    b64 = re.search(r"FROZEN_B64 = '([^']+)'", s).group(1)
    old_sha = re.search(r"FROZEN_SHA256 = '([0-9a-f]{64})'", s).group(1)
    raw = base64.b64decode(b64)
    assert hashlib.sha256(raw).hexdigest() == old_sha
    fr = json.loads(raw)
    a, b = probe_core_patch(False, args.edit_seeds)
    cells = [c for c in fr["cells"] if c["cell_type"] == "code" and a in src_of(c)]
    assert len(cells) == 1 and src_of(cells[0]).count(a) == 1
    cells[0]["source"] = src_of(cells[0]).replace(a, b, 1)
    new_raw = json.dumps(fr, ensure_ascii=False).encode()
    new_sha = hashlib.sha256(new_raw).hexdigest()
    pin[0]["source"] = s.replace(b64, base64.b64encode(new_raw).decode(), 1).replace(old_sha, new_sha)
    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                           "source": ENV_CELL.replace("__EDIT_SEEDS__", str(args.edit_seeds))})

    meta = json.loads((SRC / "kernel-metadata.json").read_text())
    meta.update(id=f"taichiiiii/{slug}", title=slug.replace("-", " "), is_private=True)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "notebook.ipynb").write_text(json.dumps(nb, ensure_ascii=False))
    (dst / "kernel-metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("wrote", dst, meta["id"], "| frozen sha", old_sha[:12], "->", new_sha[:12])


if __name__ == "__main__":
    main()
