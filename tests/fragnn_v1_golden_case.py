"""#196 (h) のテスト (vi) の作り物の入力: v1 の fragnn_score（`import fragnn_train as T` の経路）の出力を、
PR #202 の学習のファイル（fragnn_bench.py・fragnn_train.py）を持ち込む前（origin/main 9f3a59e）と後で比べる。

    PYTHONPATH=tests .venv/bin/python -c "import fragnn_v1_golden_case as G; G.write_golden()"

で tests/golden/fragnn_v1_golden.npz を書く（持ち込む前の木で 1 回だけ実行した）。テストは持ち込んだ後の木で同じ関数を
呼び、バイト一致を確かめる。v3・学習の重みは使わない（seed 0 の乱数の重み、作り物のスペクトル・断片）。
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/fragnn_v1_golden.npz"


def _fs():
    for p in (ROOT / "scripts/train", ROOT / "scripts/local_eval"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import fragnn_score as FS
    return FS


def case_inputs():
    """作り物の候補の断片（1 つは読めない）と 4 本のスペクトル（1 本は使えない、1 本は CE が空）。"""
    rng = np.random.default_rng(11)
    levels = [[np.sort(rng.random(1) * 200 + 100), np.sort(rng.random(7) * 150 + 20), np.sort(rng.random(15) * 150 + 15)]
              for _ in range(6)]
    levels[4] = None
    sps = []
    for j, (n, prec, adduct, mode, ce) in enumerate([(50, 181.07, "[M+H]+", 1, 20.0), (9, 181.07, "[M+Na]+", 1, 35.0),
                                                     (3, 179.06, "[M-H]-", -1, 0.0), (1, 181.07, "[M+H]+", 1, 10.0)]):
        mz, it = rng.random(n) * 170 + 15, rng.random(n) + 0.01
        if j == 3:
            it = np.zeros(n)            # 使えない（強度 0）
        sps.append(dict(mz=mz, it=it, prec=prec, adduct=adduct, mode=mode, ce=ce))
    return levels, sps


def compute():
    """v1 の経路の出力: pack_spectra・batch_features（numpy/numba）と score_candidates（seed 0 の v1 のモデル、CPU）。"""
    FS = _fs()
    B = FS.B
    levels, sps = case_inputs()
    spec = B.pack_spectra(FS.usable(sps))
    ok = [lv for lv in levels if lv is not None]
    masses, off = B.levels_to_csr(ok)
    cs, sp = FS.pair_layout(len(ok), len(FS.usable(sps)))
    feats, fs = B.batch_features(masses, off, cs, sp, spec)
    out = {f"pack_{k}": np.asarray(v) for k, v in spec.items()}
    out.update(feats=np.asarray(feats), frag=np.asarray(fs))
    try:
        import torch
    except ImportError:
        return out, None
    torch.manual_seed(0)
    model = B.build_model().eval()
    score = FS.score_candidates(model, levels, sps, chunk=5)
    return out, np.asarray(score, np.float64)


def write_golden(path: Path = GOLDEN) -> None:
    import torch
    out, score = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, score=score, torch=np.array(torch.__version__), machine=np.array(platform.machine()),
             system=np.array(platform.system()), **out)
