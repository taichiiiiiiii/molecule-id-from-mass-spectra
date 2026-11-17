#!/usr/bin/env python3
"""#120: 断片の説明で異性体を並べ替える学習器（fragnn）の本番の学習（ローカル MPS、内部検証の損失が下がり切るまで）。

事前登録: Issue #120（入力・モデル・学習データ・止め方・経過時間の上限は Issue のとおり。ここでは変えない、選ばない）。
- 入力・特徴・モデル・1 ステップは PR #121 の `fragnn_bench.py`（同じディレクトリ）の関数をそのまま使う:
  断片 = data/ext/frag/frag_rescore.py の fragment_levels（DEFAULT_CFG、コードは変えない）、ピーク = 強度の上位 32 本、
  特徴 17 + 組 1 + スペクトル 3、φ/ρ の MLP、候補 128（0 番 = 正解）の softmax 交差エントロピー、組の数は 2,048 の倍数に詰める
- 学習データ（fold-safe、#109 と同じ DROP）: DROP 55,369（split の全行 ∪ v3 の 4 集合の正解）、DROP_KEY 55,038。
  負例 = design: ±10 ppm 窓のうち DROP_KEY に無く、train_structs の正準キーを持つ行（train 由来の構造）だけ。
  内部検証 = 学習データの正準キーの 2%（seed 0）、学習の正解にも負例にも使わない。内部検証の候補は seed 1 で一度だけ固定
- 最適化: AdamW lr 1e-3・wd 1e-5、warmup 2,000（線形）、clip 5.0、batch 128 スペクトル、seed 42、fp32
- 止め方（2026-11-17 の版 `STOP_RULE`、#120 の 2026-11-16T14:23Z のコメント。定数は #120 のまま）: 2,000 ステップごとに
  内部検証の損失。「改善」の定義は EarlyStop の 1 つだけ（最良値からの相対改善が STOP_REL 0.005 未満なら bad + 1、
  以上なら bad 0）。warmup の後、評価ごとに bad > PLATEAU_PATIENCE 3（= bad 4）なら学習率を PLATEAU_FACTOR 0.5 倍
  （下限 MIN_LR 1e-6）にして bad を 0 に戻す（`lr_rule`）。学習率がすでに MIN_LR で下げられないまま bad が
  STOP_PATIENCE 5 に達したら止める（`should_stop`）。上限 300,000 ステップ。使う重み = 内部検証の損失が最良。
  旧版（22k で停止した回）は torch の ReduceLROnPlateau（threshold 既定 rel 1e-4）を使っていて、「改善」の定義が
  止め方とずれて学習率が一度も下がらなかった。再開のときは今の state に lr_rule を 1 回当てる（過去の評価はなぞらない）。
  meta の差が stop_rule だけなら記録して再開を許す（他の meta が違えば拒む）
- 経過時間の上限: 速度測定の report.json（sha256 9778ea20…）の 4 通りの合計（28,000 ステップ）の最大 × 1.5 と 96 h の
  小さい方（= 10.04 h）。数えるのは学習のループ（ステップと評価）の秒で、再開をまたいで state に足していく
  （memguard に止められていた間・起動の前処理は数えない）。超えたら state を書いて止め、stop_reason = wallclock_cap
- 上限の延長（#120 2026-11-17 昼の ② の決定、`--extend-cap`）: report が status ok・stop_reason wallclock_cap で
  state.pt がある場合だけ、上限を 1 回だけ延ばして再開する。延長の秒 =（目標ステップ − state のステップ）÷
  train_log.jsonl の直近 5 評価の steps_per_s の中央値 × CAP_FACTOR 1.5。目標ステップ = 最後の評価のステップ +
  （STOP_PATIENCE − その評価の bad）× EVAL_EVERY（converged に要る残りの評価の数）。延長は state.pt の meta
  （cap_base_s・cap_extension_s・cap_extension）に固定し、学習の前に state.pt を atomic に書き直す。その後の再開
  （memguard・paused・作り直し）では計算し直さず記録した値を使い、延長の後にまた上限で止まった report は延ばさない。
  止め方の定数・stopper・学習率はそのまま
- 再開: state.pt を tmp → os.replace で評価ごとと 200 ステップごとに書く。バッチの順・負例の乱数はステップ番号から決まり、
  dropout の乱数は state に入れるので、止めても同じ列で続く
- コードの root とデータの root は別: import はこのファイルのディレクトリ（git archive で固めた先）からだけ行う。
  データ（data/ext・data/holdouts・outputs/align_embed・data/train.parquet）は --data-root から読む。
  DROP の json（kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json）はコードの側から読み、sha256 を記録して
  再開のたびに照合する。frag_rescore.py は git に無いのでデータの側（data/ext/frag）から読み、sha256 を同じく照合する
- v3 の集合は除外のキーとしてだけ使う（スペクトル・点・順位は読まない）

純粋な関数（CI でテストする）は torch・rdkit を import しない。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
CODE_ROOT = CODE_DIR.parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import fragnn_bench as B  # noqa: E402  同じディレクトリ（PR #121）

ISSUE = 120
# ---- #120 本文の定数（選ばない）
SEED = B.SEED                       # 42
LR, WD, WARMUP, CLIP = B.LR, B.WD, B.WARMUP, B.CLIP   # 1e-3, 1e-5, 2,000, 5.0
BATCH, K_CAND = B.BATCH, B.K_CAND   # 128, 128
EVAL_EVERY = B.EVAL_EVERY           # 2,000
STOP_REL, STOP_PATIENCE = 0.005, 5
MAX_STEPS = 300_000
PLATEAU_FACTOR, PLATEAU_PATIENCE, MIN_LR = 0.5, 3, 1e-6
# 止め方の版（meta に入れる）。学習率の調整を EarlyStop と同じ「改善」の定義の bad で行う版（#120 2026-11-17）
STOP_RULE = "unified_rel_bad_v2"
META_RESUMABLE_KEYS = ("stop_rule",)   # 再開で違っていてもよい meta（記録はする）
VAL_FRAC, VAL_SEED, VAL_CAND_SEED = B.VAL_FRAC, B.VAL_SEED, 1
PPM = B.PPM
CAP_FACTOR, CAP_MAX_H = 1.5, 96.0
# 上限の延長（#120 2026-11-17 昼の決定。止め方の定数ではなく経過時間の上限だけに効く）
CAP_EXT_MEDIAN_N = 5      # 直近 5 評価の steps_per_s の中央値
CAP_EXT_EXPECT_TOL = 0.01  # --extend-cap-expect-s との相対の食い違いがこれ以上なら止める
CAP_EXT_FORMULA = ("cap_extension_s = (target_step - state_step) / median(steps_per_s of last 5 evals) * CAP_FACTOR; "
                   "target_step = last_eval_step + (STOP_PATIENCE - last_eval_bad) * EVAL_EVERY")
SPEED_REPORT_SHA256 = "9778ea20f5243c2e10f3aaf41a8bca0e7f03607493c0ca045ba900ee02db36ec"
SPEED_REPORT_SRC = "/Users/taichi/.hermes/profiles/massspec/cache/scratch/n1116x_fragnn/report.json"
EXPECTED_DROP, EXPECTED_DROP_KEY = B.EXPECTED_DROP, B.EXPECTED_DROP_KEY          # 55,369 / 55,038
EXPECTED_TRAIN_STRUCTS, EXPECTED_VAL_STRUCTS = B.EXPECTED_TRAIN_STRUCTS, B.EXPECTED_VAL_STRUCTS   # 214,772 / 4,383
EXPECTED_TRAIN_SPECTRA, EXPECTED_VAL_SPECTRA = B.EXPECTED_TRAIN_SPECTRA, B.N_VAL_SPECTRA         # 1,942,968 / 39,437
EXPECTED_POOL_ALLOWED = B.EXPECTED_POOL_ALLOWED                                  # 655,670
EXPECTED_STRUCTS_DESIGN = 219_155   # 速度測定の count.json の design/n_union_val（断片を作る構造の数）
HOLDOUTS = B.HOLDOUTS
KEYS_REL = "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json"
# ---- 実装の定数（モデル・学習・止め方の定義ではない。#109 と同じ値）
CKPT_EVERY = 200          # 評価の間にも state.pt を書く間隔
MEM_CHECK_EVERY = 200     # 学習中に空きを確かめる間隔
MIN_RUN_FREE_GB = 2.0     # 学習中に avail（free+speculative+inactive）がこれを下回ったら state を書いて止める（paused）
CONFIG = {"ISSUE": ISSUE, "SEED": SEED, "LR": LR, "WD": WD, "WARMUP": WARMUP, "CLIP": CLIP, "BATCH": BATCH,
          "K_CAND": K_CAND, "EVAL_EVERY": EVAL_EVERY, "STOP_REL": STOP_REL, "STOP_PATIENCE": STOP_PATIENCE,
          "MAX_STEPS": MAX_STEPS, "PLATEAU_FACTOR": PLATEAU_FACTOR, "PLATEAU_PATIENCE": PLATEAU_PATIENCE,
          "MIN_LR": MIN_LR, "VAL_FRAC": VAL_FRAC, "VAL_SEED": VAL_SEED, "VAL_CAND_SEED": VAL_CAND_SEED, "PPM": PPM,
          "TOP_PEAKS": B.TOP_PEAKS, "TOL": B.TOL, "H_SHIFTS": list(B.H_SHIFTS), "LEVEL_W": list(B.LEVEL_W),
          "F_PEAK": B.F_PEAK, "F_PAIR": B.F_PAIR, "F_SPEC": B.F_SPEC, "PAD_ROWS": B.PAD_ROWS, "negatives": "design",
          "CAP_FACTOR": CAP_FACTOR, "CAP_MAX_H": CAP_MAX_H, "weights": "best internal-val loss"}


# ---------------------------------------------------------------- 純粋な関数（CI）

_PERM_CACHE: dict = {}


def epoch_batch(step: int, n: int, batch: int = BATCH, seed: int = SEED) -> np.ndarray:
    """#109 align_train.epoch_batch と同じ定義（PR #110 未マージのため写し）。ステップ番号から決まるバッチ
    （エポック e = default_rng([seed, e]) の順列、batch 本ずつ、端数は捨てる）。再開しても同じ。"""
    per = n // batch
    if per == 0:
        raise ValueError("fewer training spectra than one batch")
    e, j = divmod(step, per)
    key = (seed, e, n)
    if key not in _PERM_CACHE:
        _PERM_CACHE.clear()
        _PERM_CACHE[key] = np.random.default_rng([seed, e]).permutation(n)
    return _PERM_CACHE[key][j * batch:(j + 1) * batch]


def step_rng(step: int, seed: int = SEED) -> np.random.Generator:
    """そのステップの負例を引く乱数（#109 と同じ、ステップ番号から決まる）。"""
    return np.random.default_rng([seed, 1, step])


class EarlyStop:
    """#109 align_train.EarlyStop と同じ定義（写し）。最良値からの相対改善が rel 未満の評価が patience 回続いたら止める。"""

    def __init__(self, rel: float = STOP_REL, patience: int = STOP_PATIENCE):
        self.rel, self.patience = rel, patience
        self.best, self.bad, self.best_step = float("inf"), 0, None

    def update(self, loss: float, step: int) -> dict:
        improved_rel = (self.best - loss) / self.best if math.isfinite(self.best) else float("inf")
        self.bad = 0 if improved_rel >= self.rel else self.bad + 1
        is_best = loss < self.best
        if is_best:
            self.best, self.best_step = loss, step
        return {"is_best": is_best, "rel_improvement": improved_rel, "bad": self.bad, "stop": self.bad >= self.patience}

    def state(self) -> dict:
        return {"best": self.best, "bad": self.bad, "best_step": self.best_step}

    def load(self, st: dict) -> None:
        self.best, self.bad, self.best_step = st["best"], st["bad"], st["best_step"]


def warmup_lr(step: int, base: float = LR, warmup: int = WARMUP) -> float:
    """warmup の間の学習率（線形、#109 と同じ）。warmup の後は lr_rule が決める。"""
    return base * min(1.0, step / max(1, warmup))


def lr_rule(lr: float, bad: int, factor: float = PLATEAU_FACTOR, patience: int = PLATEAU_PATIENCE,
            min_lr: float = MIN_LR) -> tuple[float, int, bool]:
    """#120 2026-11-17 の止め方の 2: EarlyStop の bad が patience を超えたら（torch の num_bad > patience と同じ
    数え方）学習率を factor 倍（下限 min_lr）にして bad を 0 に戻す。lr ≤ min_lr なら下げられないので何もしない。
    返り値 = (新しい lr, 新しい bad, 下げたか)。"""
    if bad > patience and lr > min_lr:
        return max(lr * factor, min_lr), 0, True
    return lr, bad, False


def should_stop(lr: float, bad: int, min_lr: float = MIN_LR, stop_patience: int = STOP_PATIENCE) -> bool:
    """#120 2026-11-17 の止め方の 3: 学習率がすでに min_lr で下げられない（lr ≤ min_lr）まま bad が stop_patience に
    達したら止める。lr_rule の後に呼ぶ（下げられる間は bad が patience + 1 で 0 に戻るので止まらない）。"""
    return lr <= min_lr and bad >= stop_patience


def meta_diff(old: dict, new: dict) -> tuple[dict, dict]:
    """再開の meta の照合。返り値 = (拒む差, 許す差)。許す差 = META_RESUMABLE_KEYS（止め方の版）だけ。"""
    keys = set(old) | set(new)
    diff = {k: (old.get(k), new.get(k)) for k in sorted(keys) if old.get(k) != new.get(k)}
    allowed = {k: v for k, v in diff.items() if k in META_RESUMABLE_KEYS}
    return {k: v for k, v in diff.items() if k not in META_RESUMABLE_KEYS}, allowed


def wallclock_cap_hours(speed_report: dict, factor: float = CAP_FACTOR, max_h: float = CAP_MAX_H) -> float:
    """#120: 速度測定の 4 通り（design/allpool × n_union/n_union_val）の 28,000 ステップの合計の最大 × factor と max_h の小さい方。"""
    ex = speed_report["extrapolation"]
    want = {"design/n_union", "design/n_union_val", "allpool/n_union", "allpool/n_union_val"}
    if set(ex) != want:
        raise ValueError(f"speed report must have exactly the 4 extrapolations {sorted(want)}, got {sorted(ex)}")
    worst = max(float(v[str(B.STEPS_JUDGE)]["total_h"]) for v in ex.values())
    return min(factor * worst, max_h)


def cap_extension_seconds(rows: list, state_step: int, eval_every: int = EVAL_EVERY, factor: float = CAP_FACTOR,
                          stop_patience: int = STOP_PATIENCE, n: int = CAP_EXT_MEDIAN_N) -> dict:
    """#120 2026-11-17 昼の延長の式（CAP_EXT_FORMULA）。rows = train_log.jsonl の行（評価の行）。
    返り値 = 延長の記録（秒・中央値・使った行・目標ステップ）。計算できなければ ValueError。"""
    ev = [r for r in rows if r.get("steps_per_s") is not None]
    if len(ev) < n:
        raise ValueError(f"steps_per_s のある評価の行が {len(ev)} < {n}")
    last = ev[-1]
    n_evals = int(stop_patience) - int(last["bad"])
    if n_evals <= 0:
        raise ValueError(f"最後の評価の bad {last['bad']} が stop_patience {stop_patience} 以上（延長の必要がない）")
    target = int(last["step"]) + n_evals * int(eval_every)
    remaining = target - int(state_step)
    if int(state_step) < int(last["step"]) or remaining <= 0:
        raise ValueError(f"state のステップ {state_step} と最後の評価 {last['step']}・目標 {target} が合わない")
    sps = [float(r["steps_per_s"]) for r in ev[-n:]]
    med = float(np.median(sps))
    if not (med > 0 and math.isfinite(med)):
        raise ValueError(f"steps_per_s の中央値が正でない: {med}")
    return {"cap_extension_s": remaining / med * factor, "formula": CAP_EXT_FORMULA,
            "median_steps_per_s": med, "steps_per_s": sps, "eval_steps": [int(r["step"]) for r in ev[-n:]],
            "last_eval_step": int(last["step"]), "last_eval_bad": int(last["bad"]), "n_evals": n_evals,
            "eval_every": int(eval_every), "target_step": target, "state_step": int(state_step),
            "remaining_steps": remaining, "factor": factor}


def resolve_cap(out: Path, meta: dict, extend: bool = False, expect_s: float | None = None,
                eval_every: int = EVAL_EVERY) -> tuple[dict, dict | None]:
    """--resume のときに上限を決める（学習の前に呼ぶ）。返り値 = (run_training に渡す meta, 今回延ばした記録 or None)。
    - state.pt の meta に延長が記録済み → 計算し直さず、記録した延長を今の meta に足して返す（元の上限が今の
      meta の wallclock_cap_s と一致し、記録の上限 = 元 + 延長のときだけ。違えば拒む）。--extend-cap の有無によらない
    - 未延長で extend → report が status ok・stop_reason wallclock_cap・state.pt があり、state の meta が今の meta と
      同じで、state が上限で止まった点のときだけ、延長を 1 回計算して state.pt を atomic に書き直す。それ以外は拒む
    - 未延長で extend なし → meta をそのまま返す（今までどおり）"""
    out = Path(out)
    statep, rpath, logp = out / "state.pt", out / "train_report.json", out / "train_log.jsonl"
    base = float(meta["wallclock_cap_s"])
    if not statep.exists():
        if extend:
            raise SystemExit(f"--extend-cap: {statep} が無い（延長しない）")
        return meta, None
    import torch
    st = torch.load(statep, map_location="cpu", weights_only=False)
    smeta = st.get("meta") or {}
    if "cap_extension_s" in smeta:   # 延長済み: 記録した値だけを使う（二重に延ばさない）
        sbase, sext = float(smeta["cap_base_s"]), float(smeta["cap_extension_s"])
        if sbase != base:
            raise SystemExit(f"state の元の上限 {sbase} が今の上限 {base} と違う（再開しない）")
        if float(smeta["wallclock_cap_s"]) != sbase + sext:
            raise SystemExit(f"state の上限 {smeta['wallclock_cap_s']} が元 {sbase} + 延長 {sext} と違う（再開しない）")
        return dict(meta, wallclock_cap_s=sbase + sext, cap_base_s=sbase, cap_extension_s=sext,
                    cap_extension=smeta["cap_extension"]), None
    if not extend:
        return meta, None
    if not rpath.exists():
        raise SystemExit(f"--extend-cap: {rpath} が無い（延長しない）")
    prev = json.loads(rpath.read_text())
    tr = prev.get("train") or {}
    if prev.get("status") != "ok" or tr.get("stop_reason") != "wallclock_cap":
        raise SystemExit(f"--extend-cap: report が status ok・stop_reason wallclock_cap でない"
                         f"（{prev.get('status')}・{tr.get('stop_reason')}）。延長しない")
    diff, allowed = meta_diff(smeta, meta)
    if diff or allowed:
        raise SystemExit(f"--extend-cap: state の meta が今の入力と違う（延長しない）: {diff or allowed}")
    if float((prev.get("meta") or {}).get("wallclock_cap_s", -1.0)) != base:
        raise SystemExit("--extend-cap: report の上限が今の上限と違う（延長しない）")
    step = int(st["step"])
    if step != int(tr.get("steps", -1)):
        raise SystemExit(f"--extend-cap: state のステップ {step} が report の {tr.get('steps')} と違う（延長しない）")
    if float(tr.get("train_elapsed_s", -1.0)) + 0.05 < float(prev.get("wallclock_cap_h", math.inf)) * 3600:
        raise SystemExit("--extend-cap: report の経過秒が上限に届いていない（延長しない）")
    rows = [json.loads(x) for x in logp.read_text().splitlines() if x.strip()] if logp.exists() else []
    if not rows or int(rows[-1]["bad"]) != int(st["stopper"]["bad"]):
        raise SystemExit("--extend-cap: train_log.jsonl の最後の評価の bad が state の stopper と違う（延長しない）")
    try:
        rec = cap_extension_seconds(rows, step, eval_every=eval_every)
    except ValueError as e:
        raise SystemExit(f"--extend-cap: 延長を計算できない: {e}")
    ext = float(rec["cap_extension_s"])
    if expect_s is not None and abs(ext - float(expect_s)) / float(expect_s) >= CAP_EXT_EXPECT_TOL:
        raise SystemExit(f"--extend-cap: 計算した延長 {ext:.1f} s が期待 {expect_s} s と相対 "
                         f"{CAP_EXT_EXPECT_TOL:.0%} 以上違う（延長しない）")
    rec.update(cap_base_s=base, wallclock_cap_s=base + ext, expect_s=expect_s, at=time.strftime("%F %T"),
               report_train=tr)
    new = dict(meta, wallclock_cap_s=base + ext, cap_base_s=base, cap_extension_s=ext, cap_extension=rec)
    st["meta"] = new
    st["wallclock_cap_s"] = base + ext
    atomic_torch_save(st, statep)    # 学習の前に固定する（最初の ckpt の前に止められても延長は失われない）
    print("CAP_EXTEND", json.dumps(rec, default=str), flush=True)
    return new, rec


def leak_counts(train_raw, train_key, split_rows_raw, split_rows_key, holdout_keys: dict, neg_keys, drop_key: set,
                pos_keys, train_neg_in_val_mask, design_neg_trainstruct_mask, train_pos_in_val_mask) -> dict:
    """#120 の漏れの数（すべて 0 でなければならない）。最初の 6 つは速度測定（fragnn_bench.phase_count）と同じ定義、
    train_pos_in_val は「学習の正解 × 内部検証」（#120 本文の列挙の残り 1 つ）。"""
    import pandas as pd
    import sanpier_fold0 as sf
    raw = sf.leak_audit(set(train_raw), split_rows_raw, holdout_keys)
    key = sf.leak_audit(set(train_key), split_rows_key, holdout_keys)
    return dict(raw=int(raw["total"]), key=int(key["total"]),
                neg_in_drop=int(pd.Series(np.asarray(neg_keys, object)).isin(drop_key).sum()),
                pos_in_drop=int(pd.Series(np.asarray(pos_keys, object)).isin(drop_key).sum()),
                train_neg_in_val=int(np.asarray(train_neg_in_val_mask, bool).sum()),
                design_neg_not_trainstruct=int((~np.asarray(design_neg_trainstruct_mask, bool)).sum()),
                train_pos_in_val=int(np.asarray(train_pos_in_val_mask, bool).sum()))


def check_leaks(leak: dict) -> None:
    """漏れが 1 件でもあれば止める（AssertionError）。"""
    bad = {k: v for k, v in leak.items() if v != 0}
    if bad:
        raise AssertionError(f"leak audit failed (must all be 0): {bad}")


def assemble(cand: np.ndarray, mask: np.ndarray, spec_idx: np.ndarray, spec_n: np.ndarray, pad_to: int = B.PAD_ROWS):
    """(b, K) の候補 → 詰めた組の並び。返り値: n（本物の組の数）, P（詰めた数）, 組の候補の構造, 組のスペクトル,
    slot（本物は i*K + k、詰め物は b*K = 捨てる slot）, valid（ピークの有効、詰め物は全部 False）。"""
    b, k = mask.shape
    vi, vk = np.nonzero(mask)
    n = len(vi)
    p = B.padded_pairs(n, pad_to)
    cs = np.zeros(p, np.int64); cs[:n] = cand[vi, vk]
    sp = np.zeros(p, np.int64); sp[:n] = spec_idx[vi]
    slot = np.full(p, b * k, np.int64); slot[:n] = vi * k + vk
    valid = np.arange(B.TOP_PEAKS)[None, :] < np.asarray(spec_n)[sp][:, None]
    valid[n:] = False
    return n, p, cs, sp, slot, valid


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_text(path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_torch_save(obj, path) -> None:
    """tmp に書いてから os.replace（保存の途中で止められても、前のファイルが壊れない）。"""
    import torch
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def trim_log(logp, step: int) -> int:
    """#109 と同じ: 再開のとき state より先の step の行と重複行を train_log.jsonl から落とす。落とした行数を返す。"""
    if not Path(logp).exists():
        return 0
    rows = [json.loads(x) for x in Path(logp).read_text().splitlines() if x.strip()]
    keep, seen = [], set()
    for r in rows:
        if r["step"] <= step and r["step"] not in seen:
            keep.append(r); seen.add(r["step"])
    if len(keep) != len(rows):
        atomic_write_text(logp, "".join(json.dumps(r) + "\n" for r in keep))
    return len(rows) - len(keep)


def free_gb() -> dict:
    """#109 と同じ: vm_stat の free+speculative と inactive（GB）。avail_gb = free + speculative + inactive。macOS 以外は None。"""
    import re
    import subprocess
    try:
        txt = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {"avail_gb": None}
    m = re.search(r"page size of (\d+) bytes", txt)
    if not m:
        return {"avail_gb": None}
    ps = int(m.group(1))
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", txt).group(1)) * ps / 1e9  # noqa: E731
    fr, ina = get("Pages free") + get("Pages speculative"), get("Pages inactive")
    return {"free_gb": round(fr, 2), "inactive_gb": round(ina, 2), "avail_gb": round(fr + ina, 2)}


def ram_peak_gb() -> float:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / (1e9 if sys.platform == "darwin" else 1e6), 3)


def curve_csv(rows: list) -> str:
    cols = ("step", "train_loss", "val_loss", "lr", "train_elapsed_s")
    lines = [",".join(cols)] + [",".join(repr(r.get(c)) if isinstance(r.get(c), float) else str(r.get(c)) for c in cols)
                                for r in rows]
    return "\n".join(lines) + "\n"


def _png_bytes(img: np.ndarray) -> bytes:
    """(H, W, 3) uint8 → PNG（zlib だけ。matplotlib が無い環境用）。"""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def curve_png(rows: list, path) -> str:
    """学習曲線の png。matplotlib があれば軸つき、無ければ zlib だけの簡易図（青 = train、赤 = 内部検証、文字なし）。"""
    xs = [r["step"] for r in rows]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, [r["train_loss"] for r in rows], label="train")
        ax.plot(xs, [r["val_loss"] for r in rows], label="internal val")
        ax.set_xlabel("step"); ax.set_ylabel("listwise CE"); ax.legend(); fig.tight_layout()
        fig.savefig(path, dpi=100); plt.close(fig)
        return "matplotlib"
    except ImportError:
        pass
    W, Hh, m = 640, 360, 20
    img = np.full((Hh, W, 3), 255, np.uint8)
    img[m, m:W - m] = img[Hh - m, m:W - m] = 0
    img[m:Hh - m, m] = img[m:Hh - m, W - m] = 0
    ys = [v for r in rows for v in (r["train_loss"], r["val_loss"]) if v is not None and math.isfinite(v)]
    if xs and ys:
        x0, x1 = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
        y0, y1 = min(ys), max(ys) if max(ys) > min(ys) else min(ys) + 1
        X = lambda x: int(round(m + (x - x0) / (x1 - x0) * (W - 2 * m)))  # noqa: E731
        Y = lambda y: int(round(Hh - m - (y - y0) / (y1 - y0) * (Hh - 2 * m)))  # noqa: E731
        for key, col in (("train_loss", (31, 119, 180)), ("val_loss", (214, 39, 40))):
            pts = [(X(r["step"]), Y(r[key])) for r in rows if r[key] is not None and math.isfinite(r[key])]
            for (ax_, ay), (bx, by) in zip(pts, pts[1:] or pts):
                n = max(abs(bx - ax_), abs(by - ay), 1)
                for t in range(n + 1):
                    px, py = ax_ + (bx - ax_) * t // n, ay + (by - ay) * t // n
                    img[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = col
    Path(path).write_bytes(_png_bytes(img))
    return "zlib"


def own_heavy_jobs() -> list[str]:
    """このリポジトリの重いジョブが自分以外に動いていないか（#109 と同じ判定。シェルや /usr/bin/time の行は数えない）。"""
    import subprocess
    try:
        txt = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    out = []
    for line in txt.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit() and int(pid) != me and is_heavy_python(cmd):
            out.append(line.strip()[:160])
    return out


HEAVY_SCRIPTS = ("fragnn_train.py", "fragnn_bench.py", "align_local_embed.py", "align_train.py", "local_fold0_logits.py")


def is_heavy_python(cmd: str) -> bool:
    """ps の command がこのリポジトリの重いジョブの python か。シェル・/usr/bin/time・memguard の行は数えない。
    python のパスに空白があっても（元のフォルダの .venv）拾う（#126 起動前レビューの軽微指摘）。"""
    cmd = cmd.strip()
    for sname in HEAVY_SCRIPTS:
        i = cmd.find(sname)
        if i < 0:
            continue
        head = cmd[:i]
        first = head.split(" ")[0]
        if os.path.basename(first) in ("bash", "sh", "zsh", "time") or first == "/usr/bin/time":
            return False
        return os.path.basename(first).startswith("python") or "/bin/python" in head
    return False


# ---------------------------------------------------------------- 学習の核（torch はここで初めて読む）

class _Abort(BaseException):
    """テスト用: 指定のステップで（state を書かずに）落ちたことにする（memguard の kill の代わり。train の except で拾わない）。"""


def run_training(D: dict, out: Path, meta: dict, dev: str = "cpu", resume: bool = False, max_steps: int = MAX_STEPS,
                 eval_every: int = EVAL_EVERY, ckpt_every: int = CKPT_EVERY, cap_s: float | None = None,
                 rep: dict | None = None, mem_check=free_gb, on_batch=None, abort_at: int | None = None,
                 val_override=None, t0: float | None = None) -> dict:
    """D: masses, off（構造ごとの断片の CSR）, spec（pack_spectra の dict、全スペクトル）, tr_i / va_i（学習・内部検証の
    スペクトルの番号）, pos_struct・cid（スペクトルごとの正解の構造と CSR の番号）, off_tr/flat_tr, off_va/flat_va（負例の CSR、
    flat は構造の番号）。meta: 再開のたびに照合する値（sha256・上限など）。返り値 = report の dict（out/train_report.json）。"""
    import torch
    t0 = time.time() if t0 is None else t0
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rpath, logp, statep, bestp = out / "train_report.json", out / "train_log.jsonl", out / "state.pt", out / "model_best.pt"
    if cap_s is None:
        cap_s = float(meta["wallclock_cap_s"])
    if rpath.exists():   # 終わった学習（status ok）は続けない・やり直さない（model_best.pt を上書きしないため）
        try:
            prev = json.loads(rpath.read_text())
        except ValueError:
            prev = {}
        if prev.get("status") == "ok":
            ptr, pmeta = prev.get("train") or {}, prev.get("meta") or {}
            # 上限の延長の後の最初の再開だけは許す（resolve_cap が state.pt に延長を固定済み。report はまだ延長の前の
            # 上限で止まった回のもの）。延長の後にまた上限で止まった report（上限 = 元 + 延長）は今までどおり拒む
            ext_resume = (resume and statep.exists() and "cap_extension_s" in meta
                          and ptr.get("stop_reason") == "wallclock_cap"
                          and float(pmeta.get("wallclock_cap_s", -1.0)) == float(meta["cap_base_s"]))
            if not ext_resume:
                raise SystemExit(f"{rpath} は status ok（{prev.get('train')}）。やり直すなら出力先を変える")
    if statep.exists() and not resume:
        raise SystemExit(f"{statep} がある。続けるなら --resume、やり直すなら出力先を変える")
    rep = dict(rep or {})
    rep.update({"issue": ISSUE, "status": "running", "device": dev, "config": CONFIG, "meta": meta,
                "wallclock_cap_h": cap_s / 3600})

    def save():
        rep["elapsed_s"] = round(time.time() - t0, 1)
        rep["ram_peak_gb"] = ram_peak_gb()
        if dev == "mps":
            rep["mps_driver_allocated_max_gb"] = max(rep.get("mps_driver_allocated_max_gb", 0.0),
                                                     round(torch.mps.driver_allocated_memory() / 1e9, 3))
        atomic_write_text(rpath, json.dumps(rep, indent=1, default=str, ensure_ascii=False))

    masses, off, spec = D["masses"], D["off"], D["spec"]
    tr_i, va_i = np.asarray(D["tr_i"], np.int64), np.asarray(D["va_i"], np.int64)
    pos_struct, cid = np.asarray(D["pos_struct"], np.int64), np.asarray(D["cid"], np.int64)
    spec_feat = torch.from_numpy(np.ascontiguousarray(spec["spec_feat"], dtype=np.float32))
    # 内部検証の候補: seed 1 で一度だけ固定（#109 と同じ）
    va_cand, va_mask = B.sample_candidates(np.random.default_rng(VAL_CAND_SEED), pos_struct[va_i], cid[va_i],
                                           D["off_va"], D["flat_va"], K_CAND)
    torch.manual_seed(SEED)
    model = B.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    stopper = EarlyStop(STOP_REL, STOP_PATIENCE)

    def cur_lr() -> float:
        return float(opt.param_groups[0]["lr"])

    def set_lr(v: float) -> None:
        for gr in opt.param_groups:
            gr["lr"] = v
    step, acc, n_acc, elapsed0, since_eval = 0, 0.0, 0, 0.0, 0.0
    if resume and statep.exists():
        st = torch.load(statep, map_location="cpu", weights_only=False)
        diff, allowed_diff = meta_diff(st["meta"], meta)
        if diff:
            raise SystemExit(f"state の meta が今の入力と違う（再開しない）: {diff}")
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        stopper.load(st["stopper"]); step = int(st["step"])
        torch.set_rng_state(st["torch_rng"])
        if dev == "mps" and st.get("mps_rng") is not None:
            torch.mps.set_rng_state(st["mps_rng"])
        acc, n_acc = float(st.get("acc", 0.0)), int(st.get("n_acc", 0))
        elapsed0, since_eval = float(st["train_elapsed_s"]), float(st.get("since_eval", 0.0))
        rep["resumed_from_step"] = step
        if rpath.exists():   # 前の回の report の再開の記録を引き継ぐ
            try:
                rep["resumes"] = list(json.loads(rpath.read_text()).get("resumes", []))
            except ValueError:
                pass
        # 止め方の 5: 今の state に lr_rule を 1 回当てる（過去の評価はなぞらない。warmup の間は当てない）
        lr0, bad0 = cur_lr(), stopper.bad
        reduced = False
        if step >= WARMUP:
            new_lr, new_bad, reduced = lr_rule(lr0, bad0)
            if reduced:
                set_lr(new_lr); stopper.bad = new_bad
        entry = {"step": step, "train_elapsed_s": elapsed0, "at": time.strftime("%F %T"), "stop_rule": STOP_RULE,
                 "meta_diff_allowed": allowed_diff, "plateau_state_ignored": "plateau" in st,
                 "lr_before": lr0, "bad_before": bad0, "lr_reduced": reduced,
                 "lr": cur_lr(), "bad": stopper.bad, "best": stopper.best, "best_step": stopper.best_step,
                 "wallclock_cap_s": cap_s, "cap_base_s": meta.get("cap_base_s"),
                 "cap_extension_s": meta.get("cap_extension_s"), "cap_extended_now": bool(rep.pop("cap_extended_now", False)),
                 "cap_extension": meta.get("cap_extension")}
        rep.setdefault("resumes", []).append(entry)
        print("RESUME", json.dumps(entry, default=str), flush=True)
    rep["log_rows_trimmed_on_resume"] = trim_log(logp, step)
    rep["params"] = int(sum(p.numel() for p in model.parameters()))
    t_run = time.time()

    def elapsed() -> float:
        return elapsed0 + (time.time() - t_run)

    def forward(cand, mask, spec_idx, train: bool):
        b = mask.shape[0]
        n, p, cs, sp, slot, valid = assemble(cand, mask, spec_idx, spec["n"])
        f0, fs0 = B.batch_features(masses, off, cs[:n], sp[:n], spec)
        feats = np.zeros((p, B.TOP_PEAKS, B.F_PEAK), np.float32); feats[:n] = f0
        fs = np.zeros((p, 1), np.float32); fs[:n, 0] = fs0
        x, vt = torch.from_numpy(feats).to(dev), torch.from_numpy(valid).to(dev)
        pair, sf = torch.from_numpy(fs).to(dev), spec_feat[sp].to(dev)
        logit = model(x, vt, pair, sf)
        flat = torch.full((b * K_CAND + 1,), float("-inf"), device=dev).index_put((torch.from_numpy(slot).to(dev),), logit)
        return flat[:b * K_CAND].view(b, K_CAND)

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.inference_mode():
            for i in range(0, len(va_i), BATCH):
                sl = slice(i, i + BATCH)
                sc = forward(va_cand[sl], va_mask[sl], va_i[sl], train=False)
                tot += float(torch.nn.functional.cross_entropy(
                    sc, torch.zeros(sc.shape[0], dtype=torch.long, device=dev), reduction="sum"))
                n += sc.shape[0]
        model.train()
        return tot / max(n, 1)

    def state_all() -> dict:
        return {"model": model.state_dict(), "opt": opt.state_dict(), "stop_rule": STOP_RULE,
                "stopper": stopper.state(), "step": step, "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if dev == "mps" else None, "acc": acc, "n_acc": n_acc,
                "train_elapsed_s": elapsed(), "since_eval": time.time() - ts, "meta": meta,
                "wallclock_cap_s": cap_s}

    def log_rows() -> list:
        return [json.loads(x) for x in logp.read_text().splitlines() if x.strip()] if logp.exists() else []

    model.train()
    ts = time.time() - since_eval
    stop_reason = None
    if step >= WARMUP and should_stop(cur_lr(), stopper.bad):
        stop_reason = "converged"        # 収束の後、出力の段の途中で止められた学習の再開
    elif step >= max_steps:
        stop_reason = "max_steps"
    elif elapsed0 >= cap_s:
        stop_reason = "wallclock_cap"
    rep["skipped_training_on_resume"] = bool(resume and stop_reason is not None)
    save()
    n_tr = len(tr_i)
    while stop_reason is None:
        if step < WARMUP:
            for gr in opt.param_groups:
                gr["lr"] = warmup_lr(step, LR, WARMUP)
        li = epoch_batch(step, n_tr, BATCH, SEED)
        bi = tr_i[li]
        cand, mask = B.sample_candidates(step_rng(step), pos_struct[bi], cid[bi], D["off_tr"], D["flat_tr"], K_CAND)
        if on_batch is not None:
            on_batch(step, bi, cand, mask)
        sc = forward(cand, mask, bi, train=True)
        loss = torch.nn.functional.cross_entropy(sc, torch.zeros(len(bi), dtype=torch.long, device=dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        step += 1
        acc += float(loss.detach()); n_acc += 1
        if abort_at is not None and step == abort_at:
            raise _Abort(step)
        is_eval = step % eval_every == 0 or step == max_steps
        if not is_eval and step % MEM_CHECK_EVERY == 0:   # 評価の step では止めない（評価を飛ばさない、#109）
            fg = (mem_check() or {}).get("avail_gb")
            if fg is not None and fg < MIN_RUN_FREE_GB:
                stop_reason = f"paused_low_memory avail={fg}GB"
                atomic_torch_save(state_all(), statep)
                break
        if is_eval:
            vl = float(val_override(step)) if val_override is not None else val_loss()
            u = stopper.update(vl, step)
            lr_used, bad_eval, reduced = cur_lr(), stopper.bad, False
            if step >= WARMUP:   # 止め方の 2（同じ bad で学習率を下げる）
                new_lr, new_bad, reduced = lr_rule(lr_used, stopper.bad)
                if reduced:
                    set_lr(new_lr); stopper.bad = new_bad
            u.update(bad=stopper.bad, bad_eval=bad_eval, lr_used=lr_used, lr_reduced=reduced,
                     stop=bool(step >= WARMUP and should_stop(cur_lr(), stopper.bad)))   # 止め方の 3
            if u["is_best"]:
                atomic_torch_save(model.state_dict(), bestp)
            row = {"step": step, "train_loss": acc / max(n_acc, 1), "val_loss": vl, "lr": cur_lr(),
                   **u, "best": stopper.best, "best_step": stopper.best_step, "train_elapsed_s": round(elapsed(), 1),
                   "steps_per_s": n_acc / max(time.time() - ts, 1e-9), "mem": mem_check()}
            with open(logp, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("EVAL", json.dumps(row), flush=True)
            acc, n_acc, ts = 0.0, 0, time.time()
            atomic_torch_save(state_all(), statep)
            rows = log_rows()
            atomic_write_text(out / "learning_curve.csv", curve_csv(rows))
            save()
            if u["stop"]:
                stop_reason = "converged"
                break
        elif step % ckpt_every == 0:
            atomic_torch_save(state_all(), statep)   # 評価の間の再開点（止め方の数え方には入らない）
        if step >= max_steps:
            stop_reason = "max_steps"
        elif elapsed() >= cap_s:
            stop_reason = "wallclock_cap"   # #120: 経過時間の上限。state を書いて止める（評価は足さない）
            atomic_torch_save(state_all(), statep)
    rep["train"] = {"steps": step, "stop_reason": stop_reason, "best_val_loss": stopper.best,
                    "best_step": stopper.best_step, "train_elapsed_s": round(elapsed(), 1),
                    "stop_rule": STOP_RULE, "final_lr": cur_lr(), "final_bad": stopper.bad}
    if stop_reason.startswith("paused"):
        rep["status"] = "paused"
        save()
        return rep
    # ---- 出力（最良の重み）
    rows = log_rows()
    atomic_write_text(out / "learning_curve.csv", curve_csv(rows))
    rep["curve_png"] = curve_png(rows, out / "learning_curve.png") if rows else None
    cfg = dict(CONFIG, best_step=stopper.best_step, best_val_loss=stopper.best, stop_reason=stop_reason, meta=meta,
               stop_rule=STOP_RULE, final_lr=cur_lr())
    atomic_write_text(out / "fragnn_config.json", json.dumps(cfg, indent=1, default=str))
    rep["outputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                      for p in sorted(out.iterdir()) if p.suffix in (".pt", ".json", ".csv", ".png")
                      and p.name not in ("train_report.json", "state.pt")}
    rep["status"] = "ok"
    save()
    return rep


# ---------------------------------------------------------------- ローカルの段（CI では動かさない）

def data_paths(data_root: Path) -> dict:
    return dict(pool=data_root / "data/ext/v4g/casmi26-v2-pool", split=data_root / "data/ext/foldsafe/split.parquet",
                embed=data_root / "outputs/align_embed", train=data_root / "data/train.parquet",
                frag=data_root / "data/ext/frag", holdouts=data_root / "data/holdouts")


def import_frag_rescore(data_root: Path):
    """frag_rescore.py（git に無い）をデータの側から読む。scripts/ ではないディレクトリだけを sys.path に足す。"""
    fdir = data_paths(data_root)["frag"]
    if str(fdir) not in sys.path:
        sys.path.insert(0, str(fdir))
    import frag_rescore as fr
    assert Path(fr.__file__).resolve() == (fdir / "frag_rescore.py").resolve(), fr.__file__
    return fr


def prepare(data_root: Path, keys_path: Path, prep_path: Path, extra: dict | None = None) -> dict:
    """速度測定の count と同じ DROP・内部検証・design の負例を作り、漏れを assert し、断片を前計算する（prep.npz、再開で使い回す）。"""
    import pandas as pd
    import sanpier_fold0 as sf
    assert Path(sf.__file__).resolve().parent == CODE_DIR, sf.__file__
    P = data_paths(data_root)
    t0 = time.time()
    meta_df = pd.read_parquet(P["pool"] / "pool_meta.parquet", columns=["key", "mass"])
    assert bool(np.all(np.diff(meta_df.mass.values) >= 0))
    ts = pd.read_parquet(P["pool"] / "train_structs.parquet", columns=["inchikey14", "key"])
    raw_to_key = dict(zip(ts.inchikey14, ts.key))
    key_to_row = dict(zip(meta_df.key.values, range(len(meta_df))))
    split = pd.read_parquet(P["split"], columns=["inchikey14", "key", "fold"])
    hk = json.loads(Path(keys_path).read_text())
    drop_raw = sf.drop_inchikey14(split.inchikey14, hk)
    assert len(drop_raw) == EXPECTED_DROP, len(drop_raw)
    drop_key = B.drop_key_space(drop_raw, raw_to_key, hk, split.key)
    assert len(drop_key) == EXPECTED_DROP_KEY, len(drop_key)
    allowed = ~meta_df.key.isin(drop_key).values
    assert int(allowed.sum()) == EXPECTED_POOL_ALLOWED, int(allowed.sum())
    h117 = {h: set((P["holdouts"] / h / "drop_keys.txt").read_text().split()) for h in HOLDOUTS}
    h117_missing = {h: len(ks - drop_key - drop_raw) for h, ks in h117.items()}
    assert all(v == 0 for v in h117_missing.values()), h117_missing
    is_trainstruct = meta_df.key.isin(set(ts.key.values)).values
    erep = json.loads((P["embed"] / "embed_report.json").read_text())
    tr = []
    for k in range(erep["n_shards"]):
        t = pd.read_parquet(P["embed"] / f"idx_{k:03d}.parquet", columns=["row", "kind", "inchikey14"])
        tr.append(t[t.kind == 0])
    tr = pd.concat(tr, ignore_index=True)
    tr["key"] = tr.inchikey14.map(raw_to_key)
    keep = tr.key.notna().values & ~tr.key.isin(drop_key).values & ~tr.inchikey14.isin(drop_raw).values
    tr = tr[keep].reset_index(drop=True)
    assert tr.row.is_unique
    val_keys = B.split_val_keys(tr.key.values)
    is_val = tr.key.isin(val_keys).values
    pos_row = np.array([key_to_row[k] for k in tr.key.values], np.int64)
    val_rows = meta_df.key.isin(val_keys).values
    up_tr, up_va = np.unique(pos_row[~is_val]), np.unique(pos_row[is_val])
    assert len(up_tr) == EXPECTED_TRAIN_STRUCTS and len(up_va) == EXPECTED_VAL_STRUCTS, (len(up_tr), len(up_va))
    assert int((~is_val).sum()) == EXPECTED_TRAIN_SPECTRA and int(is_val.sum()) == EXPECTED_VAL_SPECTRA, \
        (int((~is_val).sum()), int(is_val.sum()))
    mass = meta_df.mass.values.astype(np.float64)
    off_tr, flat_tr = B.negative_csr(up_tr, mass, allowed & ~val_rows & is_trainstruct)   # design（#120）
    off_va, flat_va = B.negative_csr(up_va, mass, allowed & is_trainstruct)
    keys_arr = meta_df.key.values
    all_neg = np.concatenate([flat_tr, flat_va])
    # 生キー・正準キーは学習と内部検証のスペクトル全部（内部検証も split・4 集合と重ならないことを確かめる、保守的）
    leak = leak_counts(tr.inchikey14.values, tr.key.values, list(zip(split.inchikey14, split.fold)),
                       list(zip(split.key, split.fold)), hk, keys_arr[all_neg], drop_key, tr.key.values,
                       val_rows[flat_tr], is_trainstruct[all_neg], val_rows[pos_row[~is_val]])
    check_leaks(leak)
    struct_rows = np.unique(np.concatenate([up_tr, up_va, flat_tr, flat_va]))
    assert len(struct_rows) == EXPECTED_STRUCTS_DESIGN, len(struct_rows)
    sidx = lambda rows: np.searchsorted(struct_rows, rows)  # noqa: E731
    # スペクトルごとの CSR の番号（学習は up_tr の中、内部検証は up_va の中）
    cid = np.where(is_val, np.searchsorted(up_va, pos_row), np.searchsorted(up_tr, pos_row)).astype(np.int64)
    assert np.all(np.where(is_val, up_va[np.minimum(cid, len(up_va) - 1)], up_tr[np.minimum(cid, len(up_tr) - 1)]) == pos_row)
    # 断片の前計算（frag_rescore.fragment_levels、DEFAULT_CFG。読めない SMILES は空）
    import pyarrow.parquet as pq
    fr = import_frag_rescore(data_root)
    smi_col = pq.read_table(P["pool"] / "pool_meta.parquet", columns=["smiles"]).column(0)
    c = fr.resolve_cfg(None)
    tf = time.time(); fr.warmup()
    lv = []
    for r in struct_rows:
        try:
            lv.append(fr.fragment_levels(smi_col[int(r)].as_py(), c["max_breaks"], c["max_bonds_2"], c["max_bonds_3"]))
        except Exception:  # noqa: BLE001
            lv.append(None)
    masses, off = B.levels_to_csr(lv)
    frag_s = time.time() - tf
    info = dict(n_train_spectra=int((~is_val).sum()), n_val_spectra=int(is_val.sum()), n_train_structs=len(up_tr),
                n_val_structs=len(up_va), drop=len(drop_raw), drop_key=len(drop_key), pool_allowed=int(allowed.sum()),
                h117_missing=h117_missing, n_structs=len(struct_rows), n_frag_none=sum(x is None for x in lv),
                neg_median_train=float(np.median(np.diff(off_tr))), neg_zero_train=int((np.diff(off_tr) == 0).sum()),
                leak=leak, frag_seconds=round(frag_s, 1), prepare_seconds=round(time.time() - t0, 1), **(extra or {}))
    tmp = Path(str(prep_path) + ".tmp.npz")
    np.savez(tmp, struct_rows=struct_rows, masses=masses, off=off, spec_row=tr.row.values.astype(np.int64),
             spec_ik=tr.inchikey14.values.astype("S14"), is_val=is_val, pos_struct=sidx(pos_row), cid=cid,
             off_tr=off_tr, flat_tr=sidx(flat_tr), off_va=off_va, flat_va=sidx(flat_va),
             info=np.array(json.dumps(info)))
    os.replace(tmp, prep_path)
    return info


def load_spectra(train_path: Path, spec_row: np.ndarray, spec_ik: np.ndarray) -> dict:
    """train.parquet を row group ごとに 1 回読み、spec_row の順に上位 32 本を詰める（pack_spectra と同じ）。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(spec_row)
    out = dict(mz=np.zeros((n, B.TOP_PEAKS)), it=np.zeros((n, B.TOP_PEAKS)), n=np.zeros(n, np.int64), prec=np.zeros(n),
               sh_v=np.zeros((n, 12)), sh_c=np.zeros((n, 12), np.int8), sh_d=np.zeros((n, 12), np.int8),
               sh_n=np.zeros(n, np.int64), spec_feat=np.zeros((n, B.F_SPEC), np.float32))
    order = np.argsort(spec_row, kind="stable")
    srow = spec_row[order]
    cols = ["inchikey14", "adduct", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities",
            "collision_energy_ev"]
    pf = pq.ParquetFile(train_path, buffer_size=1 << 20, pre_buffer=False)
    base, filled = 0, 0
    for b in pf.iter_batches(batch_size=8192, columns=cols):
        lo, hi = np.searchsorted(srow, base), np.searchsorted(srow, base + b.num_rows)
        if hi > lo:
            sub = b.take(pa.array(srow[lo:hi] - base)).to_pydict()
            rows = []
            for j in range(hi - lo):
                ce = sub["collision_energy_ev"][j]
                if sub["inchikey14"][j].encode() != spec_ik[order[lo + j]]:
                    raise AssertionError(f"inchikey14 mismatch at train row {srow[lo + j]}")
                rows.append(dict(adduct=sub["adduct"][j], prec=sub["precursor_mz"][j],
                                 mode=1 if sub["ionization_mode"][j] == "positive" else -1,
                                 mz=sub["ms2_mzs"][j], it=sub["ms2_normalized_intensities"][j],
                                 ce=float(np.mean(ce)) if ce else 0.0))
            pk = B.pack_spectra(rows)
            dst = order[lo:hi]
            for k in out:
                out[k][dst] = pk[k]
            filled += hi - lo
        base += b.num_rows
    if filled != n:
        raise AssertionError(f"spectra filled {filled} != {n}")
    return out


def ensure_speed_report(out: Path, src: str | Path = SPEED_REPORT_SRC) -> dict:
    """速度測定の report.json を out/speed_report.json に写して（既にあればそれを）sha256 を照合する。"""
    dst = out / "speed_report.json"
    if not dst.exists():
        if not Path(src).exists():
            raise SystemExit(f"speed report not found: {src} (and {dst} is missing)")
        tmp = Path(str(dst) + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    h = sha256_file(dst)
    if h != SPEED_REPORT_SHA256:
        raise SystemExit(f"speed report sha256 {h} != {SPEED_REPORT_SHA256}")
    return json.loads(dst.read_text())


def assert_code_data_separation(data_root: Path) -> None:
    """データの root の scripts/ が sys.path に無いこと、読んだ自前のモジュールがこのディレクトリのものであること。"""
    bad = [p for p in sys.path if p and Path(p).resolve().is_relative_to((data_root / "scripts").resolve())]
    if bad:
        raise SystemExit(f"data-root scripts/ is on sys.path: {bad}")
    if Path(B.__file__).resolve().parent != CODE_DIR:
        raise SystemExit(f"fragnn_bench imported from {B.__file__}, not {CODE_DIR}")


def train(data_root: Path, out: Path, device: str = "mps", resume: bool = False,
          speed_src: str | Path = SPEED_REPORT_SRC, extend_cap: bool = False, extend_cap_expect_s: float | None = None,
          **run_kw) -> dict:
    """本番の入口。run_kw はテスト用（max_steps・eval_every・mem_check など）で、main からは渡さない。
    extend_cap: 上限の延長（resolve_cap）。延長の計算と state.pt の書き直しは前処理より前に行う。"""
    t0 = time.time()
    data_root, out = Path(data_root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    assert_code_data_separation(data_root)
    keys_path = CODE_ROOT / KEYS_REL
    sp = ensure_speed_report(out, speed_src)
    cap_h = wallclock_cap_hours(sp)
    frag_py = data_paths(data_root)["frag"] / "frag_rescore.py"
    meta = {"keys_json_sha256": sha256_file(keys_path), "frag_rescore_sha256": sha256_file(frag_py),
            "speed_report_sha256": SPEED_REPORT_SHA256, "wallclock_cap_s": cap_h * 3600, "config": CONFIG,
            "stop_rule": STOP_RULE}
    rep = {"code_root": str(CODE_ROOT), "data_root": str(data_root), "keys_json": str(keys_path),
           "mem_start": free_gb(), "numba_threads": os.environ.get("NUMBA_NUM_THREADS")}
    if extend_cap and not resume:
        raise SystemExit("--extend-cap は --resume と一緒に使う")
    if resume:
        meta, ext_rec = resolve_cap(out, meta, extend=extend_cap, expect_s=extend_cap_expect_s,
                                    eval_every=run_kw.get("eval_every", EVAL_EVERY))
        rep["cap_extended_now"] = ext_rec is not None
    rpath = out / "train_report.json"
    try:
        prep_path = out / "prep.npz"
        tp = time.time()
        if prep_path.exists():
            z = np.load(prep_path)
            info = json.loads(str(z["info"]))
            bad = {k: (info.get(k), meta[k]) for k in ("keys_json_sha256", "frag_rescore_sha256") if info.get(k) != meta[k]}
            if bad:
                raise SystemExit(f"prep.npz は今の入力と違う（使い回さない）: {bad}")
            rep["prep_reused"] = True
        else:
            info = prepare(data_root, keys_path, prep_path,
                           extra={k: meta[k] for k in ("keys_json_sha256", "frag_rescore_sha256")})
            z = np.load(prep_path)
            rep["prep_reused"] = False
        rep["data"] = info
        check_leaks(info["leak"])
        D = {k: z[k] for k in ("masses", "off", "pos_struct", "cid", "off_tr", "flat_tr", "off_va", "flat_va")}
        is_val = z["is_val"]
        D["tr_i"], D["va_i"] = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        ts = time.time()
        D["spec"] = load_spectra(data_paths(data_root)["train"], z["spec_row"], z["spec_ik"])
        rep["stages"] = {"prep_s": round(ts - tp, 1), "spectra_s": round(time.time() - ts, 1)}
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        rep.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        atomic_write_text(rpath, json.dumps(rep, indent=1, default=str, ensure_ascii=False))
        return rep
    try:
        return run_training(D, out, meta, dev=device, resume=resume, rep=rep, t0=t0, **run_kw)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        prev = json.loads(rpath.read_text()) if rpath.exists() else rep
        prev.update(status="error", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-4000:])
        atomic_write_text(rpath, json.dumps(prev, indent=1, default=str, ensure_ascii=False))
        return prev


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="#120 fragnn の本番の学習（ローカル、内部検証の損失が下がり切るまで）")
    ap.add_argument("--data-root", required=True, help="データ（data/ext・data/holdouts・outputs/align_embed）のあるクローン")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--resume", action="store_true", help="out/state.pt があれば続ける（無ければ最初から）")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--speed-report", default=SPEED_REPORT_SRC)
    ap.add_argument("--extend-cap", action="store_true",
                    help="#120 2026-11-17 昼: status ok・wallclock_cap で止まった学習の上限を 1 回だけ延ばして再開する"
                         "（延長済みなら記録した上限を使い、計算し直さない）")
    ap.add_argument("--extend-cap-expect-s", type=float, default=None,
                    help="延長の秒の期待値（照合だけ。相対 1%% 以上違えば止まる）")
    a = ap.parse_args()
    for k, v in (("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")):
        if os.environ.get(k) != v:
            raise SystemExit(f"set {k}={v} before starting (torch reads it at import)")
    fg = free_gb()
    if fg.get("avail_gb") is not None and fg["avail_gb"] < a.min_free_gb:
        raise SystemExit(f"vm_stat avail(free+speculative+inactive) {fg} < {a.min_free_gb} GB")
    own = own_heavy_jobs()
    if own:
        raise SystemExit(f"another heavy job of this repo is running: {own}")
    r = train(Path(a.data_root), Path(a.out), device=a.device, resume=a.resume, speed_src=a.speed_report,
              extend_cap=a.extend_cap, extend_cap_expect_s=a.extend_cap_expect_s)
    print(json.dumps({k: r.get(k) for k in ("status", "error", "device", "train", "elapsed_s", "wallclock_cap_h")},
                     indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
