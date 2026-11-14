#!/usr/bin/env python3
"""整列チャネルの本番 その 2: 凍結 DreaMS の埋め込み（scripts/train/align_local_embed.py、または予備の Kaggle カーネル
casmi26-align-embed の出力）を読み、MSAlign と同じ形の MLP を**内部検証の損失が下がり切るまで**学習する（ローカル、MPS/CPU）。

- 事前登録: Issue #109（docs/research/contrastive-alignment.md §(e) の定義、オーナーの指示 11/14「Loss が下がり切るまで」）
- モデルと 1 ステップの定義は MSAlign の massspecgym_formula.yaml（commit 97e201b）のまま: seed 42、d_shared 512、
  d_hidden 2048、d_metadata 64、ms_pre 0 層・ms_post 2 層・mol 3 層、dropout 0.30、temperature_init 0.07、
  k_candidates 128（正解を含む）、batch 128、AdamW lr 1e-4・weight_decay 1e-6、warmup 2000、gradient clip 5.0
- 止め方（#109 で固定。v3 や LB は使わない）: 学習データの正準キーの 2%（seed 0）を内部検証に分けて学習から外す
  （正解としても負例としても使わない）。2,000 ステップごとに内部検証の InfoNCE 損失（候補は seed 1 で一度だけ固定）を測る。
  最良値からの相対改善が 0.5% 未満の評価が 5 回続いたら止める。上限 300,000 ステップ。学習率は warmup 2000 のあと
  ReduceLROnPlateau（factor 0.5、patience 3、min_lr 1e-6、ほかは torch の既定）を評価ごとに。使う重み = 内部検証の損失が最良
- 各評価で train_log.jsonl に 1 行追記し、再開用の state.pt を書く（--resume で続きから。バッチの順と候補の乱数は
  ステップ番号から決まるので、途中で止めても同じ列になる）
- 精度: MPS/CPU は fp32。CUDA のときだけ MLP を fp16 autocast＋GradScaler（Kaggle で回す場合。数値は fp32 と同じにならない）
- 出力（--out）: model_best.pt、mol_mlp_best.pt、align_config.json、g_query.npy、query_index.parquet、h_pool.npy、
  train_log.jsonl、learning_curve.csv（.png は matplotlib があれば）、train_report.json

torch はモジュールの先頭で import しない（CI は純粋な関数だけをテストする）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

MSALIGN_COMMIT = "97e201b1dd61e792d9086815192f7518a0b62ed8"
SEED = 42
D_SHARED, D_HIDDEN, D_META = 512, 2048, 64
N_MS_PRE, N_MS_POST, N_MOL = 0, 2, 3
DROPOUT, TEMP_INIT = 0.30, 0.07
K_CAND, BATCH = 128, 128
LR, WD, WARMUP, MAX_STEPS, CLIP = 1e-4, 1e-6, 2000, 300_000, 5.0   # MAX_STEPS = 上限（#109）
VAL_FRAC, VAL_SEED, VAL_CAND_SEED = 0.02, 0, 1
EVAL_EVERY = 2_000
STOP_REL, STOP_PATIENCE = 0.005, 5
PLATEAU_FACTOR, PLATEAU_PATIENCE, MIN_LR = 0.5, 3, 1e-6
PPM = 10.0
FP_BITS = 10_226
D_EMB = 1024
POOL_ROWS = 710_701
POOL_META_SHA256 = "82b8c40315611289b9a4004b3af18d6d3402b3cd5baa4ee5723da07b7e2f6bb0"
TRAIN_STRUCTS_SHA256 = "f20aa1cac23af42e430a6633707fda34e48dea97114bdc3e443897f692df3656"
FP_BITS_SHA256 = "c3869e86449d6fed6e79e9655f8a7d35bd55c3138ecddfc1dae5072627e78677"
POOL_FP_SHA256 = "f02d91eeb4e2be5687ce678c199aa09a3acda1e821478372fc23fd1c7edf55e8"   # kaggle datasets download（11/14）
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
EXPECTED_DROP = 55_369
N_QUERY = 4_756
LOG_EVERY = 100
NUMERICS_NOTE = {"cuda": "MLP trained with torch.autocast(fp16) + GradScaler; numerics differ from fp32.",
                 "mps": "fp32 on Apple MPS.", "cpu": "fp32 on CPU."}
MIN_RUN_FREE_GB = 2.0      # 学習中に空き（free+speculative）がこれを下回ったら state を書いて止める
ADDUCT_ALIASES = {"[M+CH3COO]-": "[M+CH3COOH-H]-", "[M+FA-H]-": "[M+HCOOH-H]-", "[M+HCOO]-": "[M+HCOOH-H]-"}


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def normalize_adduct(a):
    """MSAlign encoders.normalize_adduct と同じ。"""
    if not isinstance(a, str):
        return a
    c = a.replace(" ", "")
    return ADDUCT_ALIASES.get(c, c)


def adduct_vocab(adducts) -> tuple[list[str], list[int]]:
    """学習スペクトルの付加イオンの語彙（sorted）と符号（+1/−1、末尾で判断、判らなければ 0）。最後に未知の枠（符号 0）を足す。"""
    vocab = sorted({normalize_adduct(a) for a in adducts if isinstance(a, str)})
    signs = [1 if a.endswith("+") else (-1 if a.endswith("-") else 0) for a in vocab]
    return vocab, signs + [0]


def adduct_ids(adducts, vocab) -> np.ndarray:
    lut = {a: i for i, a in enumerate(vocab)}
    return np.array([lut.get(normalize_adduct(a), len(vocab)) for a in adducts], np.int64)


def drop_key_space(drop_raw: set, raw_to_key: dict, holdout_keys: dict, split_keys) -> set:
    """負例と学習から外す正準キー: split の key ∪ 4 集合の drop_keys ∪ DROP の生キーの key。"""
    out = set(map(str, split_keys))
    for ks in holdout_keys.values():
        out |= set(map(str, ks))
    out |= {raw_to_key[k] for k in drop_raw if k in raw_to_key}
    return out


def negative_csr(pos_rows: np.ndarray, mass: np.ndarray, allowed: np.ndarray, ppm: float = PPM):
    """各正解の pool 行について、±ppm 窓の allowed の行（自分を除く）を CSR で返す。mass は昇順。"""
    pos_rows = np.asarray(pos_rows, np.int64)
    tgt = mass[pos_rows]
    tol = tgt * ppm * 1e-6
    a = np.searchsorted(mass, tgt - tol, "left")
    b = np.searchsorted(mass, tgt + tol, "right")
    lens = b - a
    starts = np.repeat(a - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
    flat = np.arange(lens.sum(), dtype=np.int64) + starts
    owner = np.repeat(np.arange(len(pos_rows)), lens)
    keep = allowed[flat] & (flat != pos_rows[owner])
    flat, owner = flat[keep], owner[keep]
    offsets = np.zeros(len(pos_rows) + 1, np.int64)
    np.add.at(offsets, owner + 1, 1)
    return np.cumsum(offsets), flat


def sample_candidates(rng: np.random.Generator, pos: np.ndarray, csr_id: np.ndarray, offsets: np.ndarray,
                      flat: np.ndarray, k: int = K_CAND):
    """候補 0 = 正解。負例は窓から重複なしに k−1 個（足りなければ全部）。残りはパディング（行 0、mask False）。"""
    b = len(pos)
    cand = np.zeros((b, k), np.int64)
    mask = np.zeros((b, k), bool)
    cand[:, 0] = pos
    mask[:, 0] = True
    for i in range(b):
        lo, hi = offsets[csr_id[i]], offsets[csr_id[i] + 1]
        n = hi - lo
        if n == 0:
            continue
        m = min(n, k - 1)
        sel = flat[lo + rng.choice(n, size=m, replace=False)] if n > m else flat[lo:hi][rng.permutation(n)]
        cand[i, 1:1 + m] = sel
        mask[i, 1:1 + m] = True
    return cand, mask


_PERM_CACHE: dict = {}


def epoch_batch(step: int, n: int, batch: int = BATCH, seed: int = SEED) -> np.ndarray:
    """ステップ番号から決まるバッチ（エポック e = default_rng([seed, e]) の順列、128 本ずつ、端数は捨てる）。再開しても同じ。"""
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
    """そのステップの負例を引く乱数（ステップ番号から決まる）。"""
    return np.random.default_rng([seed, 1, step])


def split_val_keys(keys, frac: float = VAL_FRAC, seed: int = VAL_SEED) -> set:
    """正準キーの集合から内部検証の 2% を選ぶ（sorted → default_rng(seed).permutation の先頭 round(frac·n)）。"""
    ks = sorted(set(keys))
    n = int(round(frac * len(ks)))
    perm = np.random.default_rng(seed).permutation(len(ks))
    return {ks[i] for i in perm[:n]}


class EarlyStop:
    """最良値からの相対改善が rel 未満の評価が patience 回続いたら止める。best は最小値（重みを選ぶ値）。"""

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
    """warmup の間の学習率（MSAlign と同じ線形）。warmup の後は ReduceLROnPlateau が決める。"""
    return base * min(1.0, step / max(1, warmup))


def free_gb() -> dict:
    """vm_stat の free と speculative（すぐ使える空き）と inactive（GB = 10^9 バイト）。macOS 以外は None。"""
    import re
    import subprocess
    try:
        txt = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return {"free_gb": None}
    ps = int(re.search(r"page size of (\d+) bytes", txt).group(1))
    get = lambda k: int(re.search(rf"{k}:\s+(\d+)", txt).group(1)) * ps / 1e9  # noqa: E731
    return {"free_gb": round(get("Pages free") + get("Pages speculative"), 2), "inactive_gb": round(get("Pages inactive"), 2)}


def unpack_bits_np(packed: np.ndarray, nbits: int = FP_BITS) -> np.ndarray:
    """Pool.fps と同じ（np.unpackbits の既定 big）。"""
    return np.unpackbits(packed, axis=1)[:, :nbits]


def sha256_file(p, chunk=1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


# ---------------------------------------------------------------- torch の部分

def unpack_bits_torch(packed, nbits: int = FP_BITS, dtype=None):
    """packed (n, ceil(nbits/8)) uint8 → (n, nbits)。unpack_bits_np と同じビット順（上位ビットが先）。"""
    import torch
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    out = bits.reshape(packed.shape[0], -1)[:, :nbits]
    return out.to(dtype or torch.float32)


def build_model(n_adducts: int, adduct_signs, d_ms: int = D_EMB, d_mol: int = FP_BITS):
    """MSAlign models/MSAlign/model.py・modules.py・encoders.py と同じ形（Lightning を使わない書き直し）。"""
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self, d_in, d_hidden, d_out, n_hidden, dropout):
            super().__init__()
            layers, w = [], d_in
            for _ in range(n_hidden):
                layers += [nn.Linear(w, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout)]
                w = d_hidden
            layers.append(nn.Linear(w, d_out))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    class CE(nn.Module):
        def __init__(self, d_out):
            super().__init__()
            pos = torch.arange(0, d_out, 2, dtype=torch.float32)
            self.register_buffer("frequencies", 10_000.0 ** (pos / d_out))
            self.missing = nn.Parameter(torch.empty(d_out))
            nn.init.normal_(self.missing, std=0.02)

        def forward(self, e):
            miss = torch.isnan(e)
            ang = e.nan_to_num().unsqueeze(-1) / 100.0 / self.frequencies
            enc = torch.stack((ang.sin(), ang.cos()), dim=-1).flatten(-2)
            return torch.where(miss.unsqueeze(-1), self.missing.to(enc.dtype), enc)

    class Adduct(nn.Module):
        def __init__(self, n, d_out, signs):
            super().__init__()
            self.category = nn.Embedding(n, d_out // 2)
            self.sign = nn.Embedding(3, d_out // 2)
            nn.init.normal_(self.category.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.sign.weight, mean=0.0, std=0.02)
            s = torch.as_tensor(list(signs))
            self.register_buffer("sign_indices", torch.where(s == 0, 2, (s + 1) // 2).long())

        def forward(self, a):
            return torch.cat((self.category(a), self.sign(self.sign_indices[a])), dim=-1)

    class Align(nn.Module):
        def __init__(self):
            super().__init__()
            part = D_META // 2
            self.ms_pre = MLP(d_ms, D_HIDDEN, D_SHARED, N_MS_PRE, DROPOUT)
            self.energy_encoder = CE(part)
            self.adduct_encoder = Adduct(n_adducts, part, adduct_signs)
            self.ms_post = MLP(D_SHARED + D_META, D_HIDDEN, D_SHARED, N_MS_POST, DROPOUT)
            self.mol = MLP(d_mol, D_HIDDEN, D_SHARED, N_MOL, DROPOUT)
            self.log_temperature = nn.Parameter(torch.tensor(float(TEMP_INIT)).log())

        def encode_spectrum(self, x, ce, adduct):
            x = self.ms_pre(x.float())
            return self.ms_post(torch.cat((x, self.energy_encoder(ce), self.adduct_encoder(adduct).to(x.dtype)), dim=-1))

        def encode_molecule(self, m):
            return self.mol(m)

        def forward(self, x, ce, adduct, mols, mask):
            g = nn.functional.normalize(self.encode_spectrum(x, ce, adduct), dim=-1)
            h = nn.functional.normalize(self.encode_molecule(mols), dim=-1)
            vi = mask.nonzero(as_tuple=False)
            scores = torch.zeros(mask.shape, dtype=torch.float32, device=x.device)
            scores[mask] = ((g[vi[:, 0]] * h).sum(-1) / self.log_temperature.exp()).float()
            return scores.masked_fill(~mask, float("-inf"))

    return Align()


# ---------------------------------------------------------------- ローカルの学習
DEFAULTS = {"embed_dir": "outputs/align_embed", "pool_dir": "data/ext/v4g/casmi26-v2-pool",
            "split": "data/ext/foldsafe/split.parquet", "keys": "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json",
            "out": "outputs/align_train"}


class ShardEmb:
    """埋め込みのシャード（.npy の memmap）から行を集める。全体を 1 本にまとめない（RAM を使わない）。"""

    def __init__(self, embed_dir: Path, n_shards: int):
        import align_embed as E
        self.a = [np.load(Path(embed_dir) / E.shard_names(k)["emb"], mmap_mode="r") for k in range(n_shards)]

    def take(self, shard: np.ndarray, pos: np.ndarray) -> np.ndarray:
        out = np.empty((len(shard), D_EMB), np.float32)
        for k in np.unique(shard):
            m = shard == k
            order = np.argsort(pos[m])
            rows = np.asarray(self.a[k][pos[m][order]], np.float32)
            out[np.flatnonzero(m)[order]] = rows
        return out


def prepare(root: Path, embed_dir: Path, holdout_keys: dict) -> dict:
    """入力を読み、DROP_KEY・内部検証・負例の CSR・leak_audit を作る（学習の前に 1 回）。"""
    import pandas as pd
    import align_embed as E
    import sanpier_fold0 as sf
    pool = Path(root) / DEFAULTS["pool_dir"]
    p_meta, p_fp, p_ts, p_bits = (pool / "pool_meta.parquet", pool / "pool_fp.npy", pool / "train_structs.parquet",
                                  pool / "fp_bits.npy")
    sha = {"pool_meta": sha256_file(p_meta), "train_structs": sha256_file(p_ts), "fp_bits": sha256_file(p_bits),
           "pool_fp": sha256_file(p_fp)}
    assert sha == {"pool_meta": POOL_META_SHA256, "train_structs": TRAIN_STRUCTS_SHA256, "fp_bits": FP_BITS_SHA256,
                   "pool_fp": POOL_FP_SHA256}, sha
    meta = pd.read_parquet(p_meta, columns=["key", "mass"])
    assert len(meta) == POOL_ROWS and bool(np.all(np.diff(meta.mass.values) >= 0))
    assert len(np.load(p_bits)) == FP_BITS
    pool_fp = np.load(p_fp)
    assert pool_fp.shape == (POOL_ROWS, (FP_BITS + 7) // 8) and pool_fp.dtype == np.uint8, pool_fp.shape
    ts = pd.read_parquet(p_ts, columns=["inchikey14", "key"])
    raw_to_key = dict(zip(ts.inchikey14, ts.key))
    key_to_row = dict(zip(meta.key.values, range(len(meta))))
    split = pd.read_parquet(Path(root) / DEFAULTS["split"], columns=["inchikey14", "key", "fold"])
    assert sorted(holdout_keys) == sorted(HOLDOUTS)
    drop_raw = sf.drop_inchikey14(split.inchikey14, holdout_keys)
    assert len(drop_raw) == EXPECTED_DROP, len(drop_raw)
    drop_key = drop_key_space(drop_raw, raw_to_key, holdout_keys, split.key)
    allowed = ~meta.key.isin(drop_key).values
    erep = json.loads((embed_dir / "embed_report.json").read_text())
    assert erep["status"] == "ok", erep["status"]
    chash, n_shards = erep["config_hash"], erep["n_shards"]
    tr, q = [], []
    for k in range(n_shards):
        assert E.shard_is_done(embed_dir, k, chash), k
        t = pd.read_parquet(embed_dir / E.shard_names(k)["idx"])
        t["shard"], t["pos"] = k, np.arange(len(t))
        q.append(t[t.kind == 1]); tr.append(t[t.kind == 0])
    tr = pd.concat(tr, ignore_index=True); q = pd.concat(q, ignore_index=True)
    tr["key"] = tr.inchikey14.map(raw_to_key)
    keep = tr.key.notna().values & ~tr.key.isin(drop_key).values & ~tr.inchikey14.isin(drop_raw).values
    tr = tr[keep].reset_index(drop=True)
    val_keys = split_val_keys(tr.key.values, VAL_FRAC, VAL_SEED)
    is_val = tr.key.isin(val_keys).values
    pos_row = np.array([key_to_row[k] for k in tr.key.values], np.int64)
    val_rows = meta.key.isin(val_keys).values
    leak = sf.leak_audit(set(tr.inchikey14), zip(split.inchikey14, split.fold), holdout_keys)
    leak_key = sf.leak_audit(set(tr.key), zip(split.key, split.fold), holdout_keys)
    # 学習: 内部検証の構造は正解にも負例にも使わない。内部検証: 負例は allowed 全体（自分以外）
    allow_tr = allowed & ~val_rows
    up_tr, cid_tr = np.unique(pos_row[~is_val], return_inverse=True)
    off_tr, flat_tr = negative_csr(up_tr, meta.mass.values.astype(np.float64), allow_tr)
    up_va, cid_va = np.unique(pos_row[is_val], return_inverse=True)
    off_va, flat_va = negative_csr(up_va, meta.mass.values.astype(np.float64), allowed)
    leak_all = {"raw_inchikey14": leak, "canonical_key": leak_key,
                "negatives_in_drop_key": int(pd.Series(meta.key.values[np.concatenate([flat_tr, flat_va])]).isin(drop_key).sum()),
                "positives_in_drop_key": int(tr.key.isin(drop_key).sum()),
                "train_negatives_in_val": int(val_rows[flat_tr].sum()),
                "train_positives_in_val": int(np.isin(up_tr, np.flatnonzero(val_rows)).sum())}
    assert leak["total"] == 0 and leak_key["total"] == 0 and all(
        leak_all[k] == 0 for k in ("negatives_in_drop_key", "positives_in_drop_key", "train_negatives_in_val",
                                   "train_positives_in_val")), leak_all
    vocab, signs = adduct_vocab(tr.adduct.values[~is_val])
    return dict(meta=meta, pool_fp=pool_fp, tr=tr, q=q, is_val=is_val, pos_row=pos_row, val_keys=val_keys,
                off_tr=off_tr, flat_tr=flat_tr, cid_tr=cid_tr, off_va=off_va, flat_va=flat_va, cid_va=cid_va,
                vocab=vocab, signs=signs, leak=leak_all, erep=erep, n_shards=n_shards, input_sha256=sha,
                drop_key=len(drop_key), allowed=int(allowed.sum()))


def pick_device(want: str = "auto") -> str:
    import torch
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:   # MPS で動かなければ CPU
            m = build_model(3, [1, -1, 0], d_ms=8, d_mol=16).to("mps")
            mask = torch.ones((2, 3), dtype=torch.bool, device="mps")
            s = m(torch.rand(2, 8, device="mps"), torch.tensor([20.0, float("nan")], device="mps"),
                  torch.tensor([0, 2], device="mps"), torch.rand(6, 16, device="mps"), mask)
            s.float().mean().backward()
            if bool(torch.isfinite(s).all()):
                return "mps"
        except Exception:  # noqa: BLE001
            pass
    return "cpu"


def train(root=".", embed_dir=None, out_dir=None, holdout_keys=None, device="auto", resume=False,
          max_steps=MAX_STEPS, eval_every=EVAL_EVERY, t0=None) -> dict:
    import pandas as pd
    import torch
    import align_speed_trial as A
    t0 = time.time() if t0 is None else t0
    root = Path(root)
    embed_dir = Path(embed_dir or root / DEFAULTS["embed_dir"])
    out = Path(out_dir or root / DEFAULTS["out"]); out.mkdir(parents=True, exist_ok=True)
    if holdout_keys is None:
        holdout_keys = json.loads((root / DEFAULTS["keys"]).read_text())
    rpath, logp, statep = out / "train_report.json", out / "train_log.jsonl", out / "state.pt"
    dev = pick_device(device)
    rep = {"issue": 109, "status": "running", "device": dev, "numerics": NUMERICS_NOTE.get(dev.split(":")[0]),
           "config": {"MSALIGN_COMMIT": MSALIGN_COMMIT, "SEED": SEED, "D_SHARED": D_SHARED, "D_HIDDEN": D_HIDDEN,
                      "D_META": D_META, "N_MS_PRE": N_MS_PRE, "N_MS_POST": N_MS_POST, "N_MOL": N_MOL,
                      "DROPOUT": DROPOUT, "TEMP_INIT": TEMP_INIT, "K_CAND": K_CAND, "BATCH": BATCH, "LR": LR,
                      "WD": WD, "WARMUP": WARMUP, "MAX_STEPS": max_steps, "CLIP": CLIP, "PPM": PPM,
                      "VAL_FRAC": VAL_FRAC, "VAL_SEED": VAL_SEED, "VAL_CAND_SEED": VAL_CAND_SEED, "EVAL_EVERY": eval_every,
                      "STOP_REL": STOP_REL, "STOP_PATIENCE": STOP_PATIENCE, "PLATEAU_FACTOR": PLATEAU_FACTOR,
                      "PLATEAU_PATIENCE": PLATEAU_PATIENCE, "MIN_LR": MIN_LR, "weights": "best internal-val loss"},
           "mem_start": free_gb()}

    def save():
        rep["elapsed_s"] = round(time.time() - t0, 1)
        rep["ram_peak"] = A.ram_peak()
        if dev == "mps":
            rep["mps_driver_allocated_max"] = max(rep.get("mps_driver_allocated_max", 0),
                                                  int(torch.mps.driver_allocated_memory()))
        rpath.write_text(json.dumps(rep, indent=1, default=str))

    try:
        ts = time.time()
        D = prepare(root, embed_dir, holdout_keys)
        tr, q, is_val = D["tr"], D["q"], D["is_val"]
        tr_i, va_i = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        rep["data"] = {"n_train_spectra": int(len(tr_i)), "n_val_spectra": int(len(va_i)),
                       "n_val_structures": len(D["val_keys"]), "n_train_structures": int(tr.key[~is_val].nunique()),
                       "n_query": len(q), "drop_key": D["drop_key"], "pool_allowed": D["allowed"],
                       "n_adducts": len(D["vocab"]), "embed_config_hash": D["erep"]["config_hash"],
                       "neg_median_train": float(np.median(np.diff(D["off_tr"]))), "input_sha256": D["input_sha256"]}
        rep["leak_audit"] = D["leak"]
        rep["stages"] = {"prepare_s": round(time.time() - ts, 1)}
        save()
        emb = ShardEmb(embed_dir, D["n_shards"])
        aid = adduct_ids(tr.adduct.values, D["vocab"])
        ce = tr.ce.values.astype(np.float32)
        shard, posv = tr.shard.values, tr.pos.values
        # 内部検証: 埋め込み・候補を一度だけ固定
        va_emb = emb.take(shard[va_i], posv[va_i])
        va_cand, va_mask = sample_candidates(np.random.default_rng(VAL_CAND_SEED), D["pos_row"][va_i], D["cid_va"],
                                             D["off_va"], D["flat_va"], K_CAND)
        torch.manual_seed(SEED)
        model = build_model(len(D["vocab"]) + 1, D["signs"], d_ms=D_EMB, d_mol=FP_BITS).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=PLATEAU_FACTOR,
                                                             patience=PLATEAU_PATIENCE, min_lr=MIN_LR)
        amp = dev == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        stopper = EarlyStop(STOP_REL, STOP_PATIENCE)
        fp_d = torch.from_numpy(D["pool_fp"]).to(dev) if dev == "cuda" else None
        step, acc, n_acc = 0, 0.0, 0
        if resume and statep.exists():
            st = torch.load(statep, map_location="cpu", weights_only=False)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); plateau.load_state_dict(st["plateau"])
            scaler.load_state_dict(st["scaler"]); stopper.load(st["stopper"]); step = st["step"]
            torch.set_rng_state(st["torch_rng"])
            rep["resumed_from_step"] = step
        rep["data"]["params"] = int(sum(p.numel() for p in model.parameters()))
        n_tr = len(tr_i)

        def mols_for(cand_rows):
            if fp_d is not None:
                return unpack_bits_torch(fp_d[torch.from_numpy(cand_rows).to(dev)], FP_BITS,
                                         torch.float16 if amp else torch.float32)
            return unpack_bits_torch(torch.from_numpy(D["pool_fp"][cand_rows]).to(dev), FP_BITS, torch.float32)

        def val_loss() -> float:
            model.eval()
            tot, n = 0.0, 0
            with torch.inference_mode():
                for i in range(0, len(va_i), BATCH):
                    sl = slice(i, i + BATCH)
                    m = va_mask[sl]
                    sc = model(torch.from_numpy(va_emb[sl]).to(dev), torch.from_numpy(ce[va_i[sl]]).to(dev),
                               torch.from_numpy(aid[va_i[sl]]).to(dev), mols_for(va_cand[sl][m]).float(),
                               torch.from_numpy(m).to(dev))
                    l = torch.nn.functional.cross_entropy(sc.float(), torch.zeros(len(m), dtype=torch.long, device=dev),
                                                          reduction="sum")
                    tot += float(l); n += len(m)
            model.train()
            return tot / n

        model.train()
        ts = time.time()
        stop_reason = None
        while step < max_steps:
            if step < WARMUP:
                for gr in opt.param_groups:
                    gr["lr"] = warmup_lr(step, LR, WARMUP)
            li = epoch_batch(step, n_tr, BATCH, SEED)
            bi = tr_i[li]
            cand, mask = sample_candidates(step_rng(step), D["pos_row"][bi], D["cid_tr"][li],
                                           D["off_tr"], D["flat_tr"], K_CAND)
            x = torch.from_numpy(emb.take(shard[bi], posv[bi])).to(dev)
            m_d = torch.from_numpy(mask).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                scores = model(x, torch.from_numpy(ce[bi]).to(dev), torch.from_numpy(aid[bi]).to(dev),
                               mols_for(cand[mask]), m_d)
            loss = torch.nn.functional.cross_entropy(scores.float(), torch.zeros(len(bi), dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt); scaler.update()
            step += 1
            acc += float(loss.detach()); n_acc += 1
            if step % 200 == 0:
                fg = free_gb().get("free_gb")
                if fg is not None and fg < MIN_RUN_FREE_GB:
                    stop_reason = f"paused_low_memory free={fg}GB"
            if stop_reason:   # 空きが足りない: 評価せずに state だけ書いて止める（止め方の数え方を変えない）
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "plateau": plateau.state_dict(),
                            "scaler": scaler.state_dict(), "stopper": stopper.state(), "step": step,
                            "torch_rng": torch.get_rng_state()}, statep)
                break
            if step % eval_every == 0 or step == max_steps:
                vl = val_loss()
                if step >= WARMUP:
                    plateau.step(vl)
                u = stopper.update(vl, step)
                if u["is_best"]:
                    torch.save(model.state_dict(), out / "model_best.pt")
                row = {"step": step, "train_loss": acc / max(n_acc, 1), "val_loss": vl, "lr": opt.param_groups[0]["lr"],
                       "temperature": float(model.log_temperature.detach().exp()), **u,
                       "best": stopper.best, "best_step": stopper.best_step, "t": round(time.time() - t0, 1),
                       "steps_per_s": n_acc / max(time.time() - ts, 1e-9), "mem": free_gb()}
                with open(logp, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print("EVAL", json.dumps(row), flush=True)
                acc, n_acc, ts = 0.0, 0, time.time()
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "plateau": plateau.state_dict(),
                            "scaler": scaler.state_dict(), "stopper": stopper.state(), "step": step,
                            "torch_rng": torch.get_rng_state()}, statep)
                save()
                if u["stop"]:
                    stop_reason = "converged"
                    break
        if stop_reason is None:
            stop_reason = "max_steps"
        rep["train"] = {"steps": step, "stop_reason": stop_reason, "best_val_loss": stopper.best,
                        "best_step": stopper.best_step}
        if stop_reason.startswith("paused"):
            rep["status"] = "paused"
            save()
            return rep
        # ---- 出力（最良の重み、fp32 の評価モード）
        model.load_state_dict(torch.load(out / "model_best.pt", map_location=dev))
        model.eval()
        torch.save(model.mol.state_dict(), out / "mol_mlp_best.pt")
        cfg = dict(rep["config"], adduct_vocab=D["vocab"], adduct_signs=D["signs"], d_ms=D_EMB, d_mol=FP_BITS,
                   embed_config_hash=D["erep"]["config_hash"], pool_meta_sha256=POOL_META_SHA256,
                   best_step=stopper.best_step)
        (out / "align_config.json").write_text(json.dumps(cfg, indent=1))
        with torch.inference_mode():
            qemb = ShardEmb(embed_dir, D["n_shards"]).take(q.shard.values, q.pos.values)
            g = torch.nn.functional.normalize(model.encode_spectrum(
                torch.from_numpy(qemb).to(dev), torch.from_numpy(q.ce.values.astype(np.float32)).to(dev),
                torch.from_numpy(adduct_ids(q.adduct.values, D["vocab"])).to(dev)).float(), dim=-1)
            np.save(out / "g_query.npy", g.cpu().numpy())
            q[["row", "holdout", "molecule_id", "spectrum_id", "inchikey14", "adduct", "ce"]].to_parquet(
                out / "query_index.parquet", index=False)
            hp = np.lib.format.open_memmap(out / "h_pool.npy", mode="w+", dtype=np.float16, shape=(POOL_ROWS, D_SHARED))
            for i in range(0, POOL_ROWS, 4096):
                rows = np.arange(i, min(i + 4096, POOL_ROWS))
                h = model.encode_molecule(mols_for(rows).float()).float()
                hp[rows] = torch.nn.functional.normalize(h, dim=-1).cpu().numpy()
            hp.flush(); del hp
        log = [json.loads(x) for x in logp.read_text().splitlines() if x.strip()]
        pd.DataFrame([{k: r[k] for k in ("step", "train_loss", "val_loss", "lr")} for r in log]).to_csv(
            out / "learning_curve.csv", index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot([r["step"] for r in log], [r["train_loss"] for r in log], label="train")
            ax.plot([r["step"] for r in log], [r["val_loss"] for r in log], label="internal val")
            ax.set_xlabel("step"); ax.set_ylabel("InfoNCE"); ax.legend(); fig.tight_layout()
            fig.savefig(out / "learning_curve.png", dpi=100); plt.close(fig)
        except ImportError:
            pass
        rep["outputs"] = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                          for p in sorted(out.iterdir()) if p.suffix in (".pt", ".npy", ".parquet", ".json", ".csv")
                          and p.name not in ("train_report.json", "state.pt")}
        rep["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        import traceback
        rep["status"] = "error"
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-4000:]
    save()
    return rep


def bench_mlp(device: str = "auto", steps: int = 30, warm: int = 3, n_val_batches: int = 5) -> dict:
    """本番と同じ形（バッチ 128 × 候補 128、10,226 ビット、fp32）の学習ステップと評価のバッチを合成データで測る。"""
    import torch
    import align_speed_trial as A
    dev = pick_device(device)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = build_model(40, [1] * 20 + [-1] * 19 + [0] + [0], d_ms=D_EMB, d_mol=FP_BITS).to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    packed = rng.integers(0, 256, (20_000, (FP_BITS + 7) // 8), dtype=np.uint8)
    mask_np = np.ones((BATCH, K_CAND), bool); mask_np[:, 80:] = rng.random((BATCH, K_CAND - 80)) < 0.3   # 負例の中央値 79 の見込み
    mask_np[:, 0] = True
    def one(train=True):
        x = torch.from_numpy(rng.normal(size=(BATCH, D_EMB)).astype(np.float32)).to(dev)
        rows = rng.integers(0, len(packed), int(mask_np.sum()))
        mols = unpack_bits_torch(torch.from_numpy(packed[rows]).to(dev), FP_BITS, torch.float32)
        m = torch.from_numpy(mask_np).to(dev)
        ce = torch.full((BATCH,), 20.0, device=dev); ad = torch.zeros(BATCH, dtype=torch.long, device=dev)
        if train:
            sc = model(x, ce, ad, mols, m)
            loss = torch.nn.functional.cross_entropy(sc, torch.zeros(BATCH, dtype=torch.long, device=dev))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
            return float(loss.detach())
        with torch.inference_mode():
            return float(model(x, ce, ad, mols, m).float().mean())
    sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (torch.cuda.synchronize if dev == "cuda" else (lambda: None))
    for _ in range(warm):
        one()
    sync(); t = time.time()
    for _ in range(steps):
        one()
    sync(); dt = time.time() - t
    model.eval(); t = time.time()
    for _ in range(n_val_batches):
        one(False)
    sync(); dv = time.time() - t
    out = {"device": dev, "steps": steps, "steps_per_s": steps / dt, "val_batches_per_s": n_val_batches / dv,
           "valid_candidates_per_step": int(mask_np.sum()), "params": int(sum(p.numel() for p in model.parameters())),
           "ram_peak": A.ram_peak()}
    if dev == "mps":
        out["mps_driver_allocated"] = int(torch.mps.driver_allocated_memory())
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="整列 MLP をローカルで学習（#109 の止め方）")
    ap.add_argument("--embed-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", help="auto（cuda → mps → cpu）| mps | cpu")
    ap.add_argument("--resume", action="store_true", help="out/state.pt から続ける")
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    ap.add_argument("--bench", type=int, default=0, help="合成データでこのステップ数だけ速度を測って終える")
    args = ap.parse_args()
    for k, v in (("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")):
        if os.environ.get(k) != v:
            raise SystemExit(f"set {k}={v} before starting (torch reads it at import)")
    fg = free_gb()
    if fg.get("free_gb") is not None and fg["free_gb"] < args.min_free_gb:
        raise SystemExit(f"vm_stat free {fg} < {args.min_free_gb} GB")
    own = own_heavy_jobs()
    if own:
        raise SystemExit(f"another heavy job of this repo is running: {own}")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts/train"))
    if args.bench:
        r = bench_mlp(args.device, args.bench)
        r["mem_start"] = fg
        (root / "outputs/align_train").mkdir(parents=True, exist_ok=True)
        (root / "outputs/align_train/bench_mlp_local.json").write_text(json.dumps(r, indent=1))
        print(json.dumps(r, indent=1))
        return
    r = train(root=root, embed_dir=args.embed_dir, out_dir=args.out, device=args.device, resume=args.resume)
    print(json.dumps({k: r.get(k) for k in ("status", "error", "device", "data", "leak_audit", "train", "elapsed_s")},
                     indent=1, default=str))


def own_heavy_jobs() -> list[str]:
    """このリポジトリの重いジョブ（align_local_embed / align_train / local_fold0_logits）が自分以外に動いていないか。"""
    import subprocess
    try:
        txt = subprocess.run(["ps", "-Ao", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    me = os.getpid()
    out = []
    for line in txt.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        argv0 = os.path.basename(cmd.strip().split(" ")[0]) if cmd.strip() else ""
        if pid.isdigit() and int(pid) != me and argv0.startswith("python") and any(
                s in cmd for s in ("align_local_embed.py", "align_train.py", "local_fold0_logits.py")):
            out.append(line.strip()[:160])   # シェルや /usr/bin/time の行（argv0 が python でない）は数えない
    return out


if __name__ == "__main__":
    main()
