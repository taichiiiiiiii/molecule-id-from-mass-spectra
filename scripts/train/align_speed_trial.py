#!/usr/bin/env python3
"""整列チャネル（凍結 DreaMS ssl_model.ckpt ＋ 整列 MLP）の速度を Kaggle T4 で 1 回測る（Issue #99、計画 2026-11-14 昼 項目 5）。

学習の本番ではない。#99 の条件 (2)「1 万スペクトルの速度を測り、外挿が 6 時間を超えるなら止める」の材料だけを作る。
事前登録: https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/99#issuecomment-6870838727

ノートの中では、このファイルを /kaggle/working に書いて import し、`run()` を呼ぶ。
出力は align_speed_report.json だけ。どの段で止まっても report を書いて終える（数値は作らない）。

torch はモジュールの先頭で import しない（CI は torch を入れないので、純粋な関数だけをテストする）。
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import json
import math
import os
import resource
import subprocess
import sys
import time
import types
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = "https://zenodo.org/api/records/10997887/files/ssl_model.ckpt/content"
CKPT_SIZE = 1_392_710_212
CKPT_MD5 = "4deee91dd4486c2b6cbcde1f200b8f50"  # Zenodo API files[].checksum（2026-11-14 12:13 JST に取得）
DREAMS_COMMIT = "dbec3a0b514a99e5056cfccde4559fda8cfe8129"
DREAMS_ZIP_URL = f"https://github.com/pluskal-lab/DreaMS/archive/{DREAMS_COMMIT}.zip"

N_SPECTRA = 10_000
EMB_BATCH = 32              # DreaMS API（dreams_intermediates / dreams_embeddings）の既定
PEAKS = (60, 100)           # 60 = ckpt の max_peaks_n・dreams_intermediates の既定、100 = PreTrainedModel.from_name の既定
TRAIN_COLS = ["inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "collision_energy_ev", "adduct"]
TRAIN_ROWS = 2_539_608
READ_BATCH = 8192          # parquet を読む単位（pyarrow iter_batches）

# 整列 MLP（#99 の文書 (e) の形で仮に置いたもの。条件 (1) の前なので本番の代表値にならない可能性がある）
MLP_BATCH = 128
MLP_K_NEG = 128
MLP_STEPS_PROD = 30_000     # massspecgym_formula.yaml の n_max_steps
MLP_WARMUP_EXCLUDED = 3
MLP_LR = 1e-4
MLP_WD = 1e-6
MLP_CLIP = 5.0
MLP_TEMP_INIT = 0.07
MLP_DROPOUT = 0.30
FP_BITS = 10_226
D_OUT = 512
MOL_HIDDEN = 2048
MOL_LAYERS = 3
MS_HIDDEN = 512
MS_POST_LAYERS = 2
META_DIM = 64
FP_DENSITY = 0.03           # 合成の指紋の 1 の割合（密な行列積なので計算量には効かない）
FP_BANK = 20_000

# 外挿の前提（#99 の数字）
PROD_TRAIN_SPECTRA = 1_982_405
PROD_TRAIN_STRUCTS = 220_443
PROD_QUERY_SPECTRA = 4_756
LIMIT_HOURS = 6.0

BUDGET_S = 1800             # ノート全体の打ち切り
STAGE_CUTOFF_S = 1500       # これを過ぎたら新しい段を始めない
DOWNLOAD_TIMEOUT_S = 720

# 前向きに要らない外部パッケージ。import の前に stub を入れる（すでに import 済みのものは本物を使う）
STUB_TOP = ("pytorch_lightning", "torchmetrics", "matchms", "pandarallel", "igraph", "umap", "pynndescent",
            "plotly", "h5py", "rdkit", "numba", "networkx", "sklearn", "scipy", "seaborn", "ase", "wandb",
            "fire", "huggingface_hub", "matplotlib", "tqdm", "pyopenms", "molplotly", "pylab", "pyteomics", "lxml",
            "click")


class Md5Mismatch(RuntimeError):
    pass


class StageTimeout(RuntimeError):
    pass


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def md5_stream(chunks, expected_md5: str, expected_size: int | None = None) -> dict:
    """バイト列の反復から md5 とサイズを計算し、完全な値で照合する。違えば Md5Mismatch。"""
    h, n = hashlib.md5(), 0
    for c in chunks:
        h.update(c)
        n += len(c)
    got = h.hexdigest()
    info = {"expected_md5": expected_md5, "md5": got, "size": n, "expected_size": expected_size}
    if len(expected_md5) != 32:
        raise ValueError(f"expected md5 must be the full 32-hex value, got {expected_md5!r}")
    if got != expected_md5 or (expected_size is not None and n != expected_size):
        raise Md5Mismatch(json.dumps(info))
    return info


def verify_md5_file(path, expected_md5: str, expected_size: int | None = None) -> dict:
    def it():
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 22)
                if not b:
                    return
                yield b
    return md5_stream(it(), expected_md5, expected_size)


def extrapolate(sps_by_peaks: dict, mlp_steps_per_s: float, read_s: float, rows_read: int,
                fixed_s: float) -> dict:
    """事前登録の式。sps_by_peaks = {ピーク数: fp32 の（前処理＋前向き）スペクトル/秒}。"""
    n_embed = PROD_TRAIN_SPECTRA + PROD_QUERY_SPECTRA
    worst_peaks = min(sps_by_peaks, key=lambda k: sps_by_peaks[k])
    sps = sps_by_peaks[worst_peaks]
    t_read = read_s * (TRAIN_ROWS / rows_read)
    t_embed = n_embed / sps
    t_mlp = MLP_STEPS_PROD / mlp_steps_per_s
    total = t_read + t_embed + t_mlp + fixed_s
    hours = total / 3600.0
    return {
        "formula": "T_total = T_read + T_embed + T_mlp + T_fixed; T_read = read_s*(2539608/rows_read); "
                   "T_embed = (1982405+4756)/min_peaks(sps_fp32); T_mlp = 30000/steps_per_s_fp32; "
                   "T_fixed = download+code+load seconds of this run",
        "premises": {"train_spectra": PROD_TRAIN_SPECTRA, "train_structures": PROD_TRAIN_STRUCTS,
                     "query_spectra": PROD_QUERY_SPECTRA, "mlp_steps": MLP_STEPS_PROD, "train_rows": TRAIN_ROWS,
                     "mlp_config": "provisional (#99 doc (e) shape, before condition (1))",
                     "precision": "fp32 (fp16 autocast is reference only)"},
        "worst_peaks": worst_peaks, "sps_used": sps,
        "T_read_s": t_read, "T_embed_s": t_embed, "T_mlp_s": t_mlp, "T_fixed_s": fixed_s,
        "T_total_s": total, "T_total_hours": hours, "limit_hours": LIMIT_HOURS,
        "decision": "candidate_for_next_plan" if hours <= LIMIT_HOURS else "stop_production",
    }


class _StubMeta(type):
    """何を取り出しても stub のクラスを返す。呼ぶと（デコレータとして関数 1 つなら）その関数、それ以外は自分を返す。"""

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _stub_class(f"{cls.__name__}.{name}")

    def __call__(cls, *a, **k):
        if len(a) == 1 and not k and callable(a[0]) and not isinstance(a[0], _StubMeta):
            return a[0]
        return cls

    def __getitem__(cls, k):
        return cls

    def __setitem__(cls, k, v):
        pass

    def __iter__(cls):
        return iter(())

    def __or__(cls, other):
        return cls

    __ror__ = __or__


def _stub_class(name: str):
    return _StubMeta(name, (), {})


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        v = _stub_class(f"{self.__name__}.{name}")
        setattr(self, name, v)
        return v


def _lightning_stub(name: str):
    import torch

    class LightningModule(torch.nn.Module):
        def save_hyperparameters(self, *a, **k):
            pass

        @property
        def device(self):  # pytorch_lightning の LightningModule.device と同じ意味（DreaMS の forward が読む）
            return next(self.parameters()).device

        @property
        def dtype(self):
            return next(self.parameters()).dtype

    m = _StubModule(name)
    m.LightningModule = LightningModule
    return m


class StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, tops):
        self.tops = set(tops)
        self.made = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.tops:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        self.made.add(spec.name)
        if spec.name == "pytorch_lightning":
            m = _lightning_stub(spec.name)
        else:
            m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        pass


def install_stubs(tops=STUB_TOP) -> StubFinder:
    active = [t for t in tops if t not in sys.modules]
    f = StubFinder(active)
    sys.meta_path.insert(0, f)
    return f


# ---------------------------------------------------------------- Kaggle で動く段

def _now(t0):
    return round(time.time() - t0, 3)


def ram_peak() -> dict:
    out = {"ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            out[p] = int(Path(p).read_text().strip())
        except Exception:
            pass
    return out


def download_ckpt(dest: Path, url=ZENODO_URL, expected_md5=CKPT_MD5, expected_size=CKPT_SIZE,
                  timeout_s=DOWNLOAD_TIMEOUT_S) -> dict:
    t0 = time.time()

    def chunks():
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    return
                if time.time() - t0 > timeout_s:
                    raise StageTimeout(f"download exceeded {timeout_s}s")
                f.write(b)
                yield b
    info = md5_stream(chunks(), expected_md5, expected_size)
    info["seconds"] = round(time.time() - t0, 3)
    return info


def fetch_dreams_code(dest: Path, url=DREAMS_ZIP_URL, commit=DREAMS_COMMIT) -> dict:
    t0 = time.time()
    zpath = dest / "dreams.zip"
    with urllib.request.urlopen(url, timeout=120) as r:
        zpath.write_bytes(r.read())
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    with zipfile.ZipFile(zpath) as z:
        tops = {n.split("/")[0] for n in z.namelist()}
        assert tops == {f"DreaMS-{commit}"}, tops
        z.extractall(dest)
    root = dest / f"DreaMS-{commit}"
    assert (root / "dreams/models/dreams/dreams.py").is_file()
    return {"root": str(root), "zip_sha256": sha, "seconds": round(time.time() - t0, 3)}


def load_dreams(dreams_root: str, ckpt, device: str = "cpu", stub_tops=STUB_TOP):
    """DreaMS（commit 固定）のモデルを作り、ssl_model.ckpt の重みを strict に読む。ckpt はパスか file-like。"""
    import torch  # noqa: F401
    if dreams_root not in sys.path:
        sys.path.insert(0, dreams_root)
    finder = install_stubs(stub_tops)
    try:
        import dreams.utils.dformats as dfm
        import dreams.utils.data as ddata
        from dreams.models.dreams.dreams import DreaMS
        msml = types.ModuleType("msml"); msml.__path__ = []
        msml_utils = types.ModuleType("msml.utils"); msml_utils.__path__ = []
        sys.modules.setdefault("msml", msml)
        sys.modules.setdefault("msml.utils", msml_utils)
        sys.modules.setdefault("msml.utils.dformats", dfm)
        sys.modules.setdefault("msml.utils.data", ddata)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        hp = state["hyper_parameters"]
        model = DreaMS(hp["args"], hp["spec_preproc"])
        missing = model.load_state_dict(state["state_dict"], strict=True)
        for name in ("ff_out", "mz_masking_loss", "ro_out"):   # PreTrainedModel.remove_unused_backbone_parameters
            if hasattr(model, name):
                delattr(model, name)
        model = model.to(device).eval()
        info = {"stubbed_modules": sorted(finder.made), "n_layers": model.n_layers, "d_model": model.d_model,
                "params_backbone": int(sum(p.numel() for p in model.parameters())),
                "ckpt_global_step": state.get("global_step"), "ckpt_pl_version": state.get("pytorch-lightning_version"),
                "load_state_dict": str(missing)}
        del state
        return model, ddata.SpectrumPreprocessor, dfm.DataFormatA, info
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)


def find_inputs(root=Path("/kaggle/input")):
    trains = [p for p in sorted(root.rglob("train.parquet")) if "casmi26" in str(p).lower()]
    splits = [p for p in sorted(root.rglob("split.parquet")) if "fold-safe-fpnet" in str(p)]
    return (trains[0] if trains else None), (splits[0] if splits else None)


def select_spectra(train_path, split_path, n=N_SPECTRA) -> tuple[list, dict]:
    import pyarrow.parquet as pq
    t0 = time.time()
    drop, source = set(), "fallback_head"
    if split_path is not None:
        try:
            drop = set(pq.read_table(split_path, columns=["inchikey14"]).column("inchikey14").to_pylist())
            source = "foldsafe_split_excluded"
        except Exception as e:  # split が読めなければ先頭から
            source = f"fallback_head ({type(e).__name__})"
            drop = set()
    pf = pq.ParquetFile(train_path)
    out, rows_read = [], 0   # rows_read = parquet から読んで展開した行数（外挿の T_read の分母）
    for batch in pf.iter_batches(batch_size=READ_BATCH, columns=TRAIN_COLS):
        t = batch.to_pydict()
        m = len(t["inchikey14"])
        rows_read += m
        for i in range(m):
            if t["inchikey14"][i] in drop:
                continue
            mz, it = t["ms2_mzs"][i], t["ms2_normalized_intensities"][i]
            if not mz:
                continue
            ce = t["collision_energy_ev"][i]
            out.append((mz, it, float(t["precursor_mz"][i]), (ce[0] if ce else None), t["adduct"][i]))
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    info = {"data_source": source, "n_drop_structures": len(drop), "rows_read": rows_read, "n_selected": len(out),
            "seconds": round(time.time() - t0, 3)}
    return out, info


def preprocess(specs, SpectrumPreprocessor, DataFormatA, n_peaks):
    import numpy as np
    pp = SpectrumPreprocessor(dformat=DataFormatA(), n_highest_peaks=n_peaks)
    t0 = time.time()
    arr = np.stack([pp(np.array([mz, it], dtype=np.float64), prec_mz=prec, high_form=False)
                    for mz, it, prec, _, _ in specs]).astype(np.float32)
    return arr, time.time() - t0


def embed_speed(model, arr, device, amp: bool, batch=EMB_BATCH, deadline=None):
    import torch
    x = torch.from_numpy(arr)
    out = torch.zeros((len(arr), model.d_model), dtype=torch.float32)
    store = {}
    last = model.transformer_encoder.ffs[model.n_layers - 1]
    h = last.register_forward_hook(lambda mod, inp, o: store.__setitem__("e", o[:, 0, :]))
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    try:
        with torch.inference_mode():   # 2 バッチの空回し（計測から除く）
            for i in range(0, min(2 * batch, len(arr)), batch):
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                    model(x[i:i + batch].to(device))
        if device == "cuda":
            torch.cuda.synchronize()
        t0, done = time.time(), 0
        with torch.inference_mode():
            for i in range(0, len(arr), batch):
                if deadline is not None and time.time() > deadline:
                    raise StageTimeout(f"embedding stopped at {done} spectra")
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
                    model(x[i:i + batch].to(device, non_blocking=True))
                out[i:i + batch] = store["e"].float().cpu()
                done = min(i + batch, len(arr))
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
    finally:
        h.remove()
    mem = ({"max_allocated": torch.cuda.max_memory_allocated(), "max_reserved": torch.cuda.max_memory_reserved()}
           if device == "cuda" else {})
    return out, {"seconds_forward": dt, "spectra": len(arr), "forward_sps": len(arr) / dt, "gpu_mem": mem,
                 "finite": bool(torch.isfinite(out).all())}


def build_mlps(device):
    import torch
    from torch import nn

    def mlp(d_in, d_hidden, n_hidden, d_out):
        layers, d = [], d_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(MLP_DROPOUT)]
            d = d_hidden
        layers.append(nn.Linear(d, d_out))
        return nn.Sequential(*layers)

    class Align(nn.Module):
        def __init__(self, n_adducts):
            super().__init__()
            self.adduct = nn.Embedding(n_adducts, META_DIM // 2)
            self.ce = nn.Linear(1, META_DIM // 2)
            self.g = mlp(1024 + META_DIM, MS_HIDDEN, MS_POST_LAYERS, D_OUT)
            self.h = mlp(FP_BITS, MOL_HIDDEN, MOL_LAYERS, D_OUT)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / MLP_TEMP_INIT)))

        def forward(self, emb, adduct_id, ce, fps):
            meta = torch.cat([self.adduct(adduct_id), self.ce(ce.unsqueeze(-1))], dim=-1)
            g = nn.functional.normalize(self.g(torch.cat([emb, meta], dim=-1)), dim=-1)
            b = emb.shape[0]
            hh = nn.functional.normalize(self.h(fps), dim=-1).view(b, -1, D_OUT)
            return (g.unsqueeze(1) * hh).sum(-1) * self.logit_scale.exp()
    return Align


def mlp_speed(emb, specs, device, amp: bool, n_steps=None, bank_size=FP_BANK, deadline=None, seed=0):
    import torch
    from torch import nn
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = emb.shape[0]
    steps_epoch = math.ceil(n / MLP_BATCH)
    n_steps = steps_epoch if n_steps is None else n_steps
    adducts = sorted({str(s[4]) for s in specs})
    aid = torch.tensor([adducts.index(str(s[4])) for s in specs])
    ce = torch.tensor([float(s[3]) if s[3] is not None else 0.0 for s in specs]) / 100.0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bank = (torch.rand((bank_size, FP_BITS), generator=g) < FP_DENSITY).float().to(device)
    pos = torch.randint(0, bank_size, (n,), generator=g)
    model = build_mlps(device)(len(adducts)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    _gs = getattr(getattr(torch, "amp", None), "GradScaler", None)
    scaler = (_gs("cuda", enabled=amp and device == "cuda") if _gs is not None
              else torch.cuda.amp.GradScaler(enabled=amp and device == "cuda"))
    emb_d, aid_d, ce_d = emb.to(device), aid.to(device), ce.to(device)
    order = torch.randperm(n, generator=g)
    losses, t_start = [], None
    total = n_steps + MLP_WARMUP_EXCLUDED
    for step in range(total):
        if step == MLP_WARMUP_EXCLUDED:
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.time()
        if deadline is not None and time.time() > deadline:
            raise StageTimeout(f"mlp stopped at step {step}")
        j = step % steps_epoch
        idx = order[j * MLP_BATCH:(j + 1) * MLP_BATCH]
        b = len(idx)
        neg = torch.randint(0, bank_size, (b, MLP_K_NEG), generator=g)
        cand = torch.cat([pos[idx].unsqueeze(1), neg], dim=1).reshape(-1).to(device)
        fps = bank[cand]
        idx_d = idx.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            logits = model(emb_d[idx_d], aid_d[idx_d], ce_d[idx_d], fps)
            loss = nn.functional.cross_entropy(logits.float(), torch.zeros(b, dtype=torch.long, device=device))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MLP_CLIP)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach()))
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t_start
    mem = ({"max_allocated": torch.cuda.max_memory_allocated(), "max_reserved": torch.cuda.max_memory_reserved()}
           if device == "cuda" else {})
    return {"steps_timed": n_steps, "steps_per_epoch_10k": steps_epoch, "seconds": dt, "steps_per_s": n_steps / dt,
            "params": int(sum(p.numel() for p in model.parameters())), "loss_first": losses[0], "loss_last": losses[-1],
            "gpu_mem": mem, "n_adducts": len(adducts)}


def _env_info():
    import torch
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count()}
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout[:3000]
    except Exception as e:
        info["lscpu"] = f"unavailable: {e}"
    return info


def run(out_dir="/kaggle/working", work_dir="/kaggle/tmp_align", input_root="/kaggle/input", t0=None,
        ckpt_override=None, dreams_root_override=None, n_spectra=N_SPECTRA, mlp_steps=None, bank_size=FP_BANK,
        device=None) -> dict:
    """全段を順に回し、段ごとに report を書き直す。どこで止まっても status と止まった段を残す。"""
    t0 = time.time() if t0 is None else t0
    deadline = t0 + STAGE_CUTOFF_S
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    rpath = out_dir / "align_speed_report.json"
    rep = {"issue": 99, "plan": "2026-11-14 noon item 5", "status": "running", "stage": None, "stages": {},
           "prereg": "https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/issues/99#issuecomment-6870838727",
           "constants": {"N_SPECTRA": n_spectra, "EMB_BATCH": EMB_BATCH, "PEAKS": list(PEAKS), "MLP_BATCH": MLP_BATCH,
                         "MLP_K_NEG": MLP_K_NEG, "MLP_STEPS_PROD": MLP_STEPS_PROD, "FP_BITS": FP_BITS,
                         "BUDGET_S": BUDGET_S, "STAGE_CUTOFF_S": STAGE_CUTOFF_S, "DREAMS_COMMIT": DREAMS_COMMIT,
                         "CKPT_MD5": CKPT_MD5, "CKPT_SIZE": CKPT_SIZE},
           "not_for_evaluation": "speed only; outputs are not used for v3 evaluation"}

    def save():
        rep["elapsed_s"] = _now(t0)
        rep["ram_peak"] = ram_peak()
        rpath.write_text(json.dumps(rep, indent=1, default=str))

    def stage(name):
        if time.time() > deadline:
            raise StageTimeout(f"not starting stage {name}: elapsed {_now(t0)}s > {STAGE_CUTOFF_S}s")
        rep["stage"] = name
        save()

    try:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        stage("env")
        rep["env"] = _env_info()
        rep["device"] = dev
        save()

        stage("download_ckpt")
        if ckpt_override is None:
            ck = work_dir / "ssl_model.ckpt"
            rep["stages"]["download_ckpt"] = download_ckpt(ck)
            rep["md5_ok"] = True
        else:
            ck = ckpt_override
            rep["stages"]["download_ckpt"] = {"override": True, "seconds": 0.0}
        save()

        stage("dreams_code")
        if dreams_root_override is None:
            rep["stages"]["dreams_code"] = fetch_dreams_code(work_dir)
            droot = rep["stages"]["dreams_code"]["root"]
        else:
            droot = dreams_root_override
            rep["stages"]["dreams_code"] = {"override": True, "root": droot, "seconds": 0.0}
        save()

        stage("load_model")
        tl = time.time()
        model, SP, DFA, linfo = load_dreams(droot, ck, dev)
        linfo["seconds"] = round(time.time() - tl, 3)
        if dev == "cuda":
            linfo["gpu_mem_after_load"] = torch.cuda.memory_allocated()
        rep["stages"]["load_model"] = linfo
        save()
        if ckpt_override is None:
            try:
                Path(ck).unlink()
            except Exception:
                pass

        stage("select_spectra")
        if input_root is None:
            raise RuntimeError("input_root is None")
        train_p, split_p = find_inputs(Path(input_root))
        rep["inputs"] = {"train": str(train_p), "split": str(split_p)}
        if train_p is None:
            raise FileNotFoundError("competition train.parquet not found")
        specs, sinfo = select_spectra(train_p, split_p, n_spectra)
        rep["stages"]["select_spectra"] = sinfo
        save()

        emb_keep = None
        for n_peaks in PEAKS:
            stage(f"preprocess_p{n_peaks}")
            arr, t_pre = preprocess(specs, SP, DFA, n_peaks)
            rep["stages"][f"preprocess_p{n_peaks}"] = {"seconds": t_pre, "sps": len(arr) / t_pre, "shape": list(arr.shape)}
            save()
            for amp in (False, True):
                tag = f"embed_p{n_peaks}_{'fp16' if amp else 'fp32'}"
                stage(tag)
                emb, einfo = embed_speed(model, arr, dev, amp, deadline=deadline)
                einfo["sps_preprocess_plus_forward"] = len(arr) / (t_pre + einfo["seconds_forward"])
                rep["stages"][tag] = einfo
                save()
                if n_peaks == PEAKS[0] and not amp:
                    emb_keep = emb
            del arr
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

        for amp in (False, True):
            tag = f"mlp_{'fp16' if amp else 'fp32'}"
            stage(tag)
            rep["stages"][tag] = mlp_speed(emb_keep, specs, dev, amp, n_steps=mlp_steps, bank_size=bank_size,
                                           deadline=deadline)
            save()
            if dev == "cuda":
                torch.cuda.empty_cache()

        stage("extrapolate")
        S = rep["stages"]
        sps = {p: S[f"embed_p{p}_fp32"]["sps_preprocess_plus_forward"] for p in PEAKS}
        fixed = S["download_ckpt"]["seconds"] + S["dreams_code"]["seconds"] + S["load_model"]["seconds"]
        rep["extrapolation"] = extrapolate(sps, S["mlp_fp32"]["steps_per_s"], S["select_spectra"]["seconds"],
                                           S["select_spectra"]["rows_read"], fixed)
        rep["summary"] = {
            "gpu_name": rep["env"]["gpu_name"],
            "forward_sps_fp32": {p: S[f"embed_p{p}_fp32"]["forward_sps"] for p in PEAKS},
            "forward_sps_fp16_reference": {p: S[f"embed_p{p}_fp16"]["forward_sps"] for p in PEAKS},
            "mlp_steps_per_s_fp32": S["mlp_fp32"]["steps_per_s"],
            "mlp_steps_per_s_fp16_reference": S["mlp_fp16"]["steps_per_s"],
            "gpu_mem_peak_max_allocated": max(v.get("gpu_mem", {}).get("max_allocated", 0) for v in S.values()
                                              if isinstance(v, dict)),
            "ram_peak": ram_peak(),
            "T_total_hours": rep["extrapolation"]["T_total_hours"],
            "decision": rep["extrapolation"]["decision"],
        }
        rep["status"] = "ok"
        rep["stage"] = "done"
    except Md5Mismatch as e:
        rep["status"] = "md5_mismatch"
        rep["error"] = str(e)
        rep["md5_ok"] = False
    except StageTimeout as e:
        rep["status"] = "timeout"
        rep["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        rep["status"] = "error"
        rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-4000:]
    save()
    return rep


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r.get(k) for k in ("status", "stage", "error", "summary", "elapsed_s")}, indent=1, default=str))
