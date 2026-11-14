#!/usr/bin/env python3
"""整列チャネルの本番 その 1: 凍結 DreaMS（ssl_model.ckpt、fp32、ピーク 100）で fold-safe の学習スペクトルと
v3 の問い合わせを埋め込み、fp16 の .npy のシャードと索引を /kaggle/working に書く。

- 事前登録: docs/research/contrastive-alignment.md §(d)(e)、速度の試走 #99（ピーク 100・fp32 120.6 スペクトル/秒）、本番の Issue
- 重みのダウンロード・DreaMS のコードの取得・読み込み・前処理・フックは align_speed_trial.py をそのまま使う（同じ md5・commit）
- シャード = train.parquet の row group 1 個。シャード k は emb_k.npy（行 = 選んだスペクトル、1,024 次元）、idx_k.parquet
  （row, kind, inchikey14, adduct, ce, precursor_mz, holdout, molecule_id, spectrum_id）、done_k.json（最後に書く）の 3 つ
- 途中で時間切れになっても、書き終えたシャード（done あり）は残る。同じカーネルの前の版の出力を kernel_sources に付けて
  流し直すと、config の hash が同じ done のシャードは写して飛ばす（再開）
- GPU が 2 枚見えれば 2 つの子プロセスに row group を k % 2 で分ける（埋め込みは 1 スペクトルずつ独立なので結果は同じ）
- 行の選び方: kind 1（問い合わせ）= align_query_manifest.json の行。kind 0（学習）= inchikey14 が DROP
  （split.parquet の全行 ∪ 4 集合の正解、55,369 構造）に無い行。ピークが空の行と前処理で落ちた行は数えて飛ばす

torch はモジュールの先頭で import しない（CI は純粋な関数だけをテストする）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PEAKS = 100                 # #99 の判定に使ったピーク数（PreTrainedModel.from_name の既定）
EMB_BATCH = 32              # DreaMS API の既定（試走と同じ）
READ_BATCH = 8192
TRAIN_ROWS = 2_539_608
EXPECTED_DROP = 55_369      # #54 と同じ（split ∪ 4 集合の正解）
N_QUERY = 4_756
HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
D_EMB = 1024
FP16_MAX = 60_000.0         # これを超える値のあるシャードは fp32 のまま保存する（fp16 の上限 65,504 の手前）
STOP_NEW_SHARD_S = 10.0 * 3600   # これを過ぎたら新しいシャードを始めない
HARD_STOP_S = 10.75 * 3600       # シャードの途中でもこれを過ぎたら捨てて終える（Kaggle の 12 時間の前に正常終了する）
POLL_S = 60
SLUG = "taichiiiii/casmi26-align-embed"
IDX_COLS = ("row", "kind", "inchikey14", "adduct", "ce", "precursor_mz", "holdout", "molecule_id", "spectrum_id")
READ_COLS = ["inchikey14", "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "collision_energy_ev", "adduct"]


# ---------------------------------------------------------------- 純粋な関数（CI でテストする）

def config_hash(manifest_rows, n_drop: int) -> str:
    """シャードの再開で同じ設定かを見る hash。定数・問い合わせ表・DROP の件数・DreaMS の固定値を含む。"""
    import align_speed_trial as A
    blob = json.dumps({"PEAKS": PEAKS, "EMB_BATCH": EMB_BATCH, "D_EMB": D_EMB, "FP16_MAX": FP16_MAX,
                       "TRAIN_ROWS": TRAIN_ROWS, "n_drop": n_drop, "md5": A.CKPT_MD5, "commit": A.DREAMS_COMMIT,
                       "manifest": [list(r) for r in manifest_rows]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def row_group_offsets(sizes) -> list[int]:
    out, s = [], 0
    for n in sizes:
        out.append(s)
        s += int(n)
    return out


def assign_shards(n_shards: int, n_workers: int, worker: int) -> list[int]:
    return [k for k in range(n_shards) if k % n_workers == worker]


def ce_value(ce) -> float:
    """collision_energy_ev（list<double>）の先頭。無ければ NaN（MSAlign は NaN を学習する欠損ベクトルにする）。"""
    if ce is None:
        return float("nan")
    try:
        if len(ce) == 0:
            return float("nan")
        v = ce[0]
    except TypeError:
        v = ce
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def classify(row: int, ik: str, query_by_row: dict, drop: set):
    """1 行の扱い: (kind, query 情報)。kind 1 = 問い合わせ、0 = 学習、None = 使わない。"""
    if row in query_by_row:
        return 1, query_by_row[row]
    if ik in drop:
        return None, None
    return 0, None


def query_rows(manifest: dict) -> dict:
    """行 → [(holdout, molecule_id, spectrum_id, inchikey14, sig), ...]（同じ行を複数の集合が使うことがある）。"""
    out: dict[int, list] = {}
    for h, mid, sid, row, ik, sig in manifest["rows"]:
        out.setdefault(int(row), []).append((h, mid, sid, ik, sig))
    return out


def shard_names(k: int) -> dict:
    return {"emb": f"emb_{k:03d}.npy", "idx": f"idx_{k:03d}.parquet", "done": f"done_{k:03d}.json"}


def shard_is_done(d: Path, k: int, chash: str) -> bool:
    n = shard_names(k)
    p = Path(d) / n["done"]
    if not (p.is_file() and (Path(d) / n["emb"]).is_file() and (Path(d) / n["idx"]).is_file()):
        return False
    try:
        info = json.loads(p.read_text())
    except (ValueError, OSError):
        return False
    return info.get("config_hash") == chash and info.get("status") == "done"


def find_resume_dirs(input_root: Path) -> list[Path]:
    """前の版の出力（kernel_sources で付けたもの）の中の done_*.json を持つディレクトリ。"""
    root = Path(input_root)
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("done_*.json")})


class ManifestMismatch(RuntimeError):
    pass


def verify_manifest(pf, manifest: dict) -> dict:
    """埋め込みの前に、問い合わせの行の train.parquet の中身（inchikey14 と spectrum_sig）が表と同じかを全件確かめる。"""
    import align_query_manifest as QM
    offs = row_group_offsets([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    by_rg: dict[int, list] = {}
    for h, mid, sid, row, ik, sig in manifest["rows"]:
        k = max(i for i, o in enumerate(offs) if o <= int(row))
        by_rg.setdefault(k, []).append((int(row) - offs[k], ik, sig))
    n = bad_sig = bad_ik = 0
    for k, items in sorted(by_rg.items()):
        t = pf.read_row_group(k, columns=READ_COLS)
        sub = t.take(sorted({i for i, _, _ in items})).to_pydict()
        pos = {i: j for j, i in enumerate(sorted({i for i, _, _ in items}))}
        for i, ik, sig in items:
            j = pos[i]
            got = QM.spectrum_sig(sub["precursor_mz"][j], sub["adduct"][j], sub["ms2_mzs"][j],
                                  sub["ms2_normalized_intensities"][j])
            n += 1
            bad_sig += int(got != sig)
            bad_ik += int(sub["inchikey14"][j] != ik)
        del t
    return {"checked": n, "sig_mismatch": bad_sig, "ik_mismatch": bad_ik}


def leak_audit_from_index(train_ik, split_rows, holdout_keys) -> dict:
    import sanpier_fold0 as sf
    return sf.leak_audit(train_ik, split_rows, holdout_keys)


# ---------------------------------------------------------------- Kaggle で動く部分
FAKE_ENV = "ALIGN_EMBED_FAKE_DREAMS"   # テスト専用: 1 なら DreaMS の代わりに小さな偽物を使う（カーネルはこれが無いことを確かめる）


def _fake_dreams():
    import numpy as np
    import torch

    class FF(torch.nn.Module):
        def forward(self, x):
            return x.sum(-1, keepdim=True).repeat(1, 1, D_EMB)

    class M(torch.nn.Module):
        n_layers = 1

        def __init__(self):
            super().__init__()
            self.transformer_encoder = torch.nn.Module()
            self.transformer_encoder.ffs = torch.nn.ModuleList([FF()])

        def forward(self, x):
            return self.transformer_encoder.ffs[0](x)

    class SP:
        def __init__(self, dformat=None, n_highest_peaks=PEAKS):
            self.n = n_highest_peaks

        def __call__(self, arr, prec_mz, high_form=False):
            out = np.zeros((self.n + 1, 2))
            out[0] = [prec_mz, 1.0]
            k = min(self.n, arr.shape[1])
            out[1:1 + k] = arr[:, :k].T
            return out
    return M(), SP, object, {"fake": True}


def load_model(dreams_root, ckpt, device):
    import align_speed_trial as A
    if os.environ.get(FAKE_ENV) == "1":
        return _fake_dreams()
    return A.load_dreams(dreams_root, ckpt, device)


def _log(*a):
    print(*a, flush=True)


def embed_shard(k: int, model, SP, DFA, pf, rg_offset: int, query_by_row: dict, drop: set, out: Path,
                device: str, chash: str, t0: float) -> dict:
    """row group k を読み、選んだ行を埋め込んで 3 つのファイルを書く。HARD_STOP を過ぎたら書かずに StageTimeout。"""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import align_speed_trial as A

    ts = time.time()
    pp = SP(dformat=DFA(), n_highest_peaks=PEAKS)
    store = {}
    last = model.transformer_encoder.ffs[model.n_layers - 1]
    hook = last.register_forward_hook(lambda mod, inp, o: store.__setitem__("e", o[:, 0, :]))
    cols = {c: [] for c in IDX_COLS}
    chunks, buf = [], []
    stats = {"rows": 0, "empty": 0, "preproc_error": 0, "dropped": 0, "query_sig_mismatch": 0, "query_ik_mismatch": 0}
    t_fwd = 0.0

    def flush():
        nonlocal t_fwd
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf).astype(np.float32))
        tf = time.time()
        with torch.inference_mode():
            for i in range(0, len(x), EMB_BATCH):
                model(x[i:i + EMB_BATCH].to(device, non_blocking=True))
                chunks.append(store["e"].float().cpu().numpy())
        t_fwd += time.time() - tf
        buf.clear()

    try:
        row = rg_offset
        for b in pf.iter_batches(batch_size=READ_BATCH, row_groups=[k], columns=READ_COLS):
            if time.time() - t0 > HARD_STOP_S:
                raise A.StageTimeout(f"hard stop inside shard {k}")
            t = b.to_pydict()
            for i in range(len(t["inchikey14"])):
                r = row + i
                ik = t["inchikey14"][i]
                kind, qs = classify(r, ik, query_by_row, drop)
                if kind is None:
                    stats["dropped"] += 1
                    continue
                mz, it = t["ms2_mzs"][i], t["ms2_normalized_intensities"][i]
                if not mz:
                    stats["empty"] += 1
                    continue
                prec = float(t["precursor_mz"][i])
                try:
                    arr = pp(np.array([mz, it], dtype=np.float64), prec_mz=prec, high_form=False)
                except Exception:  # noqa: BLE001  前処理で落ちる行は数えて飛ばす
                    stats["preproc_error"] += 1
                    continue
                if kind == 1:
                    import align_query_manifest as QM
                    sig = QM.spectrum_sig(prec, t["adduct"][i], mz, it)
                    for h, mid, sid, qik, qsig in qs:
                        stats["query_sig_mismatch"] += int(sig != qsig)
                        stats["query_ik_mismatch"] += int(ik != qik)
                    entries = [(h, mid, sid) for h, mid, sid, _, _ in qs]
                else:
                    entries = [(None, None, None)]
                for h, mid, sid in entries:   # 同じ行を 2 集合が使うときは 2 行にする（埋め込みは同じ）
                    buf.append(arr)
                    cols["row"].append(r); cols["kind"].append(kind); cols["inchikey14"].append(ik)
                    cols["adduct"].append(t["adduct"][i]); cols["ce"].append(ce_value(t["collision_energy_ev"][i]))
                    cols["precursor_mz"].append(prec); cols["holdout"].append(h)
                    cols["molecule_id"].append(mid); cols["spectrum_id"].append(sid)
                stats["rows"] += 1
            row += len(t["inchikey14"])
            if len(buf) >= 4096:
                flush()
        flush()
    finally:
        hook.remove()
    emb = np.concatenate(chunks) if chunks else np.zeros((0, D_EMB), np.float32)
    assert emb.shape == (len(cols["row"]), D_EMB), (emb.shape, len(cols["row"]))
    finite = bool(np.isfinite(emb).all())
    max_abs = float(np.abs(emb).max()) if len(emb) else 0.0
    dtype = "float16" if (finite and max_abs <= FP16_MAX) else "float32"
    names = shard_names(k)
    tmp = out / (names["emb"] + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, emb.astype(np.float16) if dtype == "float16" else emb)
    os.replace(tmp, out / names["emb"])
    table = pa.table({c: cols[c] for c in IDX_COLS})
    pq.write_table(table, out / (names["idx"] + ".tmp"))
    os.replace(out / (names["idx"] + ".tmp"), out / names["idx"])
    info = dict(status="done", shard=k, config_hash=chash, n=len(cols["row"]), n_query=int(sum(cols["kind"])),
                dtype=dtype, finite_fp32=finite, max_abs=max_abs, seconds=round(time.time() - ts, 1),
                seconds_forward=round(t_fwd, 1), device=device, **stats)
    (out / names["done"]).write_text(json.dumps(info))
    return info


def worker_main(argv) -> None:
    """子プロセス: --shards の row group を順に埋め込む（CUDA_VISIBLE_DEVICES は親が決める）。"""
    import argparse
    import pyarrow.parquet as pq
    import align_speed_trial as A
    ap = argparse.ArgumentParser()
    for a in ("--dreams-root", "--ckpt", "--train", "--work", "--out", "--shards", "--t0", "--tag"):
        ap.add_argument(a, required=True)
    args = ap.parse_args(argv)
    t0 = float(args.t0)
    work, out = Path(args.work), Path(args.out)
    ctx = json.loads((work / "embed_ctx.json").read_text())
    manifest = json.loads((work / "align_query_manifest.json").read_text())
    drop = set((work / "drop_ik.txt").read_text().split())
    qrows = query_rows(manifest)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, SP, DFA, linfo = load_model(args.dreams_root, args.ckpt, dev)
    pf = pq.ParquetFile(args.train)
    offs = row_group_offsets([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    log_p = work / f"worker_{args.tag}.jsonl"
    for k in [int(s) for s in args.shards.split(",") if s != ""]:
        if shard_is_done(out, k, ctx["config_hash"]):
            continue
        if time.time() - t0 > STOP_NEW_SHARD_S:
            with open(log_p, "a") as f:
                f.write(json.dumps({"shard": k, "status": "not_started_time"}) + "\n")
            continue
        try:
            info = embed_shard(k, model, SP, DFA, pf, offs[k], qrows, drop, out, dev, ctx["config_hash"], t0)
        except A.StageTimeout as e:
            info = {"shard": k, "status": "hard_stop", "error": str(e)}
        except Exception as e:  # noqa: BLE001
            import traceback
            info = {"shard": k, "status": "error", "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()[-3000:]}
        info["elapsed_s"] = round(time.time() - t0, 1)
        _log("SHARD", json.dumps(info)[:400])
        with open(log_p, "a") as f:
            f.write(json.dumps(info) + "\n")
        if info.get("status") == "hard_stop":
            break


def finalize(rep: dict, out: Path, n_shards: int, chash: str, split, holdout_keys: dict) -> dict:
    """done のシャードの索引だけを読み、件数・leak_audit・status を rep に書く（Kaggle とローカルで共通）。"""
    import pyarrow.parquet as pq
    done = [k for k in range(n_shards) if shard_is_done(out, k, chash)]
    infos = [json.loads((out / shard_names(k)["done"]).read_text()) for k in done]
    train_ik, n_train, q_keys = set(), 0, set()
    for k in done:
        t = pq.read_table(out / shard_names(k)["idx"], columns=["kind", "inchikey14", "holdout", "spectrum_id"])
        kind = t.column("kind").to_pylist()
        ik = t.column("inchikey14").to_pylist()
        for j, kd in enumerate(kind):
            if kd == 0:
                train_ik.add(ik[j]); n_train += 1
        q_keys |= {(h, s) for h, s, kd in zip(t.column("holdout").to_pylist(), t.column("spectrum_id").to_pylist(),
                                              kind) if kd == 1}
    leak = leak_audit_from_index(train_ik, zip(split.inchikey14, split.fold), holdout_keys)
    (out / "train_keys.txt").write_text("\n".join(sorted(train_ik)) + "\n")
    rep.update(n_shards=n_shards, config_hash=chash, shards_done=len(done), n_train_spectra=n_train,
               n_train_structures=len(train_ik), n_query_embedded=len(q_keys), n_query_expected=N_QUERY,
               leak_audit=leak,
               totals={k: sum(i.get(k, 0) for i in infos) for k in
                       ("rows", "empty", "preproc_error", "dropped", "query_sig_mismatch", "query_ik_mismatch")},
               dtypes=sorted({i["dtype"] for i in infos}), seconds_forward=sum(i["seconds_forward"] for i in infos))
    complete = len(done) == n_shards
    ok = complete and leak["total"] == 0 and rep["totals"]["query_sig_mismatch"] == 0 \
        and rep["totals"]["query_ik_mismatch"] == 0 and len(q_keys) == N_QUERY
    rep["status"] = "ok" if ok else ("partial_resume_needed" if not complete else "check_failed")
    return rep


def run(out_dir="/kaggle/working", work_dir="/kaggle/tmp_align", input_root="/kaggle/input", t0=None,
        holdout_keys=None, manifest=None, src_dir=None) -> dict:
    """親: 重みとコードを 1 回だけ取り、DROP を作り、GPU ごとに子プロセスを立てて待ち、report を書く。"""
    import pandas as pd
    import pyarrow.parquet as pq
    import align_speed_trial as A
    t0 = time.time() if t0 is None else t0
    out, work = Path(out_dir), Path(work_dir)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rpath = out / "embed_report.json"
    rep = {"issue": None, "kernel": SLUG, "status": "running", "stages": {},
           "constants": {"PEAKS": PEAKS, "EMB_BATCH": EMB_BATCH, "precision": "fp32 (DreaMS forward), stored fp16",
                         "FP16_MAX": FP16_MAX, "STOP_NEW_SHARD_S": STOP_NEW_SHARD_S, "HARD_STOP_S": HARD_STOP_S,
                         "CKPT_MD5": A.CKPT_MD5, "DREAMS_COMMIT": A.DREAMS_COMMIT, "EXPECTED_DROP": EXPECTED_DROP}}

    def save():
        rep["elapsed_s"] = round(time.time() - t0, 1)
        rep["ram_peak"] = A.ram_peak()
        rpath.write_text(json.dumps(rep, indent=1, default=str))

    try:
        import torch
        rep["env"] = A._env_info()
        n_gpu = torch.cuda.device_count()
        rep["n_gpu"] = n_gpu
        save()
        train_p = [p for p in sorted(Path(input_root).rglob("train.parquet")) if "casmi26" in str(p).lower()]
        split_p = [p for p in sorted(Path(input_root).rglob("split.parquet")) if "fold-safe" in str(p)]
        assert len(train_p) == 1 and len(split_p) == 1, (train_p, split_p)
        rep["inputs"] = {"train": str(train_p[0]), "split": str(split_p[0])}
        pf = pq.ParquetFile(train_p[0])
        assert pf.metadata.num_rows == TRAIN_ROWS, pf.metadata.num_rows
        n_shards = pf.metadata.num_row_groups
        split = pd.read_parquet(split_p[0], columns=["inchikey14", "fold"])
        assert sorted(holdout_keys) == sorted(HOLDOUTS), sorted(holdout_keys)
        import sanpier_fold0 as sf
        drop = sf.drop_inchikey14(split.inchikey14, holdout_keys)
        assert len(drop) == EXPECTED_DROP, len(drop)
        assert manifest["train_rows"] == TRAIN_ROWS and manifest["n"] == N_QUERY == len(manifest["rows"])
        chash = config_hash(manifest["rows"], len(drop))
        rep.update(n_shards=n_shards, n_drop=len(drop), config_hash=chash)
        tm = time.time()
        rep["manifest_check"] = verify_manifest(pf, manifest)
        rep["manifest_check"]["seconds"] = round(time.time() - tm, 1)
        save()
        if rep["manifest_check"]["sig_mismatch"] or rep["manifest_check"]["ik_mismatch"] \
                or rep["manifest_check"]["checked"] != N_QUERY:
            raise ManifestMismatch(json.dumps(rep["manifest_check"]))
        (work / "drop_ik.txt").write_text("\n".join(sorted(drop)) + "\n")
        (work / "align_query_manifest.json").write_text(json.dumps(manifest))
        (work / "embed_ctx.json").write_text(json.dumps({"config_hash": chash}))
        save()

        # 再開: 前の版の done のシャードを写す
        resumed = []
        for d in find_resume_dirs(input_root):
            for k in range(n_shards):
                if not shard_is_done(out, k, chash) and shard_is_done(d, k, chash):
                    for n in shard_names(k).values():
                        shutil.copy2(d / n, out / n)
                    resumed.append(k)
        rep["resumed_shards"] = sorted(resumed)
        todo = [k for k in range(n_shards) if not shard_is_done(out, k, chash)]
        rep["todo_shards"] = todo
        save()

        if todo:
            if os.environ.get(FAKE_ENV) == "1":
                rep["stages"]["download_ckpt"] = {"fake": True}
                rep["stages"]["dreams_code"] = {"fake": True, "root": str(work)}
            else:
                rep["stages"]["download_ckpt"] = A.download_ckpt(work / "ssl_model.ckpt")
                rep["md5_ok"] = True
                rep["stages"]["dreams_code"] = A.fetch_dreams_code(work)
            save()
            n_workers = max(1, min(n_gpu, 2))
            procs = []
            for w in range(n_workers):
                shards = [k for j, k in enumerate(todo) if j % n_workers == w]
                env = dict(os.environ)
                if n_gpu:
                    env["CUDA_VISIBLE_DEVICES"] = str(w)
                cmd = [sys.executable, str(Path(src_dir or work) / "align_embed.py"), "--worker",
                       "--dreams-root", rep["stages"]["dreams_code"]["root"], "--ckpt", str(work / "ssl_model.ckpt"),
                       "--train", str(train_p[0]), "--work", str(work), "--out", str(out),
                       "--shards", ",".join(map(str, shards)), "--t0", repr(t0), "--tag", str(w)]
                procs.append(subprocess.Popen(cmd, env=env, cwd=str(Path(src_dir or work))))
            rep["workers"] = n_workers
            save()
            while any(p.poll() is None for p in procs):
                time.sleep(POLL_S)
                rep["shards_done"] = sum(shard_is_done(out, k, chash) for k in range(n_shards))
                save()
            rep["worker_returncodes"] = [p.returncode for p in procs]
            rep["worker_logs"] = []
            for w in range(n_workers):
                lp = work / f"worker_{w}.jsonl"
                if lp.exists():
                    rep["worker_logs"] += [json.loads(x) for x in lp.read_text().splitlines() if x.strip()]

        # 集計（索引だけ読む）
        finalize(rep, out, n_shards, chash, split, holdout_keys)
    except A.Md5Mismatch as e:
        rep["status"] = "md5_mismatch"; rep["error"] = str(e); rep["md5_ok"] = False
    except ManifestMismatch as e:
        rep["status"] = "manifest_mismatch"; rep["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        import traceback
        rep["status"] = "error"; rep["error"] = f"{type(e).__name__}: {e}"
        rep["traceback"] = traceback.format_exc()[-4000:]
    try:
        (work / "ssl_model.ckpt").unlink()
    except OSError:
        pass
    save()
    return rep


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        worker_main(sys.argv[2:])
