#!/usr/bin/env python3
"""整列チャネルの本番 その 1（ローカル、MPS/CPU）: 凍結 DreaMS（ssl_model.ckpt、fp32、ピーク 100）で fold-safe の学習スペクトルと
v3 の問い合わせを埋め込み、fp16 の .npy シャードに書く。Issue #109。

    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5
    .venv/bin/python scripts/train/align_local_embed.py --speed-test 10000   # 速度だけ（1 万スペクトル、書き出しなし）
    .venv/bin/python scripts/train/align_local_embed.py                      # 本番（done のシャードは飛ばすので、止めても再開できる）

- シャード・索引・done の形、行の選び方、問い合わせの照合、集計（embed_report.json）は Kaggle 版（align_embed.py）と同じ関数
- 重み: data/ext/dreams/ssl_model.ckpt（md5 4deee91d… を起動のたびに照合）、コード: data/ext/dreams/DreaMS-<commit>
- 起動の前に: 上の 2 つの環境変数、vm_stat の空き（free+speculative）≥ --min-free-gb（既定 10 GB）、このリポジトリの
  ほかの重いジョブが無いこと。どれかが満たされなければ起動しない。走っている間も、シャードの切れ目で空きが 2 GB を
  下回ったら次のシャードを始めずに止める（再開できる）
- MPS で前向きが動かない・有限でない・CPU と大きく違う（2 本で max|Δ| > 1e-2）ときは CPU に戻す
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/train"))

DREAMS_DIR = ROOT / "data/ext/dreams"
OUT = ROOT / "outputs/align_embed"
MIN_RUN_FREE_GB = 2.0
MPS_TOL = 1e-2


def check_env(min_free_gb: float) -> dict:
    import align_train as T
    for k, v in (("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6"), ("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")):
        if os.environ.get(k) != v:
            raise SystemExit(f"set {k}={v} before starting (torch reads it at import)")
    fg = T.free_gb()
    if fg.get("free_gb") is not None and fg["free_gb"] < min_free_gb:
        raise SystemExit(f"vm_stat free {fg} < {min_free_gb} GB")
    own = T.own_heavy_jobs()
    if own:
        raise SystemExit(f"another heavy job of this repo is running: {own}")
    return fg


def choose_device(model, SP, DFA, want: str):
    """MPS の前向きを CPU と 2 本で比べる。だめなら CPU。"""
    import numpy as np
    import torch
    info = {"want": want}
    if want == "cpu" or not torch.backends.mps.is_available():
        return "cpu", dict(info, reason="cpu requested or mps unavailable")
    pp = SP(dformat=DFA(), n_highest_peaks=100)
    x = torch.from_numpy(np.stack([pp(np.array([[100.0, 150.0, 200.0], [1.0, 0.5, 0.2]]), prec_mz=250.0, high_form=False),
                                   pp(np.array([[60.0, 90.0], [0.3, 1.0]]), prec_mz=120.0, high_form=False)]).astype(np.float32))
    store = {}
    h = model.transformer_encoder.ffs[model.n_layers - 1].register_forward_hook(
        lambda m, i, o: store.__setitem__("e", o[:, 0, :].float().cpu()))
    try:
        with torch.inference_mode():
            model.to("cpu"); model(x); ref = store["e"].clone()
            model.to("mps"); model(x.to("mps")); got = store["e"].clone()
        d = float((ref - got).abs().max())
        info.update(max_abs_diff_vs_cpu=d, finite=bool(torch.isfinite(got).all()))
        if info["finite"] and d <= MPS_TOL:
            return "mps", info
        model.to("cpu")
        return "cpu", dict(info, reason="mps output differs from cpu")
    except Exception as e:  # noqa: BLE001
        model.to("cpu")
        return "cpu", dict(info, reason=f"mps error {type(e).__name__}: {e}")
    finally:
        h.remove()


def load_inputs():
    import pandas as pd
    import sanpier_fold0 as sf
    import align_embed as E
    keys = json.loads((ROOT / "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json").read_text())
    manifest = json.loads((ROOT / "scripts/train/align_query_manifest.json").read_text())
    split = pd.read_parquet(ROOT / "data/ext/foldsafe/split.parquet", columns=["inchikey14", "fold"])
    drop = sf.drop_inchikey14(split.inchikey14, keys)
    assert len(drop) == E.EXPECTED_DROP, len(drop)
    return keys, manifest, split, drop


def speed_test(model, SP, DFA, dev, n: int, drop: set) -> dict:
    """#99 の試走と同じ測り方（先頭から DROP に無い n 本、前処理＋前向き、バッチ 32、ピーク 100、fp32）。"""
    import torch
    import align_speed_trial as A
    import align_embed as E
    import align_train as T
    import pyarrow.parquet as pq
    t0 = time.time()
    pf = pq.ParquetFile(ROOT / "data/train.parquet")
    specs, rows_read = [], 0
    for b in pf.iter_batches(batch_size=A.READ_BATCH, columns=A.TRAIN_COLS):
        t = b.to_pydict()
        rows_read += len(t["inchikey14"])
        for i in range(len(t["inchikey14"])):
            if t["inchikey14"][i] in drop or not t["ms2_mzs"][i]:
                continue
            specs.append((t["ms2_mzs"][i], t["ms2_normalized_intensities"][i], float(t["precursor_mz"][i]), None, None))
            if len(specs) >= n:
                break
        if len(specs) >= n:
            break
    t_read = time.time() - t0
    arr, t_pre = A.preprocess(specs, SP, DFA, E.PEAKS)
    _, info = A.embed_speed(model, arr, dev, False)
    sps = n / (t_pre + info["seconds_forward"])
    n_all = 1_982_405 + E.N_QUERY
    t_read_all = t_read * (E.TRAIN_ROWS / rows_read)
    rep = {"device": dev, "n": n, "peaks": E.PEAKS, "precision": "fp32", "batch": E.EMB_BATCH,
           "seconds_read": round(t_read, 2), "rows_read": rows_read, "seconds_preprocess": round(t_pre, 2),
           "seconds_forward": round(info["seconds_forward"], 2), "forward_sps": info["forward_sps"],
           "sps_preprocess_plus_forward": sps, "finite": info["finite"],
           "extrapolation": {"spectra": n_all, "T_embed_h": n_all / sps / 3600, "T_read_h": t_read_all / 3600,
                             "T_total_h": (n_all / sps + t_read_all) / 3600,
                             "formula": "(1982405+4756)/sps + read_s*(2539608/rows_read)"},
           "ram_peak": A.ram_peak(), "mem_after": T.free_gb()}
    if dev == "mps":
        rep["mps_driver_allocated"] = int(torch.mps.driver_allocated_memory())
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed-test", type=int, default=0, help="この本数だけ速度を測って終える（書き出しなし）")
    ap.add_argument("--device", default="auto", help="auto（mps → cpu）| cpu")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    args = ap.parse_args()
    mem0 = check_env(args.min_free_gb)
    import torch
    import pyarrow.parquet as pq
    import align_speed_trial as A
    import align_embed as E
    import align_train as T
    t0 = time.time()
    ck = DREAMS_DIR / "ssl_model.ckpt"
    md5 = A.verify_md5_file(ck, A.CKPT_MD5, A.CKPT_SIZE)
    droot = DREAMS_DIR / f"DreaMS-{A.DREAMS_COMMIT}"
    model, SP, DFA, linfo = A.load_dreams(str(droot), str(ck), "cpu")
    dev, dinfo = choose_device(model, SP, DFA, args.device)
    keys, manifest, split, drop = load_inputs()
    if args.speed_test:
        rep = speed_test(model, SP, DFA, dev, args.speed_test, drop)
        rep.update(md5=md5["md5"], device_check=dinfo, mem_start=mem0, load=linfo, torch=torch.__version__)
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        (out / "speed_local.json").write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep, indent=1, default=str))
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(ROOT / "data/train.parquet")
    assert pf.metadata.num_rows == E.TRAIN_ROWS
    n_shards = pf.metadata.num_row_groups
    chash = E.config_hash(manifest["rows"], len(drop))
    rep = {"issue": 109, "where": "local", "status": "running", "device": dev, "device_check": dinfo, "md5": md5["md5"],
           "mem_start": mem0, "torch": torch.__version__, "n_drop": len(drop), "n_shards": n_shards, "config_hash": chash}
    rpath = out / "embed_report.json"
    mc = E.verify_manifest(pf, manifest)
    rep["manifest_check"] = mc
    if mc["sig_mismatch"] or mc["ik_mismatch"] or mc["checked"] != E.N_QUERY:
        rep["status"] = "manifest_mismatch"
        rpath.write_text(json.dumps(rep, indent=1, default=str))
        raise SystemExit(json.dumps(mc))
    offs = E.row_group_offsets([pf.metadata.row_group(i).num_rows for i in range(n_shards)])
    qrows = E.query_rows(manifest)
    logs = []
    for k in range(n_shards):
        if E.shard_is_done(out, k, chash):
            continue
        fg = T.free_gb().get("free_gb")
        if fg is not None and fg < MIN_RUN_FREE_GB:
            logs.append({"shard": k, "status": "not_started_low_memory", "free_gb": fg})
            break
        info = E.embed_shard(k, model, SP, DFA, pf, offs[k], qrows, drop, out, dev, chash, time.time())
        info["elapsed_s"] = round(time.time() - t0, 1)
        info["ram_peak"] = A.ram_peak()
        if dev == "mps":
            info["mps_driver_allocated"] = int(torch.mps.driver_allocated_memory())
        logs.append(info)
        print("SHARD", json.dumps(info)[:300], flush=True)
        with open(out / "embed_local_log.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")
    rep["this_run"] = logs
    E.finalize(rep, out, n_shards, chash, split, keys)
    rep["elapsed_s"] = round(time.time() - t0, 1)
    rep["ram_peak"] = A.ram_peak()
    rpath.write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: rep.get(k) for k in ("status", "shards_done", "n_train_spectra", "n_query_embedded",
                                              "leak_audit", "elapsed_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()
