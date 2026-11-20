"""段 1 の 2 シャード同時の試走の部品（Issue #88、2026-11-14 夜）。

標準ライブラリだけで書く。2 シャードのノートには `sim1_trial.py` に続けてこのファイルのソースを 1 セルとして
埋め込む（1 シャードの既定のノートは変えないため、`sim1_trial.py` には手を入れない）。

- 減速倍率 s の定義（#88 の事前登録）: 同時実行のシャード 0 の t_query 合計 ÷ 昼の試走のシャード 0 の t_query 合計
- 2 本の subprocess を同じ締め切りで待ち、超えたものだけ止める
- nvidia-smi の全 GPU の行を読む
"""

# 2 シャードを同時に回す: シャード番号 → 見せる GPU（CUDA_VISIBLE_DEVICES）
SHARD_GPU = ((0, 0), (1, 1))
NSHARD = 6
MAX_STRUCTS = 40
BUDGET2_S = 60 * 60       # ノート全体
STAGE2_CAP_S = 45 * 60    # 段 1（2 本とも同じ締め切り）
STAGE2_MARGIN_S = 180


def shard_args(shard, nshard=NSHARD, max_structs=MAX_STRUCTS):
    """TC 既定に足す引数（昼の試走の `--shard 0 --nshard 6 --max_structs 40` のシャード番号だけを変える）。"""
    return ["--shard", str(shard), "--nshard", str(nshard), "--max_structs", str(max_structs)]


def shard_env(base_env, tc_env, gpu):
    env = dict(base_env)
    env.update(tc_env)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def stage2_timeout(elapsed, budget=BUDGET2_S, cap=STAGE2_CAP_S, margin=STAGE2_MARGIN_S):
    """段 1 の締め切り（秒）。残り時間 − 余裕 と cap の短い方。60 以下なら回さない。"""
    return int(min(cap, budget - elapsed - margin))


def parse_gpu_query(text):
    """`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits` → {index: (mem, util)}。"""
    out = {}
    for ln in (text or "").splitlines():
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != 3:
            continue
        try:
            out[int(parts[0])] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    return out


def update_gpu_peak(peak, sample):
    """peak: {str(index): {mem_mb, util}} を sample で更新する。"""
    for i, (mem, util) in sample.items():
        e = peak.setdefault(str(i), dict(mem_mb=0.0, util=0.0))
        e["mem_mb"] = max(e["mem_mb"], mem)
        e["util"] = max(e["util"], util)
    return peak


def wait_all(procs, deadline, now, wait_one, kill_one):
    """procs: {key: proc}。deadline（now() と同じ時計）までに終わらないものだけ kill_one で止める。

    wait_one(proc, timeout) は返り値を返すか TimeoutError 相当の例外を投げる。戻り値 {key: (rc, timed_out)}。
    """
    res = {}
    for k, p in procs.items():
        left = deadline - now()
        try:
            res[k] = (wait_one(p, max(1.0, left)), False)
        except Exception:
            res[k] = (kill_one(p), True)
    return res


def t_query_sum(values):
    vals = [float(v) for v in values]
    return dict(n=len(vals), sum=sum(vals), mean=(sum(vals) / len(vals)) if vals else None)


def slowdown(concurrent_shard0_sum, day_shard0_sum):
    """s = 同時実行のシャード 0 の t_query 合計 ÷ 昼の試走のシャード 0 の t_query 合計。"""
    if not concurrent_shard0_sum or not day_shard0_sum:
        return None
    return concurrent_shard0_sum / day_shard0_sum


def query_signature(sids, regimes, qidxs):
    """クエリの並び（sid, regime, qidx）の sha256。昼と同じクエリかを確かめるのに使う。"""
    import hashlib
    import json
    rows = [[int(s), str(r), [int(x) for x in q]] for s, r, q in zip(sids, regimes, qidxs)]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def cpu_model(lscpu_text):
    """lscpu の出力 → (型番, CPU(s))。"""
    model, n = None, None
    for ln in (lscpu_text or "").splitlines():
        if ":" not in ln:
            continue
        k, v = [x.strip() for x in ln.split(":", 1)]
        if k == "Model name" and model is None:
            model = v
        elif k == "CPU(s)" and n is None:
            try:
                n = int(v)
            except ValueError:
                pass
    return model, n
