"""Issue #82: キットの train_fpnet.py のログから it/s と時間を読み、本番 82,855 ステップの時間を外挿する。

ノートブック（kernels/fpnet_full_train）にこのファイルの中身を埋め込んで使う。外挿の式は Issue #82 の事前登録のとおり:
    本番の時間 = 準備とデータの読み込みの時間 + 82,855 / 定常の it/s + 16 x 最後の評価にかかった時間
定常の it/s = ログの step 1000 の行から最後の step の行までのステップの差 / 時刻の差。
"""
import json
import re

PROD_STEPS = 82855
N_EVALS = 16
SESSION_LIMIT_H = 11.0
STEADY_FROM = 1000

_TS = r'(\d\d):(\d\d):(\d\d)'
_STEP = re.compile(_TS + r' step (\d+) lr \S+ .* ([0-9.]+) it/s queue (\d+)\s*$')
_DATA = re.compile(r'\[Data\] train spectra ([\d,]+) groups ([\d,]+) pool ([\d,]+) nbits (\d+)')
_SETS = re.compile(_TS + r" (\{.*\}) train spectra (\d+)\s*$")
_PARAMS = re.compile(_TS + r' parameters (\d+) ')
_EVAL = re.compile(_TS + r' EVAL (\{.*\})\s*$')


def _sec(h, m, s):
    return int(h) * 3600 + int(m) * 60 + int(s)


def _unwrap(times):
    """HH:MM:SS の秒を、日付をまたいでも増えるように直す。"""
    out, add, prev = [], 0, None
    for t in times:
        if prev is not None and t + add < prev:
            add += 86400
        out.append(t + add)
        prev = t + add
    return out


def parse_log(text):
    steps, params_t, eval_t, eval_rec, data, sets = [], None, None, None, None, None
    for line in text.splitlines():
        m = _STEP.match(line)
        if m:
            steps.append((_sec(*m.group(1, 2, 3)), int(m.group(4)), float(m.group(5)), int(m.group(6))))
            continue
        m = _DATA.search(line)
        if m:
            data = dict(train_spectra=int(m.group(1).replace(',', '')), groups=int(m.group(2).replace(',', '')),
                        pool=int(m.group(3).replace(',', '')), nbits=int(m.group(4)))
            continue
        m = _SETS.match(line)
        if m:
            sets = dict(eval_sets=m.group(4), train_spectra=int(m.group(5)), t=_sec(*m.group(1, 2, 3)))
            continue
        m = _PARAMS.match(line)
        if m:
            params_t = _sec(*m.group(1, 2, 3))
            continue
        m = _EVAL.match(line)
        if m:
            eval_t = _sec(*m.group(1, 2, 3))
            eval_rec = json.loads(m.group(4))
    return dict(steps=steps, params_t=params_t, eval_t=eval_t, eval=eval_rec, data=data, sets=sets)


def steady_rate(steps, start=STEADY_FROM):
    """step >= start の最初の行から最後の行までの (ステップの差 / 時刻の差)。行が足りなければ None。"""
    if not steps:
        return None
    ts = _unwrap([s[0] for s in steps])
    idx = [i for i, s in enumerate(steps) if s[1] >= start]
    if len(idx) < 2:
        return None
    a, b = idx[0], len(steps) - 1
    dt = ts[b] - ts[a]
    return (steps[b][1] - steps[a][1]) / dt if dt > 0 else None


def extrapolate(setup_seconds, rate, eval_seconds, prod_steps=PROD_STEPS, n_evals=N_EVALS):
    """事前登録の式。rate が無ければ None。"""
    if not rate:
        return None
    total = setup_seconds + prod_steps / rate + n_evals * eval_seconds
    return dict(setup_seconds=setup_seconds, steady_it_per_s=rate, eval_seconds=eval_seconds, prod_steps=prod_steps,
                n_evals=n_evals, train_seconds=prod_steps / rate, total_seconds=total, total_hours=total / 3600.0,
                one_session=bool(total / 3600.0 <= SESSION_LIMIT_H), session_limit_hours=SESSION_LIMIT_H)


def summarize(text, proc_start_sec_of_day):
    """ログ全体から報告用の数をまとめる。proc_start_sec_of_day はログと同じ時計の、学習プロセスを起動した時刻（その日の秒）。"""
    p = parse_log(text)
    steps = p['steps']
    rate = steady_rate(steps)
    setup = None
    if p['params_t'] is not None:
        setup = p['params_t'] - proc_start_sec_of_day
        if setup < 0:
            setup += 86400
    eval_s, lower_bound = 0.0, True
    if p['eval_t'] is not None and steps:
        eval_s = p['eval_t'] - steps[-1][0]
        if eval_s < 0:
            eval_s += 86400
        lower_bound = False
    ext = extrapolate(setup or 0.0, rate, eval_s)
    if ext is not None:
        ext['lower_bound'] = lower_bound or setup is None
    return dict(last_step=steps[-1][1] if steps else 0, last_logged_it_per_s=steps[-1][2] if steps else None,
                last_queue=steps[-1][3] if steps else None, min_queue=min(s[3] for s in steps) if steps else None,
                data=p['data'], sets=p['sets'], final_eval=p['eval'], extrapolation=ext)
