"""#220 (#195 A15): 進め方の数値を、Issue の時刻の写しと学習ログの時刻から数える（読むだけ）。

入力: docs/research/schedule/a15_inputs.json（時刻は UTC、出どころのパスつき）
出力: JSON（標準出力、--out があればファイルにも）

(a) 直近 5 件の実験: 着手（事前登録の Issue を作った時刻）から結論（判定のコメント）までの時間と、
    そのうちログに記録された計算の時間（学習・採点のジョブの壁時計）。残り = 手続き（事前登録・レビュー・
    PR・待ち）の時間とみなす。
(b) window（168 h）の中で、このプロジェクトのローカル学習（速度測定を含む）が走っていた時間の割合。
    区間は重なりを 1 回だけ数える（和集合）。自前設計の学習だけの割合も出す。
(c) window の中に作られた Issue の 3 区分の件数と割合。
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "docs/research/schedule/a15_inputs.json"


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hours(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 3600.0


def union_hours(intervals, lo=None, hi=None) -> float:
    """区間 [(start, end), ...] の和集合の長さ（時間）。lo/hi があればその範囲に切る。"""
    segs = []
    for s, e in intervals:
        if e < s:
            raise ValueError(f"end < start: {s} {e}")
        if lo is not None:
            s = max(s, lo)
        if hi is not None:
            e = min(e, hi)
        if e > s:
            segs.append((s, e))
    segs.sort()
    total = 0.0
    cur_s = cur_e = None
    for s, e in segs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += hours(cur_s, cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += hours(cur_s, cur_e)
    return total


def compute(inp: dict) -> dict:
    w0 = parse_ts(inp["window"]["start"])
    w1 = parse_ts(inp["window"]["end"])
    win_h = hours(w0, w1)

    # (a)
    exps = []
    for x in inp["experiments"]:
        s, c = parse_ts(x["start"]), parse_ts(x["conclusion"])
        if c < s:
            raise ValueError(f"#{x['issue']}: conclusion before start")
        comp = [(parse_ts(k["start"]), parse_ts(k["end"])) for k in x["compute"]]
        comp_h = union_hours(comp, s, c)
        tot_h = hours(s, c)
        exps.append({
            "issue": x["issue"], "title": x["title"],
            "total_h": round(tot_h, 3), "compute_h": round(comp_h, 3),
            "procedure_h": round(tot_h - comp_h, 3),
            "procedure_share": round((tot_h - comp_h) / tot_h, 3) if tot_h > 0 else None,
            "conclusion_ref": x["conclusion_ref"],
        })
    tot = sum(e["total_h"] for e in exps)
    comp_sum = sum(e["compute_h"] for e in exps)

    # (b)
    tr = inp["training_intervals"]
    all_iv = [(parse_ts(t["start"]), parse_ts(t["end"])) for t in tr]
    own_iv = [(parse_ts(t["start"]), parse_ts(t["end"])) for t in tr if t["own_design"]]
    tr_h = union_hours(all_iv, w0, w1)
    own_h = union_hours(own_iv, w0, w1)

    # (c)
    wb = inp["work_breakdown"]
    cats = {k: sorted(v) for k, v in wb.items() if not k.startswith("_")}
    flat = [n for v in cats.values() for n in v]
    if len(flat) != len(set(flat)):
        raise ValueError("an issue is in two categories")
    n_all = len(flat)

    return {
        "window": {"start": inp["window"]["start"], "end": inp["window"]["end"], "hours": round(win_h, 3)},
        "a_experiments": exps,
        "a_summary": {
            "n": len(exps),
            "total_h_sum": round(tot, 3),
            "compute_h_sum": round(comp_sum, 3),
            "procedure_h_sum": round(tot - comp_sum, 3),
            "procedure_share_pooled": round((tot - comp_sum) / tot, 3) if tot > 0 else None,
            "median_total_h": sorted(e["total_h"] for e in exps)[len(exps) // 2],
        },
        "b_training": {
            "train_h": round(tr_h, 3), "share": round(tr_h / win_h, 4),
            "own_design_h": round(own_h, 3), "own_design_share": round(own_h / win_h, 4),
            "n_intervals": len(tr),
        },
        "c_breakdown": {
            "n_issues": n_all,
            **{k: {"n": len(v), "share": round(len(v) / n_all, 3), "issues": v} for k, v in cats.items()},
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = compute(json.loads(Path(a.input).read_text()))
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
