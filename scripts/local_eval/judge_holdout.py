#!/usr/bin/env python3
"""12/6（中間）・12/11（最終）の holdout の判定（Issue #249。規則は事前の記録 issuecomment-7234059974、
type-weights.md §3.2、オーナー 11/26 の承認 4、#290 の (a)〜(d)）。

holdout のモード（既定のパスは持たない。holdout のパスを引数で与えない限り holdout を読めない）:

    python scripts/local_eval/judge_holdout.py --cand D0 D1 D2 --baseline-dir outputs/local_eval/baseline_1206 \\
        --msg0 docs/experiments/holdout_msg0_eval.csv --msg1 docs/experiments/holdout_msg1_exclude.csv \\
        --label interim_1206 --out outputs/local_eval/judge/interim_1206

  実行のたびに、holdout の入力を開く前に outputs/local_eval/judge/reads.jsonl に 1 行足す。既に 2 行あれば
  何も開かずに止まる（3 回目は止まる。読んだ後に例外で止まった回も 1 回と数える）。上限を外す引数は無い。

dev の試し（採否には使わない。reads.jsonl を読まない・書かない）:

    python scripts/local_eval/judge_holdout.py --dry-run-dev --cand exp016 exp017 exp018 --base exp007 exp014 exp015 \\
        --msg0 docs/experiments/msg0_eval.csv --msg1 docs/experiments/msg1_exclude.csv --out <dir>

判定: 集合ごとに Δ = mean(版 A の 3 seed 平均の rr) − mean(基準の rr)、区間は evaluate.paired_bootstrap
（2,000 回・seed 0・百分位）。(i) すべての集合で下限 ≥ −0.02、かつ (ii) 1 つ以上で下限 > 0。丸める前の値で比べる。
判定に使う分子は MSG と重ならない正解（msg0）。msg0 が 150 分子未満の集合だけ全集合で判定する（分子数だけで決まる）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validate"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))
from evaluate import BOOT_SEED, N_BOOT, paired_bootstrap  # noqa: E402

GATE = -0.02            # (i) の下限。playbook §4.1 の値（引数で変えられない）
MIN_REMAIN = 150        # #290 (d): msg0 がこれ未満の集合は全集合で判定
MAX_READS = 2           # playbook §4.1: holdout は 12/6 の中間と 12/11 の最終の 2 回まで
SEEDS = (0, 1, 2)       # #290 (b): 版 A の 3 本（seed は選ばない）
SAME_KEYS = ("table_sha256", "features_sha256", "rows", "num_boost_round", "lightgbm", "exclude_features",
             "extra_features", "train_types", "n_rows")   # feature_groups.SAME_KEYS と同じ
# 読み出しの記録はクローンの実ディレクトリに固定する（worktree・git archive の写しから動かしても同じ記録で数える）
MAIN_CLONE = Path("/Users/taichi/work/molecule-id-dev")
READS_REL = "outputs/local_eval/judge/reads.jsonl"


def reads_log_path() -> Path:
    """本番の reads.jsonl。クローン（.git のある実ディレクトリ）が無ければ止める（別の場所に新しい記録を作らない）。"""
    if not (MAIN_CLONE / ".git").exists():
        raise JudgeError(f"クローン {MAIN_CLONE} が無い。reads.jsonl を別の場所に作らないため止める")
    return MAIN_CLONE / READS_REL

# 判定用 4 集合（この順）と型
HOLDOUT_SETS = (("v3_c1u_t10", "C1 型 (a)"), ("enveda_v1u_t10", "(d) プールにあり・参照なし"),
                ("v3_c2_t10", "C2 型 (c)"), ("v3_c2nu_t10", "c2n 型 (b)"))
# 基準 score+prior の分子ごとの rr（#249 の結果のコメント issuecomment-7185304784 の sha256、全桁）
BASELINE_RR_SHA256 = {
    "v3_c1u_t10": "db9ec1f9a5aa656b8a9cd724380f19b600d5d51a57b166725b33d30206349012",
    "enveda_v1u_t10": "5ffac6bf02f3f60da5316b15b02ea1e0b97dae8ba6e382c7ff9943af586de22c",
    "v3_c2_t10": "08dbddfef6ce243240f6c55386c6c0293afd9b672c6f1b33337b7386ccca5d05",
    "v3_c2nu_t10": "fe42d688b45436fe90e484c592bb8033cb54057a09b1196f41bea86e59727f85",
}
# docs/experiments/holdout_msg_lists.json（#290）の値
HOLDOUT_MSG_SHA256 = {"msg0": "2834ef555aa09d2ff83ed6532b5849edd726261ad03d90ee6ee03a25bc01fcd1",
                      "msg1": "5bd180b8851af95b0dbe60eb4eae6ca1eed167c593e8fb58fa1db5ed8b218340"}
# dev の 5 型（a ≒ C1、c ≒ C2、d ≒ enveda_v1）
DEV_TYPES = (("a", "dev 型 a（≒ C1）"), ("b", "dev 型 b"), ("c", "dev 型 c（≒ C2）"),
             ("d", "dev 型 d（≒ enveda_v1、プールにあり参照なし）"), ("e", "dev 型 e"))
HOLDOUT_MARKERS = tuple(s for s, _ in HOLDOUT_SETS) + ("baseline_1206", "holdout")

USE_PASS = "満たしたので、版 A を最終候補の 1 つとして保持する（playbook §4.1）。"
USE_FAIL = ("満たさない。dev での開発は続けてよいが、holdout の値・分子別の結果は開発に使わない。"
            "両方の holdout で満たさない場合の代替は最終 2 本の選び方の規則（#195 B4）に従う（playbook §4.1）。")


class JudgeError(RuntimeError):
    pass


class ReadLimitError(JudgeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return "unknown"


# ---------- 読み出しの上限 ----------

def count_reads(log: Path) -> int:
    if not Path(log).exists():
        return 0
    return sum(1 for ln in Path(log).read_text().splitlines() if ln.strip())


def register_read(log: Path, entry: dict, max_reads: int = MAX_READS) -> int:
    """holdout の入力を開く前に呼ぶ。既に max_reads 回あれば止める。追記した回の番号（1 始まり）を返す。"""
    n = count_reads(log)
    if n >= max_reads:
        raise ReadLimitError(f"holdout の読み出しは既に {n} 回（上限 {max_reads}）。何も開かずに止める")
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    rec = dict(entry, read_no=n + 1, utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    with open(log, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return n + 1


# ---------- 読み込み ----------

def _check_seeds(name: str, metas: list, seeds=SEEDS) -> None:
    if len(metas) != len(seeds):
        raise JudgeError(f"{name}: {len(metas)} 本（{len(seeds)} 本が要る）")
    got = sorted(int(m["seed"]) for m in metas)
    if got != sorted(seeds):
        raise JudgeError(f"{name}: seed の組が {got}（{sorted(seeds)} が要る）")
    for m in metas[1:]:
        for k in SAME_KEYS:
            if m.get(k) != metas[0].get(k):
                raise JudgeError(f"{name}: {k} が 3 本で同じでない")


def load_runs(name: str, dirs, seeds=SEEDS, dev: bool = False) -> pd.DataFrame:
    """3 本の per_molecule.csv を読み、(set, molecule_id) ごとの rr の 3 本の平均と type を返す。"""
    metas = [json.loads((Path(d) / "meta.json").read_text()) for d in dirs]
    if dev:   # dev の試しでは per_molecule を開く前に meta の集合が dev（b8_）だけであることを確かめる
        hold = {s for s, _ in HOLDOUT_SETS}
        for d, m in zip(dirs, metas):
            sets = list((m.get("sets") or {}).keys())
            if not sets or any((not s.startswith("b8_")) or s in hold for s in sets):
                raise JudgeError(f"{d}: dev の試しに dev（b8_）以外の集合が入っている")
    _check_seeds(name, metas, seeds)
    pms = [pd.read_csv(Path(d) / "per_molecule.csv", dtype={"molecule_id": str, "set": str}) for d in dirs]
    idx0 = pms[0].set_index(["set", "molecule_id"]).index
    if idx0.has_duplicates:
        raise JudgeError(f"{name}: per_molecule.csv に重複した分子")
    rr = []
    for d, pm in zip(dirs, pms):
        pm = pm.set_index(["set", "molecule_id"])
        if len(pm) != len(idx0) or set(pm.index) != set(idx0):
            raise JudgeError(f"{name}: {d} の分子の集合が他の本と違う")
        if not np.isfinite(pm["rr"].to_numpy(float)).all():
            raise JudgeError(f"{name}: {d} の rr に NaN/inf")
        rr.append(pm.loc[idx0, "rr"].to_numpy(float))
    out = pd.DataFrame({"rr": np.mean(rr, axis=0)}, index=idx0)
    if "type" in pms[0].columns:
        out["type"] = pms[0].set_index(["set", "molecule_id"]).loc[idx0, "type"].astype(str).to_numpy()
    out.attrs["meta"] = [{k: m.get(k) for k in ("seed",) + SAME_KEYS} for m in metas]
    return out


def load_baseline(baseline_dir: Path, pins=None) -> pd.DataFrame:
    """基準 score+prior の <集合>_rr.csv（4 集合）。sha256 が固定の値と違えば止める。"""
    pins = BASELINE_RR_SHA256 if pins is None else pins
    parts = []
    for s, _ in HOLDOUT_SETS:
        p = Path(baseline_dir) / f"{s}_rr.csv"
        got = sha256_file(p)
        if got != pins[s]:
            raise JudgeError(f"基準 {p} の sha256 {got} が固定の値 {pins[s]} と違う")
        d = pd.read_csv(p, dtype={"molecule_id": str})
        d.insert(0, "set", s)
        parts.append(d)
    b = pd.concat(parts, ignore_index=True).set_index(["set", "molecule_id"])
    if b.index.has_duplicates or not np.isfinite(b["rr"].to_numpy(float)).all():
        raise JudgeError("基準の rr に重複か NaN/inf")
    return b


def load_list(p: Path, pin: str | None = None) -> set:
    if pin is not None and sha256_file(p) != pin:
        raise JudgeError(f"{p} の sha256 が固定の値 {pin} と違う")
    d = pd.read_csv(p, dtype={"molecule_id": str, "set": str})
    return set(zip(d["set"], d["molecule_id"]))


# ---------- 判定 ----------

def judge_groups(groups, cand: pd.DataFrame, base: pd.DataFrame, msg0: set, msg1: set,
                 min_remain: int = MIN_REMAIN, gate: float = GATE, exact_lists: bool = True) -> dict:
    """groups: [(名前, 表示名, 全集合の分子の index)]。cand/base: index (set, molecule_id) の列 rr。"""
    if msg0 & msg1:
        raise JudgeError("msg0 と msg1 が重なる")
    rows = []
    for name, label, full in groups:
        full = list(full)
        if not full:
            raise JudgeError(f"{name}: 分子が無い")
        fs = set(full)
        miss_c = fs - set(cand.index)
        miss_b = fs - set(base.index)
        if miss_c or miss_b:
            raise JudgeError(f"{name}: 版 A に無い分子 {len(miss_c)}、基準に無い分子 {len(miss_b)}")
        unlisted = [m for m in full if m not in msg0 and m not in msg1]
        if unlisted:
            raise JudgeError(f"{name}: msg0 にも msg1 にも無い分子 {len(unlisted)}")
        if exact_lists:
            listed = {m for m in msg0 | msg1 if m[0] == name}
            if listed != fs:
                raise JudgeError(f"{name}: msg0 ∪ msg1 の分子が全集合と一致しない")
        sub = [m for m in full if m in msg0]
        n_msg1 = len(full) - len(sub)
        on = "subset" if len(sub) >= min_remain else "full"
        use = sub if on == "subset" else full
        rc = cand.loc[use, "rr"].to_numpy(float)
        rb = base.loc[use, "rr"].to_numpy(float)
        lo, hi = paired_bootstrap(rc, rb)
        rows.append(dict(set=name, label=label, n_full=len(full), n_msg1=n_msg1, n_remain=len(sub), judge_on=on,
                         n=len(use), base_mrr=float(rb.mean()), cand_mrr=float(rc.mean()),
                         delta=float(rc.mean() - rb.mean()), lo=lo, hi=hi,
                         lo_ge_gate=bool(lo >= gate), lo_gt_0=bool(lo > 0)))
    fail_i = [r["set"] for r in rows if not r["lo_ge_gate"]]
    pos = [r["set"] for r in rows if r["lo_gt_0"]]
    ok = (not fail_i) and bool(pos)
    return dict(gate=gate, min_remain=min_remain, n_boot=N_BOOT, boot_seed=BOOT_SEED, sets=rows,
                cond_i=not fail_i, cond_i_fail=fail_i, cond_ii=bool(pos), cond_ii_sets=pos, pass_=ok)


def render(res: dict, dry_run: bool, c2_only_name: str = "v3_c2_t10") -> str:
    L = []
    unit = "型" if dry_run else "集合"
    base_name = "基準" if dry_run else "基準 score+prior"
    g = f"{res['gate']:+.2f}".replace("-", "−")
    if dry_run:
        L.append("【dev の試し。採否には使わない（holdout は読んでいない）】")
    for r in res["sets"]:
        where = (f"全集合 {r['n_full']} のうち MSG と重ならない部分集合で判定" if r["judge_on"] == "subset"
                 else f"msg0 が {r['n_remain']} 分子で {res['min_remain']} 未満なので全集合で判定・MSG との重なり {r['n_msg1']}")
        L.append(f"- {r['set']}（{r['label']}）: n ＝ {r['n']}（{where}）、{base_name} ＝ {r['base_mrr']:.4f}、"
                 f"版 A ＝ {r['cand_mrr']:.4f}、Δ ＝ {r['delta']:+.4f}（記録用）、95% 区間 [{r['lo']:+.4f} , {r['hi']:+.4f}]、"
                 f"L ≥ {g}: {'満たす' if r['lo_ge_gate'] else '満たさない'}、L > 0: {'はい' if r['lo_gt_0'] else 'いいえ'}")
    n = len(res["sets"])
    L.append(f"- (i) {n} {unit}すべてで L ≥ {g}: " + ("満たす" if res["cond_i"] else
             f"満たさない（満たさない{unit}: " + "・".join(res["cond_i_fail"]) + "）"))
    pos = res["cond_ii_sets"]
    kind = "無し" if not pos else (f"{c2_only_name} だけ" if pos == [c2_only_name] else "ほかを含む")
    L.append(f"- (ii) L > 0 の{unit}: " + ("・".join(pos) if pos else "無し") + f"（{kind}）")
    L.append(f"- 判定: {'条件を満たす' if res['pass_'] else '条件を満たさない'}")
    use = USE_PASS if res["pass_"] else USE_FAIL
    L.append(f"- 結果の使い方: （dev の試しなので適用しない。holdout の判定なら次の文）{use}" if dry_run
             else f"- 結果の使い方: {use}")
    L.append(f"- 区間: 分子単位の対応のある bootstrap {res['n_boot']} 回・seed {res['boot_seed']}・百分位"
             f"（evaluate.paired_bootstrap）。比べは丸める前の値")
    return "\n".join(L) + "\n"


def _write(out: Path, res: dict, text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "judge.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (out / "judge.md").write_text(text)


# ---------- 2 つのモード ----------

def run_dev(cand_dirs, base_dirs, msg0_path, msg1_path, out: Path | None) -> tuple[dict, str]:
    for p in list(cand_dirs) + list(base_dirs) + [msg0_path, msg1_path] + ([out] if out else []):
        if any(k in str(p) for k in HOLDOUT_MARKERS):
            raise JudgeError(f"dev の試しに holdout らしいパス {p}")
    cand = load_runs("候補", cand_dirs, dev=True)
    base = load_runs("基準", base_dirs, dev=True)
    if set(cand.index) != set(base.index):
        raise JudgeError("候補と基準で分子の集合が違う")
    if not (base.loc[cand.index, "type"].to_numpy() == cand["type"].to_numpy()).all():
        raise JudgeError("候補と基準で型が違う")
    msg0, msg1 = load_list(msg0_path), load_list(msg1_path)
    groups = [(t, lab, list(cand.index[(cand["type"] == t).to_numpy()])) for t, lab in DEV_TYPES]
    res = judge_groups(groups, cand, base, msg0, msg1, exact_lists=False)
    res.update(mode="dry-run-dev", commit=git_commit(), cand=[str(d) for d in cand_dirs],
               base=[str(d) for d in base_dirs], cand_meta=cand.attrs["meta"], base_meta=base.attrs["meta"],
               inputs_sha256={str(p): sha256_file(Path(p)) for p in (msg0_path, msg1_path)})
    text = render(res, dry_run=True, c2_only_name="c")
    if out:
        _write(Path(out), res, text)
    return res, text


def run_holdout(cand_dirs, baseline_dir, msg0_path, msg1_path, label: str, out: Path, reads_log: Path | None = None,
                baseline_pins=None, msg_pins=None) -> tuple[dict, str]:
    reads_log = reads_log_path() if reads_log is None else reads_log
    # 中身を開かずに確かめられることだけ先に（パスの打ち間違いで 1 回を使わないため）
    need = [Path(d) / f for d in cand_dirs for f in ("meta.json", "per_molecule.csv")]
    need += [Path(baseline_dir) / f"{s}_rr.csv" for s, _ in HOLDOUT_SETS] + [Path(msg0_path), Path(msg1_path)]
    missing = [str(p) for p in need if not p.is_file()]
    if missing:
        raise JudgeError(f"入力が無い: {missing}")
    if len(cand_dirs) != len(SEEDS):
        raise JudgeError(f"--cand は {len(SEEDS)} 本")
    no = register_read(reads_log, dict(label=label, commit=git_commit(), cand=[str(d) for d in cand_dirs],
                                       baseline_dir=str(baseline_dir), msg0=str(msg0_path), msg1=str(msg1_path),
                                       out=str(out)))
    mp = HOLDOUT_MSG_SHA256 if msg_pins is None else msg_pins
    msg0, msg1 = load_list(msg0_path, mp["msg0"]), load_list(msg1_path, mp["msg1"])
    base = load_baseline(baseline_dir, baseline_pins)
    cand = load_runs("版 A", cand_dirs)
    hold = {s for s, _ in HOLDOUT_SETS}
    extra = {s for s, _ in cand.index} - hold
    if extra:
        raise JudgeError(f"版 A に判定用 4 集合以外の集合 {sorted(extra)}")
    if {s for s, _ in msg0 | msg1} != hold:
        raise JudgeError("msg0 ∪ msg1 の集合名が判定用 4 集合とちょうど一致しない")
    groups = [(s, lab, list(base.index[base.index.get_level_values(0) == s])) for s, lab in HOLDOUT_SETS]
    if set(cand.index) != set(base.index):
        raise JudgeError("版 A と基準で分子の集合が違う")
    res = judge_groups(groups, cand, base, msg0, msg1, exact_lists=True)
    res.update(mode="holdout", label=label, read_no=no, commit=git_commit(), cand=[str(d) for d in cand_dirs],
               cand_meta=cand.attrs["meta"], baseline_dir=str(baseline_dir),
               inputs_sha256={str(p): sha256_file(p) for p in need})
    text = render(res, dry_run=False)
    _write(Path(out), res, text)
    return res, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run-dev", action="store_true")
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--base", nargs="+", help="dev の試しの基準の 3 本（holdout のモードでは使わない）")
    ap.add_argument("--baseline-dir", help="holdout のモード: 基準 score+prior の <集合>_rr.csv の場所")
    ap.add_argument("--msg0", required=True)
    ap.add_argument("--msg1", required=True)
    ap.add_argument("--label")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.dry_run_dev:
        if a.baseline_dir or a.label:
            ap.error("dev の試しに --baseline-dir / --label は与えない")
        if not a.base:
            ap.error("dev の試しには --base が要る")
        _, text = run_dev(a.cand, a.base, a.msg0, a.msg1, Path(a.out) if a.out else None)
    else:
        if a.base:
            ap.error("holdout のモードに --base は与えない（基準は --baseline-dir）")
        if not (a.baseline_dir and a.label and a.out):
            ap.error("holdout のモードには --baseline-dir・--label・--out が要る")
        _, text = run_holdout(a.cand, a.baseline_dir, a.msg0, a.msg1, a.label, Path(a.out))
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
