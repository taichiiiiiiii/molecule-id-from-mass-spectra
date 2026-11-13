"""段 1（TC `simulate_v1ctx.py`）の Kaggle T4 試走の部品（Issue #88）。

標準ライブラリだけで書く。生成するノートブックにこのファイルのソースをそのまま 1 セルとして埋め込み、
同じ関数をローカルのテストでも使う。

- 入力の照合表を配布元の記録（spec の README、v4g の MANIFEST、v4b の MANIFEST）から作る
- 段 1 の標準出力（進み具合の行）から処理数と秒数を読む
- 本番の GPU 時間の外挿（式は Issue #88 の事前登録のとおり）
"""
import hashlib
import json
import re

# 段 1 の入力: work からの相対パス → (配布元のデータセット, データセットの中の相対パス, 記録の出どころ)
SPEC_FILES = ("spec_off.npy", "spec_mz.npy", "spec_it.npy", "spec_meta.parquet", "split.parquet")
POOL_FILES = ("train_structs.parquet", "pool_meta.parquet", "pool_fp.npy", "fp_bits.npy",
              "pool_frag_off.npy", "pool_frag_mass.npy", "train_fp_sel.npy")
MODEL_FILES = (("fpnet_a.pt", "fe_A.pt"), ("fpnet_b.pt", "fe_B.pt"),
               ("analysis/dreams_ft/s2/pilot_d1_step20k_ema.pt", "fe_D.pt"))
FE_V4_CODE = ("ctxlib.py", "v1engine.py", "hooks_v1.py", "derive6.py", "fragcaps.py", "chemfeat.py")
N_V1_CODE = 17

# 本番の大きさ（作者の記録。TC `fe_v4/after_sim1x_d2.log` の 15,701 クエリ、`v1fe_sim1x/log_*.txt` の 892×4＋891×2 構造）
TOTAL_QUERIES = 15701
TOTAL_STRUCTS = 5350
# 作者のシャード 0 の記録（`v1fe_sim1x/log_0.txt`）: 最初の 40 構造で 200 クエリ 641 s、6 シャード合計 36,099 シャード秒
AUTHOR_FIRST40 = dict(structs=40, queries=200, seconds=641.0)
AUTHOR_TOTAL_SHARD_SECONDS = 36099.0

_SPEC_ROW = re.compile(r"^\|\s*cache/(\S+)\s*\|\s*([\d,]+)\s*\|\s*([0-9a-f]{64})\s*\|")
_PROGRESS = re.compile(r"shard (\d+): (\d+)/(\d+) structs, (\d+) queries, (\d+)s \(([\d.]+)s/q")
_HEADER = re.compile(r"\[sim1x\] shard (\d+): (\d+) structures, (\d+) done, seed (\d+), init (\d+)s")
_DONE = re.compile(r"\[sim1x\] done shard (\d+): (\d+) new queries (\d+)s")


def parse_spec_readme(text):
    """spec cache の docs/README.md の表 → {名前: (バイト数, sha256)}。"""
    out = {}
    for line in text.splitlines():
        m = _SPEC_ROW.match(line.strip())
        if m:
            out[m.group(1)] = (int(m.group(2).replace(",", "")), m.group(3))
    return out


def expected_inputs(spec_readme_text, v4g_manifest, v4b_manifest):
    """段 1 のデータ入力の照合表。記録が無いものは sha256=None（照合せず値だけ残す）。"""
    spec = parse_spec_readme(spec_readme_text)
    out = {}
    for n in SPEC_FILES:
        b, s = spec.get(n, (None, None))
        out[n] = dict(dataset="ahmedberatozer/casmi26-spec-cache", rel=f"cache/{n}", bytes=b, sha256=s,
                      record="casmi26-spec-cache/docs/README.md" if s else "記録なし")
    gf = v4g_manifest["files"]
    for n in POOL_FILES:
        e = gf.get(f"v2_pool/{n}")
        out[n] = dict(dataset="ahmedberatozer/casmi26-v2-pool", rel=n, bytes=e["bytes"] if e else None,
                      sha256=e["sha256"] if e else None,
                      record="casmi26-attributed-v4g-reproduction-assets/MANIFEST.json" if e else "記録なし")
    bf = v4b_manifest["files"]
    for work_rel, name in MODEL_FILES:
        s = bf.get(name)
        out[work_rel] = dict(dataset="ahmedberatozer/casmi26-v4b-models", rel=name, bytes=None, sha256=s,
                             record="casmi26-v4b-models/MANIFEST.json" if s else "記録なし")
    return out


def expected_code(v4b_manifest):
    """TC の写し（README §4 の配置先）→ v4b MANIFEST の sha256。v1 casmi 17 本＋fe_v4 6 本＝23 本。"""
    bf = v4b_manifest["files"]
    out = {}
    for k, s in bf.items():
        if k.startswith("code/casmi/") and k.endswith(".py"):
            out["work/analysis/v1fe/code/casmi/" + k[len("code/casmi/"):]] = s
    for n in FE_V4_CODE:
        out["work/analysis/fe_v4/" + n] = bf["code/fe_v4/" + n]
    assert len(out) == N_V1_CODE + len(FE_V4_CODE), len(out)
    return out


def sha256_file(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compare(expected, actual):
    """expected/actual: {名前: {sha256, bytes}}。食い違いの一覧（記録が無いものは照合しない）。"""
    bad = []
    for n, e in expected.items():
        a = actual.get(n)
        if a is None:
            bad.append(dict(name=n, why="見つからない"))
            continue
        if e.get("sha256") and a.get("sha256") != e["sha256"]:
            bad.append(dict(name=n, why="sha256 が違う", expected=e["sha256"], actual=a.get("sha256")))
        elif e.get("bytes") is not None and a.get("bytes") != e["bytes"]:
            bad.append(dict(name=n, why="大きさが違う", expected=e["bytes"], actual=a.get("bytes")))
    return bad


def stage_timeout(elapsed, budget=25 * 60, cap=20 * 60, margin=120):
    """段 1 の subprocess の timeout（秒）。残り時間 − 余裕 と cap の短い方。0 以下なら回さない。"""
    return int(min(cap, budget - elapsed - margin))


def parse_sim_log(lines):
    """段 1 の標準出力 → 処理数と秒数。打ち切られても、最後の進み具合の行の値を返す。"""
    r = dict(header=None, progress=[], done=None)
    for ln in lines:
        m = _HEADER.search(ln)
        if m:
            r["header"] = dict(shard=int(m.group(1)), structs=int(m.group(2)), done_before=int(m.group(3)),
                               seed=int(m.group(4)), init_s=int(m.group(5)))
        m = _PROGRESS.search(ln)
        if m:
            r["progress"].append(dict(structs=int(m.group(2)), of=int(m.group(3)), queries=int(m.group(4)),
                                      seconds=int(m.group(5)), s_per_q=float(m.group(6))))
        m = _DONE.search(ln)
        if m:
            r["done"] = dict(queries=int(m.group(2)), seconds=int(m.group(3)))
    return r


def summarize(parsed, rows_meta_queries=None):
    """処理した構造数・クエリ数・秒数（構造の処理ループの秒。初期化は含まない）。

    完走なら done 行（構造数は見出しの行の対象数 − 処理済み）、打ち切りなら最後の進み具合の行を使う。
    """
    if parsed["done"] is not None and parsed["header"] is not None:
        n_s = parsed["header"]["structs"] - parsed["header"]["done_before"]
        n_q, sec, src = parsed["done"]["queries"], float(parsed["done"]["seconds"]), "done 行"
        if rows_meta_queries is not None:
            n_q = rows_meta_queries
    elif parsed["progress"]:
        p = parsed["progress"][-1]
        n_s, n_q, sec, src = p["structs"], p["queries"], float(p["seconds"]), "最後の進み具合の行（打ち切り）"
    else:
        return dict(structs=0, queries=0, seconds=None, s_per_struct=None, s_per_query=None, source="なし")
    return dict(structs=n_s, queries=n_q, seconds=sec, s_per_struct=sec / n_s if n_s else None,
                s_per_query=sec / n_q if n_q else None, source=src)


def extrapolate(s_per_query, s_per_struct, parallel=(1, 2, 6)):
    """本番の段 1 の GPU 時間（時間）。並行で 1 本あたりが遅くなる分は入れていない（2・6 本は楽観寄り）。

    補正: 作者の記録でも最初の 40 構造は全体平均より重い（641/200 s/q 対 36,099/15,701 s/q）ので、
    その比を掛けた値も出す（推定）。
    """
    author_first = AUTHOR_FIRST40["seconds"] / AUTHOR_FIRST40["queries"]
    author_all = AUTHOR_TOTAL_SHARD_SECONDS / TOTAL_QUERIES
    corr = author_all / author_first
    out = dict(total_queries=TOTAL_QUERIES, total_structs=TOTAL_STRUCTS, author_first40_s_per_q=round(author_first, 4),
               author_all_s_per_q=round(author_all, 4), correction=round(corr, 4), by_parallel={})
    for k in parallel:
        e = {}
        if s_per_query:
            e["by_query_h"] = round(s_per_query * TOTAL_QUERIES / k / 3600, 3)
            e["by_query_corrected_h"] = round(s_per_query * corr * TOTAL_QUERIES / k / 3600, 3)
        if s_per_struct:
            e["by_struct_h"] = round(s_per_struct * TOTAL_STRUCTS / k / 3600, 3)
        out["by_parallel"][str(k)] = e
    return out


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
