"""Issue #187: 分子ごとのクエリのスペクトルが「どの付加イオンの組」で測られたかを数える（読むだけ・採点しない）。

付加イオンの組を構造の証拠にする段（0.498 の公開ノートの APRIOR と同じ種類の証拠）を v3 で測る価値があるかを決める
ための数え上げ。MRR の採点はしない（submission の CSV も truth も読まない）。

集合ごとに（v3_c1 / v3_c2 / enveda_v1 / v3_c2n の test.parquet と可視の test.parquet）:
- 分子ごとの付加イオンの種類の数（1 / 2 / 3 以上の分子の数と割合）、スペクトルの数の平均
- 主な組（付加イオンの文字列を並べた組）の度数（上位 --top 件）
- APRIOR が使う 4 つのラベル（[M+H]+・[M-H]-・[M+CH2O2-H]-・[M+Na]+）の有無の組の度数と、4 つのどれかを持つ分子の割合
- スペクトルの付加イオンごとの度数と、多量体（[2M…・[3M…）の付加イオンを持つ分子の割合
- 極性（正のみ / 負のみ / 両方）、全クエリが timsTOF の分子の割合とスペクトル単位の timsTOF の割合（enveda-180 の測りかたの目安。test.parquet には
  ingest_lib が無いので代わりに instrument_type を使う）

--train を渡すと、train.parquet の enveda-180（構造 = inchikey14）について同じ数え上げと、「同じ分子式の構造どうしで
4 ラベルの組が違う対の割合」（付加イオンの組が同じ分子式の異性体を分けうる度合いの粗い目安。測定の揺れも含む）を出す。
train.parquet は `pq.ParquetFile(p, buffer_size=1<<20, pre_buffer=False)` で row group ごとに 5 列だけ読む。

    .venv/bin/python scripts/research/adduct_set_counts.py --train data/train.parquet \
        --out outputs/local_eval/adduct_set_counts.json
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import pandas as pd

HOLDOUTS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
OBS = ("[M+H]+", "[M-H]-", "[M+CH2O2-H]-", "[M+Na]+")   # APRIOR のラベル（ノートの _AP_OBS と同じ並び）
TIMS = "timsTOF"


def polarity(adduct):
    a = str(adduct).strip()
    if a.endswith("+"):
        return "+"
    if a.endswith("-"):
        return "-"
    return "?"


def obs_pattern(adducts):
    """4 ラベルの有無を '1010' のような文字列にする（OBS の並び）。"""
    s = set(adducts)
    return "".join("1" if a in s else "0" for a in OBS)


def summarize(df, key="molecule_id", top=10):
    """df（key・adduct・instrument_type 列）から分子ごとの付加イオンの組の分布を数える。"""
    df = df.assign(adduct=df["adduct"].fillna("").astype(str))
    g = df.groupby(key, sort=True)
    sets = g["adduct"].agg(lambda s: tuple(sorted(set(s))))
    n_spec = g.size()
    n = int(len(sets))
    k = sets.map(len)
    dist = {"1": int((k == 1).sum()), "2": int((k == 2).sum()), "3+": int((k >= 3).sum())}
    frac = {a: round(b / n, 4) if n else None for a, b in dist.items()}
    combos = Counter(" ".join(s) for s in sets)
    pats = Counter(obs_pattern(s) for s in sets)
    pol = Counter("".join(sorted({polarity(a) for a in s})) for s in sets)
    out = dict(
        n_molecules=n,
        n_spectra=int(len(df)),
        spectra_per_molecule_mean=round(float(n_spec.mean()), 3) if n else None,
        n_adduct_types=dist,
        n_adduct_types_frac=frac,
        ge2_frac=round((dist["2"] + dist["3+"]) / n, 4) if n else None,
        top_sets=[[c, int(m)] for c, m in combos.most_common(top)],
        n_distinct_sets=int(len(combos)),
        spectra_by_adduct=[[x, int(m)] for x, m in Counter(df["adduct"]).most_common()],
        has_multimer_frac=round(float(sets.map(lambda s: any(x.startswith("[2M") or x.startswith("[3M")
                                                             for x in s)).mean()), 4) if n else None,
        obs_pattern_order=list(OBS),
        obs_patterns=[[p, int(m)] for p, m in pats.most_common()],
        any_obs_label_frac=round(sum(m for p, m in pats.items() if p != "0000") / n, 4) if n else None,
        polarity={p: int(m) for p, m in sorted(pol.items())},
    )
    if "instrument_type" in df.columns:
        all_tims = g["instrument_type"].agg(lambda s: bool((s.astype(str) == TIMS).all()))
        out["all_timstof_frac"] = round(float(all_tims.mean()), 4) if n else None
        out["spectra_timstof_frac"] = round(float((df["instrument_type"].astype(str) == TIMS).mean()), 4) if n else None
    return out


def same_formula_pairs(struct_formula, struct_pattern):
    """同じ分子式の構造の対のうち、4 ラベルの組が違う対の数と割合（構造 2 つ以上の分子式だけ）。"""
    by_f = {}
    for s, f in struct_formula.items():
        by_f.setdefault(f, Counter())[struct_pattern[s]] += 1
    pairs = same = groups = groups_diff = 0
    for c in by_f.values():
        m = sum(c.values())
        if m < 2:
            continue
        groups += 1
        groups_diff += len(c) >= 2
        pairs += m * (m - 1) // 2
        same += sum(v * (v - 1) // 2 for v in c.values())
    return dict(formula_groups_ge2=groups, formula_groups_with_ge2_patterns=groups_diff,
                pairs=pairs, pairs_diff_pattern=pairs - same,
                pairs_diff_frac=round((pairs - same) / pairs, 4) if pairs else None)


def read_enveda_train(path, lib="enveda-180", batch=8192):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    f = pq.ParquetFile(path, buffer_size=1 << 20, pre_buffer=False)
    cols = ["ingest_lib", "inchikey14", "molecular_formula", "adduct", "instrument_type"]
    parts = []
    for b in f.iter_batches(batch_size=batch, columns=cols):
        b = b.filter(pc.equal(b["ingest_lib"], lib))
        if b.num_rows:
            parts.append(b.to_pandas()[cols[1:]])
    df = pd.concat(parts, ignore_index=True)
    return df


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-dir", default="data/holdouts")
    ap.add_argument("--test", default="data/test.parquet")
    ap.add_argument("--train", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cols = ["molecule_id", "adduct", "instrument_type"]
    res = {"inputs": {}, "sets": {}}
    paths = {h: os.path.join(a.holdout_dir, h, "test.parquet") for h in HOLDOUTS}
    paths["visible_test"] = a.test
    for name, p in paths.items():
        df = pd.read_parquet(p, columns=cols)
        res["inputs"][name] = dict(path=p, sha256=sha256(p))
        res["sets"][name] = summarize(df, top=a.top)
    # 可視の test に現れる付加イオンだけに絞った見かた（可視の test の語彙に無い付加イオンのスペクトルを除く。
    # 除いた結果スペクトルが 0 になる分子は数えない）
    vocab = sorted(x for x, _ in res["sets"]["visible_test"]["spectra_by_adduct"])
    res["visible_vocab"] = vocab
    res["sets_visible_vocab"] = {}
    for name in HOLDOUTS:
        df = pd.read_parquet(paths[name], columns=cols)
        n0 = df["molecule_id"].nunique()
        df = df[df["adduct"].astype(str).isin(vocab)]
        s = summarize(df, top=a.top)
        s["molecules_dropped_no_vocab_spectra"] = int(n0 - s["n_molecules"])
        res["sets_visible_vocab"][name] = s
    if a.train:
        df = read_enveda_train(a.train)
        res["inputs"]["train_enveda180"] = dict(path=a.train, lib="enveda-180")
        s = summarize(df.rename(columns={"inchikey14": "molecule_id"}), top=a.top)
        g = df.assign(adduct=df["adduct"].fillna("").astype(str)).groupby("inchikey14")
        pat = g["adduct"].agg(lambda x: obs_pattern(set(x))).to_dict()
        frm = g["molecular_formula"].first().to_dict()
        s["same_formula"] = same_formula_pairs(frm, pat)
        res["sets"]["train_enveda180"] = s
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    for name, s in res["sets"].items():
        print(name, s["n_molecules"], s["n_adduct_types_frac"], "ge2", s["ge2_frac"],
              "tims", s.get("all_timstof_frac"), s.get("same_formula", ""))
    for name, s in res["sets_visible_vocab"].items():
        print("vocab", name, s["n_molecules"], s["n_adduct_types_frac"], "ge2", s["ge2_frac"])


if __name__ == "__main__":
    main()
