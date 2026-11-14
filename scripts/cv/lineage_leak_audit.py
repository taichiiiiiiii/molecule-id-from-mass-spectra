#!/usr/bin/env python3
"""
0.487 系統を fold-safe に組み直して v3 で測る評価の、漏れの監査（Issue #114、設計 docs/design/lineage487-foldsafe-eval.md §4・§2.1）。

読むだけ。学習・推論はしない。v3 の 4 集合（v3_c1 / v3_c2 / enveda_v1 / v3_c2n）の正解キーと、§4 の 1〜12 の各集合との積を数え、
outputs/research/lineage_leak_audit.json に書く。分母はすべて outputs/holdouts/<集合>/truth.csv の分子数。

正解のキーは 2 種類:
  ik14   : truth.csv の inchikey14
  scorer : truth.csv の smiles を scripts/validate/mrr25.py の to_key（公式の scorer と同じ正準化）で InChIKey14 にしたもの
積は「どちらかのキーが相手の集合に入る分子の数」（n）と、キーごとの数（n_ik14 / n_scorer）を出す。相手の集合が
scorer のキーを持つときは、それも相手のキーに足す。

手元で数えられない項目は status = "not_computable" と理由（reason）を書く。

入力（リポジトリには入れない。data/ は追跡しない）:
  data/ext/foldsafe/split.parquet                      fold-safe の hold / np（55,367 構造）
  data/ext/panel/casmi26-identity-disjoint-validation-folds/structures.parquet   fold_identity（#74 の除外に使う）
  kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json   #54 の除外（split と合わせて 55,369）
  outputs/engine2_build/trial/own/build_report.json   #74 の試走のクエリ（本番の表はまだ無い）
  data/ext/msg/MassSpecGym.tsv, MassSpecGym1.5.tsv     MSG
  data/ext/v4g/casmi26-v2-pool/{train_structs,pool_meta}.parquet, casmi-chebi-lmsd-structures/bio_structures.parquet
  data/ext/panel/pubchemlite_annotations.parquet       PubChemLite の候補キー（Engine 2 の候補バンクの一部）
  --dl-dir（既定 data/ext/lineage_audit）に、次を Kaggle から読むだけで取ってきて置く:
    ahmedberatozer/casmi26-spec-cache      cache/split.parquet（作者の split。sha256 d05a2f9f…）
    takumuhata/casmi26-extlib              ext_meta.parquet, ext_fp_keys.parquet
    samartalwar/casmi-2026-spectral-library-massbankharmonized   structures_with_fingerprints.parquet
    dmitriigluzdov/casmi26-natural-product-knowledge-table        external_reference_spectra/external_{delta,scorer_sid_links}.parquet
  data/train.parquet は inchikey14・ingest_lib の 2 列だけを流し読みする（§4-9 のパッチ後の表を作るため）

    .venv/bin/python scripts/cv/lineage_leak_audit.py [--skip-tanimoto] [--out outputs/research/lineage_leak_audit.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
ENVEDA = {"enveda-180", "enveda-np-examples"}
C_MAX = 0.05            # §2.1 の閾値（事前登録。結果を見て変えない）
AUTHOR_QUERY_FOLDS = ("np", "nplib", "syn", "plusk")   # TC simulate_v1ctx.py の --folds の既定（ranker_v4b の模擬クエリ）
AUTHOR_SPLIT_SHA = "d05a2f9f71cffd480b92581c458547f7aeb782436b07af4fbf8f696f96649656"
TRAIN_STRUCTS_PIN = "f20aa1cac23af42e430a6633707fda34e48dea97114bdc3e443897f692df3656"   # F c36 192 行の pin
# ランカーの特徴が train 由来の表（train_structs / train_fp_sel）に依存するかの分類。名前からの推定（casmi.engine の
# ソースは手元に無いので未確認）。fe_v4 の族は ctxlib.py 187 行（analog_struct が train_fp を sid で引く）と族の名前による。
TRAIN_TABLE_FEATURES = {
    "analog_engine": ("ap", "a1", "best_tan", "top_tan", "mean_tan", "ap_rank", "ap_gap", "ap_grp_max", "top_sim", "ap_np",
                      "best_tan_np", "top_sim_np", "ap_np_gap", "ap_sum", "n_analog_hi", "tan_zero_shift", "ap_x_fz", "agree_ap"),
    "generate": ("is_gen", "gen_parent_sim", "gen_steps", "gen_parent_tan", "n_gen", "gen_rank_fz"),
}
TRAIN_TABLE_FAMILIES = ("analog_struct", "derivation")       # 表のキーが train の構造（推定）
TRAINED_FAMILIES = ("fragnet", "model_views")               # 学習済みの重みに依存（fragnet は作者の train fold で学習）
GREP_PATTERNS = ("train_fp", "train_structs", "in_train_structure", "an_sid", "generate=True", "train_sid")


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）

def overlap(truth: dict[str, dict[str, tuple[str, str | None]]], target: set, target_scorer: set | None = None) -> dict:
    """集合ごとに、正解のキー（ik14, scorer）のどちらかが target（∪ target_scorer）に入る分子の数。

    truth: {集合: {molecule_id: (ik14, scorer_key or None)}}
    """
    tall = set(target) | set(target_scorer or ())
    out = {}
    for s, mols in truth.items():
        n = len(mols)
        hit = sorted(m for m, (ik, sk) in mols.items() if ik in tall or (sk is not None and sk in tall))
        n_ik = sum(1 for ik, _ in mols.values() if ik in target)
        n_sk = sum(1 for _, sk in mols.values() if sk is not None and sk in tall)
        out[s] = dict(n=len(hit), denom=n, frac=round(len(hit) / n, 4) if n else 0.0, n_ik14=n_ik, n_scorer=n_sk)
    return out


def overlap_after(truth: dict, target: set, excluded: dict[str, set] | None) -> dict | None:
    """overlap の分子数を、excluded（集合ごとの molecule_id）を除いたあとで数える。"""
    if excluded is None:
        return None
    h = hit_mids(truth, target)
    return {s: len(h[s] - excluded[s]) for s in truth}


def hit_mids(truth: dict, target: set, target_scorer: set | None = None) -> dict[str, set]:
    tall = set(target) | set(target_scorer or ())
    return {s: {m for m, (ik, sk) in mols.items() if ik in tall or (sk is not None and sk in tall)} for s, mols in truth.items()}


def c_rule(truth: dict, components: dict[str, dict[str, set] | None], c_max: float = C_MAX) -> dict:
    """§2.1: 集合ごとに、正解が components のどれかに入る分子の割合 c。

    components: {名前: {集合: 当たった molecule_id の集合}}。None の成分は「一覧を作れない」。
    判定: どの集合でも c ≤ c_max かつ一覧を作れない成分が無い → usable。それ以外 → not_usable。
    """
    missing = sorted(k for k, v in components.items() if v is None)
    per = {}
    for s, mols in truth.items():
        u = set().union(*[v.get(s, set()) for v in components.values() if v is not None]) if components else set()
        n = len(mols)
        per[s] = dict(n=len(u), denom=n, c=round(len(u) / n, 4) if n else 0.0,
                      by_component={k: len(v.get(s, set())) for k, v in components.items() if v is not None})
    worst = max((r["c"] for r in per.values()), default=0.0)
    usable = not missing and worst <= c_max
    reasons = []
    if missing:
        reasons.append(f"一覧を作れない成分: {missing}")
    if worst > c_max:
        reasons.append(f"c の最大 {worst} > {c_max}")
    return dict(c_max=c_max, per_set=per, missing_components=missing, worst_c=worst,
                decision="usable" if usable else "not_usable", reasons=reasons)


def kept_after_hide(rows_libs_keys, drop: dict[str, set], rules: dict[str, str]) -> dict[str, set]:
    """train の行（(ingest_lib, ik14) の列）に集合ごとの hide_rule を当てたあと、drop の構造のうち 1 行でも残る構造。

    rule = all: drop の構造の行を全部落とす。enveda_libs: drop の構造の enveda 系の行だけ落とす。
    """
    for r in rules.values():
        if r not in ("all", "enveda_libs"):
            raise ValueError(r)
    alld = set().union(*drop.values()) if drop else set()
    kept = {s: set() for s in drop}
    for lib, k in rows_libs_keys:
        if k not in alld:
            continue
        for s, ks in drop.items():
            if k in ks and rules[s] == "enveda_libs" and lib not in ENVEDA:
                kept[s].add(k)
    return kept


def not_computable(reason: str, **extra) -> dict:
    return dict(status="not_computable", reason=reason, **extra)


def feature_dependency(features: list[str]) -> dict:
    """ランカーの特徴を、train 由来の表・学習済みの重みへの依存で数える（名前からの推定）。"""
    fam = lambda f: f.split("__", 1)[0] if "__" in f else None
    groups = {k: [f for f in features if f in set(v)] for k, v in TRAIN_TABLE_FEATURES.items()}
    for F in TRAIN_TABLE_FAMILIES + TRAINED_FAMILIES:
        groups[F] = [f for f in features if fam(f) == F]
    groups["library_spectra"] = [f for f in features if fam(f) is None and (f.startswith("lib_") or f in ("lib_x_fz", "agree_lib"))]
    table = sorted(set(sum((groups[k] for k in list(TRAIN_TABLE_FEATURES) + list(TRAIN_TABLE_FAMILIES)), [])))
    return dict(n_features=len(features), counts={k: len(v) for k, v in groups.items()},
                n_train_table_dependent=len(table), n_trained_family=sum(len(groups[F]) for F in TRAINED_FAMILIES),
                note="名前からの推定。casmi.engine のソースは手元に無い。library_spectra は train.parquet 由来で hide_rule が効く")


def grep_usages(paths: list[Path], patterns=GREP_PATTERNS, base: Path | None = None) -> list[dict]:
    rx = re.compile("|".join(re.escape(p) for p in patterns))
    out = []
    for p in paths:
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            m = rx.search(line)
            if m:
                rel = str(p.relative_to(base)) if base and p.is_relative_to(base) else str(p)
                out.append(dict(file=rel, line=i, pattern=m.group(0), text=line.strip()[:160]))
    return out


# ---------------------------------------------------------------- 読み込み

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_truth(holdouts: Path) -> dict:
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    import pandas as pd
    from mrr25 import to_key
    out = {}
    for s in SETS:
        t = pd.read_csv(holdouts / s / "truth.csv", dtype=str)
        out[s] = {m: (ik, to_key(smi) if isinstance(smi, str) else None) for m, ik, smi in zip(t.molecule_id, t.inchikey14, t.smiles)}
    return out


def hide_rule(data_holdouts: Path, s: str) -> dict:
    p = data_holdouts / s / "hide_rule.json"
    return json.loads(p.read_text()) if p.is_file() else dict(rule="all", pool="keep", default=True)


def keyset(df, *cols) -> set:
    out = set()
    for c in cols:
        if c in df:
            out |= {str(x)[:14] for x in df[c].dropna() if len(str(x)) >= 14}
    return out


# ---------------------------------------------------------------- 本体

def main(argv=None) -> None:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT), help="data/ と outputs/ を読む場所")
    ap.add_argument("--dl-dir", default=None, help="Kaggle から取ってきた小さいファイルの置き場（既定 <root>/data/ext/lineage_audit）")
    ap.add_argument("--frozen-dir", default=str(Path.home() / ".hermes/profiles/massspec/workspace/casmi_daily/ref/frozen_487"))
    ap.add_argument("--tc-dir", default=str(Path.home() / ".hermes/profiles/massspec/cache/scratch/tc1114"))
    ap.add_argument("--skip-tanimoto", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    R = Path(a.root)
    DL = Path(a.dl_dir) if a.dl_dir else R / "data/ext/lineage_audit"
    out_p = Path(a.out) if a.out else R / "outputs/research/lineage_leak_audit.json"
    T0 = time.time()
    truth = load_truth(R / "outputs/holdouts")
    dk = {s: set((R / "data/holdouts" / s / "drop_keys.txt").read_text().split()) for s in SETS}
    rules = {s: hide_rule(R / "data/holdouts", s) for s in SETS}
    res: dict = dict(issue=114, design="docs/design/lineage487-foldsafe-eval.md §4・§2.1",
                     denominators={s: len(truth[s]) for s in SETS},
                     hide_rule={s: rules[s] for s in SETS},
                     key_check={s: dict(ik14_ne_scorer=sum(1 for ik, sk in truth[s].values() if sk != ik),
                                        scorer_none=sum(1 for _, sk in truth[s].values() if sk is None),
                                        truth_in_drop_keys=sum(1 for ik, _ in truth[s].values() if ik in dk[s]),
                                        n_drop_keys=len(dk[s])) for s in SETS},
                     items={})
    I = res["items"]
    # 4 集合が構造で重ならないこと（§3.1 の前提）
    pair = {f"{x}&{y}": len({k for k, _ in truth[x].values()} & {k for k, _ in truth[y].values()})
            for i, x in enumerate(SETS) for y in SETS[i + 1:]}
    res["set_pairwise_overlap_ik14"] = pair

    # 1. fold-safe split
    sp = pd.read_parquet(R / "data/ext/foldsafe/split.parquet", columns=["inchikey14", "key", "fold"])
    hold, npk = keyset(sp[sp.fold == "hold"], "inchikey14", "key"), keyset(sp[sp.fold == "np"], "inchikey14", "key")
    i1 = dict(hold=overlap(truth, hold), np=overlap(truth, npk), n_hold=int((sp.fold == "hold").sum()), n_np=int((sp.fold == "np").sum()))
    i1["assert_subset_of_hold"] = all(i1["hold"][s]["n"] == len(truth[s]) for s in SETS)
    I["1_foldsafe_split"] = i1

    # 2. #54 の除外集合（split の hold/np ∪ 4 集合の drop_keys。期待 55,369）
    kf = R / "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json"
    if not kf.is_file():
        kf = ROOT / "kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json"
    hk = json.loads(kf.read_text())
    drop54 = set(sp.inchikey14.astype(str)) | set().union(*[set(v) for v in hk.values()])
    I["2_sanpier54_exclusion"] = dict(n_exclusion=len(drop54), expected=55369, matches_expected=len(drop54) == 55369,
                                      overlap=overlap(truth, drop54),
                                      assert_all_excluded=all(overlap(truth, drop54)[s]["n"] == len(truth[s]) for s in SETS),
                                      note="除外はキットの側で inchikey14 と kit key の両方に当てる（#54 の CELL_DROP）。ここでは inchikey14 だけ")

    # 3. #74 の自前表のクエリ
    br = R / "outputs/engine2_build/trial/own/build_report.json"
    if br.is_file():
        b = json.loads(br.read_text())
        qk = {q["ik"] for r in b.get("runs", []) for q in r.get("queries", [])}
        I["3_engine2_own_queries"] = dict(status="partial", source="試走（outputs/engine2_build/trial、50 分子）のクエリだけ",
                                         n_queries=len(qk), overlap=overlap(truth, qk),
                                         exclusions=b.get("exclusions"), query_exclusion_overlap=b.get("query_exclusion_overlap"),
                                         production=not_computable("#74 の本番の表（run1）はまだ手元に無い。届いたら build_report のクエリで数え直す"))
    else:
        I["3_engine2_own_queries"] = not_computable("outputs/engine2_build/trial/own/build_report.json が無い")

    # 作者の split（4・5 に使う）
    asp_p = DL / "split.parquet"
    ts = pd.read_parquet(R / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet", columns=["inchikey14", "key", "smiles", "sid"])
    if asp_p.is_file():
        asp = pd.read_parquet(asp_p)
        asp_sha = sha256(asp_p)
        afold = {f: keyset(asp[asp.fold == f], "inchikey14", "key") for f in sorted(asp.fold.unique())}
        q4 = set().union(*[afold.get(f, set()) for f in AUTHOR_QUERY_FOLDS])
        n_q_struct = int(asp.fold.isin(AUTHOR_QUERY_FOLDS).sum())
        I["4_ranker_v4b_sim_queries"] = dict(
            status="reconstructed", source="ahmedberatozer/casmi26-spec-cache cache/split.parquet", sha256=asp_sha,
            sha256_matches=asp_sha == AUTHOR_SPLIT_SHA, folds=list(AUTHOR_QUERY_FOLDS), n_structures=n_q_struct,
            expected_structures=5350, overlap=overlap(truth, q4),
            note="TC simulate_v1ctx.py は --folds np,nplib,syn,plusk の全構造を回す（146 行）。#88 の作者のシャードと一致を確かめた選び方")
        fn_train = afold.get("train", set())
        I["5_fe_v4_tables"] = dict(
            fragnet_train_structures=dict(status="approximated", definition="作者の split の fold == 'train'（twin も別に出す）",
                                          n=int((asp.fold == "train").sum()), overlap=overlap(truth, fn_train),
                                          overlap_train_or_twin=overlap(truth, fn_train | afold.get("twin", set())),
                                          by_author_fold={f: overlap(truth, ks) for f, ks in afold.items()},
                                          note="fragnet の学習構造の一覧そのものは無い。作者の train fold で学習（部品表 5 行）を前提に近似"),
            analog_struct=not_computable("fe_v4 の analog_struct の表・コードは手元に無い（ctxlib.py 187 行で train_fp を sid で引くので、キーは 9・10 と同じ train_structs と推定）"),
            frag2=not_computable("fe_v4 の frag2/* の表・コードは手元に無い"),
            derivation=not_computable("prior_tables.pkl・derivation のコードは手元に無い（self_test だけ F c37 にある）"),
            stageA_stageB=not_computable("fragnet の stageA.npz / stageB.txt は手元に無い"))
    else:
        asp, fn_train, q4 = None, None, None
        I["4_ranker_v4b_sim_queries"] = not_computable(f"作者の split が無い（{asp_p}）")
        I["5_fe_v4_tables"] = not_computable(f"作者の split が無い（{asp_p}）。fe_v4 の表も手元に無い")

    # 6. MSG
    sys.path.insert(0, str(ROOT / "scripts/research"))
    from msg_overlap import read_msg
    msg = {}
    for name in ("MassSpecGym.tsv", "MassSpecGym1.5.tsv"):
        p = R / "data/ext/msg" / name
        if p.is_file():
            folds, how, rows = read_msg(str(p))
            msg[name] = dict(rows=rows, key_from=how, overlap={f: overlap(truth, ks) for f, ks in sorted(folds.items())},
                             _folds=folds)
    if msg:
        uni = {}
        for v in msg.values():
            for f, ks in v.pop("_folds").items():
                uni.setdefault(f, set()).update(ks)
        allk = set().union(*uni.values())
        msg_any = hit_mids(truth, allk)
        I["6_msg"] = dict(versions=msg, union_1_0_1_5={f: overlap(truth, ks) for f, ks in sorted(uni.items())},
                          union_any_fold=overlap(truth, allk),
                          scored_after_msg_exclusion={s: len(truth[s]) - len(msg_any[s]) for s in SETS})
    else:
        msg_any = None
        I["6_msg"] = not_computable("data/ext/msg の tsv が無い")

    # 7. 外部ライブラリ・候補バンク（hide_rule を当てる対象を決めるため）
    i7: dict = {}

    def ov(keys: set) -> dict:
        return dict(overlap=overlap(truth, keys), after_msg_exclusion=overlap_after(truth, keys, msg_any))
    pm = pd.read_parquet(R / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key", "src", "train_sid"])
    i7["v2_pool"] = dict(n=len(pm), **ov(keyset(pm, "key")),
                         overlap_train_rows=overlap(truth, keyset(pm[pm.train_sid >= 0], "key")))
    bio = R / "data/ext/v4g/casmi-chebi-lmsd-structures/bio_structures.parquet"
    i7["engine2_bank_chebi_lmsd"] = dict(**ov(keyset(pd.read_parquet(bio, columns=["inchikey14"]), "inchikey14"))) \
        if bio.is_file() else not_computable("bio_structures.parquet が無い")
    pcl = R / "data/ext/panel/pubchemlite_annotations.parquet"
    i7["engine2_bank_pubchemlite"] = dict(**ov(keyset(pd.read_parquet(pcl, columns=["ik14", "candidate_key"]), "ik14", "candidate_key")),
                                          source="data/ext/panel/pubchemlite_annotations.parquet（PubChemLite の注記表。Engine 2 のバンクと同じ版かは未確認）") \
        if pcl.is_file() else not_computable("pubchemlite_annotations.parquet が無い")
    i7["engine2_bank_coconut"] = not_computable("COCONUT の候補表（casmi26-attributed-v4g-reproduction-assets/coconut-casmi26-candidates）は手元に無い")
    i7["hengck23_dreams_structures"] = not_computable("hengck23 の DreaMS 構造表は手元に無い")
    em, ek = DL / "ext_meta.parquet", DL / "ext_fp_keys.parquet"
    if em.is_file():
        e = pd.read_parquet(em, columns=["inchikey14"])
        i7["extlib_spectrum_rows"] = dict(n_rows=len(e), n_struct=e.inchikey14.nunique(), **ov(keyset(e, "inchikey14")),
                                          note="ext_meta の行。#52 では構造表（ext_mass / ext_fp_keys）に train の構造は 0 だが、スペクトル行には train の構造 6,414 が残る")
    else:
        i7["extlib_spectrum_rows"] = not_computable(f"{em} が無い")
    i7["extlib_structure_table"] = dict(**ov(keyset(pd.read_parquet(ek), "inchikey14"))) if ek.is_file() \
        else not_computable(f"{ek} が無い")
    mb = DL / "structures_with_fingerprints.parquet"
    i7["massbank_harmonized_structures"] = dict(**ov(keyset(pd.read_parquet(mb, columns=["inchikey"]), "inchikey")),
                                                note="structures_with_fingerprints.parquet（is_casmi_subset で落とす前の全構造。上限）") \
        if mb.is_file() else not_computable(f"{mb} が無い")
    kd, kl = DL / "external_delta.parquet", DL / "external_scorer_sid_links.parquet"
    if kd.is_file() and kl.is_file():
        d = pd.read_parquet(kd, columns=["scorer_key", "in_train_structure", "original_inchikey"])
        links = pd.read_parquet(kl)
        kq = DL / "external_record_quality.parquet"
        d["mass_ok"] = (pd.read_parquet(kq, columns=["mass_check_status"]).mass_check_status.values == "within_diagnostic_threshold") \
            if kq.is_file() else True
        i7["knowledge_table_external_reference"] = dict(
            sha256=dict(delta=sha256(kd), links=sha256(kl)), n_rows=len(d), n_keys=int(d.scorer_key.nunique()),
            **ov(keyset(d, "scorer_key", "original_inchikey")))
    else:
        d = links = None
        i7["knowledge_table_external_reference"] = not_computable("知識表の external_reference_spectra が無い")
    I["7_external_libraries_and_banks"] = i7

    # 8. in_train_structure の印
    if d is not None:
        lk = dict(zip(links.scorer_key, links.sid))
        flagged = d[d.in_train_structure.astype(bool)]
        by = {}
        for s in SETS:
            ks = {sk for _, sk in truth[s].values() if sk} | {ik for ik, _ in truth[s].values()}
            rows = flagged[flagged.scorer_key.isin(ks)]
            linked = rows.loc[rows.scorer_key.map(lambda k: bool(pd.notna(lk.get(k)))).astype(bool).to_numpy()]
            used = linked.loc[linked.mass_ok.astype(bool).to_numpy()]
            mids = {m for m, (ik, sk) in truth[s].items() if ik in set(used.scorer_key) or sk in set(used.scorer_key)}
            by[s] = dict(rows_flagged=len(rows), keys_flagged=int(rows.scorer_key.nunique()), rows_flagged_with_sid=len(linked),
                         rows_used_by_c36=len(used), molecules_used_by_c36=len(mids),
                         molecules_used_after_msg_exclusion=len(mids - msg_any[s]) if msg_any is not None else None,
                         contradicts_hide_rule=bool(rules[s]["rule"] == "all" and len(used) > 0))
        I["8_in_train_structure_flag"] = dict(per_set=by, note="印は表を作ったときの全 train で立てた固定値。hide_rule で train から消した構造でも立ったまま。"
                                              "F c36 85 行はこの印と sid のある行だけを参照に足すので、rule=all の集合で 1 以上なら正解の参照が戻る（パッチの対象）。"
                                              "rows_used_by_c36 は F c36 85 行の条件（印・sid・mass_check_status）を当てた行（付加イオン・モードの条件は数に入れていない）")
    else:
        I["8_in_train_structure_flag"] = not_computable("知識表の external_reference_spectra が無い")

    # 9. train_structs.parquet（パッチ前 = 全 train のまま、パッチ後 = hide_rule を当てた train から作り直す）
    ts_sha = sha256(R / "data/ext/v4g/casmi26-v2-pool/train_structs.parquet")
    ts_keys = keyset(ts, "inchikey14", "key")
    pf = pq.ParquetFile(R / "data/train.parquet")
    alld = set().union(*dk.values())
    n_rows = [0]

    def _rows():
        for bt in pf.iter_batches(batch_size=262144, columns=["ingest_lib", "inchikey14"]):
            iks = bt.column("inchikey14").to_pylist()
            n_rows[0] += len(iks)
            yield from zip(bt.column("ingest_lib").to_pylist(), iks)
    kept = kept_after_hide(_rows(), dk, {s: rules[s]["rule"] for s in SETS})
    n_rows = n_rows[0]
    # rule=all の集合は drop の構造の行が全部消える。他の集合の drop は他の集合の正解と重ならない（pairwise を参照）
    post = {s: overlap({s: truth[s]}, kept[s])[s] for s in SETS}
    I["9_train_structs"] = dict(n_rows=len(ts), sha256=ts_sha, sha256_matches_pin=ts_sha == TRAIN_STRUCTS_PIN,
                                pre_patch=overlap(truth, ts_keys), post_patch=post, train_rows_streamed=n_rows,
                                note="パッチ後 = その集合の hide_rule を当てた train.parquet に 1 行でも残る正解。rule=enveda_libs（v3_c1）では"
                                     "公開ライブラリの参照が残るのが本来の設計（隠れテストの『train に参照あり』と同じ）")
    # 10. train_fp_sel.npy
    I["10_train_fp_sel"] = dict(status="not_computable_locally", reason="train_fp_sel.npy は手元に無い（casmi26-v2-pool の大きいファイル）",
                                row_mapping="ctxlib.py 187 行 `train_fp[self.an_sid]` と F c36 の pin structure_rows = 275,810 から、行 i = train_structs の sid i と推定",
                                inferred_rows_for_truth=overlap(truth, ts_keys),
                                sid_is_rowindex=bool((ts.sid.to_numpy() == np.arange(len(ts))).all()))
    # 11. 使う所の一覧
    fz, tc = Path(a.frozen_dir), Path(a.tc_dir)
    files = sorted(fz.glob("c*.py")) + sorted((fz / "eng").glob("*.py")) + sorted(tc.rglob("*.py")) \
        + sorted((R / "data/ext/v4g").rglob("casmi/*.py"))
    files = [p for p in files if "__pycache__" not in str(p)]
    rk = tc / "v4b/ranker_v4b.pkl"
    if rk.is_file():
        import pickle
        with open(rk, "rb") as f:
            feats = list(pickle.load(f)["features"])
        dep = feature_dependency(feats)
    else:
        dep = not_computable(f"{rk} が無い")
    I["11_train_table_users"] = dict(grep=grep_usages(files, base=Path.home()), n_files_scanned=len(files),
                                     ranker_feature_dependency=dep,
                                     not_scanned="casmi.engine（EngineCfg(generate=True) の候補生成）、fe_v4 の analog_struct / frag2 / derivation / fragnet の族のコードは手元に無い")

    # 12. 最近傍 Tanimoto（記録だけ）
    if a.skip_tanimoto:
        I["12_nearest_tanimoto"] = not_computable("--skip-tanimoto")
    else:
        from rdkit import Chem, DataStructs, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        excl = set(sp.inchikey14.astype(str)) | alld
        ref = ts[~ts.inchikey14.isin(excl)]
        fps = [gen.GetFingerprint(m) for m in (Chem.MolFromSmiles(x) for x in ref.smiles) if m is not None]
        tt = {s: pd.read_csv(R / "outputs/holdouts" / s / "truth.csv", dtype=str) for s in SETS}
        per, rows = {}, []
        for s in SETS:
            v = []
            for mid, smi in zip(tt[s].molecule_id, tt[s].smiles):
                m = Chem.MolFromSmiles(smi)
                x = max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps)) if m is not None else float("nan")
                v.append(x); rows.append((s, mid, round(float(x), 4)))
            v = np.array(v)
            per[s] = dict(median=round(float(np.nanmedian(v)), 4), q10=round(float(np.nanpercentile(v, 10)), 4),
                          q90=round(float(np.nanpercentile(v, 90)), 4),
                          bands={"ge0.9": int((v >= 0.9).sum()), "0.7-0.9": int(((v >= 0.7) & (v < 0.9)).sum()),
                                 "0.5-0.7": int(((v >= 0.5) & (v < 0.7)).sum()), "lt0.5": int((v < 0.5).sum())})
        csvp = out_p.with_name("lineage_leak_audit_tanimoto.csv")
        csvp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["set", "molecule_id", "nearest_tanimoto"]).to_csv(csvp, index=False)
        I["12_nearest_tanimoto"] = dict(reference="train_structs から fold-safe split（hold ∪ np）と 4 集合の drop_keys を除いた構造",
                                        n_reference=len(fps), fingerprint="Morgan r=2, 2048 bit", per_set=per, per_molecule_csv=csvp.name)

    # §2.1 の判定
    comp_pre = {"fragnet_train_structures": hit_mids(truth, fn_train) if fn_train is not None else None,
                "ranker_v4b_sim_queries": hit_mids(truth, q4) if q4 is not None else None,
                "analog_struct_frag2_derivation_keys": None,
                "train_structs_and_fp_sel(9-10)": hit_mids(truth, ts_keys)}
    comp_post = dict(comp_pre)
    comp_post["train_structs_and_fp_sel(9-10)"] = {s: hit_mids({s: truth[s]}, kept[s])[s] for s in SETS}
    ext_hits = None
    if d is not None:
        fl = keyset(d[d.in_train_structure.astype(bool)], "scorer_key")
        ext_hits = hit_mids(truth, fl)
        comp_pre["in_train_structure_external_refs(8)"] = ext_hits
        comp_post["in_train_structure_external_refs(8)"] = {s: set() for s in SETS}   # パッチで v3 のキーの行を落とす（§2-15）
    res["tier0_rule_2_1"] = dict(
        pre_patch_record_only=c_rule(truth, comp_pre),
        post_patch=c_rule(truth, comp_post),
        note="判定は post_patch（§4 の 9〜10 と 8 の表を hide_rule に合わせて作り直した後）。pre_patch は記録だけ。"
             "analog_struct / frag2 / derivation のキーは手元で一覧を作れないので missing。v3_c1 の 9〜10 の残りは hide_rule どおり。"
             "sensitivity_* は fragnet の近似（作者の train fold）を使わず、模擬クエリの構造だけで数えた c（判定には使わない）")
    only_q = {"ranker_v4b_sim_queries": comp_post["ranker_v4b_sim_queries"]}
    res["tier0_rule_2_1"]["sensitivity_ranker_queries_only"] = c_rule(truth, only_q) if only_q["ranker_v4b_sim_queries"] is not None else None
    if msg_any is not None:
        if only_q["ranker_v4b_sim_queries"] is not None:
            res["tier0_rule_2_1"]["sensitivity_ranker_queries_only_after_msg_exclusion"] = c_rule(
                {s: {m: v for m, v in truth[s].items() if m not in msg_any[s]} for s in SETS},
                {"ranker_v4b_sim_queries": {s: only_q["ranker_v4b_sim_queries"][s] - msg_any[s] for s in SETS}})
        res["tier0_rule_2_1"]["post_patch_after_msg_exclusion"] = c_rule(
            {s: {m: v for m, v in truth[s].items() if m not in msg_any[s]} for s in SETS},
            {k: (None if v is None else {s: v[s] - msg_any[s] for s in SETS}) for k, v in comp_post.items()})
    res["seconds"] = round(time.time() - T0, 1)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    print(json.dumps(dict(denominators=res["denominators"], post=res["tier0_rule_2_1"]["post_patch"]["per_set"],
                          decision=res["tier0_rule_2_1"]["post_patch"]["decision"], seconds=res["seconds"]), ensure_ascii=False))
    print("wrote", out_p)


if __name__ == "__main__":
    main()
