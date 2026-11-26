"""台帳の行（既定 exp001〜exp003）の漏れの点検（playbook §4.2 (a)〜(e)、#255）。値（rr・MRR）は読まない。

    python scripts/ranker/leak_check_ledger.py [exp_id ...] > docs/experiments/<dir>/leak_check.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ranker")); sys.path.insert(0, str(ROOT / "scripts/validate"))
import lambdarank_oof as LR

out = {}
for exp in (sys.argv[1:] or ["exp001", "exp002", "exp003"]):  # 引数で exp004〜 も
    d = ROOT / "outputs/local_eval/ranker/lambdarank" / exp
    meta = json.loads((d / "meta.json").read_text())
    tab = pd.read_parquet(meta["table"], columns=list(LR.CHECK_READ_COLS) + ["is_tier"])
    r = {}
    r["a_model_fold_eq_query_fold"] = bool(((tab.model_fold == tab.query_fold) & (tab.np_model_fold == tab.query_fold)).all())
    if meta["rows"] == "pool":
        tab = tab[tab.is_tier == 0].reset_index(drop=True)
    oof = pd.read_parquet(d / "oof.parquet", columns=["set", "molecule_id", "query_fold", "type"])
    r["oof_rows_eq_table_rows"] = bool(len(oof) == len(tab) and (oof["set"].values == tab["set"].values).all()
                                       and (oof.molecule_id.values == tab.molecule_id.values).all())
    g = LR.group_ids(tab)
    # #259: 学習から外した分子（meta の一覧、sha256 を確かめる）は fold_masks にも同じものを渡し、学習に残らないことを見る
    excl = None
    if meta.get("exclude_train_molecules"):
        excl, rec = LR.read_molecule_list(Path(meta["exclude_train_molecules"]["path"]))
        assert rec["sha256"] == meta["exclude_train_molecules"]["sha256"], "学習の除外の一覧が当てはめの時と違う"
    b, c = {}, {}
    for k in LR.FOLDS:
        tr, pr = LR.fold_masks(tab, k, exclude_train=excl)
        b[k] = dict(overlap=len(set(g[tr]) & set(g[pr])), n_train_mol=len(set(g[tr])),
                    excluded_in_train=len(set(g[tr]) & excl) if excl else 0,
                    meta_n_train_mol=meta["fold_info"][str(k)]["n_train_molecules"],
                    train_folds=sorted(int(x) for x in np.unique(tab.query_fold.values[tr])))
        c[k] = int((tab.type.values[tr] == "e").sum())
    r["b_leave_one_fold_out"] = b
    r["b_ok"] = all(v["overlap"] == 0 and v["n_train_mol"] == v["meta_n_train_mol"] and k not in v["train_folds"]
                     and v["excluded_in_train"] == 0 for k, v in b.items())
    if meta.get("eval_molecules"):   # #259: 評価の分子 = 評価の一覧、学習の除外と重ならない
        pm = pd.read_csv(d / "per_molecule.csv", dtype={"molecule_id": str})
        ev, _ = LR.read_molecule_list(Path(meta["eval_molecules"]["path"]))
        got = set(pm["set"] + "\t" + pm["molecule_id"])
        r["f_eval_molecules"] = dict(n_eval=len(got), eq_list=got == ev, overlap_excluded=len(got & excl) if excl else 0)
        r["b_ok"] = r["b_ok"] and got == ev and (not excl or not (got & excl))
    r["c_type_e_rows_in_train"] = c
    r["c_ok"] = all(v == 0 for v in c.values()) and meta["train_types"] == ["a", "b", "c", "d"]
    feats = meta["features"]
    models_feats = []
    for k in LR.FOLDS:
        t = (d / f"model_fold{k}.txt").read_text()
        models_feats.append(t.split("feature_names=")[1].split("\n")[0].split())
    # #257（#256 の minor 2）: 外した列は meta の exclude_features と、求めていない登録済みの列から読む
    excluded = list(meta.get("exclude_features", [])) + list(meta.get("ignored_registered_columns", []))
    expect = [f for f in LR.FEATURES if f not in meta.get("exclude_features", [])] + list(meta.get("extra_features", []))
    bad = [f for f in feats if f in LR.FORBIDDEN or f in LR.ID_COLS or f in excluded]
    r["d_excluded_not_in_features"] = dict(bad=bad, excluded=excluded, features_eq_expected=feats == expect,
                                          models_eq_meta=all(m == feats for m in models_feats), n_features=len(feats),
                                          features_sha256=meta["features_sha256"], sha_ok=LR.features_sha256(feats) == meta["features_sha256"])
    r["d_ok"] = (not bad and feats == expect and all(m == feats for m in models_feats)
                 and r["d_excluded_not_in_features"]["sha_ok"])
    sets = sorted(meta["sets"])
    r["e_tables"] = meta["table"]
    r["e_ok"] = ("features_holdout" not in meta["table"] and "holdout" not in Path(meta["table"]).name
                 and all(s.startswith("b8_f") and s[4] in "1234" for s in sets) and len(sets) == 20)
    r["table_sha256_ok"] = LR.sha256_file(Path(meta["table"])) == meta["table_sha256"]
    r["all_ok"] = all(r[x] for x in ("a_model_fold_eq_query_fold", "oof_rows_eq_table_rows", "b_ok", "c_ok", "d_ok", "e_ok", "table_sha256_ok"))
    out[exp] = r
print(json.dumps(out, indent=1, ensure_ascii=False))
