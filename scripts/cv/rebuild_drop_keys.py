#!/usr/bin/env python3
"""
v3 の 4 集合の drop_keys.txt を、正準キーと別名まで含めて作り直す（Issue #116）。

今の drop_keys.txt は truth.csv の inchikey14（tautomer の正準化なし）だけなので、公式の scorer のキー
（scripts/validate/mrr25.py の to_key）や、train_structs で同じ構造の別の inchikey14（別名）が漏れる。
新しい drop は集合ごとに

    drop = 今の drop_keys ∪ truth の inchikey14 ∪ to_key(truth の smiles)
           ∪ train_structs.key（上の各キーの ik14 の行の key）
           ∪ train_structs で key が上のどれかと同じ inchikey14（別名）

hide_rule.json と pool: remove の当て方は変えない（library.load_refs と run.py の hidden は drop_keys.txt を読むだけなので、
新しいキーで当たる）。

    rebuild : data/holdouts/<集合>/drop_keys.txt を書き換え、drop_keys_manifest.json（前後の件数・sha256・足したキー）を書く。
              2 回目以降は、今のファイルの sha256 が manifest の after と同じなら何も書かない（冪等）
    check   : 受け入れの数え上げ（#116 の受け入れ条件）を json に書く。読むだけ

    .venv/bin/python scripts/cv/rebuild_drop_keys.py rebuild
    .venv/bin/python scripts/cv/rebuild_drop_keys.py check --out outputs/cv116/drop_keys_acceptance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETS = ("v3_c1", "v3_c2", "enveda_v1", "v3_c2n")
ENVEDA = {"enveda-180", "enveda-np-examples"}
TS_REL = "data/ext/v4g/casmi26-v2-pool/train_structs.parquet"
DEFINITION = ("drop = 今の drop_keys ∪ truth.inchikey14 ∪ mrr25.to_key(truth.smiles) ∪ train_structs.key(上の ik14) "
              "∪ train_structs で key が上のどれかと同じ inchikey14（別名）")


# ---------------------------------------------------------------- 純粋な関数（単体テストの対象）

def rebuild_keys(truth: list[tuple[str, str, str | None]], old: set[str], ts_pairs, to_key) -> tuple[set[str], dict[str, list[str]]]:
    """truth = [(molecule_id, inchikey14, smiles)]、ts_pairs = [(inchikey14, key)]（train_structs）。

    返り値: (新しい drop, {足したキー: 理由の一覧})。新しい drop は old を必ず含む（減らさない）。
    理由: "scorer:<mid>"（to_key(smiles)）、"truth_ik14:<mid>"、"ts_key_of:<ik14>"、"alias_of:<key>"。
    """
    reasons: dict[str, list[str]] = {}

    def add(k, why):
        if k and k not in old:
            reasons.setdefault(k, []).append(why)

    base = set(old)
    for mid, ik, smi in truth:
        if ik:
            base.add(ik); add(ik, f"truth_ik14:{mid}")
        sk = to_key(smi) if isinstance(smi, str) and smi else None
        if sk:
            base.add(sk); add(sk, f"scorer:{mid}")
    pairs = [(str(i), str(k)) for i, k in ts_pairs]
    key_of = dict(pairs)
    canon = set(base)
    for k in sorted(base):
        c = key_of.get(k)
        if c and c != k:
            canon.add(c); add(c, f"ts_key_of:{k}")
    new = set(canon)
    for ik, k in pairs:
        if k in canon and ik not in canon:
            new.add(ik); add(ik, f"alias_of:{k}")
    return new, {k: sorted(set(v)) for k, v in sorted(reasons.items())}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def format_keys(keys: set[str]) -> str:
    return "\n".join(sorted(keys)) + "\n"


def rebuild_set(hdir: Path, truth: list, ts_pairs, to_key, meta: dict | None = None) -> dict:
    """hdir/drop_keys.txt を作り直す。manifest の after と今のファイルが同じなら書かない（冪等）。"""
    p, mp = hdir / "drop_keys.txt", hdir / "drop_keys_manifest.json"
    raw = p.read_bytes() if p.is_file() else b""
    cur = set(raw.decode().split())
    if mp.is_file():
        m = json.loads(mp.read_text())
        if m.get("after", {}).get("sha256") == sha256_bytes(raw):
            return dict(m, status="unchanged")
    new, added = rebuild_keys(truth, cur, ts_pairs, to_key)
    out = format_keys(new).encode()
    m = dict(issue=116, definition=DEFINITION,
             before=dict(n=len(cur), sha256=sha256_bytes(raw)),
             after=dict(n=len(new), sha256=sha256_bytes(out)),
             n_added=len(added), added=added,
             n_added_by_reason={r: sum(1 for v in added.values() if any(x.startswith(r + ":") for x in v))
                                for r in ("truth_ik14", "scorer", "ts_key_of", "alias_of")},
             **(meta or {}))
    p.write_bytes(out)
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    return dict(m, status="written")


def train_rows_left(rows_libs_keys, drop: set[str], rule: str, struct_keys: set[str], canon: dict[str, str]) -> dict:
    """hide_rule を drop で当てた（今と同じ: ik14 ∈ drop の行を落とす。enveda_libs は enveda 系の行だけ）あとに残る、
    struct_keys の構造（ik14 か train_structs.key が struct_keys に入る）の行を数える。"""
    if rule not in ("all", "enveda_libs"):
        raise ValueError(rule)
    left, by_lib, structs = 0, {}, set()
    alias_rows = 0
    for lib, k in rows_libs_keys:
        c = canon.get(k, k)
        if k not in struct_keys and c not in struct_keys:
            continue
        hidden = k in drop
        if hidden and rule == "enveda_libs":
            hidden = lib in ENVEDA
        if hidden:
            continue
        left += 1; structs.add(c); by_lib[lib] = by_lib.get(lib, 0) + 1
        if k not in struct_keys:      # ik14 が正解のキーでなく、scorer のキーだけで当たる行（別名）
            alias_rows += 1
    return dict(rows=left, structures=len(structs), rows_by_lib=dict(sorted(by_lib.items())), alias_rows=alias_rows,
                enveda_rows=sum(v for l, v in by_lib.items() if l in ENVEDA))


# ---------------------------------------------------------------- 読み込み

def load_truth(root: Path, s: str) -> list:
    import pandas as pd
    t = pd.read_csv(root / "outputs/holdouts" / s / "truth.csv", dtype=str)
    return list(zip(t.molecule_id, t.inchikey14, t.smiles))


def _to_key():
    sys.path.insert(0, str(ROOT / "scripts/validate"))
    from mrr25 import to_key
    return to_key


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cmd_rebuild(a) -> None:
    import pandas as pd
    import rdkit
    R = Path(a.root)
    to_key = _to_key()
    ts = pd.read_parquet(R / TS_REL, columns=["inchikey14", "key"])
    pairs = list(zip(ts.inchikey14.astype(str), ts.key.astype(str)))
    meta = dict(rdkit=rdkit.__version__, train_structs_sha256=sha256_file(R / TS_REL),
                script_sha256=sha256_file(Path(__file__)), created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out = {}
    for s in SETS:
        r = rebuild_set(R / "data/holdouts" / s, load_truth(R, s), pairs, to_key, meta)
        out[s] = dict(status=r["status"], before=r["before"], after=r["after"], n_added=r["n_added"],
                      n_added_by_reason=r.get("n_added_by_reason"))
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_check(a) -> None:
    """#116 の受け入れの数え上げ。定義は lineage_leak_audit.py の 9（hide_rule の後の train）・9b（縮小版）と同じ。"""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    R = Path(a.root)
    to_key = _to_key()
    ts = pd.read_parquet(R / TS_REL, columns=["inchikey14", "key"])
    canon = {ik: k for ik, k in zip(ts.inchikey14.astype(str), ts.key.astype(str)) if ik != k}
    pool_keys = set(pd.read_parquet(R / "data/ext/v4g/casmi26-v2-pool/pool_meta.parquet", columns=["key"]).key.astype(str))
    truth = {s: load_truth(R, s) for s in SETS}
    drop = {s: set((R / "data/holdouts" / s / "drop_keys.txt").read_text().split()) for s in SETS}
    rules = {}
    for s in SETS:
        p = R / "data/holdouts" / s / "hide_rule.json"
        rules[s] = json.loads(p.read_text()) if p.is_file() else dict(rule="all", pool="keep")
    tkeys = {s: {m: (ik, to_key(smi) if isinstance(smi, str) else None) for m, ik, smi in truth[s]} for s in SETS}
    # 正解の構造のキー: ik14・scorer・その train_structs.key
    struct = {s: {x for ik, sk in tkeys[s].values() for x in (ik, sk, canon.get(ik), canon.get(sk or "")) if x} for s in SETS}
    watch = set().union(*struct.values())
    rows = []
    n_rows = 0
    for bt in pq.ParquetFile(R / "data/train.parquet").iter_batches(batch_size=262144, columns=["ingest_lib", "inchikey14"]):
        iks = bt.column("inchikey14").to_pylist()
        n_rows += len(iks)
        rows += [(lib, k) for lib, k in zip(bt.column("ingest_lib").to_pylist(), iks) if k in watch or canon.get(k, k) in watch]
    res: dict = dict(issue=116, root=str(R), train_rows_streamed=n_rows, sets={})
    for s in SETS:
        d, r = drop[s], rules[s]
        miss_sk = sorted(m for m, (ik, sk) in tkeys[s].items() if sk is None or sk not in d)
        miss_ik = sorted(m for m, (ik, sk) in tkeys[s].items() if ik not in d)
        alias_ik = sorted({ik for ik, k in canon.items() if k in struct[s] and ik not in {x for v in tkeys[s].values() for x in v if x}})
        # 分子ごとの残り（hide_rule の後の train）
        per_mol_left = {}
        for m, (ik, sk) in tkeys[s].items():
            keys_m = {x for x in (ik, sk, canon.get(ik), canon.get(sk or "")) if x}
            t = train_rows_left([x for x in rows if x[1] in keys_m or canon.get(x[1], x[1]) in keys_m], d, r["rule"], keys_m, canon)
            if t["rows"]:
                per_mol_left[m] = t
        tot = train_rows_left(rows, d, r["rule"], struct[s], canon)
        pool_left = sorted(m for m, (ik, sk) in tkeys[s].items()
                           if r.get("pool") == "remove" and any(x in pool_keys and x not in d for x in (ik, sk) if x))
        res["sets"][s] = dict(
            hide_rule=r, n_truth=len(tkeys[s]), n_drop_keys=len(d),
            truth_scorer_key_not_in_drop=len(miss_sk), truth_scorer_key_not_in_drop_ids=miss_sk,
            truth_ik14_not_in_drop=len(miss_ik),
            alias_ik14=alias_ik, alias_ik14_not_in_drop=sum(1 for k in alias_ik if k not in d),
            train_after_hide=dict(molecules_with_rows_left=len(per_mol_left), **tot,
                                  alias_molecules_with_rows_left=sum(1 for v in per_mol_left.values() if v["alias_rows"])),
            pool_remove=r.get("pool") == "remove", truth_left_in_pool=len(pool_left), truth_left_in_pool_ids=pool_left,
            alias_ik14_that_are_pool_keys=len(set(alias_ik) & pool_keys))
        # load_refs のキャッシュ（refs_top60.parquet）の中: 正解の構造の行と別名の行
        cp = R / "outputs/local_eval" / s / "refs_top60.parquet"
        if cp.is_file():
            ref = pd.read_parquet(cp, columns=["inchikey14", "ingest_lib"])
            ik = ref.inchikey14.astype(str)
            is_truth = ik.isin(struct[s]) | ik.map(lambda k: canon.get(k, k)).isin(struct[s])
            is_alias = ik.isin(set(alias_ik))
            res["sets"][s]["refs_top60"] = dict(
                rows=len(ref), sha256=sha256_file(cp), truth_struct_rows=int(is_truth.sum()),
                truth_struct_enveda_rows=int((is_truth & ref.ingest_lib.isin(ENVEDA)).sum()),
                alias_rows=int(is_alias.sum()), drop_key_rows=int(ik.isin(d).sum()),
                drop_key_enveda_rows=int((ik.isin(d) & ref.ingest_lib.isin(ENVEDA)).sum()),
                truth_struct_molecules=len({m for m, (a_, b_) in tkeys[s].items()
                                            if np.isin([x for x in (a_, b_) if x], ik[is_truth].unique()).any()}))
        else:
            res["sets"][s]["refs_top60"] = dict(status="missing", path=str(cp))
        # load_refs に届く別名: load_refs は候補（pool key）と ik14 が等しい行を読み、ik14 ∈ drop を隠す
        res["sets"][s]["alias_molecules_reachable_by_load_refs"] = sum(
            1 for m, (ik, sk) in tkeys[s].items()
            if any(a_ in pool_keys and a_ not in d for a_ in alias_ik if canon.get(a_) in {ik, sk, canon.get(ik)}))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    summ = {s: {k: v for k, v in r.items() if k in ("truth_scorer_key_not_in_drop", "alias_ik14_not_in_drop", "truth_left_in_pool",
                                                   "alias_molecules_reachable_by_load_refs")}
            | dict(train_after_hide={k: r["train_after_hide"][k] for k in ("molecules_with_rows_left", "rows", "alias_rows",
                                                                           "alias_molecules_with_rows_left", "enveda_rows")},
                   refs_top60={k: r["refs_top60"].get(k) for k in ("rows", "truth_struct_rows", "alias_rows", "drop_key_enveda_rows")})
            for s, r in res["sets"].items()}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print("wrote", out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("rebuild", "check"))
    ap.add_argument("--root", default=str(ROOT), help="data/ と outputs/ を読む（rebuild は data/holdouts に書く）場所")
    ap.add_argument("--out", default=str(ROOT / "outputs/cv116/drop_keys_acceptance.json"))
    a = ap.parse_args(argv)
    (cmd_rebuild if a.cmd == "rebuild" else cmd_check)(a)


if __name__ == "__main__":
    main()
