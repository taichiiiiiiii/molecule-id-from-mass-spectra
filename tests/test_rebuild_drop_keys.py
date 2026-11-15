"""rebuild_drop_keys（#116）の単体テスト。合成の truth・train_structs・train.parquet で、別名と正準化の違いがある構造が
drop に入ること、件数、冪等、library.load_refs と run.py の hidden が新しいキーで正解を外すことを確かめる。"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


RB = _load("rebuild_drop_keys", "scripts/cv/rebuild_drop_keys.py")

# 正解 3 分子。m1: ik14 = scorer。m2: truth の ik14（非正準）と scorer のキーが違う（#92 の v3_c2 の 3 件の形）。
# m3: ik14 = scorer だが、train_structs に別の ik14（別名 ALIAS3、key = K3）でも載っている（v3_c2n の 8 件の形）
TO_KEY = {"smi1": "K1", "smi2": "K2CANON", "smi3": "K3"}.get
TRUTH = [("m1", "K1", "smi1"), ("m2", "K2RAW", "smi2"), ("m3", "K3", "smi3")]
TS = [("K1", "K1"), ("K2RAW", "K2CANON"), ("K3", "K3"), ("ALIAS3", "K3"), ("OTHER", "OTHER")]


def test_rebuild_keys_adds_scorer_key_and_alias():
    new, added = RB.rebuild_keys(TRUTH, {"K1", "K2RAW", "K3"}, TS, TO_KEY)
    assert new == {"K1", "K2RAW", "K2CANON", "K3", "ALIAS3"}
    assert added == {"K2CANON": ["scorer:m2", "ts_key_of:K2RAW"], "ALIAS3": ["alias_of:K3"]}
    assert "OTHER" not in new


def test_rebuild_keys_never_drops_old_and_handles_missing_smiles():
    new, added = RB.rebuild_keys([("m1", "K1", None)], {"EXTRA"}, TS, TO_KEY)
    assert new == {"EXTRA", "K1"} and added == {"K1": ["truth_ik14:m1"]}   # 古いキー（正解以外）も残す
    # train_structs の key しか知らない（smiles が変換できない）でも ts の key から別名まで辿る
    new2, _ = RB.rebuild_keys([("m2", "K2RAW", "bad")], set(), TS + [("K2ALIAS", "K2CANON")], lambda s: None)
    assert new2 == {"K2RAW", "K2CANON", "K2ALIAS"}


def test_rebuild_set_counts_manifest_and_idempotent(tmp_path: Path):
    h = tmp_path / "h"; h.mkdir()
    (h / "drop_keys.txt").write_text("K1\nK2RAW\nK3\n")
    r = RB.rebuild_set(h, TRUTH, TS, TO_KEY, meta=dict(rdkit="x"))
    assert r["status"] == "written" and r["before"]["n"] == 3 and r["after"]["n"] == 5 and r["n_added"] == 2
    assert (h / "drop_keys.txt").read_text() == "ALIAS3\nK1\nK2CANON\nK2RAW\nK3\n"
    m = json.loads((h / "drop_keys_manifest.json").read_text())
    assert m["after"]["sha256"] == RB.sha256_bytes((h / "drop_keys.txt").read_bytes())
    assert m["n_added_by_reason"] == {"truth_ik14": 0, "scorer": 1, "ts_key_of": 1, "alias_of": 1} and m["rdkit"] == "x"
    before = (h / "drop_keys.txt").read_bytes(), (h / "drop_keys_manifest.json").read_bytes()
    r2 = RB.rebuild_set(h, TRUTH, TS, TO_KEY)
    assert r2["status"] == "unchanged" and r2["before"]["n"] == 3     # 2 回目は書かない。manifest の「前」は最初のまま
    assert ((h / "drop_keys.txt").read_bytes(), (h / "drop_keys_manifest.json").read_bytes()) == before
    # manifest が無くても、作り直した drop に同じ定義を当てると同じ集合（固定点）
    new, added = RB.rebuild_keys(TRUTH, set((h / "drop_keys.txt").read_text().split()), TS, TO_KEY)
    assert new == set((h / "drop_keys.txt").read_text().split()) and added == {}


def test_train_rows_left_old_vs_new_drop():
    rows = [("gnps", "K3"), ("riken", "ALIAS3"), ("mona", "ALIAS3"), ("enveda-180", "K1"), ("gnps", "K1"), ("riken", "OTHER")]
    canon = {"ALIAS3": "K3", "K2RAW": "K2CANON"}
    old = RB.train_rows_left(rows, {"K3"}, "all", {"K3"}, canon)
    assert old["rows"] == 2 and old["alias_rows"] == 2 and old["rows_by_lib"] == {"mona": 1, "riken": 1}
    new = RB.train_rows_left(rows, {"K3", "ALIAS3"}, "all", {"K3"}, canon)
    assert new["rows"] == 0 and new["alias_rows"] == 0
    c1 = RB.train_rows_left(rows, {"K1"}, "enveda_libs", {"K1"}, canon)   # enveda_libs: 公開ライブラリの行は設計どおり残る
    assert c1["rows"] == 1 and c1["enveda_rows"] == 0 and c1["rows_by_lib"] == {"gnps": 1}
    with pytest.raises(ValueError):
        RB.train_rows_left(rows, set(), "pool", set(), canon)


def _fake_root(tmp_path: Path, drop: set, rule: dict) -> Path:
    h = tmp_path / "data/holdouts/h"; h.mkdir(parents=True)
    (h / "drop_keys.txt").write_text("\n".join(sorted(drop)) + "\n")
    (h / "hide_rule.json").write_text(json.dumps(rule))
    pd.DataFrame(dict(inchikey14=["K3", "ALIAS3", "ALIAS3", "K1", "OTHER"],
                      ingest_lib=["gnps", "riken", "mona", "enveda-180", "gnps"],
                      ms2_mzs=[np.array([50.0, 60.0])] * 5,
                      ms2_normalized_intensities=[np.array([1.0, 0.5])] * 5)).to_parquet(tmp_path / "data/train.parquet")
    return tmp_path


def test_library_load_refs_hides_alias_rows_with_new_keys(tmp_path: Path, monkeypatch):
    L = _load("local_eval_library_t116", "scripts/local_eval/library.py")
    monkeypatch.setattr(L, "_prep", lambda mz, it: (tuple(mz), tuple(it)))   # casmi に依らない
    new, _ = RB.rebuild_keys([("m3", "K3", "smi3")], {"K3"}, TS, TO_KEY)
    for drop, expect in (({"K3"}, {"ALIAS3", "K1", "OTHER"}), (new, {"K1", "OTHER"})):
        root = _fake_root(tmp_path / ("new" if drop is new else "old"), drop, dict(rule="all", pool="remove"))
        monkeypatch.setattr(L, "ROOT", root)
        refs = L.load_refs("h", {"K3", "ALIAS3", "K1", "OTHER"}, root / "refs.parquet")
        assert set(refs) == expect          # 古い drop（ik14 だけ）では別名の行が参照に残る、新しい drop では残らない
        assert L.drop_keys("h") == drop


def test_run_hidden_removes_truth_from_pool_with_new_keys(tmp_path: Path):
    le = _load("local_eval_run_t116", "scripts/local_eval/run.py")
    L = _load("local_eval_library_t116b", "scripts/local_eval/library.py")
    # プールの key は scorer のキー（正準）。正解 m2 の truth の ik14（K2RAW）はプールに無い
    meta = pd.DataFrame(dict(key=["K2CANON", "OTHER", "K1"], smiles=["smi2", "x", "smi1"],
                             mass=[100.0, 100.0002, 100.0004], pid=[0, 1, 2]))
    pool = le.Pool(meta, None)
    old = {"K1", "K2RAW"}
    new, _ = RB.rebuild_keys([("m1", "K1", "smi1"), ("m2", "K2RAW", "smi2")], old, TS, TO_KEY)
    for drop, expect in ((old, {"K2CANON", "OTHER"}), (new, {"OTHER"})):
        root = _fake_root(tmp_path / ("new" if drop is new else "old"), drop, dict(rule="all", pool="remove"))
        L.ROOT = root
        hidden = L.drop_keys("h") if L.hide_rule("h").get("pool") == "remove" else set()
        idx = le.visible_window(pool, 100.0002, hidden)
        assert set(pool.key[idx]) == expect   # 古い drop では正解（K2CANON）がプールに残る（#92）、新しい drop では外れる
    assert set(pool.key[le.visible_window(pool, 100.0002, set())]) == {"K2CANON", "OTHER", "K1"}   # hidden 無し = 窓のまま
