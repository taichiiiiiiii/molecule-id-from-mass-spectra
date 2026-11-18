"""Issue #146: d_seen（scripts/local_eval/dseen.py）の合成データのテスト。実データは読まない。"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
sys.path.insert(0, str(ROOT / "scripts/ranker"))

import dseen as D  # noqa: E402

NAN = math.nan


# ---------------------------------------------------------------- 群の分け方
def test_group_S_by_canon_or_raw_U_by_canon_or_raw_else_O():
    T_canon, T_raw, TR = {"TC"}, {"TR1"}, {"TC", "TR1", "UC", "UR"}
    assert D.group_of("TC", ["x"], T_canon, T_raw, TR) == "S"            # 正準化後で T
    assert D.group_of("zz", ["x", "TR1"], T_canon, T_raw, TR) == "S"     # 生（どれかの行）で T
    assert D.group_of("UC", ["x"], T_canon, T_raw, TR) == "U"            # T でなく train の key
    assert D.group_of("zz", [None, "UR"], T_canon, T_raw, TR) == "U"     # 生だけで train
    assert D.group_of("zz", [None, "q"], T_canon, T_raw, TR) == "O"
    # T は U より先（T ⊂ train でも S）
    assert D.group_of("TC", ["UR"], T_canon, T_raw, TR) == "S"


# ---------------------------------------------------------------- 正解の除き方（生・正準化後・別名）
def test_truth_exclusion_uses_raw_canon_and_aliases():
    fns = dict(raw=lambda s: "RAW_" + s, to_key=lambda s: "TOK_" + s, canon=lambda s: "CAN_" + s)
    # train_structs の行 (inchikey14, rawkey, key): 正解の生の key を inchikey14 に持つ行の key が別名
    ts = [("IK_T", "IK_T", "ALIAS_K"), ("OTHER", "OTHER", "OTHER_K"), ("IK2", "IK2", "CAN_s")]
    E = D.truth_exclusion("IK_T", "s", fns, D.alias_index(ts))
    assert {"IK_T", "RAW_s", "TOK_s", "CAN_s", "ALIAS_K", "IK2"} <= E   # CAN_s の行の inchikey14 IK2 も別名
    assert "OTHER_K" not in E and "OTHER" not in E
    # 候補は pool の key か、どれかの行の raw_key で E に入れば除く
    assert D.is_truth("ALIAS_K", ["q"], E)
    assert D.is_truth("zz", [None, "RAW_s"], E)
    assert not D.is_truth("zz", [None, "q"], E)


def test_truth_exclusion_without_smiles_or_ik():
    fns = dict(raw=lambda s: None, to_key=lambda s: None, canon=lambda s: None)
    assert D.truth_exclusion(None, None, fns, {}) == set()
    assert D.truth_exclusion("A", "s", fns, {}) == {"A"}   # None の key は入れない


# ---------------------------------------------------------------- 分子ごとの d と外す規則
def _c(group, n, nb, f, s=0.0):
    return dict(group=group, n=n, nb=nb, f=f, s=s)


def test_molecule_d_value_ddof0_and_finite_filter():
    cands = [_c("S", 3.0, 1.0, 1.0, 1.0), _c("S", 1.0, 2.0, 2.0, 2.0), _c("U", 0.0, 3.0, 3.0, 3.0),
             _c("O", 0.0, 4.0, 2.0, 4.0),
             _c("S", NAN, 1.0, 1.0), _c("U", 1.0, 1.0, None), _c("U", 1.0, math.inf, 1.0)]   # 点の無い候補は外れる
    rec, why = D.molecule_d(cands)
    assert why is None and rec["n_cands"] == 4 and (rec["n_S"], rec["n_U"], rec["n_O"]) == (2, 1, 1)
    n = np.array([3.0, 1.0, 0.0, 0.0]); zn = (n - n.mean()) / n.std()          # ddof 0
    assert rec["d_seen"]["n"] == pytest.approx(zn[:2].mean() - zn[2])
    assert rec["d_dom"]["n"] == pytest.approx(zn[2] - zn[3])
    f = np.array([1.0, 2.0, 3.0, 2.0]); zf = (f - f.mean()) / f.std()
    assert rec["d_seen"]["f"] == pytest.approx(zf[:2].mean() - zf[2])


def test_molecule_d_excludes_sd0_one_group_no_candidates():
    # fragnn の sd 0
    assert D.molecule_d([_c("S", 1.0, 1.0, 1.0), _c("U", 1.0, 2.0, 2.0)]) == (None, "sd0")
    # MetFrag-lite の sd 0
    assert D.molecule_d([_c("S", 1.0, 1.0, 5.0), _c("U", 2.0, 2.0, 5.0)]) == (None, "sd0")
    # U が無い（O だけ）
    assert D.molecule_d([_c("S", 1.0, 1.0, 1.0), _c("O", 2.0, 2.0, 2.0)]) == (None, "one_group")
    # 3 つの点がそろう候補が 0
    assert D.molecule_d([_c("S", 1.0, NAN, 1.0), _c("U", 2.0, 2.0, None)]) == (None, "no_candidates")
    # blind の sd 0 は外さず、blind の d だけ None（記録だけの対比で数えない）
    rec, why = D.molecule_d([_c("S", 1.0, 7.0, 1.0), _c("U", 2.0, 7.0, 2.0)])
    assert why is None and rec["d_seen"]["nb"] is None and rec["d_dom"]["n"] is None   # O が無いので d_dom は無し


def test_summarize_counts_reasons_and_blind():
    recs = {"m1": D.molecule_d([_c("S", 2.0, 1.0, 1.0), _c("U", 1.0, 2.0, 2.0)])[0],
            "m2": D.molecule_d([_c("S", 1.0, 7.0, 2.0), _c("U", 2.0, 7.0, 1.0)])[0]}
    out = D.summarize_set(recs, {"m3": "sd0", "m4": "one_group", "m5": "sd0"})
    assert out["n"] == 2 and out["excluded"] == dict(no_logits=0, no_candidates=0, one_group=1, sd0=2)
    # m1: d_seen(n) = 2, d_seen(f) = −2 → 4 ; m2: d_seen(n) = −2, d_seen(f) = 2 → −4
    assert out["delta_seen"]["mean"] == 0.0 and out["delta_seen"]["n"] == 2
    assert out["blind_minus_metfrag"]["n"] == 1 and out["blind_sd0_excluded"] == 1
    assert out["band"] in ("higher_seen", "lower_seen", "none")


# ---------------------------------------------------------------- bootstrap の再現
def test_bootstrap_matches_train_eval_and_is_reproducible():
    from train_eval import bootstrap_diff
    rng = np.random.default_rng(5)
    x = {f"m_{i:05d}": float(v) for i, v in enumerate(rng.normal(0.1, 0.4, 137))}
    a = D.bootstrap_ci(x)
    b = bootstrap_diff(x, {m: 0.0 for m in x})
    assert round(a["mean"], 4) == b["diff"] and round(a["lo"], 4) == b["lo"] and round(a["hi"], 4) == b["hi"]
    assert D.bootstrap_ci(x) == a                                  # seed 0 で同じ
    shuffled = dict(sorted(x.items(), key=lambda kv: -kv[1]))      # 入力の並びに依らない（molecule_id の昇順）
    assert D.bootstrap_ci(shuffled) == a
    assert a["n"] == 137


# ---------------------------------------------------------------- 帯と使い道（#146 A §3.4）
def test_band_uses_interval_ends():
    assert D.band(0.001, 0.2) == "higher_seen"
    assert D.band(-0.2, -1e-9) == "lower_seen"
    assert D.band(-0.1, 0.1) == "none" and D.band(0.0, 0.3) == "none" and D.band(-0.3, 0.0) == "none"


@pytest.mark.parametrize("b1,b2,want", [
    ("higher_seen", "higher_seen", "neg_outside_T"),
    ("lower_seen", "lower_seen", "freeze_no_gate"),
    ("lower_seen", "none", "freeze_no_gate"),
    ("none", "lower_seen", "freeze_no_gate"),
    ("higher_seen", "lower_seen", "no_decision"),
    ("lower_seen", "higher_seen", "no_decision"),
    ("higher_seen", "none", "no_treatment"),
    ("none", "higher_seen", "no_treatment"),
    ("none", "none", "no_treatment"),
])
def test_usage_four_branches(b1, b2, want):
    assert D.usage(b1, b2) == want


# ---------------------------------------------------------------- e180_c2k の窓の一致・準備
def test_windows_match_reports_differences():
    a = {"m1": np.array([1, 2, 3]), "m2": np.array([5]), "m3": np.array([7])}
    assert D.windows_match(a, {k: v.copy() for k, v in a.items()}) == []
    b = {"m1": np.array([1, 2]), "m2": np.array([5]), "m4": np.array([9])}
    assert D.windows_match(a, b) == ["m3", "m4", "m1"]


def test_prepare_e180c2k_copies_with_sha_and_keep_rule(tmp_path, monkeypatch):
    src = tmp_path / "data/holdouts/e180_c2"
    src.mkdir(parents=True)
    shas = {}
    for name in ("test.parquet", "truth.csv", "drop_keys.txt"):
        (src / name).write_bytes(name.encode())
        shas[name] = hashlib.sha256(name.encode()).hexdigest()
    (src / "hide_rule.json").write_text(json.dumps({"rule": "all", "pool": "remove"}))
    monkeypatch.setattr(D, "E180_SHA", shas)
    rep = D.prepare_e180c2k(tmp_path)
    dst = tmp_path / "data/holdouts/e180_c2k"
    assert json.loads((dst / "hide_rule.json").read_text()) == {"rule": "all", "pool": "keep"}
    assert all((dst / n).read_bytes() == n.encode() for n in shas) and rep["files"] == shas
    (src / "truth.csv").write_bytes(b"changed")
    with pytest.raises(SystemExit):
        D.prepare_e180c2k(tmp_path)


def test_read_paths_for_e180_c2k_stay_in_clone():
    assert D.truth_path("e180_c2k", Path("/r")) == Path("/r/data/holdouts/e180_c2k/truth.csv")
    assert D.logits_path("e180_c2k", Path("/r")) == Path("/r/outputs/local_eval/e180_c2k/pc_logits.npz")
    assert D.truth_path("v3_c1", Path("/r")) == Path("/r/outputs/holdouts/v3_c1/truth.csv")
    assert D.logits_path("v3_c1", Path("/r")) == Path("/r/outputs/holdout_eval/v3_c1_fold0/pc_logits.npz")
    assert D.READ_SETS == ("v3_c1", "e180_c2k") and set(D.RECORD_SETS) == {"enveda_v1", "v3_c2n", "v3_c2"}


def test_fragnn_score_accepts_logits_and_truth_paths():
    import fragnn_score as FS
    p = inspect.signature(FS.score_holdout).parameters
    assert p["logits"].default is None and p["truth"].default is None   # 既定は今までの読み先
