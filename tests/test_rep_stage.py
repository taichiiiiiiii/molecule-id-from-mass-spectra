"""Issue #171: rep 段（家族に特有の断片・中性損失）の合成データでのテスト（事前登録 §6 の (i)〜(iv) と段の組み合わせ）。"""
import hashlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import rep as RP  # noqa: E402
import run as R  # noqa: E402

FR = lambda b: (RP.KIND_FRAG << RP.CODE_SHIFT) + b  # noqa: E731
LS = lambda b: (RP.KIND_LOSS << RP.CODE_SHIFT) + b  # noqa: E731

# main（a40c87d = 53e4d5e）の scripts/local_eval/run.py の sha256。rep の印の間を除いた run.py がこれと同じ
MAIN_RUN_PY_SHA256 = "75a8cdc41f8ad46f17be25ac5cefaea9cd1ce9ac4e3355668c2c25cab8d20c3f"


# ---------------------------------------------------------------- 合成の参照ライブラリ
# スペクトル（行）: (sid, mode, prec, mz の列)。sid 0,1,2 は家族 0、sid 3,4 は家族 1、sid 5 は骨格 None（表に入らない）
SPECS = [
    (0, 1, 200.0, [100.0, 150.0]),
    (1, 1, 210.0, [100.0, 160.0]),
    (2, 1, 220.0, [100.0, 150.0, 170.0]),
    (3, 1, 200.0, [150.0, 120.0]),
    (4, 1, 230.0, [100.0, 130.0]),
    (5, 1, 200.0, [100.0, 150.0]),
    (0, -1, 200.0, [100.0]),
    (3, -1, 190.0, [100.0, 90.0]),
]
SID_FAM = np.array([0, 0, 0, 1, 1, -1], np.int64)


def _csr(specs):
    off = np.zeros(len(specs) + 1, np.int64)
    for j, s in enumerate(specs):
        off[j + 1] = off[j] + len(s[3])
    mz = np.concatenate([np.asarray(s[3], np.float32) for s in specs])
    prec = np.array([s[2] for s in specs], np.float64)
    return off, mz, prec


def _tables(specs=SPECS, sid_fam=SID_FAM, drop_sids=()):
    keep = [j for j, s in enumerate(specs) if s[0] not in set(drop_sids)]
    off, mz, prec = _csr(specs)
    sid = np.array([s[0] for s in specs], np.int64)
    mode = np.array([s[1] for s in specs], np.int8)
    return RP.build_tables(np.array(keep, np.int64), sid[keep], mode[keep], off, mz, prec, sid_fam, 2)


def _feat(specs, sid_fam, mode):
    """手で数える: {sid: 特徴の集合}（表に入る構造だけ）。"""
    out = {}
    for s, m, p, mz in specs:
        if m != mode or sid_fam[s] < 0:
            continue
        f = {FR(int(round(x / 0.01))) for x in mz}
        f |= {LS(int(round((p - x) / 0.01))) for x in mz if round((p - x) / 0.01) > 0}
        out[s] = f
    return out


# ---------------------------------------------------------------- 特徴
def test_spectrum_codes_bins_and_losses():
    c = RP.spectrum_codes([100.004, 150.0, 200.0, 201.0], 200.0)
    # 断片 4 つ、損失は 200−100.004 = 99.996 → 9999.6 → 10000、50 → 5000、0 と負は捨てる
    assert set(c.tolist()) == {FR(10000), FR(15000), FR(20000), FR(20100), LS(10000), LS(5000)}
    assert list(c) == sorted(set(c.tolist()))


def test_query_codes_union_per_mode():
    q = [dict(mz=np.array([100.0]), prec=150.0, mode=1), dict(mz=np.array([100.0, 120.0]), prec=150.0, mode=1),
         dict(mz=np.array([80.0]), prec=100.0, mode=-1)]
    Q = RP.query_codes(q)
    assert set(Q) == {1, -1}
    assert set(Q[1].tolist()) == {FR(10000), FR(12000), LS(5000), LS(3000)}
    assert set(Q[-1].tolist()) == {FR(8000), LS(2000)}


# ---------------------------------------------------------------- (i) 合成の表で r(f,F) が式どおり
def test_i_table_counts_and_r_formula():
    T = _tables()
    t = T[1]
    feats = _feat(SPECS, SID_FAM, 1)
    assert t.N == 5 and t.nF.tolist() == [3, 2]   # 骨格 None の sid 5 は入らない
    allf = sorted(set().union(*feats.values()))
    q = np.array(allf, np.int64)
    c = t.c_of(q)
    for f, cv in zip(allf, c):
        assert cv == sum(1 for s in feats if f in feats[s])
    a0 = t.a_of(0, q)
    for f, av in zip(allf, a0):
        assert av == sum(1 for s in feats if SID_FAM[s] == 0 and f in feats[s])
    # 候補の家族 0、自分なし → Q のすべての特徴の r の平均
    Q = {1: q}
    got = RP.rep_score(T, Q, 0, set())
    exp = np.mean([math.log((a + 0.5) / (3 + 1)) - math.log((cv + 0.5) / (5 + 1)) for a, cv in zip(a0, c)])
    assert got == pytest.approx(exp, rel=1e-12)


def test_i_r_value_formula_and_two_modes():
    assert RP.r_value(2, 3, 4, 10) == pytest.approx(math.log(2.5 / 4) - math.log(4.5 / 11))
    T = _tables()
    qp = np.array([FR(10000), FR(99999)], np.int64)   # 99999 は c_f = 0 → 平均から外れる
    qn = np.array([FR(10000)], np.int64)
    got = RP.rep_score(T, {1: qp, -1: qn}, 0, set())
    # 正: a = 3（sid 0,1,2）、n_F 3、c 4（0,1,2,4）、N 5 ／ 負: a = 1、n_F 1、c 2、N 2
    exp = np.mean([math.log(3.5 / 4) - math.log(4.5 / 6), math.log(1.5 / 2) - math.log(2.5 / 3)])
    assert got == pytest.approx(exp, rel=1e-12)


# ---------------------------------------------------------------- (ii) leave-own-structure-out
@pytest.mark.parametrize("own", [{0}, {3}, {0, 3}, {0, 1, 2}, {5}])
def test_ii_leave_out_equals_table_without_own(own):
    q = np.array(sorted(set().union(*_feat(SPECS, SID_FAM, 1).values())) + [FR(77777)], np.int64)
    qn = np.array([FR(10000), FR(9000), LS(10000)], np.int64)
    Q = {1: q, -1: qn}
    full, cut = _tables(), _tables(drop_sids=own)
    for fam in (0, 1):
        got = RP.rep_score(full, Q, fam, own)
        exp = RP.rep_score(cut, Q, fam, set())
        if math.isnan(exp):
            assert math.isnan(got)
        else:
            assert got == pytest.approx(exp, rel=1e-12)
    # 家族 0 の構造を全部引くと n_F = 0 → 点なし（そのモードで）
    assert math.isnan(RP.rep_score(full, {1: q}, 0, {0, 1, 2}))


def test_ii_own_index_includes_alias_keys():
    # sid 0: ik A / key KA / raw RA、sid 1: 別の ik B で key が KA（別名）、sid 2: raw が RA2、key K2（無関係）
    # sid 3: ik C で rawkey が RA（sid 0 の rawkey の別名）
    ik = np.array(["A", "B", "Z", "C"], object)
    key = np.array(["KA", "KA", "K2", "KC"], object)
    raw = np.array(["RA", "RB", "RA2", "RA"], object)
    oi = RP.OwnIndex(ik, key, raw)
    assert oi.sids({"A"}) == {0, 1, 3}   # A → 行 0 の 3 つの key {A, KA, RA} → KA の行 1、RA の行 3
    assert oi.sids({"KA", None}) == {0, 1, 3}   # KA → 行 0・1 の key {A, B, KA, RA, RB}（1 段）→ RA を持つ行 3 も入る
    assert oi.sids({"nothing"}) == set()


def test_ii_own_index_one_step_only():
    ik = np.array(["A", "B", "C"], object)
    key = np.array(["KA", "KB", "KC"], object)
    raw = np.array(["RA", "KA", "KB"], object)   # 行 1 の raw が行 0 の key、行 2 の raw が行 1 の key
    oi = RP.OwnIndex(ik, key, raw)
    # K = {A}: E0 の別名 = 行 0 の key {A, KA, RA}。KA を持つ行 = 0, 1 → 行 2（KB 経由、2 段目）は入らない
    assert oi.sids({"A"}) == {0, 1}


# ---------------------------------------------------------------- (iii) 家族が表に無い候補は点なし（z = 0）で数が記録される
def _fake_ctx(T, fam_id, smiles, keys, scaf, train_sid, truth=("TIK", "t"), spectra=None):
    ctx = RP.RepContext.__new__(RP.RepContext)
    ctx.tables, ctx.fam_id = T, fam_id
    ctx.pool = SimpleNamespace(smiles=np.array(smiles, object), key=np.array(keys, object))
    ctx.scaffold_fn = lambda s: scaf[s]
    ctx.to_key = lambda s: "TK_" + s
    ctx.raw_key = lambda s: None
    ctx.own_index = RP.OwnIndex(np.array(["TIK"], object), np.array(["TK_t"], object), np.array([None], object))
    ctx.train_sid = np.array(train_sid, np.int64)
    ctx.ik2sid = {"TIK": 0}
    ctx.truth_rows = {"m": truth}
    ctx.spectra = spectra or {"m": [dict(mz=np.array([100.0, 150.0]), prec=200.0, mode=1)]}
    ctx._scaf, ctx._own, ctx._tk, ctx.window_scores = {}, {}, {}, {}
    ctx.cnt = dict(window_rows=0, window_no_score=0, top_rows=0, top_no_score=0, molecules=0, molecules_unchanged=0,
                   train_sid_checked=0, same_scaffold_diff_rep=0, decoy_overtakes_truth=0, truth_in_window=0,
                   truth_sid_known=0, truth_sid_in_leaveout=0, no_query_features=0)
    return ctx


def test_iii_family_absent_no_score_counted_and_z_zero():
    T = _tables()
    # 候補: 0 = 家族 f0、1 = 家族 f1、2 = 表に無い家族、3 = 骨格 None
    ctx = _fake_ctx(T, {"f0": 0, "f1": 1}, ["a", "b", "c", "d"], ["k0", "k1", "k2", "k3"],
                    {"a": "f0", "b": "f1", "c": "fX", "d": None}, [-1, -1, -1, -1])
    sc = ctx.score_window_rows("m", [0, 1, 2, 3])
    assert math.isfinite(sc[0]) and math.isfinite(sc[1])
    assert math.isnan(sc[2]) and math.isnan(sc[3])
    assert ctx.cnt["window_rows"] == 4 and ctx.cnt["window_no_score"] == 2
    rows = [(4.0, 0), (3.0, 2), (2.0, 1), (1.0, 3)]
    out = ctx.mix("m", rows)
    assert ctx.cnt["top_rows"] == 4 and ctx.cnt["top_no_score"] == 2 and ctx.cnt["molecules"] == 1
    # 点なしの行は z(rep) = 0、点のある 2 行だけで mean・std
    rep = np.array([sc[0], np.nan, sc[1], np.nan])
    ok = np.isfinite(rep)
    zr = np.zeros(4)
    zr[ok] = (rep[ok] - rep[ok].mean()) / (rep[ok].std() + 1e-9)
    f = RP.zscore([4.0, 3.0, 2.0, 1.0]) + zr
    exp = np.argsort(-f, kind="stable")
    assert [i for _, i in out] == [rows[j][1] for j in exp]
    assert [v for v, _ in out] == pytest.approx(list(f[exp]))


def test_iii_train_sid_assert_and_truth_sid_record():
    T = _tables()
    # 行 0 は正解（to_key = TK_t、train_sid 0 は own に入る）、行 1 は train_sid 0 を持つのに own に入らない → assert
    ctx = _fake_ctx(T, {"f0": 0}, ["t", "x"], ["kt", "kx"], {"t": "f0", "x": "f0"}, [0, -1])
    ctx.score_window_rows("m", [0, 1])
    assert ctx.cnt["truth_in_window"] == 1 and ctx.cnt["truth_sid_known"] == 1 and ctx.cnt["truth_sid_in_leaveout"] == 1
    assert ctx.cnt["train_sid_checked"] == 1
    bad = _fake_ctx(T, {"f0": 0}, ["t", "x"], ["kt", "kx"], {"t": "f0", "x": "f0"}, [-1, 0])
    with pytest.raises(AssertionError):
        bad.score_window_rows("m", [0, 1])


def test_rep_mix_keeps_rows_when_too_few_or_constant():
    rows = [(3.0, 10), (2.0, 11), (1.0, 12)]
    out, applied = RP.rep_mix(rows, np.array([np.nan, 0.4, np.nan]))
    assert out is rows and not applied
    out, applied = RP.rep_mix(rows, np.array([0.4, 0.4, np.nan]))
    assert out is rows and not applied
    out, applied = RP.rep_mix([(1.0, 1)], np.array([1.0]))
    assert not applied
    out, applied = RP.rep_mix([(1.0, 1), (1.0, 2), (0.0, 3)], np.array([0.0, 0.0, np.nan]))
    assert not applied   # std 0


def test_rep_mix_stable_ties_and_no_top1_lock():
    rows = [(1.0, 1), (1.0, 2), (0.0, 3), (0.0, 4)]
    out, applied = RP.rep_mix(rows, np.array([0.0, 0.0, 1.0, 1.0]))
    assert applied and [i for _, i in out] == [1, 2, 3, 4]   # z の和が全部同じ → 元の順
    out, _ = RP.rep_mix([(1.0, 1), (0.9, 2)], np.array([0.0, 5.0]))
    assert [i for _, i in out] == [2, 1]   # 1 位を固定しない


def test_same_scaffold_none_is_not_same():
    assert RP.same_scaffold("c1ccccc1", "c1ccccc1") and RP.same_scaffold("", "")
    assert not RP.same_scaffold(None, None) and not RP.same_scaffold(None, "") and not RP.same_scaffold("a", "b")


def test_decoy_overtake_record():
    T = _tables()
    ctx = _fake_ctx(T, {"f0": 0, "f1": 1}, ["t", "d", "e"], ["kt", "kd", "ke"], {"t": "f0", "d": "f0", "e": "f1"},
                    [-1, -1, -1])
    ctx.window_scores["m"] = {0: 0.0, 1: 10.0, 2: 0.0}
    ctx.mix("m", [(1.0, 0), (0.99, 1), (0.0, 2)])
    assert ctx.cnt["decoy_overtakes_truth"] == 1 and ctx.cnt["same_scaffold_diff_rep"] == 1


# ---------------------------------------------------------------- 段の組み合わせ
def test_check_stages_rep():
    R.check_stages(["score", "prior", "rep", "frag"])
    R.check_stages(["score", "prior", "frag"])
    for bad in (["score", "prior", "rep"], ["score", "rep", "frag"], ["score", "prior", "rep", "frag", "elem"],
                ["score", "nbz", "prior", "rep", "frag"], ["score", "pmi", "prior", "rep", "frag"],
                ["score", "fragnn", "prior", "rep", "frag"], ["score", "sanpier", "prior", "rep", "frag"],
                ["score", "prior", "lfus", "rep", "frag"], ["score", "prior", "rep", "frag", "lib"],
                ["score", "prior", "rep", "frag", "pubchem"], ["score", "prior", "frag", "rep"]):
        with pytest.raises(AssertionError):
            R.check_stages(bad)


# ---------------------------------------------------------------- (iv) score,prior,frag の経路は rep を足す前と同じ
def _strip_rep(src: str) -> str:
    out, skip = [], False
    for line in src.splitlines(keepends=True):
        if "# rep>>" in line:
            assert not skip
            skip = True
            continue
        if "# <<rep" in line:
            assert skip
            skip = False
            continue
        if not skip:
            out.append(line)
    assert not skip
    return "".join(out)


def test_iv_run_py_without_rep_blocks_is_main_bytes():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    assert src.count("# rep>>") == src.count("# <<rep") == 5
    assert hashlib.sha256(_strip_rep(src).encode()).hexdigest() == MAIN_RUN_PY_SHA256


def test_iv_rep_blocks_only_act_with_rep_stage():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    blocks = src.split("# rep>>")[1:]
    for b in blocks:
        body = b.split("# <<rep")[0]
        # check_stages の中の block 以外は rep_ctx を作る / 使う条件付きの block だけ
        assert ("RP.check_stages_rep(stages)" in body) or ('if "rep" in stages:' in body) or ("if rep_ctx is not None:" in body)


def test_run_order_rep_after_prior_before_frag():
    src = (ROOT / "scripts/local_eval/run.py").read_text()
    loop = src[src.index("    for mid in truth:\n        if mid not in targets:"):]
    i_win = loop.index("rep_ctx.score_window_rows(")
    i_prior = loop.index("rows = rerank_with_prior(rows, pool)")
    i_rep = loop.index("rows = rep_ctx.mix(mid, rows)")
    i_frag = loop.index("rows = frag_rerank(")
    assert loop.index("rows = score_window(") < i_win < i_prior < i_rep < i_frag


# ---------------------------------------------------------------- 1 回だけの起動スクリプト（#171 §4、レビュー記録の決めごと 1・2）
def test_run_rep_once_script_shape():
    import subprocess
    p = ROOT / "scripts/local_eval/run_rep_once.sh"
    src = p.read_text()
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    assert "SETS=(v3_c1 v3_c2 enveda_v1 v3_c2n e180_c2k)" in src
    # python を memguard に直接渡す（間に何も挟まない）
    assert 'MIN_FREE_PCT=15 /bin/bash "$MEMGUARD" "$log" "${cmd[@]}"' in src
    assert 'cmd=("$PY" scripts/local_eval/run.py --holdout "$h" --stages score,prior,rep,frag)' in src
    assert 'cmd=("$PY" scripts/local_eval/rep.py window --holdout "$h")' in src
    for bad in ("uv run", "caffeinate", "bash -c", "env "):
        assert bad not in src.split("set -u", 1)[1]
    assert 'if [ -e "$done_f" ]' in src and "rep_once.done" in src
    assert 'if [ "$rc" -ne 0 ]' in src and "exit 3" in src
