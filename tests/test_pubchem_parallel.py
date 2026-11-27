"""Issue #278: pubchem.run_holdout の workers（fork した複数プロセス）が workers=1 と完全に同じ結果を返すこと。

小さな合成の層（数百行の pc_mass / pc_off / pc_smiles の npy）と、casmi.chem の代わりの小さなモジュールで確かめる。
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem import Descriptors, rdFingerprintGenerator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local_eval"))
import pubchem as PC  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fork が要る")

BASE = ["CCO", "CCN", "CCCO", "CC(C)O", "OCCO", "CC(=O)O", "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "CCOC",
        "CC(N)C(=O)O", "NCC(=O)O", "OC(=O)CCC(=O)O", "c1ccncc1", "C1CCCCC1", "C1CCCCC1O", "CCCCCCO",
        "OC1C(O)C(O)C(O)C(O)C1O", "c1ccc2ccccc2c1", "CC(=O)Nc1ccc(O)cc1", "O=C(O)c1ccccc1O", "CN1CCCC1",
        "CCOC(=O)C", "CCCC(=O)O", "OCC(O)CO", "Oc1ccc(O)cc1", "Oc1cccc(O)c1", "Oc1ccccc1O", "CCC(C)O", "CCCCO"]


def _fake_chem() -> types.ModuleType:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    gen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)

    def raw_fingerprint(s):
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return np.concatenate([gen.GetFingerprintAsNumPy(m), gen3.GetFingerprintAsNumPy(m)]).astype(np.uint8)

    def score_key(s):
        m = Chem.MolFromSmiles(s)
        return None if m is None else Chem.MolToSmiles(m)

    mod = types.ModuleType("casmi.chem")
    mod.raw_fingerprint, mod.score_key = raw_fingerprint, score_key
    return mod


def _tier(d: Path, n_rep: int = 12):
    """同じ構造の非正準な書き方を混ぜて数百行（質量順）。"""
    rng = np.random.default_rng(0)
    rows = []
    for s in BASE:
        m = Chem.MolFromSmiles(s)
        mass = Descriptors.ExactMolWt(m)
        for r in range(n_rep):
            smi = Chem.MolToSmiles(m, doRandom=True) if r % 3 else s
            rows.append((mass + rng.uniform(-1e-4, 1e-4), smi))
    rows.append((Descriptors.ExactMolWt(Chem.MolFromSmiles("CCO")), "not_a_smiles"))
    rows.sort(key=lambda r: r[0])
    enc = [s.encode("ascii") for _, s in rows]
    off = np.zeros(len(enc) + 1, np.int64)
    off[1:] = np.cumsum([len(e) for e in enc])
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "pc_mass.npy", np.array([m for m, _ in rows], np.float64))
    np.save(d / "pc_off.npy", off)
    np.save(d / "pc_smiles.npy", np.frombuffer(b"".join(enc), np.uint8))
    return len(rows)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    fake = _fake_chem()
    pkg = types.ModuleType("casmi")
    pkg.chem = fake
    monkeypatch.setitem(sys.modules, "casmi", pkg)
    monkeypatch.setitem(sys.modules, "casmi.chem", fake)
    tier_dir = tmp_path / "tier"
    n = _tier(tier_dir)
    assert n > 300
    bits = np.array(sorted(set(range(0, 4096, 3)) | set(range(4096, 5120, 5))), np.int32)
    rng = np.random.default_rng(1)
    targets, zf = {}, {}
    for i, s in enumerate(BASE[:23]):
        m = f"m{i:02d}"
        targets[m] = Descriptors.ExactMolWt(Chem.MolFromSmiles(s))
        zf[m] = rng.normal(size=len(bits)).astype(np.float32)
    targets["m_empty"] = 9999.0     # 窓が空
    zf["m_empty"] = rng.normal(size=len(bits)).astype(np.float32)
    pool_keys = {Chem.MolToSmiles(Chem.MolFromSmiles("CCN"))}
    return dict(tmp=tmp_path, tier_dir=tier_dir, targets=targets, zf=zf, bits=bits, pool_keys=pool_keys)


def _run(st, cache, workers):
    return PC.run_holdout("h", st["targets"], st["zf"], st["bits"], st["pool_keys"], workers=workers,
                          cache=cache, tier_dir=st["tier_dir"])


def test_workers2_identical_to_serial(setup):
    st = setup
    c1, c2 = st["tmp"] / "s/pubchem_fold0.json", st["tmp"] / "p/pubchem_fold0.json"
    r1 = _run(st, c1, 1)
    r2 = _run(st, c2, 2)
    assert list(r1) == list(st["targets"]) == list(r2)
    # 候補の列・順位・f·z の値まで完全一致（JSON にすると tuple と list の違いも消える）
    assert json.dumps(r1) == json.dumps(r2)
    assert c1.read_bytes() == c2.read_bytes()
    assert sum(len(v) for v in r1.values()) > 0 and r1["m_empty"] == []
    assert all(k not in st["pool_keys"] for v in r1.values() for _s, _f, k in v)
    # worker のキャッシュはまとめた後に消える
    assert sorted(p.name for p in c2.parent.iterdir()) == ["pubchem_fold0.json"]


def test_workers3_more_than_needed_and_resume_from_cache(setup):
    st = setup
    c1 = st["tmp"] / "s/pubchem_fold0.json"
    r1 = _run(st, c1, 1)
    # 半分だけ済んだキャッシュから再開しても、順も値も同じ
    c3 = st["tmp"] / "p3/pubchem_fold0.json"
    c3.parent.mkdir(parents=True)
    half = list(r1)[:10]
    c3.write_text(json.dumps({m: r1[m] for m in half}))
    r3 = _run(st, c3, 3)
    assert json.dumps(r3) == json.dumps(r1)
    # 分子より worker が多くても動く
    small = {m: st["targets"][m] for m in list(st["targets"])[:2]}
    c4 = st["tmp"] / "p4/pubchem_fold0.json"
    r4 = PC.run_holdout("h", small, st["zf"], st["bits"], st["pool_keys"], workers=4, cache=c4, tier_dir=st["tier_dir"])
    assert json.dumps(r4) == json.dumps({m: r1[m] for m in small})


def test_worker_cache_is_read_and_merged(setup):
    st = setup
    c = st["tmp"] / "p/pubchem_fold0.json"
    wps = PC.worker_cache_paths(c, 2)
    assert [p.name for p in wps] == ["pubchem_fold0.w0of2.json", "pubchem_fold0.w1of2.json"]
    mids = list(st["targets"])
    # 途中で止まった worker 0 のキャッシュ（印の値）が残っていれば、その分子は採点し直さずにまとめに入る
    wps[0].parent.mkdir(parents=True)
    wps[0].write_text(json.dumps({mids[0]: [["SENTINEL", 1.5, "K"]]}))
    r = _run(st, c, 2)
    assert r[mids[0]] == [["SENTINEL", 1.5, "K"]]
    assert list(r) == mids
    assert json.loads(c.read_text()) == r
    assert not wps[0].exists() and not wps[1].exists()


def test_worker_exception_stops_everything(setup, monkeypatch):
    st = setup
    bad = st["targets"]["m05"]
    orig = PC.probe

    def probe(tier, target, *a, **kw):
        if target == bad:
            raise ValueError("boom")
        return orig(tier, target, *a, **kw)

    monkeypatch.setattr(PC, "probe", probe)
    c = st["tmp"] / "p/pubchem_fold0.json"
    with pytest.raises(RuntimeError, match="失敗"):
        _run(st, c, 2)
    assert not c.exists()   # 欠けたまとめは書かない


def test_default_is_serial_path(setup, monkeypatch):
    """workers の既定は 1 で、fork を使わない（今までと同じ経路）。"""
    st = setup
    import multiprocessing as mp

    def no(*a, **kw):
        raise AssertionError("workers=1 で multiprocessing を使った")

    monkeypatch.setattr(mp, "get_context", no)
    c = st["tmp"] / "s/pubchem_fold0.json"
    r = PC.run_holdout("h", st["targets"], st["zf"], st["bits"], st["pool_keys"], cache=c, tier_dir=st["tier_dir"])
    assert list(r) == list(st["targets"])
