"""Issue #54: sanpier キットの prepare.py への 2 点の変更と、リーク確認の数え方。"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/train"))
import sanpier_fold0 as sf  # noqa: E402
import make_sanpier_fold0_kernel as mk  # noqa: E402

# キット prepare.py の該当部分と同じ形（インデント・アンカー）の縮小版
MINI = '''import collections
ALIASES={'smiles':['smiles','canonical_smiles'],'mzs':['ms2_mzs','mzs']}
def run(frames, structure):
    reject=collections.Counter();rows=[];mapping={'smiles':'smiles'}
    for source_index,frame in enumerate(frames):
            accepted=0;total=0
            if True:
                for row in frame:
                    total+=1;s=str(row[mapping['smiles']])
                    try:
                        st=structure(s)
                        key,canonical,formula,mass,fp=st
                        rows.append(key)
                        accepted+=1
                    except (ValueError,TypeError):reject['invalid_spectrum']+=1
    return rows,reject
'''


def load(src):
    ns = {}
    exec(compile(src, "prepare_patched", "exec"), ns)
    return ns


def test_patch_drops_by_inchikey14_and_kitkey():
    ns = load(sf.patch_prepare_source(MINI))
    assert "normalized_smiles" in ns["ALIASES"]["smiles"]
    ns["DROP_IK"].update({"AAA"})
    ns["DROP_KIT"].update({"K_TWIN"})
    frame = [{"smiles": "s1", "inchikey14": "AAA"},   # 競技キーで落ちる
             {"smiles": "s2", "inchikey14": "BBB"},   # 互変異性体のキットキーで落ちる
             {"smiles": "s3", "inchikey14": "CCC"}]   # 残る
    kit = {"s1": "K1", "s2": "K_TWIN", "s3": "K3"}
    rows, reject = ns["run"]([frame], lambda s: (kit[s], s, "F", 1.0, None))
    assert rows == ["K3"]
    assert reject["dropped_heldout_inchikey14"] == 1 and reject["dropped_heldout_kitkey"] == 1
    assert ns["TRAIN_IK"] == {"CCC"}


def test_patch_only_source0_uses_inchikey14():
    ns = load(sf.patch_prepare_source(MINI))
    ns["DROP_IK"].add("AAA")
    rows, _ = ns["run"]([[], [{"smiles": "s1", "inchikey14": "AAA"}]], lambda s: ("K1", s, "F", 1.0, None))
    assert rows == ["K1"]  # 外部ソース（ここでは使わない）は競技キーでは落とさない


def test_patch_requires_unique_anchor():
    with pytest.raises(ValueError):
        sf.patch_prepare_source(MINI.replace("accepted+=1", "accepted+=1;accepted+=1"))
    with pytest.raises(ValueError):
        sf.patch_prepare_source(MINI.replace("'smiles':['smiles','canonical_smiles']", "'smiles':['smiles']"))


def test_drop_and_leak_audit():
    drop = sf.drop_inchikey14(["A", "B"], {"h1": ["B", "C"], "h2": ["D"]})
    assert drop == {"A", "B", "C", "D"}
    leak = sf.leak_audit({"A", "X", "D"}, [("A", "hold"), ("B", "np")], {"h1": ["B", "C"], "h2": ["D"]})
    assert leak == {"split_hold": 1, "split_np": 0, "h1": 0, "h2": 1, "total": 2}


def test_kernel_files_match_generator_and_compile():
    nb = mk.build_notebook()
    on_disk = json.loads((ROOT / "kernels/sanpier_fold0/notebook.ipynb").read_text())
    assert nb == on_disk
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")
    meta = json.loads((ROOT / "kernels/sanpier_fold0/kernel-metadata.json").read_text())
    assert meta == mk.metadata() and meta["enable_gpu"] and meta["is_private"]
    keys = json.loads((ROOT / "kernels/sanpier_fold0_drop" / mk.KEYS_FILE).read_text())
    assert sorted(keys) == sorted(mk.HOLDOUTS)
    assert {h: len(v) for h, v in keys.items()} == {"v3_c1": 200, "v3_c2": 200, "enveda_v1": 202, "v3_c2n": 200}
    if (ROOT / "data/holdouts/v3_c1/drop_keys.txt").exists():  # data/ は git に無いので CI では数だけ確かめる
        assert keys == mk.holdout_keys()
