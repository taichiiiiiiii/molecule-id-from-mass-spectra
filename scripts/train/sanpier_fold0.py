"""sanpier キット（casmi-spectrum-model-v1）を、検証の正解構造を外して学習するための部品（Issue #54）。

キットへの変更は 2 点だけ:
  1. SMILES 列の別名に normalized_smiles を足す（キットの説明文にある互換の直し、v80 と同じ 1 行）
  2. 行選択: 競技の inchikey14 が DROP_IK に入る行と、キットの構造キーが DROP_KIT に入る行を落とす

カーネル（kernels/sanpier_fold0）はこのファイルをそのまま埋め込んで使う。
"""
import json

ALIAS_OLD = "'smiles':['smiles','canonical_smiles']"
ALIAS_NEW = "'smiles':['smiles','canonical_smiles','normalized_smiles']"

# 行ごとの最初の処理。ここで競技の inchikey14 で落とす。
ROW_OLD = "total+=1;s=str(row[mapping['smiles']])"
ROW_NEW = ("total+=1;s=str(row[mapping['smiles']]);_ik=str(row.get('inchikey14',''))\n"
           "                    if source_index==0 and _ik in DROP_IK:reject['dropped_heldout_inchikey14']+=1;continue")
# キットの構造キーができた直後。ここで互変異性体のキーで落とす。
KEY_OLD = "key,canonical,formula,mass,fp=st"
KEY_NEW = ("key,canonical,formula,mass,fp=st\n"
           "                        if key in DROP_KIT:reject['dropped_heldout_kitkey']+=1;continue")
ACC_OLD = "accepted+=1"
ACC_NEW = "accepted+=1;TRAIN_IK.add(_ik)"
HEADER = "\nDROP_IK=set();DROP_KIT=set();TRAIN_IK=set()\n"


def patch_prepare_source(src: str) -> str:
    """キットの prepare.py の文字列に 2 点の変更を当てる。各アンカーはちょうど 1 回現れることを確かめる。"""
    for old in (ALIAS_OLD, ROW_OLD, KEY_OLD, ACC_OLD):
        n = src.count(old)
        if n != 1:
            raise ValueError(f"anchor must appear exactly once: {old!r} ({n})")
    out = src.replace(ALIAS_OLD, ALIAS_NEW).replace(ROW_OLD, ROW_NEW).replace(KEY_OLD, KEY_NEW).replace(ACC_OLD, ACC_NEW)
    # モジュールの大域変数として置く（import 行の後、ALIASES の前）。
    i = out.index("ALIASES=")
    return out[:i] + HEADER.lstrip("\n") + out[i:]


def drop_inchikey14(split_keys, holdout_keys: dict) -> set:
    """split.parquet の全行の inchikey14 ∪ 各検証の正解 inchikey14。"""
    out = set(map(str, split_keys))
    for keys in holdout_keys.values():
        out |= set(map(str, keys))
    return out


def leak_audit(train_ik, split_frame_rows, holdout_keys: dict) -> dict:
    """学習に使った inchikey14 と、外すべき各集合の重なり件数。split_frame_rows は (inchikey14, fold) の並び。"""
    train_ik = set(map(str, train_ik))
    folds = {}
    for ik, fold in split_frame_rows:
        folds.setdefault(str(fold), set()).add(str(ik))
    out = {f"split_{f}": len(train_ik & ks) for f, ks in sorted(folds.items())}
    for name, keys in sorted(holdout_keys.items()):
        out[name] = len(train_ik & set(map(str, keys)))
    out["total"] = sum(out.values())
    return out


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps({"alias": ALIAS_NEW}))
