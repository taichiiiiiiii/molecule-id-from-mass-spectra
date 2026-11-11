"""msg_overlap の単体テスト（合成の tsv・drop_keys・split）。"""
import importlib.util
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("msg_overlap", os.path.join(HERE, "..", "scripts", "research", "msg_overlap.py"))
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def _setup(tmp_path):
    k = lambda i: f"K{i:013d}"   # 14 文字
    rows = [(k(i), "train") for i in range(0, 50)] + [(k(i), "val") for i in range(50, 60)] \
        + [(k(i), "test") for i in range(60, 70)] + [(k(1) + "-UHFFFAOYSA-N", "train")]
    msg = tmp_path / "msg.tsv"
    pd.DataFrame(dict(identifier=range(len(rows)), inchikey=[r[0] for r in rows], smiles="C",
                      fold=[r[1] for r in rows])).to_csv(msg, sep="\t", index=False)
    hd = tmp_path / "holdouts"
    sizes = dict(v3_c1=(0, 150), v3_c2=(100, 300), enveda_v1=(55, 205), v3_c2n=(40, 160))
    for h, (a, b) in sizes.items():
        (hd / h).mkdir(parents=True)
        (hd / h / "drop_keys.txt").write_text("\n".join(k(i) for i in range(a, b)) + "\n")
    sp = tmp_path / "split.parquet"
    pd.DataFrame(dict(inchikey14=[k(i) for i in (0, 1, 65, 999)], fold=["hold", "hold", "np", "hold"])).to_parquet(sp)
    return str(msg), str(hd), str(sp), k


def test_read_msg_truncates_to_14(tmp_path):
    msg, _, _, k = _setup(tmp_path)
    folds, how, rows = M.read_msg(msg, chunksize=7)
    assert how == "inchikey[:14]" and rows == 71
    assert len(folds["train"]) == 50 and len(folds["val"]) == 10 and len(folds["test"]) == 10
    assert k(1) in folds["train"]


def test_main_overlap_and_rule(tmp_path):
    msg, hd, sp, k = _setup(tmp_path)
    out = tmp_path / "o.json"
    M.main(["--msg", msg, "--holdout-dir", hd, "--split", sp, "--out", str(out)])
    d = json.load(open(out))
    ov = d["union"]["overlap"]
    assert ov["v3_c1"]["train"]["n"] == 50 and ov["v3_c1"]["val"]["n"] == 10 and ov["v3_c1"]["test"]["n"] == 10
    assert ov["split_hold"]["n"] == 3 and ov["split_hold"]["any_fold"]["n"] == 2
    assert ov["split_np"]["test"]["n"] == 1
    r = d["remaining_rule"]
    assert r["v3_c1"] == dict(n=150, overlap=60, remain=90, measurable=False)   # train ∪ val = 0..59
    assert r["v3_c2"]["remain"] == 200 and r["v3_c2"]["measurable"]
    assert r["enveda_v1"]["overlap"] == 5                                        # 55..59
    assert d["remaining_conservative"]["v3_c1"]["remain"] == 80                  # test も除く
    assert d["all_measurable_rule"] is False
