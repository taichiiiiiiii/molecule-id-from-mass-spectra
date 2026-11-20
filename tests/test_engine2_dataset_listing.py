"""#165: push 前の照合（check_dataset_listing.py）と見張り（poll_kernel.py）、manifest の単体テスト。kaggle・gh は呼ばない。"""
import importlib.util
import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "engine2", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CL = _load("check_dataset_listing")
PK = _load("poll_kernel")
MK = _load("make_build_kernel")

MAN = dict(dataset="x/y", files=[dict(path="build_tables.py", size=10, sha256="a"),
                                 dict(path="drop_keys/v3_c1/drop_keys.txt", size=5, sha256="b"),
                                 dict(path="eng/pv.py", size=7, sha256="c")])
HDR = "name,size,creationDate\r\n"


def _csv(rows):
    return HDR + "".join(f"{n},{s},2026-11-20 03:00:00.000000\r\n" for n, s in rows)


# ---- 照合 ----
def test_listing_all_present_passes():
    lst = CL.parse_listing(_csv([("build_tables.py", 10), ("drop_keys/v3_c1/drop_keys.txt", 5), ("eng/pv.py", 7),
                                 ("manifest.json", 99)]))
    r = CL.compare(MAN, lst)
    assert r["ok"] and r["missing"] == [] and r["n_present"] == 3 and r["extra_in_listing"] == ["manifest.json"]


def test_listing_missing_subfolders_detected():
    # 前回（#165 b_run1 v1）と同じ形: 上の階層のファイルだけ
    lst = CL.parse_listing(_csv([("build_tables.py", 10)]))
    r = CL.compare(MAN, lst)
    assert not r["ok"] and r["missing"] == ["drop_keys/v3_c1/drop_keys.txt", "eng/pv.py"]


def test_listing_size_mismatch_detected():
    lst = CL.parse_listing(_csv([("build_tables.py", 11), ("drop_keys/v3_c1/drop_keys.txt", 5), ("eng/pv.py", 7)]))
    r = CL.compare(MAN, lst)
    assert not r["ok"] and r["size_mismatch"][0]["path"] == "build_tables.py"


def test_basename_only_listing_is_missing():
    lst = CL.parse_listing(_csv([("build_tables.py", 10), ("drop_keys.txt", 5), ("pv.py", 7)]))
    assert not CL.compare(MAN, lst)["ok"]


@pytest.mark.parametrize("text", ["", "   \n", "Authentication required\n", "404 Client Error: Not Found\n",
                                  "name,size\nfoo,1\n", HDR + "a,1,x\nNext Page Token = abc\n"])
def test_listing_unreadable(text):
    with pytest.raises(CL.ListingUnreadable):
        CL.parse_listing(text)


def test_listing_full_page_is_unreadable():
    with pytest.raises(CL.ListingUnreadable):
        CL.parse_listing(_csv([(f"f{i}.py", 1) for i in range(100)]))


def test_main_rc(tmp_path):
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(MAN))
    ok = tmp_path / "ok.csv"
    ok.write_text(_csv([("build_tables.py", 10), ("drop_keys/v3_c1/drop_keys.txt", 5), ("eng/pv.py", 7)]))
    ng = tmp_path / "ng.csv"
    ng.write_text(_csv([("build_tables.py", 10)]))
    bad = tmp_path / "bad.csv"
    bad.write_text("Authentication required")
    assert CL.main(["--manifest", str(mp), "--listing-file", str(ok)]) == 0
    assert CL.main(["--manifest", str(mp), "--listing-file", str(ng)]) == 1
    assert CL.main(["--manifest", str(mp), "--listing-file", str(bad)]) == 2


def test_empty_manifest_is_not_ok():
    assert not CL.compare(dict(files=[]), {})["ok"]


# ---- manifest ----
def test_write_manifest(tmp_path):
    (tmp_path / "eng").mkdir()
    (tmp_path / "eng" / "pv.py").write_bytes(b"abc")
    (tmp_path / "eng" / "__pycache__").mkdir()
    (tmp_path / "eng" / "__pycache__" / "pv.pyc").write_bytes(b"x")
    (tmp_path / "build_tables.py").write_bytes(b"hello")
    (tmp_path / "dataset-metadata.json").write_text("{}")
    man = MK.write_manifest(str(tmp_path))
    assert [f["path"] for f in man["files"]] == ["build_tables.py", "eng/pv.py"]
    assert man["files"][1]["size"] == 3
    assert man["files"][1]["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert json.load(open(tmp_path / "manifest.json")) == man


# ---- 見張り ----
@pytest.mark.parametrize("text,want", [
    ('taichiiiii/k has status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
    ('taichiiiii/k has status "KernelWorkerStatus.ERROR"', "ERROR"),
    ('taichiiiii/k has status "KernelWorkerStatus.CANCEL_REQUESTED"', "CANCEL"),
    ('taichiiiii/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "CANCEL"),
    ('taichiiiii/k has status "KernelWorkerStatus.RUNNING"', "RUNNING"),
    ('taichiiiii/k has status "KernelWorkerStatus.QUEUED"', "QUEUED"),
    ('taichiiiii/k has status "complete"', "COMPLETE"),
    ("", "UNREADABLE"),
    ("   ", "UNREADABLE"),
    ("Authentication required. Please run kaggle auth login", "UNREADABLE"),
    ("401 Client Error: Unauthorized", "UNREADABLE"),
    ("something unexpected", "UNREADABLE"),
    ('taichiiiii/k has status "KernelWorkerStatus.WEIRD"', "UNREADABLE"),
])
def test_classify(text, want):
    assert PK.classify(text) == want


def test_log_tail_json_stream():
    arr = [dict(stream_name="stdout", time=1, data=f"line{i}\n") for i in range(50)]
    t = PK.log_tail(json.dumps(arr))
    assert t.split("\n") == [f"line{i}" for i in range(20, 50)]


def test_log_tail_plain():
    assert PK.log_tail("a\nb\nc\n", n=2) == "b\nc"


def test_comment_body_error_written_to_file(tmp_path):
    body = PK.comment_body("u/k", "ERROR", 'u/k has status "KernelWorkerStatus.ERROR"', "2026-11-20 13:00:00",
                           ["t0 RUNNING", "t1 ERROR"], tail="AssertionError: x", files=["k.log"], issue=165)
    p = PK.write_body(str(tmp_path / "c.md"), body)
    s = open(p).read()
    assert "**ERROR**" in s and "AssertionError: x" in s and "最後 30 行" in s and "#165" in s and "`k.log`" in s


def test_comment_body_complete_and_unreadable():
    c = PK.comment_body("u/k", "COMPLETE", "x", "t", ["t RUNNING"], files=[], check='{"a": 1}', issue=7)
    assert "**COMPLETE**" in c and '{"a": 1}' in c and "最後 30 行" not in c
    u = PK.comment_body("u/k", "UNREADABLE", "", "t", ["a", "b", "c"], issue=7)
    assert "3 回続けて読めなかった" in u


def test_issue_is_required():
    with pytest.raises(SystemExit):
        PK.main(["--kernel", "u/k", "--log", "/dev/null"])
