"""Issue #127: DreaMS の ckpt から学習の状態を外すスクリプト（scripts/ship/strip_dreams_ckpt.py）。

torch は CI に無いので、純粋な関数（キーの選び方・大きさの数え方・出力先の守り・md5/sha256）を小さな偽の ckpt で確かめる。
torch があるときだけ、torch.save した偽の ckpt で保存・読み直しまでを確かめる。
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/ship"))

import strip_dreams_ckpt as S  # noqa: E402


def fake_ckpt():
    sd = {"ff_fourier.0.weight": np.zeros((10, 10), np.float32),          # 400 B
          "transformer_encoder.layers.0.w": np.zeros((250, 1000), np.float32),   # 1,000,000 B
          "transformer_encoder.layers.0.b": np.zeros(1000, np.float32),    # 4,000 B
          "ff_out.weight": np.zeros((5, 5), np.float64)}                    # 200 B
    return {"state_dict": sd, "hyper_parameters": {"args": {"d_model": 8}, "spec_preproc": "x"},
            "optimizer_states": [{"state": {0: {"exp_avg": np.zeros(500_000, np.float32)}}}],
            "lr_schedulers": [{}], "callbacks": {}, "loops": {}, "hparams_name": "kwargs",
            "global_step": 71500, "epoch": 3, "pytorch-lightning_version": "1.9"}


def test_strip_keeps_forward_keys_and_drops_optimizer():
    ck = fake_ckpt()
    out, keys = S.strip_checkpoint(ck)
    assert set(out) == {"state_dict", "hyper_parameters", "global_step", "epoch", "pytorch-lightning_version"}
    assert keys["dropped"] == ["callbacks", "hparams_name", "loops", "lr_schedulers", "optimizer_states"]
    assert out["state_dict"] is ck["state_dict"]      # 値はそのまま（写さない）
    assert "optimizer_states" in ck                   # 入力の dict は変えない


def test_strip_requires_state_dict_and_hparams():
    ck = fake_ckpt()
    del ck["hyper_parameters"]
    with pytest.raises(KeyError):
        S.strip_checkpoint(ck)
    with pytest.raises(TypeError):
        S.strip_checkpoint([1, 2])


def test_strip_optional_keys_may_be_absent():
    out, keys = S.strip_checkpoint({"state_dict": {}, "hyper_parameters": {}, "optimizer_states": []})
    assert keys["kept"] == ["state_dict", "hyper_parameters"] and keys["dropped"] == ["optimizer_states"]


def test_size_by_prefix_counts_bytes_per_top_name():
    r = S.size_by_prefix(fake_ckpt()["state_dict"])
    assert r["total_bytes"] == 400 + 1_000_000 + 4_000 + 200
    assert r["per_prefix_mb"] == {"ff_fourier": 0.0, "transformer_encoder": 1.0, "ff_out": 0.0}
    assert r["n_tensors"] == 4


def test_dropped_sizes_walk_nested_containers():
    ck = fake_ckpt()
    n = sum(S.tensor_nbytes(t) for t in S._iter_tensors(ck["optimizer_states"]))
    assert n == 2_000_000


def test_tensor_nbytes_prefers_numel_times_element_size():
    class T:
        def numel(self):
            return 7

        def element_size(self):
            return 2
    assert S.tensor_nbytes(T()) == 14
    assert S.tensor_nbytes(np.zeros(3, np.int16)) == 6
    assert S.tensor_nbytes("not a tensor") == 0


def test_check_out_path_refuses_input_folder(tmp_path):
    src = tmp_path / "orig" / "data" / "ext" / "dreams"
    src.mkdir(parents=True)
    ck = src / "ssl_model.ckpt"
    ck.write_bytes(b"x")
    with pytest.raises(ValueError):
        S.check_out_path(src / "out.pt", ck)
    with pytest.raises(ValueError):
        S.check_out_path(src / "sub" / "out.pt", ck)
    with pytest.raises(ValueError):
        S.check_out_path(ck, ck)
    # 入力のフォルダへのシンボリックリンク経由でも拒む（クローンの data/ext は元のフォルダへのリンク）
    link = tmp_path / "clone_ext"
    link.symlink_to(src)
    with pytest.raises(ValueError):
        S.check_out_path(link / "out.pt", ck)
    S.check_out_path(tmp_path / "scratch" / "out.pt", ck)   # 別のフォルダなら通る


def test_file_digest_matches_hashlib(tmp_path):
    p = tmp_path / "a.bin"
    data = bytes(range(256)) * 1000
    p.write_bytes(data)
    assert S.file_digest(p, "md5", chunk=1000) == (hashlib.md5(data).hexdigest(), len(data))
    assert S.file_digest(p, "sha256") == (hashlib.sha256(data).hexdigest(), len(data))


def test_expected_constants_match_issue_109():
    assert S.CKPT_MD5 == "4deee91dd4486c2b6cbcde1f200b8f50" and S.CKPT_SIZE == 1_392_710_212


def test_torch_roundtrip_with_fake_ckpt(tmp_path):
    torch = pytest.importorskip("torch")
    ck = {"state_dict": {"a.w": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b.w": torch.ones(4)},
          "hyper_parameters": {"args": {"k": 1}}, "optimizer_states": [{"m": torch.zeros(100)}], "global_step": 5}
    out, _ = S.strip_checkpoint(ck)
    p = tmp_path / "s.pt"
    torch.save(out, p)
    back = torch.load(p, map_location="cpu", weights_only=False)
    v = S.verify_same(ck, back)
    assert v["ok"] and v["tensors_differing"] == 0
    back["state_dict"]["b.w"] = torch.zeros(4)
    assert not S.verify_same(ck, back)["ok"]
