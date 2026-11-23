"""#152 speed2: 候補プールの指紋 PFP を packbits のまま持ち、取り出した行だけ unpack する（自前のコード）。

上流 upstream/train_fp3.py（変えない）の 20 行は
    PFP = np.unpackbits(np.load(f'{pooldir}/pool_fp.npy'), axis=1)[:, :self.nbits]
で (711,705 × 6,930) を 1 バイト/ビット（約 4.93 GB）で CPU に持つ。使い道は行の取り出し（上流 72・73 行、
train_s2_resumable.py の D.PFP[cand]）だけなので、ここでは packed (N, ceil(nbits/8)) の uint8 のまま持ち、
`PFP[idx]` で該当行だけを上流と同じ式（unpackbits(axis=1)[:, :nbits]）で戻す。返す値は上流の
`torch.from_numpy(np.ascontiguousarray(unpacked))[idx]` と dtype（uint8）・形（idx.shape + (nbits,)）・値が同じ。

numpy の部分（rows_np）は torch 無しでも使える（CI のテスト用）。torch は __getitem__ の中でだけ読む。
"""
import numpy as np


class PackedPFP:
    """packed の uint8 行列を持ち、行の取り出しで unpack する入れ物。`.device` は cpu（上流も mps・cpu では CPU）。"""

    def __init__(self, packed, nbits):
        packed = np.asarray(packed)
        if packed.dtype != np.uint8 or packed.ndim != 2:
            raise ValueError(f'packed は 2 次元の uint8 が要る: {packed.dtype} {packed.shape}')
        if not (0 < nbits <= packed.shape[1] * 8):
            raise ValueError(f'nbits {nbits} が packed の幅 {packed.shape[1]} バイトに合わない')
        self.packed = packed
        self.nbits = int(nbits)
        self.shape = (packed.shape[0], self.nbits)

    def __len__(self):
        return self.packed.shape[0]

    @property
    def device(self):
        import torch
        return torch.device('cpu')

    def rows_np(self, idx):
        """idx（整数の配列、形は任意）の行を unpack して idx.shape + (nbits,) の uint8 で返す。"""
        idx = np.asarray(idx)
        if idx.dtype.kind not in 'iu':
            raise TypeError(f'idx は整数の配列: {idx.dtype}')
        flat = idx.reshape(-1)
        rows = np.unpackbits(self.packed[flat], axis=1)[:, :self.nbits]
        return np.ascontiguousarray(rows).reshape(idx.shape + (self.nbits,))

    def __getitem__(self, idx):
        import torch
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        return torch.from_numpy(self.rows_np(idx))
