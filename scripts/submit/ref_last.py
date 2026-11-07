"""
参照スペクトルのある候補を後ろに回す（Issue #22）。

根拠: 競技のクラス 2/3 の正解は「公開スペクトルが無い」構造。一方、候補プールの大部分は学習データ（= 公開スペクトル）
由来で、その候補は自分の参照スペクトルを持つ。ライブラリ一致が弱い（library_max < LIB_TAU）分子では、
参照を持つのに一致しなかった候補は正解である見込みが低い。そこで、そうした分子に限り、
候補リストを「学習データに参照の無い候補 → 参照のある候補」の順に安定分割する（各グループ内の順序は保つ）。

LIB_TAU = 0.7 は 0.479 版パイプラインの既存定数（ALPHA_LIB_TAU）。新しい調整値は入れない。
このファイルは提出ノートブックにソースごと埋め込まれる（make_ref_last_kernel.py）。
"""
from __future__ import annotations

LIB_TAU = 0.7


def ref_last(candidates, keys_of, ref_keys, lib_max, tau=LIB_TAU):
    """candidates: 順位順の SMILES。keys_of(smiles) -> そのキー集合（生 InChIKey14 と採点用キーなど）。
    ref_keys: 学習データの InChIKey14 集合。lib_max: この分子のライブラリ一致の最大値（None は 0 扱い）。
    戻り値: (新しいリスト, 変えたか)"""
    if (lib_max or 0.0) >= tau or len(candidates) < 2:
        return list(candidates), False
    is_ref = [bool(keys_of(s) & ref_keys) for s in candidates]
    new = [s for s, r in zip(candidates, is_ref) if not r] + [s for s, r in zip(candidates, is_ref) if r]
    return new, new != list(candidates)


def make_keys_of():
    """RDKit で SMILES -> {生 InChIKey14, 互変異性体正準化後の InChIKey14}。失敗は空集合。"""
    from functools import lru_cache
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    te = rdMolStandardize.TautomerEnumerator()

    @lru_cache(maxsize=None)
    def keys_of(smi):
        out = set()
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                return frozenset()
            out.add(Chem.MolToInchiKey(m)[:14])
            out.add(Chem.MolToInchiKey(te.Canonicalize(m))[:14])
        except Exception:
            pass
        return frozenset(k for k in out if k)
    return keys_of
