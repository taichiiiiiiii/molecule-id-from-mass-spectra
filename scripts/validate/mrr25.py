#!/usr/bin/env python3
"""
Enveda CASMI 2026 - Molecule ID From Mass Spectra
MRR@25 評価と提出形式チェック（docs/competition/overview.md の仕様どおり）

- 分子ごとに最大 25 候補。正解が最初に出た順位 r で 1/r、25 位以内に無ければ 0。
- 最初の正解だけが効く。
- 同一判定は InChIKey 先頭 14 文字（connectivity）。SMILES は RDKit で
  tautomer 正準化してから InChIKey14 に変換する。

使い方:
    python scripts/validate/mrr25.py --truth truth.csv --pred submission.csv
    truth.csv は `molecule_id` と `smiles` または `inchikey` の列を持つ。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

TOP_K = 25
SEP = ";"
_INCHIKEY = re.compile(r"^[A-Z]{14}(-[A-Z]{10}-[A-Z])?$")


class SubmissionError(ValueError):
    """Kaggle が拒否する形式の提出物。"""


@lru_cache(maxsize=None)
def to_key(structure: str) -> str | None:
    """SMILES または InChIKey を、比較用の InChIKey14 に変換する。変換できなければ None。"""
    s = structure.strip()
    if not s:
        return None
    if _INCHIKEY.match(s):
        return s[:14]
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as e:  # pragma: no cover - CI では rdkit を入れる
        raise RuntimeError("SMILES の比較には rdkit が必要です") from e
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:  # 正準化に失敗しても元の分子で比較する
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


def reciprocal_rank(truth_key: str | None, candidates: Sequence[str], top_k: int = TOP_K) -> float:
    """1 分子分のスコア。candidates は順位順。"""
    if truth_key is None:
        return 0.0
    for rank, cand in enumerate(candidates[:top_k], start=1):
        if to_key(cand) == truth_key:
            return 1.0 / rank
    return 0.0


def calculate_mrr25(
    truth: Mapping[str, str],
    preds: Mapping[str, Sequence[str]],
    top_k: int = TOP_K,
) -> float:
    """
    truth: molecule_id -> 正解構造（SMILES か InChIKey）
    preds: molecule_id -> 候補構造のリスト（順位順）
    予測の無い分子は 0 点。truth に無い molecule_id は無視する。
    """
    if not truth:
        raise ValueError("truth が空です")
    total = 0.0
    for mid, t in truth.items():
        cands = preds.get(mid, [])
        if len(cands) > top_k:
            raise SubmissionError(f"{mid}: 候補が {len(cands)} 件（上限 {top_k}）")
        total += reciprocal_rank(to_key(t), cands, top_k)
    return total / len(truth)


def parse_candidates(cell: str) -> list[str]:
    return [c.strip() for c in (cell or "").split(SEP) if c.strip()]


def read_submission(path: str | Path) -> dict[str, list[str]]:
    """submission.csv を読み、形式を確かめて molecule_id -> 候補 を返す。"""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return validate_submission(rows)


def validate_submission(
    rows: Sequence[Sequence[str]],
    expected_ids: set[str] | None = None,
    top_k: int = TOP_K,
) -> dict[str, list[str]]:
    """rows はヘッダーを含む CSV の行。Kaggle が拒否する形式なら SubmissionError。"""
    if not rows or [h.strip() for h in rows[0]] != ["molecule_id", "smiles"]:
        raise SubmissionError("ヘッダーは `molecule_id,smiles` である必要があります")
    out: dict[str, list[str]] = {}
    for i, row in enumerate(rows[1:], start=2):
        if len(row) != 2:
            raise SubmissionError(f"{i} 行目: 列数が {len(row)}（2 列が必要）")
        mid = row[0].strip()
        if not mid:
            raise SubmissionError(f"{i} 行目: molecule_id が空です")
        if mid in out:
            raise SubmissionError(f"{i} 行目: molecule_id {mid} が重複しています")
        cands = parse_candidates(row[1])
        if len(cands) > top_k:
            raise SubmissionError(f"{i} 行目: 候補が {len(cands)} 件（上限 {top_k}）")
        out[mid] = cands
    if expected_ids is not None:
        missing = expected_ids - out.keys()
        extra = out.keys() - expected_ids
        if missing or extra:
            raise SubmissionError(f"molecule_id の不一致: 不足 {len(missing)} 件、余分 {len(extra)} 件")
    return out


def read_truth(path: str | Path) -> dict[str, str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        col = "inchikey" if "inchikey" in (reader.fieldnames or []) else "smiles"
        return {r["molecule_id"].strip(): r[col] for r in reader}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="MRR@25 評価")
    parser.add_argument("--truth", required=True, help="正解 CSV（molecule_id と smiles/inchikey）")
    parser.add_argument("--pred", required=True, help="submission.csv")
    parser.add_argument("--output", default="results/validation.json")
    args = parser.parse_args()

    truth = read_truth(args.truth)
    preds = read_submission(args.pred)
    mrr = calculate_mrr25(truth, preds)
    covered = sum(1 for m, t in truth.items() if reciprocal_rank(to_key(t), preds.get(m, [])) > 0)
    result = {"mrr25": mrr, "n": len(truth), "hit_at_25": covered / len(truth)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    logger.info("MRR@25=%.4f hit@25=%.3f n=%d -> %s", mrr, result["hit_at_25"], len(truth), args.output)


if __name__ == "__main__":
    main()
