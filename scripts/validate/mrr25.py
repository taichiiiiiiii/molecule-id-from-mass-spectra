#!/usr/bin/env python3
"""
Enveda CASMI 2026 - Molecule ID From Mass Spectra
MRR@25 評価と提出形式チェック（docs/competition/overview.md の仕様どおり）

- 分子ごとに最大 25 候補。正解が最初に出た順位 r で 1/r、25 位以内に無ければ 0。
- 最初の正解だけが効く。
- 同一判定は公式の metric ノート `metric/casmi-mean-reciprocal-rank`（ipynb sha256 bf8535f0…09dd、
  写し `scripts/validate/casmi_metric_official.py`）と同じ手順（#197）:
  SMILES → 立体を外す（RemoveStereochemistry）→ 互変異性体の正準化 → InChIKey 先頭 14 文字。
  正準化に失敗した候補は不一致。正解はカンマ区切りで複数の SMILES を受け付ける（どれかに一致すれば正解）。
- 既知の差: metric は RDKit 2026.03.3 に固定している。手元の .venv（2026.09.1）や CI の版では
  互変異性体の正準化の結果が変わりうる（版を固定できない場合の既知の差。#197 に数を記録）。
- metric の「重原子の組成が違う候補は正準化を飛ばす」前段の絞り込みは点を変えない（InChIKey14 が同じなら
  重原子の組成も同じ）ので実装しない。

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


_TAUTOMER = None


@lru_cache(maxsize=None)
def to_key(structure: str) -> str | None:
    """SMILES または InChIKey を、比較用の InChIKey14 に変換する。変換できなければ None。

    SMILES は metric の `canonicalize_smiles_to_ik14` と同じ手順: 立体を外してから互変異性体を正準化する
    （立体中心が二重結合の位置を縛ると、立体あり・なしで別の互変異性体になるため）。正準化に失敗したら None。
    """
    global _TAUTOMER
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
    if _TAUTOMER is None:
        _TAUTOMER = rdMolStandardize.TautomerEnumerator()
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        mol = _TAUTOMER.Canonicalize(mol)
        key = Chem.MolToInchiKey(mol)
    except (ValueError, RuntimeError, TypeError):  # metric と同じ: 正準化の失敗は不一致
        return None
    return key[:14] if key else None


def truth_keys(truth: str | None) -> set[str]:
    """正解の文字列（カンマ区切りで複数の SMILES 可、metric の `accept_smiles.split(',')` と同じ）のキーの集合。"""
    if not truth:
        return set()
    return {k for k in (to_key(t) for t in truth.split(",") if t.strip()) if k}


def reciprocal_rank(truth_key: str | set[str] | None, candidates: Sequence[str], top_k: int = TOP_K) -> float:
    """1 分子分のスコア。candidates は順位順。truth_key は 1 つのキーかキーの集合（どれかに一致すれば正解）。"""
    keys = {truth_key} if isinstance(truth_key, str) else set(truth_key or ())
    keys.discard(None)
    if not keys:
        return 0.0
    for rank, cand in enumerate(candidates[:top_k], start=1):
        if to_key(cand) in keys:
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
        total += reciprocal_rank(truth_keys(t), cands, top_k)
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
    covered = sum(1 for m, t in truth.items() if reciprocal_rank(truth_keys(t), preds.get(m, [])) > 0)
    result = {"mrr25": mrr, "n": len(truth), "hit_at_25": covered / len(truth)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    logger.info("MRR@25=%.4f hit@25=%.3f n=%d -> %s", mrr, result["hit_at_25"], len(truth), args.output)


if __name__ == "__main__":
    main()
