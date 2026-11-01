#!/usr/bin/env python3
"""
Envela CASMI 2026 - Molecule ID From Mass Spectra
MRR@25 検証スクリプト
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/validation.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def calculate_mrr25(y_true: list, y_pred: list, top_k: int = 25) -> float:
    """
    MRR@25 計算
    y_true: 正解リスト（分子ごとに1つの正解構造）
    y_pred: 予測リスト（分子ごとに25候補）
    """
    logger.info("=== MRR@25 計算開始 ===")
    logger.info(f"候補数: {top_k}")

    # TODO: 実際の評価ロジック
    # - 正解が上位に出現した順位を計算
    # - MRR = 1/rank の平均
    # - 候補にない場合は 0

    mrr = 0.0
    logger.info("=== MRR@25 計算完了 ===")
    return mrr


def evaluate_submission(y_true: str, y_pred: str, output: str) -> float:
    """
    提出物の評価
    """
    logger.info(f"y_true: {y_true}")
    logger.info(f"y_pred: {y_pred}")
    logger.info(f"output: {output}")
    return 0.0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRR@25 検証")
    parser.add_argument("--y_true", required=True, help="正解データのパス")
    parser.add_argument("--y_pred", required=True, help="予測データのパス")
    parser.add_argument("--output", default="results/validation.json", help="出力ディレクトリ")
    args = parser.parse_args()

    mrr = calculate_mrr25(args.y_true, args.y_pred)
    result = {"mrr25": mrr}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"検証結果を保存しました: {args.output}")
