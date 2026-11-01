#!/usr/bin/env python3
"""
Enveda CASMI 2026 - Molecule ID From Mass Spectra
提出物（submission.csv）生成スクリプト
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/submit.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def generate_submission(predictions: str, sample_submission: str, output: str) -> None:
    """
    提出物生成
    - predictions: モデル予測結果
    - sample_submission: サンプル提出ファイル
    - output: 出力ファイルパス
    """
    logger.info("=== 提出物生成開始 ===")
    logger.info(f"predictions: {predictions}")
    logger.info(f"sample_submission: {sample_submission}")
    logger.info(f"output: {output}")

    # TODO: 実際の生成ロジック
    # - 予測結果をサンプル提出形式に整形
    # - molecule_id, smiles 形式に変換
    # - 最大25候補をセミコロン区切りで結合

    logger.info("=== 提出物生成完了 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="提出物生成")
    parser.add_argument("--predictions", required=True, help="予測結果のパス")
    parser.add_argument("--sample_submission", required=True, help="サンプル提出ファイルのパス")
    parser.add_argument("--output", default="submission.csv", help="出力ファイルパス")
    args = parser.parse_args()

    generate_submission(args.predictions, args.sample_submission, args.output)
