#!/usr/bin/env python3
"""
Envela CASMI 2026 - Molecule ID From Mass Spectra
ベースラインモデル学習スクリプト
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/training.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def train_baseline(train_data: str, output_dir: str) -> None:
    """
    ベースラインモデル学習
    - ライブラリ検索ベースライン (class 1)
    - 参考スペクトルとの類似度計算
    - 類似度上位 N 個の構造を候補として返す
    """
    logger.info("=== ベースラインモデル学習開始 ===")
    logger.info(f"train: {train_data}")
    logger.info(f"output: {output_dir}")

    # TODO: 実際の学習ロジック
    # - ライブラリ検索
    # - 類似度計算 (コサイン類似度など)
    # - 候補ランキング
    # - モデル保存

    logger.info("=== ベースラインモデル学習完了 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ベースラインモデル学習")
    parser.add_argument("--train", required=True, help="学習用データのパス")
    parser.add_argument("--output", default="models/baseline", help="出力ディレクトリ")
    args = parser.parse_args()

    train_baseline(args.train, args.output)
