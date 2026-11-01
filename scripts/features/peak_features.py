#!/usr/bin/env python3
"""
Envela CASMI 2026 - Molecule ID From Mass Spectra
ピーク特徴量抽出スクリプト
"""
import json
import logging
import sys
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/peak_features.log", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_peak_features(train_parquet: str, test_parquet: str, output_dir: str) -> None:
    """
    ピーク特徴量抽出
    - スペクトルのピーク強度、質量、幅などを抽出
    """
    logger.info("=== ピーク特徴量抽出開始 ===")
    logger.info(f"train: {train_parquet}")
    logger.info(f"test: {test_parquet}")
    logger.info(f"output: {output_dir}")

    # TODO: 実際の特徴量抽出ロジック
    # - スペクトルのピーク強度の分布
    # - 質量の最大値、最小値、平均値
    # - ピークの数、幅、間隔など
    # - マルチスペクトル集約（平均、最大、最小）

    # Example placeholder: create dummy output for now
    import pandas as pd

    output_path = Path(output_dir) / "peak_features.parquet"
    # TODO: 実際の特徴量を計算して保存
    # df_peak = pd.DataFrame(...)
    # df_peak.to_parquet(output_path, index=False)
    logger.info(f"ピーク特徴量を抽出しました: {output_path}")
    logger.info("=== ピーク特徴量抽出完了 ===")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ピーク特徴量抽出")
    parser.add_argument("--train", required=True, help="train.parquet のパス")
    parser.add_argument("--test", required=True, help="test.parquet のパス")
    parser.add_argument("--output", default="features/peak", help="出力ディレクトリ")
    args = parser.parse_args()

    extract_peak_features(args.train, args.test, args.output)
