#!/usr/bin/env python3
"""
Enveda CASMI 2026 - Molecule ID From Mass Spectra
改善ループ マスターピアニスト

改善ループ全体を実行するエントリーポイント:
  1. データ取得 (bash scripts/download.sh)
  2. 特徴量抽出 (peak, molecule, fusion)
  3. モデル学習 (baseline)
  4. 検証 (MRR@25)
  5. 提出物生成 (submission.csv)
  6. Kaggle提出
  7. 誤差分析
  8. 反復

使用方法:
  python scripts/run_improvement_loop.py        # フルパイプライン
  python scripts/run_improvement_loop.py --stage features
  python scripts/run_improvement_loop.py --stage train
  python scripts/run_improvement_loop.py --stage validate
  python scripts/run_improvement_loop.py --stage submit
  python scripts/run_improvement_loop.py --stage all
"""
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/improvement_loop.log", mode="a"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_command(cmd, check=True):
    """コマンドを実行"""
    logger.info(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"コマンド失敗 (exit={result.returncode})")
        logger.error(f"stderr: {result.stderr}")
        if check:
            sys.exit(result.returncode)
    else:
        logger.info("成功")
    return result


def step1_data_acquisition():
    """ステップ1: データ取得"""
    logger.info("=" * 60)
    logger.info("STEP 1: データ取得")
    logger.info("=" * 60)

    # Kaggle CLI ウォレット確認（追加でチェック）
    result = subprocess.run(
        ["kaggle", "competitions", "list", "--page-size", "200"],
        capture_output=True, text=True,
    )
    if "enveda" not in result.stdout.lower():
        logger.error("Kaggle CLI ウォレットにコンペ未登録です")
        logger.error("Kaggleウェブサイトで登録してください")
        logger.error("再ログイン: kaggle auth login --force")
        sys.exit(1)

    run_command(["bash", "scripts/download.sh"])


def step2_feature_engineering():
    """ステップ2: 特徴量抽出"""
    logger.info("=" * 60)
    logger.info("STEP 2: 特徴量抽出")
    logger.info("=" * 60)

    run_command(["python", "scripts/features/peak_features.py",
                 "--train", "data/train.parquet",
                 "--test", "data/test.parquet",
                 "--output", "features/peak"])
    run_command(["python", "scripts/features/molecule_features.py",
                 "--train", "data/train.parquet",
                 "--test", "data/test.parquet",
                 "--output", "features/molecule"])
    run_command(["python", "scripts/features/fusion_features.py",
                 "--train", "data/train.parquet",
                 "--test", "data/test.parquet",
                 "--output", "features/fusion"])


def step3_model_training():
    """ステップ3: モデル学習"""
    logger.info("=" * 60)
    logger.info("STEP 3: モデル学習")
    logger.info("=" * 60)

    run_command(["python", "scripts/train/baseline.py",
                 "--train", "data/train.parquet",
                 "--output", "models/baseline"])


def step4_validation():
    """ステップ4: 検証 (MRR@25)"""
    logger.info("=" * 60)
    logger.info("STEP 4: 検証 (MRR@25)")
    logger.info("=" * 60)

    run_command(["python", "scripts/validate/mrr25.py",
                 # 検証 fold の正解と予測（形式は mrr25.py の docstring を参照）
                 "--truth", "outputs/valid_truth.csv",
                 "--pred", "outputs/valid_submission.csv",
                 "--output", "results/validation.json"])


def step5_submit():
    """ステップ5: 提出物生成"""
    logger.info("=" * 60)
    logger.info("STEP 5: 提出物生成")
    logger.info("=" * 60)

    run_command(["python", "scripts/submit/generate.py",
                 "--predictions", "outputs/test_predictions.parquet",
                 "--sample_submission", "data/sample_submission.csv",
                 "--output", "submission.csv"])


def step6_kaggle_submit():
    """ステップ6: Kaggle提出"""
    logger.info("=" * 60)
    logger.info("STEP 6: Kaggle提出")
    logger.info("=" * 60)

    run_command([
        "kaggle", "competitions", "submit",
        "-c", "enveda-CASMI26-molecule-id-mass-spectra",
        "-f", "submission.csv",
        "-m", "改善ループ v1.0"
    ])


def step7_error_analysis():
    """ステップ7: 誤差分析"""
    logger.info("=" * 60)
    logger.info("STEP 7: 誤差分析")
    logger.info("=" * 60)

    # TODO: 誤差分析を作成する
    logger.info("誤差分析を実行してください")


def step8_iteration():
    """ステップ8: 反復"""
    logger.info("=" * 60)
    logger.info("STEP 8: 反復")
    logger.info("=" * 60)
    logger.info("次の改善イテレーションを計画してください")


def main():
    parser = argparse.ArgumentParser(description="改善ループマスターピアニスト")
    parser.add_argument(
        "--stage",
        choices=["data", "features", "train", "validate", "submit", "kaggle", "all"],
        default="all",
        help="実行ステージ (default: all)"
    )
    args = parser.parse_args()

    # タイムスタンプディレクトリ作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results") / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("改善ループ開始")
    logger.info(f"実行時刻: {datetime.now().isoformat()}")
    logger.info(f"結果保存先: {results_dir}")
    logger.info("=" * 60)

    if args.stage in ["data", "all"]:
        step1_data_acquisition()
    if args.stage in ["features", "all"]:
        step2_feature_engineering()
    if args.stage in ["train", "all"]:
        step3_model_training()
    if args.stage in ["validate", "all"]:
        step4_validation()
    if args.stage in ["submit", "all"]:
        step5_submit()
    if args.stage in ["kaggle", "all"]:
        step6_kaggle_submit()
    if args.stage in ["error_analysis", "all"]:
        step7_error_analysis()
    if args.stage in ["iteration", "all"]:
        step8_iteration()

    logger.info("=" * 60)
    logger.info("改善ループ完了")
    logger.info(f"結果保存先: {results_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
