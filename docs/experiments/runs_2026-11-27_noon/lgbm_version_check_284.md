# lightgbm 4.6.0 と 4.7.0 の予測の同一性の点検（Issue #284）

2026-11-27 昼、計画の項目 6。定義と補足の点検は数える前に #284 に書いた（本文とコメント）。

## 本体: 同じ機械（macOS arm64）で版だけを変える
- モデル `outputs/local_eval/ranker/submit_e2e_274/ranker_submit.txt`（sha256 54cb7d8d…、4.7.0 で学習）、特徴 64 列（features.json sha256 701b396a…、features_sha256 e0b73e5c…）
- 表 `outputs/local_eval/features/dev_b8_cc.parquet`（sha256 304f50c3…）の全 430,979 行・3,901 分子
- 4.7.0 は .venv のもの。4.6.0 は `uv pip install --target <scratch>/lgbm460 --no-deps lightgbm==4.6.0` で入れ、`PYTHONPATH` の先頭に置いた別プロセスで動かした（`--expect-version` で版を assert）
  - 4.6.0 の wheel の `lib_lightgbm.dylib` は `@rpath/libomp.dylib` を読み込めなかった（Homebrew の libomp が無い）。そこで .venv の 4.7.0 と同じ rpath（.venv の torch/lib、同じ libomp）を `install_name_tool -add_rpath` で足し、`codesign -f -s -` で署名し直した（scratch の複製だけを変えた。.venv は変えていない）
- 結果（`lgbm_version_check_284.json`）:
  - 最大の絶対差 0.0。430,979 行すべてで float64 の値が完全一致した（予測の配列の sha256 はどちらも 9898cbe5…）
  - 上位 25 の順が完全一致した分子は 3,901 / 3,901（100%）、1 位が違う分子は 0
  - **判定: 同じ**（事前の定義「全分子で上位 25 の順が 100% 一致」を満たす）。wheel を同梱する案は要らない
- 時間と資源: 予測はどちらも 0.3 s（プロセス全体で 0.62 s）、maxRSS 0.99 GB / 1.00 GB、比べる段 0.34 GB

## 補足: Kaggle（Linux、4.6.0）の出力を手元の 4.7.0 で再現する
本体は同じ機械なので、Kaggle の機械（Linux x86_64）との違いは見えない。そこで補足を 1 回だけ行った（`replay`、#284 のコメントで事前登録）。
- 入力: `taichiiiii/casmi26-e2e-timing-274` 版 2 の出力 `features.parquet`（sha256 e5a5f48f…）と `parts_cc/sub_test.parquet`（sha256 83833143…）、計 45,173 行。比べる相手は同じ版の `submission.csv`（sha256 6fcc3353…。項目 5 で記録したものと同じ）
- 結果（`lgbm_version_replay_284.json`）: 400 / 400 id で上位 25 の SMILES の並びが一致した。手元で書いた submission の sha256 も 6fcc3353… で、バイト単位で一致した。CCO で埋めた id は 0
- 言えること: 見える test の 400 分子では、Kaggle の 4.6.0（Linux）と手元の 4.7.0（macOS）で提出の中身が同じ。言えないこと: 隠れた test の特徴の値の範囲で同じかは測っていない。dev の 430,979 行での一致は macOS 上の版の違いだけについての結果である
