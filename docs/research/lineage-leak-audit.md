# 0.487 系統の fold-safe 評価: 漏れの監査（1 日目、#114）

設計 `docs/design/lineage487-foldsafe-eval.md`（#112）§4 の 1〜12 と §2.1 の判定。生の出力は `outputs/research/lineage_leak_audit.json`（追跡しない）と、分子ごとの最近傍 Tanimoto `outputs/research/lineage_leak_audit_tanimoto.csv`。

    .venv/bin/python scripts/cv/lineage_leak_audit.py      # 約 40 秒、RSS 最大 1.39 GB（2026-11-14 に手元で測った）

学習・推論・Kaggle の push はしていない。Kaggle からは小さいファイル 7 本を読むだけで取ってきた（`data/ext/lineage_audit/`、置き場はスクリプトの docstring）。

## 数（分母は truth.csv: v3_c1 200、v3_c2 200、enveda_v1 200、v3_c2n 195）

| # | 対象 | v3_c1 | v3_c2 | enveda_v1 | v3_c2n | 備考 |
|---|---|---:|---:|---:|---:|---|
| 1 | fold-safe split の hold | 200 | 200 | 200 | 195 | 全部 hold（assert 通過）。np とは 0 |
| 2 | #54 の除外（55,369） | 200 | 200 | 200 | 195 | 除外の数は 55,369 で期待どおり |
| 3 | #74 の試走のクエリ（48） | 0 | 0 | 0 | 0 | 本番の表は手元に無い（not_computable） |
| 4 | ranker_v4b の模擬クエリの構造（5,350） | 7 | 3 | 4 | 13 | 作者の split（sha256 一致）の np・nplib・syn・plusk |
| 5 | fragnet の学習構造（近似: 作者の train fold 270,304） | 193 | 197 | 196 | 182 | 近似。analog_struct・frag2・derivation・stageA/B は手元に無い |
| 6 | MSG 1.0 ∪ 1.5（どれかの fold） | 1 | 0 | 0 | 116 | §2-10 の値を再現。採点に残るのは 199・200・200・79 |
| 7 | ExtLib のスペクトル行（ext_meta） | 0 | 0 | 0 | 18（MSG で外したあと 7） | 構造表（ext_fp_keys）とは 0 |
| 7 | MassBank harmonized の構造 | 0 | 0 | 0 | 44（同 12） | 上限（is_casmi_subset で落とす前） |
| 7 | 知識表の外部参照 | 0 | 0 | 0 | 5（同 2） | |
| 7 | ChEBI/LMSD・PubChemLite | 0 / 0 | 0 / 0 | 0 / 1 | 111 / 96 | COCONUT・hengck23 は手元に無い |
| 8 | in_train_structure の印で F c36 が参照に足す行 | 0 | 0 | 0 | 5 行・4 分子（同 2） | v3_c2n は hide_rule = all なので食い違い（パッチの対象） |
| 9 | train_structs.parquet（パッチ前） | 200 | 200 | 200 | 195 | sha256 は pin と一致 |
| 9 | 同（hide_rule を当てた train から作り直した後） | 200 | 0 | 0 | 0 | v3_c1 は公開ライブラリの参照が残るのが本来の設計 |
| 10 | train_fp_sel.npy | — | — | — | — | 手元に無い。行 = sid と推定（train_structs の sid は行番号と一致） |
| 11 | ランカーの 160 特徴のうち train 由来の表に依存 | | | | | 56（名前からの推定）、学習済みの族（fragnet・model_views）21 |
| 12 | 最近傍 Tanimoto の中央値（残った train 220,443 構造） | 0.634 | 0.650 | 0.647 | 0.727 | ≥ 0.9 は 1・2・2・32 分子 |

## §2.1 の判定（事前登録どおり）
- パッチの後の c: 4 集合とも 1.0（fragnet の近似と模擬クエリの和）。analog_struct・frag2・derivation の一覧も作れない → **Tier 0 は較正にも採否にも使わない**
- パッチの前の c（記録だけ）: 4 集合とも 1.0
- 感度（判定には使わない）: fragnet の近似を使わず模擬クエリだけでも、v3_c2n は 13 / 195 = 6.7%（MSG で外したあと 4 / 79 = 5.1%）で 5% を超える

## 設計との食い違い
- §3.1 の「4 集合は構造で重ならない」は成り立たない: v3_c1 と enveda_v1 で 1 構造（VHNSOMLGZMGWNH）、v3_c2 と enveda_v1 で 2 構造（LBLAVYDUPXGKQX、MUAYFYDPOHQAJL）が重なる。データセット A（v3_c1 + enveda_v1）では、この構造に 2 つの hide_rule が当たる
- §2-15 の「ExtLib は v3 の正解を 1 位に上げられない」は構造表についてだけ正しい。スペクトル行（ext_meta）には v3_c2n の正解が 18 構造（MSG で外したあと 7）ある
- §4-8: 知識表の外部参照は、v3_c2n の正解 4 分子について hide_rule = all に反して参照を戻す（F c36 85 行の条件を当てた数。付加イオン・モードの条件は数えていない）
