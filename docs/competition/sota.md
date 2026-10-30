# 他参加者の手法と現状（State of the Art — 外部参照）

※ ここは「公式ページ」ではなく、公開リポジトリ・ノートブック・ベンチマークからの要約です。
  規約解釈に関わる箇所は公式原文を優先してください。

## 参照した参加者

| リポジトリ | 特徴 | Public LB（目安） |
|---|---|---|
| `dmitriigluzdov/CasMI26-…` / ベンチマーク | 250分子ベンチ、identity-disjoint folds、FPNet | パネル 0.589（FPNet A+B） |
| `nghianguyen7171/casmi26` | 4チャネル＋ranker、Class3列挙 | 0.328 |
| `gnshx/Enveda-CASMI-2026` | 8フェーズ、Engine Fusion + forward rescoring | 0.420（1位相当） |
| `billylu24/…`（flexonafft） | V30再現、コード全6セル | 0.441（最上位） |
| `ibnmarzuk/casmi26-molecule-id` | ライブラリ検索ベースライン + HOW_TO_SUBMIT | 0.441 |
| `sinhaaditya5/kaggle-enveda-casmi26` | 質量窓＋コサインライブラリ、class1検証 | ~0.916（class1相当） |
| `pilkwangkim` | spectrum-to-fingerprint + ranker、Two Rankers One Engine | 0.342 |

## 核心技術の分解

### 1. 候補生成（candidate generation）
- **質量窓削減**: 中性質量 ±10ppm（フォールバック30ppm）で COCONUT ∪ train（約71万構造）から候補化。
- → **70.8% の候補が真の分子式を共有**。つまり「質量・式だけでは解けず、異性体判別が本体」。
- 中央値約107候補/分子。5ppm窓は真構造を98.8%保持（中央値12候補）。

### 2. 証拠チャネル（evidence channels）

| チャンネル | 内容 | 有効 |
|---|---|---|
| ライブラリ一致 | スペクトルエントロピー類似＋付加体シフト | class1: 0.917 |
| 質量シフトanalog伝播 | 近傍スペクトルから `similarity^3 × Tanimoto` | class2: 0.465 |
| MetFrag-lite | 1–2結合フラグメントで強度説明割合 | 0.263（相関弱い） |
| FPNet | ピーク→6,930bitフィンガープリントlogit `z`、候補 `f` に対し `f·z`（ビット独立のベイズ対数尤度） | class2: 0.477 |

### 3. ランク付け
- 31–51特徴量のGradientBoosting ranker（class prior 0.30/0.60、複数seed）。
- 独立チャネルの出力を「within-molecule正規化ランク」でブレンド（0.88/0.12等）。

### 4. クラス3（de-novo）
- 公開FPNet/transformerでフィンガープリント生成、候補を再順位づけ。
- 予約済みtailスロットにClass3候補を差し込む。

### 5. 融合/再ランキング（上位の手法）
- **Engine Fusion**: 検索エンジン（v4n）と2ラーナ（BIO+AFIX）をRRFで統合。
- **Union Forward Rescoring**: ベースリストのtop-40をforward MS/MS予測（ICEBERG/GLACIER）に丸ごと投入。
- 同じ式グループ内で再順位づけし、popularity priorとlibrary gateで異性体を分離。

## 公共最上位（公開時点）
- **flexonafft V15**: 0.441（正規コード再現）
- **gnshx**: 0.420（1位相当、PubChem Join）

---
出典: 各リポジトリ README / HOW_TO_SUBMIT / GitHub Pages。ただし「他者の主張」であり、
本リポジトリの公式評価ではない。独自検証では上の旨を常に測定すること。
