# Molecule ID From Mass Spectra（Kaggle）

**Envela CASMI 2026 — Molecule ID From Mass Spectra**（コードコンペ）

- URL: https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra
- スラッグ: `envela-CASMI26-molecule-id-mass-spectra`
- 開催期間: 2026-09-14 〜
- エントリー締切: 2026-12-07 23:59 UTC
- 最終提出締切: 2026-12-14 23:59 UTC
- 賞金: **$50,000**（1位 $16,000 / 2位 $12,000 / 3位 $9,000 / 4位 $7,000 / 5位 $6,000）

## 方針・状況

- 評価指標は **MRR@25**。
- 任は「未知・新規分子の同定」であり、参照ライブラリの類似比較では不十分な
  クラス（3: novel）が最も難しい。
- 方針:
  1. まず有効なライブラリ検索ベースライン（class 1）を作る
  2. class 2 は式・analog伝播で補強
  3. class 3 は de-novo（フィンガープリント生成 → 候補選択）で対応
  4. マルチスペクトル集約 + 学習済みランク付けで融合
- 検証は **アイデンティティ分離（分子が学習と重複しない）hold-out** を使う。
  パブリックLBの波（再ランによるスパーキング）を鵜呑みにしない。

## 公式記述

公式ページ（概要 / データ / 評価 / タイムライン / 賞金 / 規約）は
`docs/competition/` に取り込んでいる。公式情報は必ず原文（上リンク）を参照。

- `docs/competition/README.md` — 公式記述の取得方法・注意点
- `docs/competition/overview.md` — 概要・評価指標・タイムライン・賞金
- `docs/competition/data.md` — データ説明
- `docs/competition/rules.md` — 参加・提出ルール

## コンペの概要（要約）

- 植物・哺乳類・微生物などの天然物様試料中の低分子を、LC-MS/MS スペクトルから
  2D 構造（SMILES）として同定する。
- 分子あたり 1〜16 スペクトル（中央値 3）を 1 つの順位付きリストに集約する。
- 3 クラス（非明瞭性）: **1** 公開ライブラリ / **2** 既知だが公開スペクトルなし
  / **3** 完全新規（de-novo）。

## データ（要約）

| ファイル | 内容 |
|---|---|
| `train.parquet` | 約 250 万 MS/MS スペクトル、約 27.5 万種の構造 |
| `test.parquet` | 約 1,500 スペクトル、約 400 分子（1〜16 スペクトル/分子、中央値 3） |
| `sample_submission.csv` | 提出形式の参考例 |

- 試験は **Bruker timsTOF**、質量 157〜1,159 Da（中央値 348）。
- 提出は `submission.csv`（`molecule_id,smiles`、1 行 1 分子、最大 25 候補・セミコロン区切り）。

## リポジトリ構成

```
.
├── README.md
├── config.toml
├── .gitignore
├── scripts/download.sh      # データ取得
└── docs/competition/        # 公式記述（概要 / データ / 規約 / 取り込み方法）
    ├── README.md
    ├── overview.md
    ├── data.md
    └── rules.md
```

## クイックスタート

1. Kaggle CLI が認証済みであることを確認:
   ```bash
   kaggle auth check
   ```
2. データを取得（参加登録後に 1 回だけ）:
   ```bash
   bash scripts/download.sh
   ```
3. 提出用カーネル（Kaggle Notebook）で `submission.csv` を作成する。
   提出は必ず **Kaggle Notebooks** から（CPU/GPU 9h 以内・インターネットオフ）。

## 参考文献（他のKaggleコンペの構成）

- RSNA: `/Users/taichi/コンペティション/Kaggle/RSNA-Knee-Abnormality-Detection/`
- biohub: `/Users/taichi/コンペティション/Kaggle/biohub-cell-tracking/`

- 他参加者の手法: `docs/competition/sota.md`（gnshx / nghianguyen7171 / dmitriigluzdov ほか）
- 検証手法: `docs/competition/validation.md`（identity-disjoint folds / パネルベンチ）

---

*このリポジトリは他のKaggleコンペの構成と、他参加者の成果物に倣って整理しています。*
