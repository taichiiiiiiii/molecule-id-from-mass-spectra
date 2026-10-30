# 公式記述の取り込み（envela-CASMI26-molecule-id-mass-spectra）

Kaggle CLI で公式ページ（概要 / データ / 評価 / タイムライン / 賞金 / 提出条件 / 規約）
と Discussion を取得し、ここに永続化する。

## 取得方針

- 公式ページ本文: `kaggle competitions pages`（slug を 2 回渡す）
- Discussion: `kaggle competitions topics list` / `show`

```bash
kaggle competitions pages envela-CASMI26-molecule-id-mass-spectra list \
    envela-CASMI26-molecule-id-mass-spectra --content --format json

kaggle competitions topics list envela-CASMI26-molecule-id-mass-spectra -s top
kaggle competitions topics show envela-CASMI26-molecule-id-mass-spectra <topic-id>
```

取得できるページ: `abstract` / `Description` / `Evaluation` / `Timeline` / `data-description` /
`Prizes` / `Code Requirements` / `rules` / `Acknowledgements` など。

## ⚠️ 注意点

- 以下は**公式ページ（公開情報）からの要約**であり、規約解釈が問題になる場面では
  必ず原文（https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra/overview）に
  当たること。各ファイル冒頭に取得日と URL を記載する。
- 現在のセッションは未参加（ウォレット未登録）のため、`kaggle competitions pages/topics` は
  参加登録後に完全な原文を取得可能。参加後の `kaggle competitions pages ... list` で
  上記ファイルを上書き更新してよい。

## ファイル一覧

| ファイル | 出典 | 最終更新 |
|---|---|---|
| [overview.md](overview.md) | Overview / Evaluation / Timeline / Prizes / Code Requirements | 2026-10-30 |
| [data.md](data.md) | data-description | 2026-10-30 |
| [rules.md](rules.md) | rules / participation | 2026-10-30 |
| [sota.md](sota.md) | 他参加者の手法・現状（外部リンク参照） | 2026-10-30 |
| [validation.md](validation.md) | 検証ベンチマーク・リーク対策 | 2026-10-30 |

## 補足

本リポジトリの構成は、既存Kaggle競수협（RSNA, biohub 等）および本コンペ参加者
（gnshx, nghianguyen7171, dmitriigluzdov ほか）の成果物に倣っています。
詳細は `sota.md` を参照してください。
