# データ（要約）

出典: https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra/data
最終更新: 2026-10-30

## 規模

| ファイル | 内容 |
|---|---|
| `train.parquet` | 約 250 万 MS/MS スペクトル、約 27.5 万種の構造（構造が与されている） |
| `test.parquet` | 識別対象スペクトル（約 1,500 スペクトル、約 400 分子、取得 1〜16 スペクトル/分子・中央値 3） |
| `sample_submission.csv` | 正常な提出形式の参考例 |

## 特徴

- 試験スペクトルは **Bruker timsTOF** で取得。
- モノアイソトピック質量 **157〜1,159 Da**（中央値 348）。
- 分子あたり複数のスペクトルが異なる衝突エネルギー / 付加体（adduct）で取得される。
- **1 分子につき 1 つの構造を予測**するため、分子に所属する全スペクトルの証拠を
  1 つの順位付きリストにまとめる必要がある。

## 3 クラス（新規性、難易度順。詳細は隠蔽）

| クラス | 定義 |
|---|---|
| 1 | 公開スペクトルライブラリにある構造（参照スペクトルと類似で探せる） |
| 2 | 既知構造だが公開スペクトルなし（PubChem / COCONUT で取り出し可能） |
| 3 | PubChem に存在しない新規構造（de novo 予測が必要） |

## データ取得（参考）

取得コマンド（参加登録後）:

```bash
kaggle competitions download -c envela-CASMI26-molecule-id-mass-spectra -p data
```

引数 `-p` はダウンロード先ディレクトリ。`data/` に `train.parquet`,
`test.parquet`, `sample_submission.csv`, 付属メタデータ JSON が置かれる。
