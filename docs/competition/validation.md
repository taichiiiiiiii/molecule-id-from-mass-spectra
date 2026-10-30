# 検証ベンチマークとリーク対策

## 参照: オープンプレミアムベンチマーク
`dmitriigluzdov/CasMI26-Natural-Product-Panel-Benchmark`（Kaggle dataset）

- 自然物パネル **250分子・正解有り・timsTOF（1,184スペクトル）**
- 候補テーブルとスコアを保存 → 検索深度・prior・マージ規則・rankerの変更を **GPUなしで再現実行**
- 公開FPNetのパネルスコア:
  - FPNet A（main engine）: パネル 0.581
  - FPNet A+B（PubChem channel）: パネル 0.589
  - FPNet full1（全fold学習）: パネル 0.928（panel top-1 89%）

## リーク対策: identity-disjoint validation folds
`dmitriigluzdov/CasMI26-Identity-Disjoint-Validation-Folds`

- スコアリングは **RDKit tautomer正準化後の14桁 InChIKey`14`**
- ランダム分割はリークする（1分子=平均9スペクトル、1.6%構造が保存key≠スコアキ）。
- 以下を「1スコア付きアイデンティティ」単位で1側へ分割:
  - `fold_identity`（277,566構造 → 274,184グループ）
  - `fold_scaffold`（Bemis–Murckoスキャフォールドを1fold: 140,160グループ）

### クリーンな評価手順
1. 除外する正体（held-out identities）を **学習・参照ライブラリ・特徴量テーブル** から全面的に外す。
2. **候補プールに何を入れるか**決める: 保持は「既知だが参照なし」分子を模擬、除去は「DBに無い」分子を模擬。
3. ランキング指標の前に **候補カバレッジ** を報告し、答案が候補に入らない分子も含めMRR@25を出す。
4. 公開チェックポイントは5foldを全て見たことが多いので、**4foldのみ**で学習したモデルで推定する。

## ローカルでの使い方（参考）
```python
import pandas as pd
F = '/kaggle/input/casmi26-identity-disjoint-validation-folds'
train = pd.read_parquet('/kaggle/input/competitions/envela-CASMI26-molecule-id-mass-spectra/train.parquet',
                        columns=['inchikey14','normalized_smiles','precursor_mz','adduct'])
folds = pd.read_parquet(f'{F}/structures.parquet').drop_duplicates('inchikey14').set_index('inchikey14')
train['fold'] = folds.fold_identity.reindex(train.inchikey14).values  # fold_scaffold も可
fit, valid = train[train.fold != 0], train[train.fold == 0]
assert not set(fit.inchikey14) & set(valid.inchikey14)
```

---
出典: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds
