# Envela CASMI 2026 — Molecule ID From Mass Spectra（概要）

出典: https://www.kaggle.com/competitions/envela-CASMI26-molecule-id-mass-spectra/overview
最終更新: 2026-10-30

## 基本情報

| 項目 | 内容 |
|---|---|
| Slug | `envela-CASMI26-molecule-id-mass-spectra` |
| 開催種別 | Featured Code Competition |
| タイトル | Enveda CASMI 2026 - Molecule ID From Mass Spectra |
| 参加チーム数 | 4,270 teams |
| 賞金 | **$50,000**（1位 $16,000 / 2位 $12,000 / 3位 $9,000 / 4位 $7,000 / 5位 $6,000） |
| 開催期間 | 2026-09-14 〜 |
| エントリー締切 | 2026-12-07 23:59 UTC |
| 最終提出締切 | 2026-12-14 23:59 UTC |

## 説明（要約）

植物・哺乳類・微生物などの天然物様試料に含まれる低分子（small molecules）を、
LC-MS/MS サン spectrophotometry（tandem mass）スペクトルから同定する。
CASMI 2026（Critical Assessment of Small Molecule Identification）の課題であり、
参加者は **2D 化学構造（SMILES）** を予測するモデルを構築する。

現代の質量分析では複雑な生物試料から数千の分子が検出されるが、参照ライブラリで
ヒットする既知化合物よりも、未知・新規分子の同定が難しい。現在の手法は
参照ライブラリとの類似比較が中心で、未知分子の同定がボトルネックになっている。
本課題では、既知・新規を問わず、正確な SMILES 表現を生成することを目指す。

## 評価指標: MRR@25

各分子 $u$ につき、最大 25 件の候補構造を「正解と思われる順」で並べる。
正解構造が最初に登場する順位を $\text{rank}_u$ とすると、

$$ \text{MRR@25} = \frac{1}{U}\sum_{u=1}^{U}\frac{1}{\text{rank}_u} $$

正解が 1 件で、その順位が 1 なら 1.0、2 なら 0.5、25 なら 0.04。
正解が 25 件中に存在しない分子は 0。**1 つの分子につき正解は 1 構造のみ**で、
**最初の正解のみ**が効く。

正解判定は **同一原子結合（connectivity）** の一致で行う（tautomer 等は
RDKit 正準化 + InChIKey14 で同一視される）。

## 提出形式

- 1 行 1 分子、ヘッダー `molecule_id,smiles`
- 各分子につき最大 25 候補（セミコロン区切り）、正解を先頭に
- `molecule_id` は重複不可、null 不可、25 個超は拒否
- 提出ファイル名は `submission.csv`
