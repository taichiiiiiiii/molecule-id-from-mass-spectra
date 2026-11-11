# Majyles の公開パイプライン（識別検索＋類縁伝播＋LightGBM lambdarank、公開 LB 0.313）

調査 Issue #75（計画 2026-11-12 夜 項目 4）。案 1（Engine 2 の 2 表を自前で作り直した 0.487 系統、#74）が「失点」と判定された場合に、「再現できる部品だけの版」（オーナー判断 4）を作る材料として読んだ。**この文書は読んだ結果と測り方の案だけ**で、コードは動かしていない。

## 出典
- リポジトリ: https://github.com/Majyles/kaggle-casmi-2026 、コミット `26aa5c6097922fd3fade397ea9d011f909dcb4cc`（2026-11-06 16:10 +0100、「CASMI 2026: identity search, analog propagation and a learned ranker」）。2026-11-12 00:44 JST に `git clone --depth 50` で取得
- 議論: トピック 756504「Open-source pipeline: identity search + analog propagation + LightGBM ranker (with notes on validation)」（Yohann Roux、2026-11-06、`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/756504`）
- 公開 LB 0.313 は作者の申告（README・experiment_log）。こちらでは確かめていない

## ライセンス
- リポジトリ: **MIT**（`LICENSE`、Copyright (c) 2026 Majyles）
- 依存（`pyproject.toml`）: 本体は rdkit==2026.3.3、ms-entropy>=1.5.2（FlashEntropy）、pebble>=5.2.2 など。lightgbm>=4.7.0 は本体の dependencies ではなく dependency-groups の experiments に入っている。Kaggle 用のオフライン wheel は rdkit 2026.3.3・ms_entropy 1.5.2・pebble 5.2.2 に固定（`kaggle/dataset-metadata.json`）。依存それぞれのライセンスは**未確認**
- COCONUT: 作者のコード（`casmi/data.py` の docstring）は「CC0」と書く。一方、こちらの部品表（`lineage-reproducibility.md` 6 行、v2-pool）は「COCONUT CC BY 4.0 由来とカード」と書いていて食い違う。COCONUT 側の表示は**未確認**。入力は COCONUT の CSV（`coconut_csv_lite-09-2026.zip`、`casmi/paths.py`）を手で取得する前提で、リポジトリには入っていない
- Kaggle の資産 dataset の雛形（`kaggle/dataset-metadata.json`）は CC0-1.0 と宣言している（作者が上げたかどうか、その slug はリポジトリからは分からない。`<kaggle-username>/casmi26-assets` のまま）

## 要点（README・コードから）
1. スペクトルの掃除（基準ピークの 1 % 未満とプリカーサ付近を除く）と中性質量
2. **識別検索**（class 1）: 同じ中性質量 ±10 ppm の参照とエントロピー類似度（FlashEntropy）
3. **ハイブリッド検索**（class 2）: すべての質量の参照と、ピークをそのまま、または中性質量の差だけずらして照合し、類縁体を探す
4. **類縁伝播**: 正しい質量のプール構造ごとに score(c) = Σₐ wₐ · Tanimoto(c, a)（Morgan 指紋）、wₐ = 各クエリスペクトルの類似度の 16 乗の和
5. **LightGBM lambdarank**（20 特徴: `casmi/ranking.py` の `FEATURES` 17 個 + 派生 3 個。an16/an4/an1 と相対値・順位、id_max、id_n_spectra、類縁体との Tanimoto、has_visible_spectra、in_coconut、np_likeness、mass_err_ppm、exact_mass、n_candidates、best_analog_sim、n_query_spectra、id_rank、id_rel、tani_rank）。識別類似度 ≥ 0.97 は必ず 1 位（「identity guard」）
6. 候補は COCONUT に限る（NP-likeness ≥ 0.5 の非 COCONUT と強い識別一致を足す）。enveda-180・drug_plus だけにある合成化合物は類縁体から外す
- 作者の記録（`docs/experiment_log.md`）: v1 0.263 → v2 0.295（プールを天然物に絞る）→ v3 0.313（ランカー）。「順位の誤りの約 98 % は同じ組成式の異性体」
- 議論 756504 のコメント（Jake、11/9）: 次の手として GLACIER の証拠と絞った PubChem 候補を勧め、作者も同意。コメントは一覧表示で途中までしか読めていない

## 学習データと検証
- ランカーの学習集合（`scripts/build_ranker_set.py`）: 天然物 2,000 分子（seed 7）。条件は「公開ライブラリ（riken, gnps, massbank, mona, spectraverse, msdial, pluskal_ms2, masaryk）にあり、COCONUT にあり、enveda-180・drug_plus・enveda-np-examples に無く、157–1,159 Da、作者の検証集合の構造でない」。分子ごとにクエリの出どころを 1 つ（QTOF 優先）
- 学習のグループ（`scripts/train_ranker.py`）: (condition, molecule)。condition は class1x（クエリと同じ装置系統の参照も隠す、交差装置の class 1）と class2（その分子の参照を全部隠す）、重み 0.16 / 0.45（作者が議論の推定から置いた test の比率）。正解に見える参照が残らない群は落とす（`reachable_only`）。木の数は学習集合の 4 分割 GroupKFold で決める
- 提出したモデル（`--final`）: 上の学習集合に**検証部分集合 A（enveda-np-examples の 250 分子、timsTOF）を重み 3 で足して**学習
- 作者の検証: A（enveda-np-examples 250）と B（他の天然物 1,000、全部 COCONUT にある）、2 回の抽選の平均、対の bootstrap 区間
- **fold で分けていない**: 学習集合は identity-disjoint folds とは無関係に選ばれている。こちらの v3 の 4 集合の正解（`data/holdouts/*/drop_keys.txt`）や split の hold・np と重なるかは**未確認**（提出モデルは np パネルを学習に含む）
- テストの範囲: `tests/` に 6 本（metric、validation、ranking、chem、spectra、data）、README によると 45 件

## 部品ごとの再現性
| 部品 | 生成コード | 備考 |
|---|---|---|
| 候補プール（train ∪ COCONUT、正規化、metric key） | あり: `scripts/build_structures.py`・`casmi/candidates.py` | 入力の COCONUT CSV は手で取得（リポジトリに無い） |
| 検証部分集合 A・B（val_structures / val_queries） | あり: `notebooks/02_validation.ipynb`（スクリプトではなくノートブック、README の手順 2） | ランカーの学習集合（`build_ranker_set.py`）が `casmi.load_validation()` で除外に使うので、作り直すにはこのノートブックも回す |
| Morgan 指紋・NP-likeness | あり: `casmi/candidates.py`（`pool_np_likeness`、COCONUT の列と RDKit の NP スコア） | |
| 識別・ハイブリッド検索 | あり: `casmi/spectra.py`（ms_entropy に依存） | 検索結果はキャッシュ（`scripts/cache_validation_hits.py`） |
| ランカーの学習集合と特徴表 | あり: `scripts/build_ranker_set.py`・`scripts/build_feature_tables.py` | 「数時間」（README） |
| LightGBM ランカー | あり: `scripts/train_ranker.py --final` | |
| 推論 | あり: `kaggle/inference.py`、資産は `kaggle/build_assets.py` で組む | Kaggle で約 28–30 分（作者の記録）。`build_assets.py` は Linux 用の third-party wheel（rdkit・ms_entropy・pebble）を手で `pip download` しておく前提で、無いと「missing third-party wheels」で止まる |
| 学習済みの重み・資産 | リポジトリには無い（探した場所: リポジトリ全体、`kaggle/` の metadata） | 自前で作り直す必要がある |

すべての部品に生成コードがあり、外部の学習済みモデル（FPNet や ICEBERG）を使わない。最終提出の「端から端まで再現できること」（主催者回答 754610）を満たしやすい系統である（ライセンスの未確認分を除く）。

## 縮小版（`scripts/local_eval`、LB 0.282）との差
| 段 | 縮小版 | Majyles |
|---|---|---|
| 候補 | pool_meta（v2-pool）の ±10 ppm 窓 | COCONUT（＋NP-likeness ≥ 0.5、強い識別一致）、合成だけの類縁体は除く |
| 主な点 | fold-safe FPNet fold 0 のロジット `fp[bits] @ z` | 識別検索と類縁伝播（スペクトルライブラリ由来。学習済みモデルなし） |
| 事前分布 | PubChem 文献 prior（0.479 版の定数） | in_coconut・np_likeness を特徴として |
| 並べ替え | MetFrag-lite の断片説明率（frag） | LightGBM lambdarank（20 特徴）、識別 ≥ 0.97 は 1 位 |
| 学習 | なし（frag も学習なし） | ランカーのみ（fold-safe ではない） |

縮小版に無いのは「ライブラリ検索の証拠」（識別・類縁伝播）で、Engine 2 の lib・anapol 列と考え方が近い。

## v3 で測る案（書くだけ。実行していない）
「Majyles のランカーの点を縮小版の新しい段として足す」1 通りだけを測る。
1. リークの確かめ方（先に行い、満たせなければ測らない）
   - 作者の提出モデルは np パネルを含み、学習集合も fold で分けていないので、**そのままでは v3 を測れない**（v3 の正解を学習で見ている可能性を消せない）
   - 作者のコードのまま、学習集合の選び方の除外に v3 の 4 集合の drop_keys・split の hold/np・`fold_identity == 0` を足してランカーを学習し直す（`--final` は使わない＝部分集合 A を入れない）。学習集合の key と除外集合の積が 0 であることを assert する
   - 識別・類縁検索のライブラリからは、各 v3 集合の条件どおりに正解の参照を隠す（v3 の `drop_keys` の行を落とした train）
   - v3 の hide_rule のうちプールとクエリの規則も同じに当てる: v3_c2 は `pool: remove`（`scripts/local_eval/run.py` 210 行付近で正解を候補窓から消す）なので、Majyles 側でも正解を COCONUT プールと「強い識別一致でプールの外から足す候補」の両方から消す。v3_c2n は `query: prefer_enveda`。確かめること: v3_c2 で正解が Majyles 側の候補にも特徴にも戻らないこと（`has_visible_spectra`・`id_max` は隠したあとの train で計算する）
2. 足し方（事前に 1 つに決める。重みや入れ方は変えて試さない）:
   - 候補は縮小版の窓（pool_meta ±10 ppm）のまま。Majyles 側にしか無い候補は足さない（点の追加だけの変更にする）
   - 縮小版の prior の段の後・frag の段の前で、分子ごとに Majyles のランカーの点を z 化し、prior 後の点の z と等重みで足す。frag は組成式グループの中での並べ替えなので、そのまま後に置く
   - Majyles の点が無い候補（Majyles の候補外）は、その分子で Majyles の点がある候補の z の最小値にする
3. 採否の規則（設計 §5、#48 と同じ）: v3_c1・v3_c2・enveda_v1・v3_c2n の 4 集合すべてで「(新 − 対照) の 95% CI 下限 ≥ −0.02」、かつ 1 つ以上で「下限 > 0」。1 集合でも欠ければ採用しない。v3_c1 と v3_c2 は必ず報告する
4. 先に数えること: v3_c2 の正解のうち COCONUT にある割合（Majyles の候補は COCONUT 中心なので、COCONUT に無い正解には届かない）
5. 費用の見積もり（作者の README）: 構造表 約 3 時間（8 コア）、ランカーの学習集合の検索「数時間」。ローカル CPU の重いジョブになるので、メモリと他のジョブを確かめてから 1 本だけ

## 未確認のこと
- 公開 LB 0.313 と各版の検証値（作者の申告のまま）
- 依存ライブラリと COCONUT CSV のライセンス
- 作者の学習集合と v3 の正解・split の重なり
- 議論 756504 のコメントの全文
