# 調査: スペクトルと分子の対照学習による整列（MSAlign・JESTR 型）を自前の証拠チャネルにできるか（#99）

計画 2026-11-14 夜 項目 2。親 #3。**読むだけ**（GPU 0、学習・推論はしていない）。2026-11-13 23:3x〜 JST に読んだ。

印: **確認** = 原文・コード・API の応答をこちらで読んだ ／ **主張** = 著者や投稿者の申告で、こちらで再現していない ／ **未確認** = 調べたが分からない（理由を書く）。見積もりの数値はすべて未測定。

## 出典
1. MSAlign: Krzakala et al., "MSAlign: Aligning Molecule and Mass Spectra Representations for Metabolite Identification", arXiv:2605.19752v3（2026-11-02）。https://arxiv.org/abs/2605.19752 （DOI 10.48550/arXiv.2605.19752、CC BY 4.0）。HTML 版の全文を取得して読んだ
2. MSAlign のコード: https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026 （GitHub API の license = MIT、最終 push 2026-11-06）。README、DATA.md、`models/MSAlign/README.md`、`models/MSAlign/configs/massspecgym_formula.yaml`、`models/MSAlign/losses.py`、`preprocessing/encode_dreams.py`、`preprocessing/encode_mol.py` を読んだ
3. MSAlign の重みとデータ: Zenodo 23372807（API の license = cc-by-4.0。ファイル 6 件: `msalign_massspecgym_formula_seed0.zip` 66.5 MB、`msalign_massspecgym_mces_1.zip` 50.9 MB、MSAlign+ 版 2 件 288〜351 MB、`massspecgym.zip` 1.08 GB、`spectraverse.zip` 2.24 GB）
4. JESTR: Kalia, Zhou Chen, Krishnan, Hassoun, "JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data", arXiv:2411.14464（v3 2025-06-07）、Bioinformatics 41(7) btaf354（DOI 10.1093/bioinformatics/btaf354）。arXiv の ID は検索結果と arXiv の著者一覧で確かめた。本文は要旨・手法の節の抜粋だけ読んだ
5. JESTR のコード: https://github.com/HassounLab/JESTR1 （license = MIT）。README と Issue #5 を読んだ
6. DreaMS: Bushuiev et al., "Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS", Nature Biotechnology (2025)、DOI 10.1038/s41587-025-02663-3（論文は CC BY 4.0、本文を取得して該当箇所を読んだ）。コード https://github.com/pluskal-lab/DreaMS （MIT）。重み Zenodo 10997887（cc-by-4.0。`ssl_model.ckpt` 1,392,710,212 バイト md5 4deee91d…、`embedding_model.ckpt` 1,241,290,418 バイト md5 4f6913c2…）
7. MassSpecGym: Bushuiev et al., NeurIPS 2024 Datasets and Benchmarks、arXiv:2410.23326（HTML 版の retrieval の節を読んだ）
8. MolDeBERTa（MSAlign の分子側）: HF https://huggingface.co/SaeedLab/MolDeBERTa-base-123M-mtr （モデルカードと API。license = cc-by-nc-nd-4.0、学習データ SaeedLab/MolDeBERTa）
9. 主催者回答（`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/<id> --format json`、2026-11-13 23:3x に取得）: 748882、757307、755373、754610
10. リポジトリ内: `docs/research/lineage-reproducibility.md`（4 行 fe_D、18 行 sanpier）、`docs/research/objective-mismatch.md`（#56、MSAlign の要点を一度読んだもの）、`docs/research/mist-massspecgym.md`（#66、MSG と v3 の重なり）、`outputs/sanpier_fold0/casmi_spectrum_model_fold0/{model.py,config_used.json,drop_audit.json}`、`outputs/local_eval/sanpier_v3_check.json`（#60）、`scripts/train/sanpier_fold0.py`、`data/ext/foldsafe/{README.md,split.parquet,train_fold0.log}`
11. 0.487 系統の fe_D の学習コード（TC = `ahmedberatozer/casmi26-training-code`、11/13 に scratch へ取得済みの写し）: `dreams_ft/review_fixes/train_dreams_fp.py` の docstring と引数、`dreams_ft/logs/d2_p100_60k_summary.json`

## 要点

### MSAlign（出典 1–3）
- 確認: 凍結した 2 つの基盤モデル（スペクトル側 DreaMS 1024 次元、分子側 MolDeBERTa 768 次元）の出力を前もって計算し、両側の軽い MLP（共有 512 次元）だけを学習する。点は温度つき cosine（§3、付録 B.3、表 8）
- 確認: 損失は「候補ベースの InfoNCE」（式 6）。正解 1 個と、その問い合わせの同質量候補から取った負例で交差エントロピー。設定は `massspecgym_formula.yaml`: `k_candidates 128`、`batch_size 128`、`n_max_steps 30000`、`learning_rate 1e-4`、`dropout 0.30`、`temperature_init 0.07`、衝突エネルギーと付加イオンを入力に使う
- 主張（表 3、MassSpecGym 組成式分割、3 分割の平均）: R@1 は MSAlign 47.5±3.4、スコア融合版 58.2±3.4。MCES 分割では 19.0 / 30.8
- 主張（表 4、組成式分割 R@1）: 分子側が指紋のとき、スペクトル側をビン化（10,050 次元）から DreaMS に替えると 34.3 → 38.6。分子側を MolDeBERTa にすると 44.5（ビン）／47.5（DreaMS）。**伸びの大半は分子側の MolDeBERTa から来ている**（著者の数値の読み）
- 主張（表 5）: 同じ構成で損失だけ替えると、候補ベース InfoNCE 47.5、バッチ内 InfoNCE 19.5、回帰 22.8 / 16.7
- 確認: 融合（式 7、付録 B.2）は 4 つの整列モデルの点と 3 つの質量の点の凸結合で、**重みと温度を検証の R@1 で選ぶ**。この競技ではハイパラ調整にあたるので写さない
- 主張（表 10、V100 1 枚、表記は hours.min）: MSAlign の学習 0.82（組成式分割）、融合版 3.29。付録 C.2「MolDeBERTa と DreaMS の埋め込みの前計算は約 5 時間、1 回だけ」。§3「1 エポック 2 分未満」
- 確認: `encode_dreams.py` は `PreTrainedModel.from_name("DreaMS_embedding")` を使う。DreaMS の `dreams/api.py`（commit dbec3a0）で、これは `embedding_model.ckpt`（対照学習で微調整した版）を読む。**自己教師ありの `ssl_model.ckpt` そのものではない**
- 確認: 配布の重み（出典 3）は MassSpecGym で学習したもの。基盤モデルの重みは再配布しない（DATA.md）。README は CUDA GPU と MolDeBERTa の HF の利用条件の承諾を要求する

### JESTR（出典 4–5）
- 確認（要旨・手法の抜粋）: スペクトルの MLP と分子のグラフ NN を一から学習し、CMC（対照多視点符号化）で共通の空間に埋め込み、cosine で候補を並べる。学習の終わりに、学習分子の同質量候補を使う正則化を入れる
- 主張: rank@1 を他手法より大きく上回る（要旨）。MSAlign の追試（出典 1 表 3）では MassSpecGym 組成式分割 R@1 13.9 で、MSAlign はこれを公開実装の問題（JESTR1 Issue #5「公開の MassSpecGym チェックポイントが公式の候補集合で Hit@1 0.00」、open）と結びつけている
- 確認: README「NPLIB1 のデータセットと学習済みの重みを公開する。他のデータセットはライセンス上の取り決めで公開できない」「A100、CUDA 11.8」。学習済みの重みは NPLIB1 と MassSpecGym（`data/MassSpecGym/`）の 2 つ。DGL が要る。表 10 の学習時間（主張）は 7.34（hours.min、1,000 エポック）
- 基盤モデルを使わない、一から学習の対照型という点で、**下の sanpier キットと同じ型**

### DreaMS（出典 6）
- 確認（論文）: 自己教師ありの事前学習は GeMS（MassIVE GNPS から集めた**ラベルの無い**スペクトル）の GeMS-A10（2,400 万スペクトル）で、隠したピークの予測と保持時間の順序の予測で学習する
- 確認（論文）: 「DreaMS 埋め込み」（`embedding_model.ckpt`）は、同じ構造の別スペクトルを正、質量差 0.05 Da 以内の別構造を負とする三つ組の損失で短く微調整したもの。「スペクトルライブラリへの偏りを避けるため MoNA の 5,500 分子だけを使う」と本文にある。DreaMS のリポジトリの `dreams/training/fine_tune.sh` にも `MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl` を使うコマンドがある（コメントアウト）
- 確認（論文 Methods）: 埋め込みの微調整について「MoNA の histogram-disjoint 分割の精選した部分集合（約 25,000 スペクトル、InChI の connectivity block で 5,500 種）で対照学習し、**NIST20 のスペクトルは学習に一切使わない**」と明記されている（同じ段落で、埋め込み空間の評価には NIST20 の高品質な部分集合を使う＝学習ではなく評価）。学習ログでは確かめていない
- 確認（論文 Methods「Linear probing」）: 自己教師ありの事前学習の途中、2,500 反復ごとに、**凍結した**モデルの precursor トークンの埋め込みから MACCS 166 ビットを当てるロジスティック回帰（線形層 1 枚）を、NIST20 と MoNA の Murcko histogram 分割から無作為に取った 6,000 件で学習して、構造の情報の現れ方を測っている。backbone に NIST20 の勾配は入らない書き方。ただし、配布の `ssl_model.ckpt` をどの反復で選んだか、その選択にこのプロービングの値を使ったかは、本文に見当たらない（未確認）

### MassSpecGym の retrieval 設定（出典 7、1）
- 確認: 候補は正解と同じ質量の分子を、metabolite としての関連の高い DB から順に取り、1 スペクトルあたり最大 256。分割は MCES 距離 10 未満の分子が学習とテストにまたがらないようにする。組成式を与える「ボーナス」設定もある
- この競技の候補（v2 プール ±10 ppm、窓の中央値 95 候補、`data/ext/foldsafe/README.md`）とは作り方が違う。論文の R@1 をこの競技の MRR@25 の見込みには使わない

### 主催者回答（出典 9）
- 確認 748882（Marie Killian、2026-10-13）: 「承認済みの MassSpecGym・GLACIER/ICEBERG・DreaMS のチェックポイントは、配布のチェックポイント、著者の学習コードと許可された学習データへの参照、自前の微調整・推論の全コードを渡せば足りる。事前学習をやり直す必要はない。DreaMS については、NIST などの禁止データ由来でない限り、自己教師ありの backbone も承認に含む」。同（10-14）「元の学習コードと許可データで再現できない第三者の重みは不可」
- 確認 754610（David Healey、2026-11-05）: 「公式に配布された DreaMS のチェックポイントは使ってよい」。Aravindhan Chandran（11-10）の「`ssl_model.ckpt` も含むか」には返答が無い（取得時点）。ただし上の 748882 が自己教師ありの backbone を明示しているので、答えは出ていると読む
- 確認 757307（David Healey、2026-11-11）: 「MassSpecGym と MIST のモデルは問題ない」。11-13 に GNPS-NIST14-MATCHES についての追加質問があり、返答は無い
- 確認 755373: コメントは参加状況の話だけで、整列・DreaMS・外部の重みに関わる裁定は無い
- 確認 748882 の 10-13 の回答 B: TabPFN（非商用ライセンス）の重みは「著者の学習コードと許可データへの参照があれば可」。非商用ライセンスそのものは除外の理由にされていない。ただし MolDeBERTa は名指しされておらず、ND（改変禁止）条項の扱いは誰も聞いていない

### 0.487 系統と縮小版にすでにある近い部品（出典 10–11）
- 確認 fe_D（`lineage-reproducibility.md` 4 行、TC の `train_dreams_fp.py`）: DreaMS の `ssl_model.ckpt` を土台に**全体を微調整**し、10,226 ビットの指紋ロジット z を出す。点は f·z（候補の指紋ビットとの内積）。損失は BCE(z, fp) + λ·窓の中の softmax 対照（正解と ±10 ppm の負例 K = 63）+ μ·元素 MSE。作者の分割（np/nplib/syn/plusk/twin を外す）で学習し、fold-safe の fold 0 は含む。`lineage-reproducibility.md` 4 行に記録された使われ方は v4b ランカーの特徴（F セル 37 の model_views）だけ（候補の点として直接並べる箇所は、その調べでは記録されていない）
  - 確認（学習ログ `d2_p100_60k_summary.json`、作者の GPU は不明）: 60,000 ステップの学習を再開した区間（40k → 60k の 20,000 ステップ × 128）で 5,974 秒、428.5 スペクトル/秒、最大メモリ 9.64 GB
- 確認 sanpier キット（`lineage-reproducibility.md` 18 行、`outputs/sanpier_fold0/.../model.py`・`config_used.json`）: スペクトルの小さな Transformer（幅 192、3 層）と、指紋 2,048 ビットの MLP を**一から**学習し、正規化した 192 次元の cosine で並べる。損失は温度 0.1 の候補ベース InfoNCE（難しい負例 16、負例プール 128）＋指紋の補助損失。**これは JESTR・Emb-Cos と同じ型の整列チャネルそのもの**
  - 確認（#54・#60）: fold-safe に学習し直した重み（hold/np と v3 の正解 55,369 構造を外す、リーク 0）を縮小版 score+prior+frag に等重みの z 合成で足した結果は**不採用**: v3_c1 +0.0194 [−0.0008, +0.0404]、v3_c2 0、enveda_v1 −0.0209 [−0.0489, +0.0048]、v3_c2n −0.0255 [−0.0565, +0.0051]（`outputs/local_eval/sanpier_v3_check.json`）
- したがって、**この競技で新しいのは「対照学習で整列すること」ではなく、「スペクトル側に GeMS で事前学習した凍結 DreaMS を使うこと」だけ**。縮小版（`scripts/local_eval/run.py`）には DreaMS 由来の部品が 1 つも無い

## 5 問への答え

### (a) 学習コードの公開とライセンス
- MSAlign: 学習・評価・前処理のコードが公開（MIT、確認）。データ・分割・候補・重みは Zenodo 23372807（CC BY 4.0、確認）
- JESTR: 学習コード公開（MIT、確認）。データは NPLIB1 だけ公開、重みは NPLIB1 と MassSpecGym（README、確認）
- DreaMS: コード MIT、重み CC BY 4.0（確認）
- MolDeBERTa: 重み CC BY-NC-ND 4.0（確認）。GitHub `pcdslab/MolDeBERTa` は API の license が null（ライセンス表示が無い。確認）
- 自前でやる場合: MSAlign のコードは MIT なので、整列 MLP の部分（`models/MSAlign/`）を参考に自前のスクリプトを書けばよい。この競技の train.parquet と v2 プールに合わせたデータ読み込みは自前で書く必要がある（MSAlign の datamodule は MassSpecGym と Spectraverse の形式向け。確認）

### (b) 事前学習の重みが 748882 の条件に当てはまるか
- DreaMS `ssl_model.ckpt`: **当てはまると読む**（裁定の文言と論文は確認、当てはめは読み）。748882 A.1 が「NIST などの禁止データ由来でない限り、自己教師ありの backbone を含む」と明示し、学習データ GeMS は MassIVE GNPS のラベル無しスペクトルで NIST ではない（論文で確認）。注意: 事前学習の途中の線形プロービングに NIST20+MoNA の 6,000 件を使っている（上の DreaMS の節）。プローブは凍結したモデルの上の別の線形層なので backbone の重みは NIST から作られていないと読むが、配布の重みの選択にプローブの値を使ったか、748882 の「NIST 由来」がこれを含むかは**未確認**（748882 で yesterdays_rebel が 2026-11-05 に「fe_D の土台の公式 SSL チェックポイントの config は NIST20+MoNA をプロービング集合として挙げている（学習の目的ではなく監視用と読める）が、許可に入るか」と質問済みで、取得時点で主催者の返答は無い。`lineage-reproducibility.md` 4 行でも要確認としていた点）。渡すもの: Zenodo 10997887 の md5 を固定した重み、DreaMS の学習コード（GitHub、commit を固定）と GeMS への参照、自前の学習・推論コード
- DreaMS `embedding_model.ckpt`（MSAlign が使う版）: 754610 の「公式に配布されたチェックポイントは可」には入る（確認）。微調整は MoNA の 5,500 分子で、NIST20 は使わないと論文に明記（確認。学習ログは見ていない）。**v3 の正解構造が MoNA の 5,500 分子に入っているかは未確認**なので、リーク無しの測定には使わない
- MolDeBERTa: **未確認・使わない**。承認済みの一覧（748882 A.1）に無い。第三者の重みとしては「元の学習コードと許可データ」が要るが、学習コードのリポジトリにライセンス表示が無く（確認）、重みは ND（改変禁止）。凍結して出力だけ使う場合に ND に触れるかは法的な解釈で、主催者にも聞かれていない。この調査では使わない側に決める
- MSAlign の配布重み（出典 3）: MassSpecGym（757307 で可）で学習しているが、MolDeBERTa に依存するので上と同じ理由で使わない。加えて v3_c2n の正解の 58.5% が MassSpecGym にある（#66、`mist-massspecgym.md`）ので、リーク無しに測れない

### (c) 0.487 系統の DreaMS（fe_D.pt）との違い
- 目的: fe_D は「スペクトル → 指紋のロジット」の回帰（BCE が主、窓の対照は補助）。整列チャネルは「スペクトルの埋め込みと分子の埋め込みの cosine」で、分子側も学習する
- 学習のしかた: fe_D は DreaMS 全体（約 9,550 万パラメータの backbone ＋ 新規 2,600 万、`results.json` の `params`）を微調整する。整列は DreaMS を凍結して埋め込みを 1 回だけ計算し、MLP だけを学習する（MSAlign の型）
- 分子側: fe_D は固定の 10,226 ビット（線形の内積）。整列は同じ 10,226 ビットを入力にした学習する MLP（MolDeBERTa を使わない版。下の (e)）
- 使われ方: fe_D は v4b ランカーの特徴の 1 つで、ランカー全体が fold-safe でないので、縮小版では測れない（`lineage-reproducibility.md` 4 行）。整列は縮小版の段として fold-safe に作れる
- 重なり: どちらも同じ `ssl_model.ckpt` を土台にする。0.487 系統に足すと、同じ DreaMS の情報が fe_D 経由（ランカー）と整列経由の 2 回入る。**0.487 系統の上での上積みは、縮小版での上積みより小さい見込み**（推測）

### (d) fold-safe に学習するときの学習データと見積もり時間
- 学習データ（確認、11/13 に `data/train.parquet` の `inchikey14` 列と `data/ext/foldsafe/split.parquet` を読んで数えた）:
  - train.parquet 全体 2,539,608 スペクトル・275,810 構造
  - split.parquet（hold 55,102 ＋ np 265）の構造を外すと 1,982,405 スペクトル・220,443 構造
  - #54 と同じく v3 の 4 集合の正解も外す（`scripts/train/sanpier_fold0.py` の `drop_inchikey14` = split ∪ 各検証の正解、計 55,369 構造）。#54 の数え方（キットの前処理後）では学習構造 220,042
  - 負例: v2 プール（`pool_meta.parquet`）の ±10 ppm 窓の構造から、上の除外 55,369 構造を除いたもの（fe_D の「holdout-clean negatives」・sanpier と同じ扱い）
  - DreaMS の埋め込み自体はラベルを使わないので、hold や v3 の問い合わせスペクトルを埋め込んでも構造の答えは漏れない。ただし GeMS に v3 の問い合わせスペクトルそのもの（ラベル無し）が入っているかは**未確認**（v3 の問い合わせは timsTOF が中心で、Enveda の自社データなら MassIVE に無い見込みだが確かめていない）
- 見積もり（すべて未測定。根拠の数値は上の確認済みのものだけ）:
  - DreaMS の埋め込み（backbone 約 9,550 万パラメータ、SSL の頭を含めると約 1.16 億。下の FLOP は頭を含む上限。ピーク 100 個まで）: 学習データ 1,982,405 ＋ v3 の問い合わせ 4,756（v3_c1 1,252、v3_c2 1,215、enveda_v1 614、v3_c2n 1,675、各 `data/holdouts/<h>/test.parquet` の行数）
    - Kaggle GPU（T4）: fe_D の学習が作者の GPU で 428.5 スペクトル/秒（前向き＋逆伝播）。前向きだけなら 2〜4 倍、T4 は作者の GPU より 2〜4 倍遅いと仮定すると 200〜850 スペクトル/秒 → **約 0.7〜2.8 時間**。幅が大きいので、本番の前に 1 万スペクトルで速度を測る
    - ローカル CPU: 1 スペクトル約 2×1.16 億×100 ≈ 23 GFLOP（backbone だけなら約 19 GFLOP）。実効 0.2〜0.5 TFLOPS と仮定して 約 9〜22 スペクトル/秒 → 学習データ全体は **約 25〜64 時間で不可**。v3 の問い合わせ 4,756 件だけなら 4〜9 分の見込み
  - 整列 MLP の学習（MSAlign の設定のまま 30,000 ステップ × 128 問い合わせ × 129 候補、分子側の入力 10,226 ビット → 2,048 × 3 層 → 512）: 分子 1 個の前向き 約 3,040 万回の積和（約 61 MFLOP）× 16,512 個 ≈ 1 ステップの前向き 約 1.0 TFLOP、前後で約 3 TFLOP、計 約 90 PFLOP（候補の重複を考えない上限）
    - Kaggle GPU（T4、fp16 実効 20 TFLOPS と仮定）: **約 75 分**。上の埋め込みと同じジョブで回す
    - ローカル CPU: 約 50 時間で不可。MPS は測っていない
  - 合計: **Kaggle GPU 1 ジョブ、約 2〜4 時間**（週 30 時間の枠を他プロファイルと共有。使う前に残りを確かめる）。重みの入手はジョブ内で Zenodo から（インターネット有り、md5 照合）。出力は MLP の重み（数十 MB）と v3 の問い合わせの埋め込み（4,756 × 1,024、約 20 MB）だけを持ち帰り、学習データの埋め込み（fp16 で約 4 GB）はジョブ内で使い捨てる
  - 提出に使う段階では、`ssl_model.ckpt`（1.39 GB）を Kaggle のデータセットに置く必要がある。private の残り約 0.9 GB に入らないので、backbone だけを fp16 で保存（約 1.9 億バイトと見込む、未確認）するか、公開データセットにする。今は決めない

### (e) v3 の 4 集合で測る設計（1 つの形）
- 新しい段 `align`（`scripts/local_eval/run.py --stages score,align,prior,frag`）
  - スペクトル側 g: DreaMS `ssl_model.ckpt`（凍結）の埋め込み 1,024 次元 → MSAlign と同じ形の MLP（`n_hidden_layers_ms_post 2`、衝突エネルギーと付加イオンの 64 次元を連結）→ 512 次元に正規化
  - 分子側 h: v2 プールの 10,226 ビット指紋（`pool_fp.npy`、FPNet と同じビット）→ MLP 3 層 → 512 次元に正規化。MolDeBERTa は使わない（(b) の理由）
  - 点: 分子の各スペクトルの g の平均を正規化したものと h の cosine
  - 学習の設定は `massspecgym_formula.yaml` の値をそのまま使う（`k_candidates 128`、`batch_size 128`、`n_max_steps 30000`、`lr 1e-4`、`weight_decay 1e-6`、`warmup 2000`、`dropout 0.30`、`temperature_init 0.07`、`gradient_clip 5.0`）。**最後のステップの重みを使い、検証で途中の重みを選ばない**。候補が 128 に満たない問い合わせは MSAlign と同じくパディングして無効扱い
  - 埋め込みの代表値は DreaMS の precursor トークンの最終層（DreaMS `dreams/api.py` commit dbec3a0 の `dreams_intermediates`（`transformer_encoder.ffs[i]` へのフック）は `precursor_only=True` で `output[:, 0, :]` を取る＝確認。`ssl_model.ckpt` を読んだときに同じ出力になるか（`DreaMS_embedding` は `embedding_model.ckpt` の頭を通す）は実装時に確かめる＝未確認）
- 混ぜ方（#60 の sanpier と同じ、1 つだけ）: 分子ごとに、±10 ppm 窓のプール候補の FPNet の点 s と整列の点 a をそれぞれ z 化し、s' = μ_s + σ_s·(z_s + z_a)/2（等重み。重みは定義から決め、検証で選ばない）。そのあと既存の prior（μ 0.15）と frag（λ 0.5）をそのまま当てる。a が付かない分子（窓の候補 1 個、σ_a = 0、使えるスペクトル無し）は s のまま
- 対照: score+prior+frag（v3_c1 0.8535 / v3_c2 0.0117 / enveda_v1 0.8376 / v3_c2n 0.6284、#97 で cmp 同一を確認済み）
- 採否（#97 と同じ型。結果を見て変えない）: `bootstrap_diff`（`scripts/ranker/train_eval.py`、2,000 回、seed 0）の「新 − 対照」の 95% 下限が **4 集合すべてで ≥ −0.02、かつ v3_c1 の下限 > 0** なら採用、それ以外は不採用。v3_c2 は対照の値が除外漏れ 3 件だけで決まる（#92）ので ≥ −0.02 の確認にだけ使う。必須の集合が欠けたら不採用
- リークの確かめ: 学習に使った inchikey14 と split（hold/np）・4 集合の正解の重なりを `leak_audit`（`scripts/train/sanpier_fold0.py`）と同じ数え方で出し、0 でない集合は測定不能
- 事前登録: 上の全部を、学習のジョブを流す前に新しい Issue に書き、独立レビュー PASS をもらう
- 期待: **小さい**。同じ型で一から学習した sanpier は enveda_v1 と v3_c2n で下限 −0.02 を割った（#60）。論文で DreaMS が足した分は指紋側で 34.3 → 38.6（MassSpecGym、主張）で、しかも MSAlign の数値は `ssl_model.ckpt` でなく `embedding_model.ckpt` のもの。v3_c1 の下限 > 0 を満たす見込みは五分以下と読む（推測）
- 縮小版の上の改善は 0.479／0.487 を超える根拠にならない（縮小版の score+prior は LB 0.282、#37）。採用になっても、0.487 系統に載せる 1 セルの診断提出は、読み方の帯を事前に固定してから別に考える

## 結論: 次の計画に学習を入れるか
**入れる（条件つき、Kaggle GPU 1 ジョブ）。** 理由:
- DreaMS の事前学習（GeMS-A10、ラベル無し 2,400 万スペクトル）は、縮小版に無い情報源で、規則上も使える（748882 A.1）。整列の MLP、データ読み込み、負例の取り方はすべて自前のコードで作れ、最終提出の「自前のコードで作り直せる」条件を満たす
- 費用は見積もりで 2〜4 時間の GPU 1 本（1 日 1 本の枠に入る）。埋め込みは一度作れば、他の DreaMS 由来の段（例: DreaMS 埋め込みでのライブラリ近傍）にも使い回せる
- 条件: (0) (b) の線形プロービングの点を事前登録の Issue に書き、最終提出に使う前に主催者の裁定で確かめる（v3 での測定そのものは規則の判断を待たずにできる）、(1) 学習の前に、上の (e) を Issue に事前登録して独立レビュー PASS、(2) 本番の前に同じカーネルで 1 万スペクトルの速度を測り、外挿が 6 時間を超えるなら止める、(3) Kaggle GPU の週の残りを確かめる、(4) v3 の問い合わせの埋め込みと MLP の重みだけを持ち帰り、ローカル CPU で 4 集合を 1 回だけ測る
- 入れない場合の理由になりうること（記録しておく）: sanpier（同型、一から学習）が不採用だった。MolDeBERTa を使わない分、論文の主要な伸びは手に入らない。期待は小さく、外れたら GPU 数時間ぶんの損になる

## 試さないこと
- MSAlign のスコア融合（重みと温度を検証の R@1 で選ぶ）: ハイパラ調整にあたる
- MolDeBERTa・MSAlign の配布重み: (b) のとおり
- JESTR の公開重み: MassSpecGym で学習、公開実装の再現に問題の報告（Issue #5）があり、DGL と GPU が要る。型は sanpier と同じで、新しい情報源を足さない
