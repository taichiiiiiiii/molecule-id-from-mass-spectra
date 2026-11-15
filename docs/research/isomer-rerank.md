# 調査: 同じ分子式の異性体の並べ替えを自前で学習できるか（#118）

計画 2026-11-15 昼 項目 4（夜の計画 `plan_2026-11-15_night.md` 項目 3 からの持ち越し）。親 #3、#108。**読むだけ**: GPU 0、学習 0、提出 0。数え上げはローカル CPU で、最大 RSS 135 MB。2026-11-15 15:00〜16:00 JST に読み、数えた。

印の意味: **確認** = 原文・コード・API の応答・こちらの出力を読んだ。**主張** = 著者や投稿者の申告で、こちらでは再現していない。**未確認** = 調べたが分からない（理由を書く）。見積もりの数値はすべて未測定で、根拠を横に書く。

## 出典

1. heliosli「casmi26-0.489」https://www.kaggle.com/code/heliosli/casmi26-0-489 （`kaggle kernels pull heliosli/casmi26-0-489 -m`、15:06 JST。ipynb は sha256 e102d021…、26 セル。11/14 に読んだ「casmi26-0-488」版 2 と同じ sha256。0-488 の slug は今は pull できない）。rr3 のセルは 14・23・24
2. ICEBERG: Goldman, Li, Coley, "Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks", arXiv:2304.13136v2（論文のライセンス CC BY-NC-SA 4.0）。HTML 版の全文を取得し、retrieval の節（図 5）、§3.5、付録 S1.1（CASMI22）を読んだ。コード https://github.com/samgoldman97/ms-pred （GitHub API の license = MIT、最終 push 2026-11-03）。README を読んだ
3. FraGNNet: Young, Wang, Wishart, Wang, Greiner, Röst, "FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction", arXiv:2404.02360v2、TMLR 08/2025（CC BY 4.0）。HTML 版の全文を取得し、§5.2・表 2・付録 A.11 を読んだ。コード https://github.com/FraGNNet/fragnnet （license = BSD-2-Clause、最終 push 2026-11-13）。README を読んだ
4. MassSpecGym（Bushuiev et al., NeurIPS 2024 D&B、arXiv:2410.23326、コード https://github.com/pluskal-lab/MassSpecGym 、license = MIT）。retrieval の節は 11/13 に読んだ（`docs/research/contrastive-alignment.md` 出典 7）。今回はライセンスだけ API で確かめ直した
5. 主催者の回答 748882・754610・756893・757307（11/11〜11/13 に `kaggle competitions topics show` で取得し、`docs/research/glacier-iceberg-provenance.md`・`contrastive-alignment.md`・`organizer-question-756893.md` に原文で記録したもの。今回は取り直していない）
6. 競技の議論の一覧: `kaggle competitions topics list … --csv -p 1〜15`（重複を除いて 161 件）の題を検索した。isomer・rerank・rr3・heliosli などを含む題は 745468、755089、744556、746878、752234、741745。745468「Problem is in Isomer identification …」は本文が CLI の json に入っておらず（コメント 1 件「Thank you!」だけ）、読めなかった（**未確認**）
7. リポジトリの中: `scripts/local_eval/run.py`（frag 段・hidden の当て方）、`scripts/train/align_train.py`（#109、feat/109 ブランチ）、`outputs/local_eval/sanpier_v3_check.json`（#60）、`docs/research/lineage-reproducibility.md` 12 行（MetFrag-lite `frag_rescore.py`）

## 要点

### rr3（出典 1）
- 確認（セル 24 の冒頭のコメント）: 「same-formula isomer reranker」。各候補を 1〜2 本の結合で切った断片に観測ピークを合わせ、グラフネット＋transformer で点をつける。FREEZE の規則（`rr3_config.json` の mode / alpha / delta、`rr_v3.py`）は、学習集合にある構造の候補を元の順位に固定し、ほかの候補だけを並べ替える。lib_max ≥ LIB_TAU の分子には触れない。ベンチの 3 分子の自己テスト（相関 ≥ 0.999）に通らなければ使わない
- 主張（同じコメント）: 「学習集合の構造に、同じリストのほかの候補より約 0.34 SD 低い点をつける」。それを直すために FREEZE を入れた、とある
- 確認（コード）: 重みとコードは、添付データセットのどこかにある `rr3_config.json` のフォルダから読む（`find("rr3_config.json")`、`.py` を `/tmp/rr3code` に写して `rr_deploy`・`rr_v3` を import）。セル 14 のコメントに「The private 0.475 model freezes known training candidates. This public analogue …」とある
- 確認: `dataset_sources` 9 件のうち 2 件が空文字（名前が出ない＝非公開と読む）。名前の出る 7 件（sanpier/casmi-spectrum-model-v1、dmitriigluzdov/casmi26-fold-safe-fpnet・…-natural-product-knowledge-table・…-natural-product-panel-benchmark・…-pubchem-popularity-prior・…-attributed-v4g-reproduction-assets、sanpier/catalogue-experiment-saner）は、`kaggle datasets files <id> --page-size 200 --csv` に rr3・rr_v3・rr_deploy・reranker を含むファイル名が無い。…-attributed-v4g-reproduction-assets は 237 ファイルで 200 件目のあとに次のページがあり、私の 1 回目の確認はそこで止まっていた。独立レビューが `--page-token` で残りを読み、そこにも無いことを確かめた。heliosli の公開データセット 9 件（`kaggle datasets list --user heliosli`）に CASMI のものは無い。セルが引く `reports/rr3_on468_criteria.md`・`reports/reranker_criteria_v3.md` も公開物の中に見つからない
- 未確認: rr3 の学習データ（v3 の正解を含むか）、学習コード、ライセンス、効果（題の 0.489 は作者の申告で、中身の説明は無い）

### ICEBERG（出典 2）
- 確認（retrieval の節、図 5 の説明）: 異性体での検索のベンチ。テストのスペクトルごとに、PubChem から正解と Tanimoto が最も近い「decoy isomer」を最大 49 個取り、予測スペクトルとの cosine で並べ、top-k の正答率を出す。NPLIB1（GNPS の天然物）と NIST20 で測っている
- 主張: NPLIB1 の top-1 で次点に約 9 ポイントの差（相対 46%）。NIST20 の top-1 は MassFormer と同等。CASMI22 の正イオン 303 スペクトル（PubChem の全異性体が候補、NPLIB1 で学習したモデル）で top-1 12.9%（付録 S1.1 表 1）
- 確認（README）: 学習・評価のコードが公開されている。MassSpecGym 1.5 用のデータ処理（`msg_all`）と MassSpecGym で学習した ICEBERG 2.1・GLACIER の重みの配布がある。NIST20/23 は有償のデータとして別扱い
- この競技との関係（確認、`glacier-iceberg-provenance.md`）: ICEBERG・GLACIER の MassSpecGym 版はすでに 0.487 系統の中にあり（模擬スペクトルの証拠）、主催者が名指しで承認している（744338・745841）。縮小版（v3）には入っていない

### FraGNNet（出典 3）
- 確認（§5.2）: 検索の評価は、テストの各分子 X に対して、PubChem から X と Tanimoto の高い 49 分子を取って候補集合を作り、予測スペクトルと実スペクトルの類似度で並べる。本文のこの段には「同じ分子式」とは書かれていない（ICEBERG の設定と同じとある）
- 主張（表 2、NIST20、5 seed）: top-1 は InChIKey 分割で FraGNNet-D4 36.0、ICEBERG 36.0、MassFormer 24.2。Scaffold 分割で 30.3、27.9、20.4
- 確認（README）: 学習スクリプトがあり、対照・ペアの学習に使う「分子式ごとの難しい負例のサンプラー」（`formula_hard_negative_dynamic_sampler`）がある。論文の学習データは NIST20
- 未確認: MassSpecGym などの許可されたデータで学習した FraGNNet の重みが配布されているか（README には見当たらない）

### MassSpecGym（出典 4）
- 確認（11/13 の記録）: retrieval の候補は正解と同じ質量の分子で、1 スペクトルあたり最大 256。組成式を与える「bonus」設定もあり、これは同じ式の異性体の並べ替えのベンチにあたる。MIT

### 主催者の条件（出典 5、原文は各文書）
- 748882（10-13）: 承認済みの MassSpecGym・GLACIER/ICEBERG・DreaMS のチェックポイントは、配布のチェックポイント、著者の学習コードと許可された学習データへの参照、自前の微調整・推論の全コードを渡せば足りる。DreaMS の自己教師ありの backbone は、NIST などの禁止データ由来でない限り承認に含む
- 748882（10-14）: 元の学習コードと許可データで再現できない第三者の重みは不可
- 756893（David Healey、11-11）: 「Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible.」後半（自前の生成器で作り直した表は可か）は未回答（下書きはまだ投稿していない）
- 757307: MassSpecGym と MIST のモデルは問題ない

## 測定（手順 3、読むだけ）

- 入力（元のフォルダ、読むだけ）: `outputs/local_eval/<集合>/submission_score+prior+frag.csv`（v3_c1 は 11/14 20:23、sha1 b9d0ddcd…、クローンのコピーと同じ）、`outputs/holdouts/<集合>/truth.csv`。分子式は候補と正解の SMILES から RDKit の `CalcMolFormula`。同一の判定は `scripts/validate/mrr25.py` の `to_key`（tautomer の正準化 → InChIKey14）
- スクリプト（scratch、PR には入れていない）: `cache/scratch/n1115n_iso/{miss_classify.py,c_similarity.py,stage_trace.py}`。出力 `miss_<集合>.json`
- 確かめ: スクリプトで計算し直した MRR@25 は、各集合の `result_score+prior+frag.json` の値と一致する（v3_c1 0.8535、enveda_v1 0.8376、v3_c2n 0.6284、v3_c2 0.0117）。v3_c2 は #116 の作り直し**前**の出力（作り直し後の対照は 0.0000、PR #117 未マージ）

### 1 位が外れた分子の内訳（score+prior+frag）
- **v3_c1**（200 分子）: 1 位の正解 151、外れ 49 = (a) 3 ＋ (b) 2 ＋ (c) **44**（合計 49 で一致）
  - (c) の正解の順位: 2 位 31、3 位 9、4 位 2、8 位 1、20 位 1
  - (a) の 3 件は、どれも上位 25 に正解と同じ分子式の候補がある（25 位より下や窓の外にいるかは数えていない）
  - (b) の 2 件の正解の順位: 3 位、5 位
- 参考 **enveda_v1**（プールにあり参照なし）: 外れ 49 = (a) 7 ＋ (b) 3 ＋ (c) 39
- 参考 **v3_c2n**（Enveda の装置の問い合わせ、参照なし）: 外れ 95 = (a) 25 ＋ (b) 6 ＋ (c) 64。(a) のうち 18 件は上位 25 に正解の分子式が無い
- 参考 **v3_c2**（正解がプールに無い、作り直し前）: 外れ 198 = (a) 197 ＋ (b) 1 ＋ (c) 0。(a) のうち 185 件は上位 25 に正解の分子式がある
- 上位 25 のうち正解と同じ分子式の候補の数（(c) の分子の中央値）: v3_c1 22、enveda_v1 23、v3_c2n 25。数えたのは (c) の分子の**上位 25 の中**だけで、窓全体は数えていない。上位 25 の中はほとんど同じ式の候補なので、**分子式の絞り込みでは (c) の外れを直せない**

### (c) の 1 位と正解の近さ（Morgan 半径 2・2,048 ビットの Tanimoto、Murcko 骨格）
- v3_c1（44 件）: Tanimoto の中央値 0.491、≥ 0.85 が 0 件、≥ 0.7 が 3 件、< 0.5 が 25 件。骨格一致 7 件
- enveda_v1（39 件）: 中央値 0.484、≥ 0.85 が 0 件、骨格一致 14 件
- v3_c2n（64 件）: 中央値 0.523、≥ 0.85 が 12 件、骨格一致 32 件
- 読み: v3_c1・enveda_v1 では、1 位に来た別の異性体は、正解の位置異性体のような「1 か所違い」ではなく、骨格から違うものが多い。断片（ピークの説明）で見分けられる余地がある、と読める（推測。断片の説明で実際に分かれるかは測っていない）

### (c) は段のどこで生まれたか（同じ分子、`submission_score.csv`・`submission_score+prior.csv`）
- v3_c1 の (c) 44 件のうち、score だけで 1 位だったのは 4 件、score+prior で 1 位は 1 件。39 件は score の段から 2 位以下（上位 25 の中）、1 件は score の段では上位 25 の外（prior・frag で上位 25 に入った）
- enveda_v1 の (c) 39 件: score だけで 1 位 6 件、score+prior で 4 件
- 読み: 外れの大半は FPNet の点（score）の段からあり、prior・frag の段が作った外れは少ない

## 5 問への答え

### (i) rr3 の学習コード・重みの出所とライセンス
- 学習コード: **見つからない**。調べた場所: ノートのセル 14・23・24（推論とオーバーレイだけ）、名前の出る添付データセット 7 件のファイル一覧、heliosli の公開データセット 9 件、議論の一覧 161 件の題（rr3・heliosli を含む題は無い）。11/12・11/14 の調べ（`top-notebooks-2026-11-12.md`）でも、heliosli の CASMI のデータセットは非公開（403）
- 重み: コードで確かめたのは、`/kaggle/input` の下の `rr3_config.json` のフォルダから読むことまで（確認）。それが名前の出ない（非公開の）添付データセットにある、というのは**推定**（名前の出る 7 件には無い、という消去法）。ライセンスは**未確認**（データセットが見えない）
- この競技で使えるか: **今の時点では使えない**。学習コードと学習データが公開されない限り、748882（10-14）「再現できない第三者の重みは不可」と 756893 に当たる（748882 の 10-21 の回答では、締切までに学習コードと許可データが揃えば可）。写すのも方針（自前の部品）に反する。使うのは考え方（候補の断片にピークを合わせて点をつける、同じリストの中で並べ替える）だけ

### (ii) 公開の学習コードがある方法とライセンス、748882・756893 の条件
- ICEBERG・GLACIER（ms-pred、MIT。論文は CC BY-NC-SA）: 学習コードと MassSpecGym の重みがある。748882 の 1 番に当てはまる。0.487 系統にすでに入っている。残る穴は `glacier-iceberg-provenance.md` のとおり（Kaggle の重みと上流の一致、推論の包みのライセンス）
- FraGNNet（BSD-2-Clause、論文 CC BY 4.0）: 学習コードと、同じ式の難しい負例のサンプラーがある。論文の学習データは NIST20 で、748882 は NIST 由来を禁じるので、使うなら**許可されたデータ（MassSpecGym か競技の train）で学習し直す**必要がある。その重みは配布されていない（未確認）
- MSAlign（MIT、重み CC BY 4.0）・JESTR（MIT）: 候補（同じ質量）を負例にした対照学習。#99 の調べで、MSAlign の分子側 MolDeBERTa は CC BY-NC-ND で使わないと決めた。この型は #109 の整列 MLP として自前で学習中
- MetFrag-lite `frag_rescore.py`（Apache-2.0 とノートのコメントにある、学習なし）: 縮小版の frag 段として入っている
- 756893 の条件: 自前のコードで作った中間物は「再現できる」側に入る。他人の生成コードの無い表や重みを読む解は入らない。**自前で学習する並べ替え器は、学習コードと学習データ（競技の train＋許可された公開データ）を解と一緒に出せば条件を満たす**（748882・756893 の文面からの読み。主催者に個別に確かめてはいない）

### (iii) 手順 3 の数から見た伸びしろ
- v3_c1 の外れ 49 のうち (c) が 44（90%）。スキルの規則の前提「外れの大半が同じ分子式の異性体」は v3_c1・enveda_v1（39/49）で成り立つ。v3_c2n は 64/95（67%）で、(a) 25 が残る
- 上限: (c) をすべて 1 位にできたとすると、v3_c1 は 0.8535 → 0.9777（+0.124）、enveda_v1 0.8376 → 0.9552（+0.118）、v3_c2n 0.6284 → 0.8549（+0.227）。実際には今の 1 位の正解（v3_c1 で 151）を崩す損があるので、これは届かない上限
- (c) は 2 位に集まる（v3_c1 で 31/44）。2 位の正解を 1 位にすると 1 件 +0.5/200 = +0.0025。31 件すべてなら +0.078
- 測れる大きさ: #60 の sanpier の v3_c1 の 95% 区間の幅（[−0.0008, +0.0404]、幅 0.041）から逆算すると、200 分子の 1 集合で検出できる差（MDE ≈ 2.8 × SE）は約 0.03（推定。変更ごとに Δrr のばらつきは違う）。2 位の正解を 1 位に上げる正味の件数が、崩す件数より 12 件ほど多くないと（12 × 0.5 / 200 = 0.03）見えない
- 前例: 同じ型（窓の中の候補を負例にした対照学習、一から学習）の sanpier は、fold-safe で v3_c1 +0.0194 [−0.0008, +0.0404]、enveda_v1 −0.0209 [−0.0489, +0.0048]、v3_c2n −0.0255 [−0.0565, +0.0051] で**不採用**（#60）。窓の候補（大半は同じ式と見込む。窓全体の式は数えていない）を負例に学習しても、それだけでは足りなかった、と読む（#60 は同じ式の効果だけを切り分けていない）
- 0.487 系統での伸びしろ: 縮小版のこの内訳が 0.487 系統（ICEBERG・GLACIER、Engine 2 のランカーを含む）でも同じかは**測っていない**。0.487 系統の候補表（`variants/*.csv`）を v3 で出す #112 Tier 1 ができるまで、系統の上の (a)(b)(c) は分からない

### (iv) fold-safe の学習の設計（案。事前登録の前に Issue でレビューする）
- 何を学習するか: 「断片の説明の並べ替え器」。候補ごとに 1〜2 本の結合を切った断片（`frag_rescore.py` の断片列挙を再利用）と観測ピークの一致を入力にした小さなネット（ピークと断片の集合を読む set/attention、または断片の一致を数値の列にした小さな MLP）で、同じ ±10 ppm の窓の候補の中の正解を当てる損失（候補ベースの InfoNCE）で学習する。点は既存の score・prior・frag に等重みの z 合成で足す（sanpier・align と同じ式、重みは選ばない）。rr3 のセルのコメントと同じ考え方だが、コード・重みは自前
  - 整列 MLP（#109、凍結 DreaMS の埋め込み × 指紋）との違い: 整列 MLP は分子を指紋だけで見る。こちらは分子の断片の質量をピークに直接合わせるので、骨格の違う異性体（v3_c1 の (c) で多い）を断片で見分ける余地がある。ただし、効果の見込みは測るまで分からない
- データ: 競技の train のうち、fold 0（`split.parquet` の fold == 0、sanpier の `sanpier_fold0_holdout_keys.json`）と v3 の 4 集合の正解（正準キー＋別名。#116 の `drop_keys` と同じ定義、`align_train.py` の `EXPECTED_DROP` 55,369 と同じ集合）を、正解としても負例としても外す。負例は v2 プールの ±10 ppm の窓の候補（`align_train.negative_csr` と同じ作り方、`leak_audit` と同じ 0 件の確認）
- 止め方（#108 の追記どおり、v3・LB を使わない）: 学習データの正準キーの 2%（seed 0）を内部検証に分ける。一定のステップごとに内部検証の InfoNCE 損失を測り、最良からの相対改善 0.5% 未満が 5 回続いたら止める。ReduceLROnPlateau（factor 0.5、patience 3、min_lr 1e-6）、上限 300,000 ステップ、最良の重みを使う。学習曲線を Issue に貼る。途中から再開できる形にする
- ローカルの時間の見込み（未測定、根拠つき）:
  - 断片の前計算: 縮小版の frag 段の秒（`result_score+prior+frag.json` − `result_score+prior.json`、起動の warmup を含む）は、約 200 分子（v3_c2n は 195）あたり 3.5〜6.8 秒（enveda_v1 3.5、v3_c2n 4.1、v3_c2 6.6、v3_c1 6.8＝37.1 − 30.3）。1 分子（60 候補まで）あたり 0.017〜0.034 秒。学習データは、同じ除外（fold 0 と v3 の正解）をした #109 の `outputs/align_train/train_report.json` で 214,772 構造・1,942,968 スペクトル（1 構造あたり約 9.0 スペクトル。v3_c1 は 6.3）。`score_molecule` はスペクトルごとに回るので、スペクトル数の比（約 1.45 倍）を掛けると、60 候補で約 1.5〜2.9 時間、窓の候補 128 個で約 3〜6 時間（CPU、推定。warmup を含む値からの外挿なので、ばらつきは大きい）。ただし `frag_rescore.py` は断片の列挙を候補の SMILES ごとにキャッシュし（`_levels_cached`）、スペクトルごとに回るのはピークの照合だけなので、費用はスペクトル数に比例せず、窓に出る異なる候補の数にも依る。enveda_v1（1 分子 3.1 スペクトル）3.5 秒、v3_c2n（8.6）4.1 秒と、秒はスペクトル数どおりに並ばない。上の値は粗い上側の見積もりで、計画した 1,000 分子の切片の測定で決める
  - 学習: 整列 MLP の MPS の速さ 1.42 ステップ/秒（#109 のベンチ、埋め込みを前計算した MLP、バッチ 128 × 候補 128）が参考。断片の集合を読むネットはこれより遅い。最短停止（評価 5 回）までの時間は、**小さな切片で速度を測ってから**決める（スキルの規則）
  - メモリ: 断片の表は、214,772 構造 × 候補 60〜128 × 断片の数（未測定）で決まる。前計算を shard に分け、memguard（空き 15%）で動かす。全部を使うか一部の構造にするかは、切片の測定のあとで、v3 を見る前に決める
- 重いジョブは同時に 1 本の規則があり、今は #109 段 b がローカルの枠を使っている。段 b が止まるまで、この学習の本番は始めない

### (v) 測り方
- v3 の 4 集合の規則（`spectrum-blind-and-np-priors.md`・`contrastive-alignment.md` と同じ型、結果を見て変えない）: `scripts/ranker/train_eval.py` の `bootstrap_diff`（分子の再抽出 2,000 回、seed 0）で (新 − 対照 score+prior+frag) の 95% 区間を出す。4 集合（v3_c1・v3_c2・enveda_v1・v3_c2n）すべてで下限 ≥ −0.02、かつ **v3_c1 の下限 > 0** で採用。1 集合でも欠ければ不採用
- C2 の条件: #116 の作り直し（PR #117）の後、v3_c2 の正解はプールに無く、対照は 0.0000。プールの中の並べ替えは v3_c2 を動かせない（正解が候補に無い）ので、v3_c2 は「≥ −0.02」の確認にだけ使い、正の条件は v3_c1 に置く。測るのは #117 の drop_keys が main に入ってから（入る前に測るなら、その旨を結果に書く）
- 学習するのが「点」（z 合成）でなくランカー・ルータになる場合: (答えの出所 × 参照の有無) の組み合わせをすべて評価し、学習にも決めた比で入れる。v3_c1 = プールにあり・参照あり（別の装置）、enveda_v1 = プールにあり・参照なし、v3_c2n = プールにあり・参照なし・Enveda の装置の問い合わせ、v3_c2 = プールに無し・参照なし。「プールに無し・参照あり」は 4 集合に無い（この組を作れるかは未確認）。今回の案は z 合成の点なので、ランカーにはしない
- 0.487 系統での確かめ: 縮小版の v3 で採用されても、それだけでは最終候補にしない（#34 の 0.246 の前例）。0.487 系統に 1 変更として足した版を、帯を Issue に固定してから 1 回だけ診断提出する。帯の案（Public の SE ≈ 0.04 から。`docs/design/engine2-rebuild.md` 145 行の推定値〈約 125〜130 分子、逆数順位の SD 約 0.45。160 分子の仮定なら 0.035、大きい方を使う〉で、測った値ではない）: Δ = LB − 0.487 が +0.08 以上（2 SE）なら改善、−0.08 以下なら失点、その間は判定なし（判定なしでは追加の提出をしない、定数を動かさない）。提出ノートの 9 時間の上限があるので、足した段の時間を先に測る

## 結論

- **v3 の縮小版の外れの大半は、同じ分子式の異性体の並べ替え**（v3_c1 で 44/49、enveda_v1 で 39/49）。(c) の分子の上位 25 はほぼ同じ式の候補なので、分子式の絞り込みでは直せない。直すには、同じ式の候補を見分ける証拠が要る。1 位に来た別の異性体は正解と骨格から違うものが多い（Tanimoto の中央値 0.49）ので、断片による証拠の余地があると読む（推測）
- rr3 は学習コードが見つからず重みは非公開なので、使わない。考え方だけを自前で作る
- **次の計画に学習を入れるか: 入れる。ただし本番の学習ではなく、小さな切片での速度とメモリの測定にとどめる**。理由:
  1. 今の自前の学習（#109 の整列 MLP）が、同じ窓の候補を負例にした異性体の並べ替えそのもので、ローカルの重い枠を使っている。重いジョブは同時に 1 本の規則があり、段 b の v3 の結果（採否）が出る前に、似た型をもう 1 本走らせない
  2. 同じ型の sanpier が不採用（#60）なので、断片の説明の並べ替え器は「指紋で見ない証拠を使う」点で違う設計にする。その違いを、学習の前に事前登録としてレビューする
  3. 速度とメモリの見込み（上の (iv)）は未測定。スキルの規則どおり、本番の前に小さな切片で測る
- 次の計画に載せる具体の項目（案）: (1) 断片の説明の並べ替え器の事前登録（設計・止め方・4 集合の規則・帯）を Issue に書いてレビュー、(2) 学習データ 1,000 分子の切片で、断片の前計算の秒・メモリと、MPS の 1 ステップの秒を測る（CPU と MPS、30 分以内）。本番の学習は #109 段 b が終わってから

## 独立レビュー
- 1 回目（Opus、作業に関わっていない delegate、読むだけ）: **FAIL**（major 1、minor 5、info 3）。数え上げ（4 集合の a/b/c、順位、上限、MRR の再計算）、rr3 のセル、論文の数値、規則の文面は一致と確認
  - major: 断片の前計算の見込みが根拠の無い「50,000 分子」だった → #109 の `train_report.json` の 214,772 構造・1,942,968 スペクトルで計算し直した
  - minor: stage_trace の 1 件の抜け／rr3 の重みの場所は推定と書く・237 ファイルのページ送り／「最終提出には使えない」を「今の時点では」に／「窓の中は」を「上位 25 の中は」に／sanpier の読みを推測と書く → すべて反映
  - info: CASMI22 は NPLIB1 で学習したモデル／frag 段の秒のばらつき 3.5〜6.8 秒／SE 0.035 の出典 → すべて反映
- 2 回目（Opus、別の delegate、読むだけ）: **PASS**（major 0、minor 2、info 1）。1 回目の指摘がすべて入っていること、新しい時間の見積もりの計算（214,772 × 0.0175〜0.034 × 1.445 = 1.5〜2.9 時間）を確認
  - minor: 断片の列挙は候補の SMILES ごとにキャッシュされるので、スペクトル数の比での外挿は粗い → 上側の見積もりと書き、切片の測定で決めると追記／帯の SE は出典が 0.04 を採っている → SE 0.04、帯 ±0.08 に直した
  - info: v3_c2n は 195 分子 → 「約 200 分子」に直した

## PR について
- 計画の手順 6 の条件どおり、項目 3（#109 段 b の学習）がクローンで動いている（`.agent.lock` の pid 6022、ブランチ feat/109-align-local-train）ので、クローンのブランチを切り替えなかった。**PR は未作成で、引き継ぎ**。この文書の全文は #118 のコメントにある。段 b が終わったあと、クローンで `docs/118-isomer-rerank` を origin/main から切り、この文書を `docs/research/isomer-rerank.md` として入れて PR を出す（数え上げのスクリプトを入れるなら `scripts/research/` に）

## 再現のコマンド
```
S=~/.hermes/profiles/massspec/cache/scratch
O='/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra'
ENV="NUMBA_CACHE_DIR=$S/n1115n_numba PYTHONPYCACHEPREFIX=$S/n1115n_pyc"
for h in v3_c1 enveda_v1 v3_c2n v3_c2; do
  env $ENV "$O/.venv/bin/python" $S/n1115n_iso/miss_classify.py "$O" /Users/taichi/work/molecule-id-dev $h score+prior+frag $S/n1115n_iso/miss_$h.json
done
env $ENV "$O/.venv/bin/python" $S/n1115n_iso/c_similarity.py "$O" v3_c1 score+prior+frag $S/n1115n_iso/miss_v3_c1.json
env $ENV "$O/.venv/bin/python" $S/n1115n_iso/stage_trace.py "$O" /Users/taichi/work/molecule-id-dev v3_c1 $S/n1115n_iso/miss_v3_c1.json
kaggle kernels pull heliosli/casmi26-0-489 -p $S/n1115n_iso/helio489 -m
kaggle datasets list --user heliosli --csv
for p in $(seq 1 15); do kaggle competitions topics list enveda-CASMI26-molecule-id-mass-spectra --csv -p $p; done
gh api repos/samgoldman97/ms-pred -q .license.spdx_id   # FraGNNet/fragnnet、pluskal-lab/MassSpecGym も同様
```

## 追記（2026-11-16 夜 項目 3、PR 化）
- 上の本文は #118 の 1 番目のコメント（独立レビュー 2 回目 PASS の版）と同じ文面のまま入れた（末尾の空白だけ違う）。数え上げのスクリプト 3 本は scratch から `scripts/research/isomer_rerank/` に写した（中身は変えていない）。再現のコマンドの `$S/n1115n_iso/<名前>.py` は `scripts/research/isomer_rerank/<名前>.py` と読み替える
- 次の段（事前登録と 1,000 構造の切片での速度測定）は別の Issue で行う
