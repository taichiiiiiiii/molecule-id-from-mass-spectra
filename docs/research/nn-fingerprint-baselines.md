# 調査: 最近傍の指紋予測の基準と、自前チャネル `nnfp` の案（#206）

Issue #206（Refs #195 B8・C3）。計画 2026-11-23 昼 項目 3（カード t_35fa39df）。読むだけの調査で、採点・提出・GPU・学習はしていない。手元で動かしたのは、出典 1 のコードの部分式の割り当て関数の速さを作り物のピークで測った 1 回だけ（§5.7、数秒）。v3 の集合は一度も採点していない。

## 1. 出典（読んだ範囲）

1. **Khoo, L. M. S.「Nearest-neighbour baselines for fingerprint prediction from MS/MS spectra under different assumptions」** arXiv:2610.02249v1（2026-09-30）。https://arxiv.org/abs/2610.02249 ・ HTML https://arxiv.org/html/2610.02249 ・ DOI https://doi.org/10.48550/arXiv.2610.02249
   - 読んだ範囲: HTML の全文（本文 §1〜§2、表 1〜4、補足 S1〜S6、参考文献）。PDF（sha256 19c0dddb…）は取得したが、HTML と同じかの突き合わせはしていない
   - コード: https://github.com/serenaklm/ML_MS_analysis （HTML の中のリンク。GitHub API の license = MIT、最後の push 2026-09-30）。`benchmarked_models/nearest_neighbour/README.md`・`run_nn.py`（sha256 e2e3a4c3…）・`utils/nn_lib.py`・`utils/subformula_fallback.py` を読んだ。`nn_driver.py`・`peak_similarity.py`・`audit_metrics.py` は**未読**
2. **Bushuiev et al.「MassSpecGym: A benchmark for the discovery and identification of molecules」** NeurIPS 2024 Datasets and Benchmarks。arXiv:2410.23326v3、https://arxiv.org/abs/2410.23326
   - 読んだ範囲: HTML の §3.2（課題と指標）・§3.3〜3.4 の一部・§4 の基準の方法の説明・表 5（検索）。表 3・4・6・7 は見出しだけ。補足は**未読**
3. **Gupta, V. et al.「Confronting spurious evaluations of computational methods in small molecule mass spectrometry」** bioRxiv v2、DOI https://doi.org/10.64898/2026.05.03.722532
   - 読んだ範囲: HTML の本文と方法のうち、取れた文（構造だけのモデル・Spectraverse・17 モデルの比較・最近傍の基準の方法）。図 4 と補足図 5 の値（棒の高さ）は画像なので**未読**。本文の中の一部（約 5 万字）は取得の出力が切れて**未読**
4. **Asher, G. et al.「LLMs Depend on Specialist Tools for Out-of-Distribution Generalization in MS/MS Structural Elucidation」** arXiv:2611.03308v1（2026-11-02）。https://arxiv.org/abs/2611.03308
   - 読んだ範囲: HTML の要旨・§1〜結論（表 1 の数は本文の文に出てくるものだけ）と付録 F の本文。付録 D・E の中身は**未読**
5. （引用のみ、本文は**未読**）Khoo & Barzilay「Why machine learning fails at mass spectrometry for small molecules」Nature Metabolism 8, 1247–1249 (2026)、https://doi.org/10.1038/s42255-026-01544-6 （購読制。出典 1 の仮定 2〈同じ分子式の中の最近傍〉の元の Comment）
6. （この repo の既存の記録）MIST の使用条件と v3 との重なり: `docs/research/mist-massspecgym.md`（#66）。lfus の結果: #150・`outputs/local_eval/lfus_v3_check.json`。fold-safe の分割が同一性だけで分けていること: `docs/research/msg-in-the-wild.md`（#154）

CSI:FingerID（Dührkop 2015）・MIST の論文そのもの（Goldman 2023）の表は今回は読んでいない。MIST の数は出典 2 の表 5 の行だけを使う。

## 2. 出典 1（Khoo）の要点 — 表の見出し・方法の文から

### 2.1 何を測っているか（候補集合の定義）
- **候補の構造を並べる検索の課題ではない。** 測っているのは、予測した指紋と正解の指紋の近さだけ: 表 1 の見出し「Binary Jaccard index (Tanimoto similarity) between the predicted and the true fingerprint (Morgan, radius 2, 4096 bits). Predictions are binarised at 0.5. Mean over the full test split.」、表 2 の見出し「Cosine similarity between the raw prediction … and the true fingerprint」
- 論文の「候補」は、構造の候補ではなく**最近傍を探す train のスペクトルの集まり** 𝒞(q):
  - 仮定 1（分子式なし、S3）: 𝒞(q) = 𝒯（train の分割全体）。最も似たスペクトルの指紋をそのまま返す。「This is the nearest-neighbour baseline of Gupta et al. (2026), which there ranks the candidate structures of a retrieval task by their similarity to FP(t̂(q)); here the retrieved fingerprint itself is evaluated.」
  - 仮定 2（分子式あり、S4）: 𝒞(q) = 同じ中性の分子式の train スペクトル（無ければ 𝒯 全体に戻る）。**質量の窓ではなく、正解の分子式との完全一致**
  - 仮定 3（正解の分子式でピークを注釈、S5）: 正解の分子式から作った前駆イオンの式で各ピークに部分式を割り当てる。割り当て自体は予測（質量の分解）。2 つの変種: (i) `+ sub-formula`（同じ分子式の train があれば仮定 2、無ければ部分式の規則）、(ii) `Sub-formula rule only`（全スペクトルを部分式の規則で。固い分子式の絞り込みは無く、分子式は α·s_F でだけ入る）
- 部分式の規則（S5）: score(q, t) = s_SF(q, t) + α·s_F(q, t)。s_SF は断片の式のトークンと中性損失の式のトークン（前駆イオンの式 − 断片の式）の疎なベクトルの cos、s_F は異種原子に重み w を掛けた Bray–Curtis 類似度（同じ式なら 1）。予測は上位 k の train の指紋の softmax 重み付き平均（w_i ∝ exp((s_i − s_1)/T)）。上位 100 本のピーク、許容 max(0.01 Da, 20 ppm)、緩い原子価の検査
- 定数（w, α, T, k）は各データ・各分割の**検証の分割**で格子から選ぶ（「the test split is never used for any choice」）。選ばれた値は表 4（例: MassSpecGym scaffold の rule only で w 8・α 1・T 0.05・k 20、NPLIB1 scaffold で w 4・α 1・T 0.05・k 10）
- データと分割（表 3）: NPLIB1（scaffold: train 12,909 / test 2,689、random: 12,800 / 2,744）、MassSpecGym（scaffold: 76,243 / 16,042、random: 75,828 / 16,250）。表 3 の「Leakage」（test の構造の第 1 ブロックが train にある割合）は NPLIB1 scaffold 1.6%・random 62.9%、MassSpecGym scaffold 14.3%・random 95.3%。「FP oracle」（train の指紋 1 つを返す方法の上限）は 0.523 / 0.870 / 0.562 / 0.977

### 2.2 主な数値（表 1、Jaccard。並びは NPLIB1 scaffold / NPLIB1 random / MSG scaffold / MSG random）
- 仮定 1 の最良（DreaMS の埋め込みの cos）: 0.263 / 0.691 / 0.299 / 0.895。修正 cos: 0.236 / 0.517 / 0.246 / 0.700
- 仮定 2 の最良（DreaMS）: 0.259 / 0.745 / 0.308 / 0.945
- 仮定 3 の rule only: 0.337 / 0.757 / 0.381 / 0.937。各列の最良は rule only（scaffold の 2 列）と DreaMS + sub-formula（random の 2 列: 0.758・0.947）
- 本文 §2 の文: 「Using the formula to annotate the peaks gives the best results on every dataset and split (best Jaccard index 0.337, 0.758, 0.381 and 0.947 …)」。仮定 2 からの上がり幅は scaffold で +0.078・+0.073、random で +0.014・+0.003（本文の値。表 1 の丸めた値からの差は random で +0.013・+0.002。random の差は小さく、区間は報告されていない）
- 表 2（cos）でも rule only が scaffold の 2 列で最良（0.528・0.556）
- 報告前の「正解の分子式でピークを注釈する変種が全データ・全分割で最良」（report_2026-11-23_noon Q5、抜粋から）は、**表 1・表 2 と本文 §2 の文で確かめた**（仮定 3 の 6 行のどれかが各列の最良）。ただし仮定 3 の中で最良の行は列で違う: 表 1 の random の 2 列は `DreaMS + sub-formula`（0.758・0.947）、表 2 では NPLIB1 random も rule only（0.842）で、MSG random は `DreaMS + sub-formula`（0.960）

### 2.3 この競技にとっての読み
- 仮定 3 は**正解の分子式**を使う。私たちの課題では正解の分子式は分からないが、**各候補の分子式は分かる**。候補 c の式を仮定として問い合わせのピークを注釈すれば、部分式の規則を「候補の式ごと」に回せる（§5 の案）。論文はこの使い方（間違った式で注釈した場合）を測っていない
- 論文の random の分割の高い値は「Leakage」95.3%（MSG random）のように構造そのものが train にある場合を含む。骨格（scaffold）で分けた値（0.34〜0.38）は「FP oracle」（0.52〜0.56）より明らかに低い（MSG scaffold でも Leakage は 14.3%）。私たちの fold-safe の分割は同一性（完全一致）だけで分けていて、正解と fold-0 の学習構造の最近傍 Tanimoto ≥ 0.7 は v3_c1 65/200・v3_c2 64/200・enveda_v1 61/200・v3_c2n 109/195（`docs/research/msg-in-the-wild.md`、#154。合併の集合では数えていない）。論文の scaffold と random の間と見る（未測定）
- 指紋の Jaccard が高いことは、候補の並べ替えが良いことを保証しない（出典 3 も方法の節で「retrieval … may not be correlated and may in fact be anticorrelated with the accuracy of fingerprint prediction」と書く。その元の De Waele ほか arXiv:2602.16507 は**未読**）
- コードと論文の違い: 論文 S5 はトークンの重みを強度 I（重複は最大）と書くが、`nn_lib.subformula_tokens` は `sqrt(intensity_norm)`（重複は最大）。README も sqrt と書く。`annotate_peaks` は正の 1 価イオン（m/z + 電子の質量）だけを扱い、負イオンの扱いは書かれていない

## 3. ほかの出典の要点

### 3.1 MassSpecGym（出典 2）の検索の基準
- 候補集合（§3.2 の文）: 「the candidate set 𝒞 can be constructed to include only molecules with same masses as the true one (within an acceptable experimental error range) … we limit |C| ≤ 256 candidates per spectrum」。ボーナスの課題は「graphs with the given nodes V」（同じ分子式）。質量の窓の幅（ppm）は読んだ範囲には書かれていない（**未確認**）
- 表 5（見出し「Baseline results for the molecule retrieval challenge … 99.9% confidence intervals upon bootstrapping (20,000 resamples)」）の hit@1（%）: 質量の候補で Random 0.37・DeepSets 1.47・Fingerprint FFN 2.54・DeepSets + Fourier 5.24・**MIST 14.64**（hit@20 59.15）。同じ分子式の候補（ボーナス）で Random 3.06・Fingerprint FFN 5.09・**MIST 9.57**（hit@20 41.12）
- MIST の検索は「ranks the candidates by cosine similarity between the fingerprints」（§4）。指紋予測 → 候補との cos という、§5 の案と同じ形の点付け。分割は MCES ≥ 10 で分けた厳しいもの（§3.4）

### 3.2 Gupta ほか（出典 3）
- 構造だけのモデル（スペクトルを見ない GNN）が MSG の質量の候補で top-1 9.5%（MSG 論文の最良 14.6%）。スペクトルの有無を問わず「参照スペクトルがありそうな構造」を当てられる、という交絡の指摘。この repo の `prior`・`dseen` の議論と同じ向き
- 最近傍の基準（方法の節）: Khoo & Barzilay の「最近傍のスペクトルの正解の指紋」を、候補の指紋との cos で候補を並べる形にして使う（4096 bit・半径 2 の Morgan）。類似度は cos（開いた検索）・中性損失 cos・修正 cos
- 結果（本文の文）: 「13 of the 17 models failed to outperform the modified cosine variant of the nearest-neighbor baseline, and only FraGNNet and ICEBERG achieved a significantly higher top-1 accuracy」。FraGNNet の top-1 は分子式の候補で 20.1%・質量の候補で 17.2%。**最近傍の基準そのものの top-1 の値は図にだけあり、未読**
- 候補集合は Spectraverse の言語モデルで作った囮（参照ライブラリの構造に似せた構造）。本文の文では、PubChem 由来の一般的な候補集合が数百〜数千で、言語モデルの囮の集合はそれと同じくらいの大きさ（別の文）。私たちの v3 の窓（プールと PubChem の段）とは作り方が違う

### 3.3 Asher ほか（出典 4）
- 課題は de novo（候補リストも参照ライブラリも無し、分子式は与える）。道具無しの LLM は文献の多い分子で top-1 52%、文献に無い分子で 2%。de novo 生成器を足すと文献に無い側で 12〜38%（本文 §1 の箇条と §結果の文）
- 誤りの大部分は「正解が候補に出てこない」こと（「at most 14% of errors occur in cases where the agent had the correct structure but failed to return it first」）。生成器の符号化器の学習データとの近縁の ECFP4 Tanimoto の中央値が文献の多い側 0.90・無い側 0.82（本文の値。付録 F では 0.903・0.817）
- この競技への含意: 最近傍・類縁体の方法は「近い仲間が train にいる分子」で強く、いない分子で弱い。§5 の案の危険（v3_c2n で lfus が負けた理由の候補）と同じ向き

## 4. ライセンス・再現性（この競技の規則に照らして）

- 出典 1 の論文: **CC BY 4.0**（HTML の「License: CC BY 4.0」、abs ページの view license のリンク先 http://creativecommons.org/licenses/by/4.0/）。コード（serenaklm/ML_MS_analysis）: **MIT**（GitHub API）。案 §5 は方法と式だけを使い、コードを写す場合も MIT の表示を残せば足りる
- 出典 1 のデータ（Google Drive の処理済み NPLIB1・MassSpecGym）と NIST2023 を使うスクリプトは**使わない**。案のライブラリはこの競技の `train.parquet` から作る（実装では lfus と同じ作業ファイル〈v2-pool の `train_structs.parquet` で sid → 構造〉を使うが、`train.parquet` の列だけで作り直せる）ので、新しい外部データ・事前学習の重みは入らない（NIST 由来のものも入らない）
- 照らし合わせた規則の元: 主催者の回答 748882（公開した元の学習コードがあれば再現性として足りる）・756893（再現できない中間の成果物を読む解は再現性が無い）・757307/757309（MSG・MIST の重み、PubChem/COCONUT の候補の索引は可）。どれも `kaggle-competition-workflow/references/casmi26.md` と既存の文書の記録からで、**今回は取り直していない**。「許可される外部データ」の投稿 750330 は本文が読めていない（`docs/research/final-repro-inventory.md`）
- ライブラリ側（代表スペクトルとその構造）は競技の `train.parquet`（`normalized_smiles`・`molecular_formula`・`adduct` の列を持つ）だけから作り直せる。候補の SMILES は既存の対照と同じ v2-pool（第三者の公開データセットで、既存の管路がすでに使うもの。競技の train ではない）
- 学習するものは無い（最近傍と固定の式だけ）。定数 (w, α, T, k) は論文の表 4 の値を写す（§5.3）。再現には「コード + train + 固定の定数」で足り、提出に使う場合も自前のコードで作り直せる（重みのファイルが無い）
- DreaMS の変種（仮定 1・2 の最良）は使わない。DreaMS の SSL の重みの事前学習データの扱い（#109 の記録で 748882 に未回答の質問がある）を持ち込まないため
- 出典 2（MassSpecGym 論文、arXiv の非独占ライセンス）・出典 3（bioRxiv、CC BY-NC-ND 4.0）・出典 4（arXiv の非独占ライセンス）は引用だけで、文や図を写さない
- 未確認: 出典 1 のコードが表 1・2 の数をそのまま出すか（こちらで走らせていない）。論文の I とコードの sqrt(I) のどちらで表が作られたか

## 5. 自前チャネル `nnfp` の案と、v3 で測る案（実行しない）

### 5.1 何をするか（変更は 1 つ）
- 段 `nnfp`: prior の段の後・frag の段の前（lfus と同じ位置）。分子 q ごとに、prior 後の上位 60（`TOPN`）の候補の**分子式ごと**に
  1. 候補の式 F と問い合わせの付加イオン（`query_spectra` の adduct）から前駆イオンの式 F⁺ を作る
  2. 問い合わせのスペクトル（分子ごとの先頭 16 本、各 100 本の強いピーク）を F⁺ で注釈し、断片・中性損失のトークンの L2 正規化ベクトル v(q, F) を作る（複数のスペクトルのトークンは最大でまとめる）
  3. 同じイオン化モードの train の代表スペクトル t（hide_rule を当てた後）について score = s_SF(v(q,F), v(t)) + α·s_F(F, F_t; w)。v(t) は t 自身の構造の式で注釈したもの
  4. 上位 k の t の Morgan 指紋の softmax(T) 平均 r(q, F)
  5. その式の候補 c ごとに N(c) = cos(r(q, F_c), fp(c))
- 合成: 新しい点 = z(prior 後の点) + z(N)（上位 60 の中で z、等重み、`sanpier_mix`・lfus と同じ形。std(N) = 0 なら z(N) = 0）。1 位の固定・上書きはしない
- 組み合わせ: `check_stages` で nnfp と併用できるのは score・prior・frag だけ（lfus・nbz・fragnn・fragnn2・sanpier・lib・reflast・pubchem・elem などとは併用しない）。prior が無ければ止める
- 正直に書く: 正解を外す分子の大半は同じ分子式の異性体（np_v2 97 対 7、enveda_v1 42 対 3、`casmi26.md`）。同じ式の候補は同じ r を共有するので、異性体どうしの区別は「r に近い指紋の候補」を上げることだけで、**集め方（N の形）は lfus（類縁体の構造との Tanimoto の s 重み付き平均）とほぼ同じ**。違いは「どの類縁体を集めるか」だけ（§5.2）

### 5.2 既存の自前チャネルとの違い
- lfus（#150）: 類縁検索は生のピークのエントロピー類似（`analog_search`、±200 Da、上位 100）。nnfp は**部分式のトークン**（質量ではなく式なので、前駆の質量が違う類縁体でも断片・損失が比べられる）と、式の近さ α·s_F で集める。上位 k の softmax で、上位に重みが集中する（lfus は s の比例の重み）
- rep（#169）: 断片・損失の特徴を Murcko 骨格の家族の表に引く。候補の構造の骨格だけを見る。nnfp は家族を使わず、近いスペクトルの指紋の合意を使う
- nbz（#103）: 問い合わせどうしの伝播。nnfp は問い合わせを他の問い合わせと比べない
- frag（MetFrag-lite）: 候補の構造そのものを切って断片を説明する。nnfp は候補の構造を切らない（候補の式でピークに式を付けるだけ）
- fragnn（#120・#196）: 学習したモデル。nnfp は学習しない
- 0.487/0.495 系統の Engine 2 の `analog_search`・`_analog_feats` も類縁体の型の特徴を持つ（#150 の本文）。部分式のトークンで集めるものは、読んだ範囲の系統のコードには見当たらない（「見当たらない」で、無いとは言わない）

### 5.3 定数（v3 を見る前に決める。動かさない）
- (w, α, T, k) = **(8, 1, 0.05, 20)**: 出典 1 表 4 の MassSpecGym scaffold・rule only の行。選んだ理由（v3 を見ずに決めた）: (i) rule only は候補の式ごとに回す §5.1 の形と同じ、(ii) scaffold の分割は構造の重複が少なく、enveda_v1・v3_c2n・v3_c2（正解の参照が train に無い）に近い（v3_c1 は C1 型で正解自身の他装置の参照がある）、(iii) MSG の方が NPLIB1 より大きい。MSG の付加イオンは [M+H]+・[M+Na]+ の 2 種（出典 2 の図 4 の文）、NPLIB1 は未確認。どちらも正イオンだけなので、負イオンでのこの定数は外挿。ほかの行は試さない（試すと v3 に合わせた定数選びになる）
- 上位 100 本のピーク、許容 max(0.01 Da, 20 ppm)、トークンの重み sqrt(強度)（コードの定義）、原子価の検査 2·RDBE ≥ −1（コードと同じ）
- 負イオン: 前駆と断片のイオンの質量を m/z − 電子の質量で扱う（出典 1 のコードには無いので自前の変更。式は付加イオンの表から作る）。付加イオンの表は `_t10` の 10 種に限る。表に無い付加イオンの分子は nnfp を掛けない（数を記録）
- 指紋: RDKit の Morgan 半径 2・4096 bit（出典 1 と同じ定義。候補はプールの SMILES、train は `train_structs.parquet` の SMILES）。作れない構造は除いて数を記録

### 5.4 リークの扱い（fold-safe のライブラリ）
- 代表スペクトル: lfus と同じ `Library.representatives(exclude=印, key=<holdout>)`（hide_rule を当てた後）。印は #150 と同じ（行の inchikey14 ∈ drop_keys、v3_c1 は enveda の 2 ライブラリの行だけ、加えて sid の key・rawkey が drop_keys に入る行）
- 加えて、**判定の 4 集合すべての正解の構造（raw・canonical・別名）を、どの集合の採点でもライブラリから除く**（ただし v3_c1 を採点するとき、v3_c1 自身の正解の他装置の参照は C1 型の定義どおり残す。集合の間で共通の正解の構造〈元の集合で enveda_v1 と v3_c1 に 1、enveda_v1 と v3_c2 に 2〈`docs/research/lineage-leak-audit.md` (a)〉。合併の集合での数は未確認〉は、採点中の集合自身の正解なら その集合の hide_rule を優先し、件数を json に出す）
- カードの文「fold 0 の hold と判定の集合の正解を除いたライブラリ」について: nnfp は学習しないので、fold 0 の hold の構造のうち判定の正解でないものは「この分子のために隠すもの」ではない（隠れた test でも train 全体がライブラリ）。そこで**主の測定は hide_rule ＋ 4 集合の正解を除いたライブラリ**とし、fold 0 の hold の構造をすべて除いた版は**記録だけ**（採否に使わない）として同じ回に 1 回だけ出す。v3_c1 では、すべて除いた版は正解自身の他装置の参照も消えるので C1 型でなくなる。記録だけの版は採否に使わず、主の版と差の符号が逆なら注意として記録するだけにする。カードの文と違う読みを採ったこととその理由（隠れた test のライブラリは train 全体）は、採点の前の事前登録の本文にも書いてオーナーが見える形にする（この文書の独立レビューでは妥当と判断された）
- 混入の検査（#150 と同じ形）: 代表の行と、上位 k に選ばれた t の行の両方で、隠すべき行が 0 件であることを assert し、件数を json に出す
- 定数は出典の表から写すので、v3 の値から何も選ばない。v3 の正解は定数の決定に使わない

### 5.5 採否の規則（今の規則、#195 A8・#196 (f)）
- 新 = `score,prior,nnfp,frag`、対照 = `score,prior,frag`。集合は `v3_c1u_t10`（C1 型、394 分子）・`v3_c2_t10`（C2 型）・`enveda_v1u_t10`（プールにあり参照なし、373 分子）・`v3_c2nu_t10`（394 分子）。対照の値は #199 の記録（0.8878 / 0.0000 / 0.8513 / 0.6357）と採点時に一致を assert（一致しなければ止める。#198 の後の扱いは #196 の決めに従う）
- 差は分子ごとの rr の paired bootstrap（`train_eval.bootstrap_diff`、2,000 回、seed 0）の「新 − 対照」95% 区間。4 集合すべてで下限 ≥ −0.02、かつ v3_c1u_t10 の下限 > 0。v3_c2 型は対照が 0 なので「0 から下がらない・候補の被覆の点検」（並べ替えだけの段なので差は 0 のはず、を assert）。集合が 1 つでも欠ければ採用しない
- 読み方（事前）: v3_c1 の改善には正解自身の他装置の参照との一致が混ざる（lfus と同じ）。類縁体から伝わる効果そのものは enveda_v1・v3_c2n で読む
- 記録だけ（採否に使わない）: 式の数・トークンの数・注釈できたピークの割合（正解の式 vs ほかの式）、r(q, F_truth) と正解の指紋の Jaccard/cos（出典 1 の指標で、論文の値と並べる）、改善・悪化の分子数、fold 0 の hold をすべて除いた版の差

### 5.6 MDE との比較と見込み
- MDE（2.8·sd(Δrr)/√n）: 合併の集合の値はまだ数えていない。参考に、同じ合併の集合での fragnn v1 − 対照の 95% 区間の半幅（v3_c1u_t10 約 0.017・enveda_v1u_t10 約 0.020、`outputs/local_eval/a0i_1123/a0i_result.json`）から換算すると MDE は約 0.025・約 0.028。v3_c2nu_t10 は同じ fragnn v1 の区間 [−0.0051, +0.034] の半幅 約 0.020 から約 0.028。lfus 型の段は動かす分子が多く（#161 の n_changed: v3_c1 で lfus 30 対 fragnn 24）、sd(Δrr) は v3_c1・enveda_v1（n 200）で lfus 0.18・0.16、fragnn 0.17・0.16 とほぼ同じなので、この 2 集合は同じ程度と見込む。v3_c2n では lfus の sd(Δrr) が fragnn の約 1.3 倍（`net_mde.json`）なので、v3_c2nu_t10 の MDE は約 0.036 と見込む（すべて未測定）
- 見込みの効果: lfus（同じ位置・同じ合成）は v3_c1 +0.025 [−0.000, +0.051]・enveda_v1 +0.003・v3_c2n −0.039 [−0.075, −0.005] で不採用（#150）。§5.1 のとおり集め方がほぼ同じなので、nnfp の差も lfus と同じ桁（±0.03）と見込み、**v3_c2n で同じように負ける危険が高い**。出典 4 の「近い仲間がいない分子で弱い」と同じ向き。MDE（約 0.025〜0.036）より小さい差は「決まらない」と読む
- それでも測る理由: 部分式のトークンは前駆の質量が違う類縁体を拾えるので、v3_c2n（正解がプールにあり参照が無い）で lfus と違う類縁体を集める可能性がある
- 採点の前の 2 つの点検（役割が違う）:
  - 点検 1（**記録だけ**）: 集めた上位 k の t の sid と lfus の類縁体の sid の重なりの割合。rr は数えない。結果がどうでも採点するかどうかは変えない
  - 点検 2（**取り下げの判断だけ**。kaggle-improvement-loop の決まり: 前の版が負けた仕組みを、前の版の保存された出力で先に確かめ、勝った集合と負けた集合の両方を説明できるかを見る）: lfus の v3_c2n の悪化 40 分子と v3_c1 の改善 21 分子で、上がった誤りの候補が「train に仲間の多い家族の構造」かを #150 の分子ごとの出力（`outputs/local_eval/v3_c2n/submission_score+prior+lfus+frag.csv`・`lfus_analogs.json`、対照の `submission_score+prior+frag.csv`。どれも元の 200 分子の集合 v3_c2n のもので、合併の集合 v3_c2nu_t10 のものではない）で数える。その仕組みで nnfp も同じように負けると判断できるなら、nnfp は採点せずに取り下げる。取り下げの基準は採点の前の事前登録の Issue で数で決める。**点検の結果で nnfp の設計・定数は変えない**（取り下げか、§5.1〜5.5 のまま 1 回の採点のどちらか）

### 5.7 費用の見積もり
- 速さの試し（実データではない）: 出典 1 の `annotate_peaks` に一様乱数の 100 本の m/z を与え、4 つの式（C20H26NO6・C30H49O10・C15H11O6・C40H65O15）で各 1 回 0.4〜1.0 ms（`.venv` の python、作り物のピーク、1 回ずつの計測。コードは下の脚注）。代表 371,900 本（#150 の v3_c1 の `n_rep`）を注釈すると 1 コアで約 2.5〜6 分の見込み。代表のトークンは 1 回だけ作り、集合ごとの hide_rule は行の印で引く（集合ごとに作り直さない）
- 問い合わせ側: 1,358 分子（394 + 197 + 373 + 394）× 上位 60 の式の数（未測定。1 分子 20 式と仮定）× 16 本 = 434,560 回の注釈 → 0.4〜1.0 ms で約 3〜7 分。疎行列の積と s_F は 1 回数 ms〜数十 ms（未測定）で数分〜20 分の見込み。ほかに `Library` の build が 1 集合 158〜260 s（#150 の実測）× 4 集合
- メモリ: `Library` の読み込みは lfus の build で 4.9〜5.1 GB（#150）。トークンの疎行列は代表 37 万 × 平均 50〜200 個の非零で 1 GB 未満の見込み。**関門は peak ≤ 6 GB**（`/usr/bin/time -l` で測る。最初の集合の 10 分子で外挿し、全体 1.5 h・6 GB を超える見込みなら採点しない）
- GPU 0、提出 0。重いローカルジョブ（fragnn v2 の学習）が動いている間は、その日の計画の共通の決まり（memory_pressure・swap・vm_stat の関門、自分の CPU 処理は 1 本）に従う
- 作業の見積もり: 事前登録の Issue（§5.6 の点検の定義を含む）と独立レビュー 1 h、実装（段・テスト〈作り物のデータで注釈・トークン・softmax 合意・負イオン・混入の検査〉）2〜3 h、採点 0.5〜1 h、レビュー 0.5 h。計 4〜5.5 h（2 枚のカードに分ける）

## 6. 結論
- 出典 1 は検索の課題ではなく指紋の近さの比較で、構造の候補集合を持たない。「正解の式でピークを注釈する変種が全データ・全分割で最良」は表 1・2 で確かめた（ただし仮定 3 の中の最良の行は列で違う。§2.2）
- この競技では候補の式ごとに部分式の規則を回せば同じ情報を使える。学習は無く、コードは MIT、データは train だけで、規則上の問題は見当たらない
- ただし、同じ式の異性体の区別は lfus と同じ「類縁体の指紋に近い候補を上げる」形になり、v3_c2n で負けた lfus と同じ危険が高い。測るなら §5.6 の事前の点検（lfus の負けの仕組み）を先に通し、通らなければ取り下げる

---
脚注（§5.7 の速さの試しのコード。`.venv` の python、出典 1 の `nn_lib.py` を scratch に置いて実行）:
```python
import time, numpy as np, nn_lib as L
rng = np.random.default_rng(0)
for f in ['C20H26NO6', 'C30H49O10', 'C15H11O6', 'C40H65O15']:
    p = L.parse_formula(f); L.heavy_subformulas(p)
    mz = np.sort(rng.uniform(50, 500, 100)); t = time.time(); a = L.annotate_peaks(mz, p)
    print(f, '%.4fs' % (time.time() - t), sum(x is not None for x in a))
# 出力: 0.0010s / 0.0008s / 0.0004s / 0.0009s（注釈できたピーク 12 / 25 / 3 / 32）
```
