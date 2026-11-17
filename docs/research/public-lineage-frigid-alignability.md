# 公開上位（heliosli 0.495・flexonafft）と FRIGID・Separability Is Not Alignability: 次に 0.487 系統へ載せる自前の部品の候補

Issue #138（親 #3、計画 2026-11-17 昼 項目 4。夜の項目 4 が Anthropic 429 で未着手になったもののやり直し）。

目的: 0.487 系統の上に載せる自前の部品の次の候補と、fragnn（#120）・整列チャネル（#109 / #127）の位置づけを、公開ノートと論文で確かめる。最後に v3 で測る案を 1 つ書く。

印の意味（#94 の文書と同じ）: **確認** は本文・コード・メタデータを自分で読んで確かめたもの。**主張** は著者が書いているが、こちらでは再現していないもの。**未確認** は読めていないか、確かめていないもの。読んだのは 2026-11-17 13:3x〜14:xx（JST）。

関連する既存の文書（重ねて書かない）:
- `docs/research/spectrum-blind-and-np-priors.md`（#94）: Separability Is Not Alignability の §5・付録 I（spectrum-blind の対照、式 (3) の分解）と、そのコード autonlab/msms-retrieval
- `docs/research/gems-isomer-search.md`: FRIGID の README（MIST 符号器の差し替えの注記）
- `docs/research/isomer-rerank.md`: heliosli の rr3（同じ分子式の異性体の並べ替え器）

## 1. 出典

### S1. heliosli「casmi26-0.495」（Kaggle ノート）
- https://www.kaggle.com/code/heliosli/casmi26-0-495 。`kaggle kernels pull heliosli/casmi26-0-495 -p <scratch>/h495 -m` で取得（13:36 JST）。ipynb 771,620 バイト、30 セル、出力 0 セル。lastRunTime 2026-11-13 09:52 UTC。**確認**
- 競技のノート一覧（`kaggle kernels list --competition … --sort-by scoreDescending`）で 1 番目。点数は CLI に出ないので、題の「0.495」以外の値は**未確認**
- 比べた相手: 11/16 夜に pull した「casmi26-0.494 public demo」（https://www.kaggle.com/code/heliosli/casmi26-0-494-public-demo 、30 セル）

### S2. heliosli のデータセット（S1 が読むもの）の README
- https://www.kaggle.com/datasets/heliosli/casmi26-analyticon-megx （ライセンス `other`、2026-11-16 14:35 更新）、https://www.kaggle.com/datasets/heliosli/casmi26-m5-v2d （CC0-1.0）、https://www.kaggle.com/datasets/heliosli/casmi26-joint-ranker （`other`）。`kaggle datasets metadata` と `kaggle datasets files` で確認。**確認**

### S3. flexonafft「CASMI26: Anchored Spectral Network」（Kaggle ノート）
- https://www.kaggle.com/code/flexonafft/casmi26-anchored-spectral-network 。11/17 の日報で pull したもの（scratch `pub1117/asn`）。lastRunTime 2026-11-16 18:18 UTC、enable_internet true、出力 0 セル。スコア順の一覧で 2 番目（点数は CLI に出ず**未確認**）。**確認**
- 注意: pull した .ipynb の中身は題と違い、最初の markdown が「CASMI26: Isomer Ranker Train」で、Anchored Spectral Network が使う異性体の joint ranker を**学習する**ノートだった（14 セル〈0〜13〉、学習モジュール `isomer_ranker.py` 18,525 文字を文字列で埋め込み）。提出側のオーバーレイ（`apply_main`）は同じモジュールの中にある。Anchored Spectral Network の提出版のほかのセルは読んでいない（**未確認**）。**確認**

### S4. FRIGID（arXiv:2604.16648）
- Bohde, Liu, Manjrekar, Lederbauer, Ji, Wang, Coley「FRIGID: Scaling Diffusion-Based Molecular Generation from Mass Spectra at Training and Inference Time」。https://arxiv.org/abs/2604.16648 、DOI https://doi.org/10.48550/arXiv.2604.16648 。v1 2026-04-17、v3 2026-10-25。論文のライセンスは CC BY-NC-SA 4.0（arXiv の表示）。**確認**（抄録のページだけ。本文は読んでいない）
- コード: https://github.com/coleygroup/FRIGID （README、最終コミット 5adba44 2026-10-21）。コードは Apache-2.0（README の License 節）。**確認**（README だけ）

### S5. Separability Is Not Alignability（arXiv:2610.37979）
- Musielinski, Teh, Koyama, Dubrawski、NeurIPS 2026 Workshop RPS のポスター（arXiv の abs ページの Comments 欄の記載）。https://arxiv.org/abs/2610.37979 、DOI https://doi.org/10.48550/arXiv.2610.37979 。CC BY 4.0。今回は HTML 版 v1（https://arxiv.org/html/2610.37979v1 ）の §1〜§7 と付録 A〜C の途中まで読んだ（web_extract が付録 C の途中で切れた。付録 D 以降は読んでいない）。**確認**

### S6. dmitriigluzdov「casmi26-pubchem-popularity-prior」の説明（S3 の文献の特徴の出どころ）
- https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior 、ライセンス `other`。`kaggle datasets metadata` の description を読んだ。**確認**

## 2. 要点

### 2.1 heliosli 0.495 は 0.494 demo から何を変えたか（S1・S2）
- セルの sha1 を 30 セルすべて比べた。**変わったのはセル 26 と 27 だけ**で、最初の markdown（セル 0）は両方とも 0.477 版の文章のまま（0.495 の変更の説明は無い）。**確認**
- `dataset_sources`: 0.494 demo から `heliosli/casmi26-denovo-v3` が消え、`heliosli/casmi26-analyticon-megx` と `heliosli/casmi26-m5-v2d` が増えた。ほかの 12 件と `kernel_sources`（空）・docker_image・machine_shape（T4）は同じ。**確認**
- セル 26（CV-JOINT、joint ranker の上書き）: コードは同じで、自己テストの参照値 `CVDEEP_SELFTEST` の `z` だけが変わった（例: 0.0139 → −6.41）。S2 の README の「デモ版に合わせて参照値を差し替えた」と合う（重みが違うことを示すだけで、何の重みかは**未確認**）。**確認**
- セル 27（M5 の part A、リストの中の異性体の判定）: 0.494 demo の「J2 multi-evidence judge」（judge_core.Judge: M3 の z、元の順位、PubChem の文献・出どころ・天然物の印、COCONUT の収載、NP-likeness、官能基 × 付加イオン）を**丸ごと消し**、「MEGx prior」に置き換えた。**確認**
  - 中身: 層「np」のリストで、各候補の z に `MEGX_BETA × (megx(c) − megx(1 位))` を足す。megx(c) は候補の InChIKey14 が AnalytiCon MEGx のカタログ（コメントでは「COCONUT 2026-11 の collection」）に載るかの 0/1。`MEGX_BETA = 5.0`
  - コメントの数値（**主張**）: 「timsTOF の NP パネル（M3 の学習外、M3 の 5-fold 平均、tau 5）で、隠れの重みつき ×1.8 で +0.012 → beta 5 で +0.026」。β = 5 はパネルの値を見て選んだ定数と読める（何の値を比べて選んだかは書かれておらず**未確認**）
- **再現できない**: S2 の README によると、`casmi26-analyticon-megx` は 3,683 件の作った分子（デモの識別子、元のカタログの収載ではない）、`casmi26-m5-v2d`・`casmi26-joint-ranker` は「0.494 に使った重みは入っていない。初期化した重み（joint ranker は合成のフィット）に置き換えた」。写しても 0.495 にはならない。11/16 夜に 0.494 demo で確かめたことと同じ。**確認**
- 分類: MEGx prior は「カタログに載るか」という新しい証拠のチャネル（構造の変更）だが、重み β = 5 はパネルで選んだ定数で、元の表は非公開。

### 2.2 flexonafft の異性体 joint ranker（S3・S6）
- 着想は heliosli の CV-JOINT（FPNet ＋ 文献 ＋ 天然物 → LightGBM が 1 位を判定し直す）。heliosli の重みが公開されていないので、**公開の入力だけで学習し直し、学習の全体をノートに載せた**と書いている（RT と PX v2 のモデルは使わない）。**確認**（markdown セル 0 とモジュールの docstring）
- 学習の流れ（セル 3〜13）。**確認**
  1. 学習分子: identity fold 0（`dmitriigluzdov/casmi26-natural-product-panel-benchmark` の `structures.parquet` の `fold_identity == 0`）の構造。fold-safe FPNet fold 0 はこれを見ていない。timsTOF または enveda のライブラリに載る構造（「test-like」）を 3,000、残りを無作為に足して計 5,000（seed 0）。スペクトルは timsTOF を先に最大 8 本
  2. FPNet fold 0 のロジット z を GPU で計算（提出側と同じ関数 `fpnet_z`。複数スペクトルの合成は `0.5 × (zs + zm)`）
  3. 候補: その分子の**分子式の正確な質量 ± 2e-4 Da** の PubChem 層（`casmi26-pubchem-popularity-prior` の `casmi26-pubchem-tier`）から、InChIKey14 で重複を除き、文献の多い順に最大 1,000。ラベルは候補の InChIKey14 が正解と同じか
  4. 一次選択: z 標準化 fzz、文献 pop = ln 物質数 + ln PubMed、`chan = fzz + 0.25 × pop` の上位 150 ∪ fz の上位 50 ∪ pop の上位 50
  5. 特徴 16: fzz・fz_gap・log 順位、pop・pop_gap・log 順位、chan・chan_gap・log 順位、ls・lp、天然物の印（isnp、COCONUT・LOTUS・NPAtlas のビット）、ln 候補数。**スペクトルから来る特徴は FPNet の z だけ**で、プールにあるか・参照スペクトルがあるかの特徴は無い
  6. LightGBM lambdarank（lr 0.05、leaves 31、min_data_in_leaf 50、400 rounds 固定、早期終了なし、deterministic）。5-fold の OOF で報告と閾値、最後に全分子で 1 本
- margin の決め方（セル 11）。**確認**
  - margin = OOF の 1 位と 2 位の点の差。test-like の分子（300 以上あれば）で margin の大きい順に並べ、1 位が正解の割合（累積の precision）が 90% 以上（30 分子以上）になる一番下の margin を `swap`、50% 以上（50 分子以上）を `insert` とする
- 提出側（`apply_main`）。**確認**
  - 今の提出の 1 位の候補の分子式で同じ PubChem 層の窓を作り、ranker の 1 位 `best` を決める。`best` が 1 位と同じなら何もしない
  - 既定は `--mode insert`: margin ≥ insert なら `best` を 2 位に入れる（すでに 2 位なら何もしない）。`swap` モードでは margin ≥ swap かつ 1 位の library_max < 0.9 のときだけ 1 位と入れ替える。1 位は insert では動かない
  - 失敗・時間切れはその分子を変えない
- 出力（`isomer_ranker_model.txt`・`_meta.json`・OOF 表）が公開されているかは、pull したノートの出力が 0 セルなので**未確認**。OOF の値（正解率など）もこちらでは見ていない
- S6 の説明（**主張**）: 「パネルの天然物の正解は、同じ質量の PubChem 構造の中で 66% が一番よく文献に載っているもの」。S3 の ranker は、この文献の偏りと FPNet の z を同じ木で組み合わせる
- 分類: 新しい並べ替え器（構造の変更）。閾値は OOF から決まった規則で決める（LB を見た調整ではない）が、0.25・K1 150/50/50・CAP 1000・400 rounds などの定数は作者が決めたもの。v3 で測るならそのまま固定して使い、動かさない

### 2.3 FRIGID（S4）
- 抄録（**主張**）: 中間の指紋（MIST）と分子式を条件にした拡散言語モデル（SAFE 列）で de novo に構造を作る。数億のラベルなし構造で事前学習。順方向の断片化モデル（ICEBERG）で、スペクトルに合わない断片を見つけて、その部分だけマスクし直して作り直す（推論時のスケーリング）。arXiv の abs ページの抄録は「MassSpecGym で Top-1 18% 超、NPLIB1 で既存法の 3 倍」と書いている（ページは v3 2026-10-25 を最新版と表示しているが、表示された抄録が v3 のものか、v1 のものが残っているのかは**未確認**）
- README（10/21 更新、**主張**）: 前の版の MassSpecGym 用 MIST の学習に、テストの分子の分子式に合う・似る PubChem 分子の合成スペクトルを使っていた（テストの情報を使っていた）ので、MIST を差し替えて出し直した。改めた値は Top-1 で NPLIB1 14.16%（base 13.31%）、MassSpecGym 9.05%（base 8.21%）。抄録の「18% 超」と README の 9.05% は食い違う。抄録の版と、本文 v3 の値がどちらかは、本文を読んでおらず**未確認**
- 推論時間（README、**主張**）: NPLIB1 で base 6.58 秒/スペクトル、10 回の作り直しで 41.3 秒/スペクトル（NVIDIA 6000 Ada 1 枚）
- 重み: FRIGID の decoder と ICEBERG は Zenodo record 23543494、MIST は Zenodo 23303289。重みのライセンスは**未確認**（README はコードの Apache-2.0 だけを書いている）。MIST は NPLIB1（CANOPUS）と MassSpecGym の学習分割と GLACIER の合成スペクトルで学習（README）
- heliosli はセル 11・28 で FRIGID の第 1 候補を条件つきで 2 位に入れている（11/16 夜に確認）。その効果は分離されていない

### 2.4 Separability Is Not Alignability（S5）
- §5・付録 I（候補リストの目的関数とスペクトルを見ない対照）は #94 の文書に書いた。ここでは今回新しく本文で確かめたことだけを書く
- §4（凍結した符号器、MassSpecGym の MCES 分割、**主張**の数値で、こちらでは再現していない）:
  - DreaMS の自己教師ありの段は分離が弱い: silhouette +0.14、kNN purity 0.68、MoLFormer との mutual-kNN 0.09。11/16 夜の日報で「検索結果の抜粋で、本文は読み切っていない（未確認）」としていた silhouette +0.14 は、**本文 §4 にあることを確認した**
  - DreaMS の対照学習の段: silhouette +0.28、purity 0.78 に上がるが、cross-modal の一致と凍結での検索 MRR（0.23）はほとんど変わらない
  - MS-JEPA（著者の新しい符号器）は分離が一番強い（silhouette +0.49、purity 0.91）のに、凍結での検索は一番悪い（MRR 0.12、Recall@1 0.05）。「分離できること」と「整列できること」は別
- §6（**主張**）: 符号器を凍結せずに end-to-end で学習すると、事前学習の差はほとんど消える（新しく作った符号器でも MRR 0.232、DreaMS 0.244〜0.255）。GLACIER と FraGNNet で作った 590 万分子の合成スペクトルで事前学習すると MRR 0.232 → 0.463（MaxSim）。候補リストの目的関数を足すと 0.566 になるが、spectrum-blind の MRR も 0.070・0.085 上がる
- 結論（§7、**主張**）: データ不足を補うのは構造（凍結した表現の整列）ではなく、分子の範囲を広げること（順方向モデルでの合成スペクトル）

## 3. この競技で使えるか

### 3.1 heliosli 0.495（S1・S2）
- 写しても 0.495 にならない（重みと MEGx の表がデモ版、§2.1）。組織側の議論 756893 の回答（再現できない中間物を使う解は再現性が無い。`docs/research/organizer-question-756893.md` に記録）にも当たる
- 使えるのは考え方だけ: 「天然物の層で、ある商用の天然物カタログに載る候補を少し上げる」。仮説として、隠しテスト（Enveda が測った天然物）の一部が購入した天然物ライブラリ由来なら効く。ただしこれは**未確認の推測**で、MEGx の収載をこちらで作れるか（手元の COCONUT に collection の列があるか）も**未確認**。β はパネルで選んだ定数なので、同じ値を持ち込めば調整の禁止に触れる。今は案にしない

### 3.2 flexonafft の joint ranker（S3）
- 入力はすべて公開データセット（fold-safe FPNet、identity folds、PubChem 層と文献の表、v4g の `casmi` コード、RDKit の wheel）で、学習コードもノートにある。再現性の条件（生成コードの公開）を満たす形。ただし入力のデータセットのライセンスは `other` のものがあり（S6 など）、各入力の条件に従う
- **v3 のリーク**: 学習分子は identity fold 0 の構造から選んだ 5,000 で、v3 の 4 集合はどれも fold-safe の fold 0（hold）から作っている（`docs/design/lineage487-foldsafe-eval.md` §「v3」）。hold の構造はすべて `fold_identity == 0` に入る（`docs/design/engine2-rebuild.md` 100 行目）。だから**公開の学習済み model をそのまま v3 で採点するとリークになる恐れがある**（v3 の正解が LightGBM の学習の正解に入っていうるが、重なりの数は数えていない）。v3 で測るには、自前のコードで v3 を外して学習し直す必要がある（§4）
- 近道の危険: 特徴にプール・参照の有無が無いので、#29/#32 の ranker が覚えた「ライブラリ一致が無ければ外部の候補」の近道は覚えようがない。代わりに文献 pop（S6 の 66%）を強く使う形なので、S5 §5 の「スペクトルに依らない事前分布」をそのまま点にする危険がある。スペクトルを入れ替えた blind 対照（#120 と同じ作り）で Δs を測る必要がある
- 学習の正解は train の構造（enveda / timsTOF を優先）で、隠しテストの天然物とは文献の多さの分布が違う恐れがある（**未測定**）

### 3.3 FRIGID（S4）
- コードは Apache-2.0。重みのライセンスと、組織側が FRIGID の重みを認めるかは**未確認**（議論 757307 で MassSpecGym・MIST のモデルは使ってよいとされた〈`docs/research/glacier-iceberg-provenance.md`・`contrastive-alignment.md` に記録〉。PubChem/COCONUT の候補索引も議論 757309 で使ってよいとされた〈`docs/research/lineage-reproducibility.md` 83〜84 行に記録〉。FRIGID の重みについての回答は見つけていない）
- v3 では測れない: MIST・ICEBERG は MassSpecGym（と NPLIB1）で学習しており、v3_c2n は MassSpecGym と 116/195 が重なる（`docs/design/lineage487-foldsafe-eval.md` の部品表の 10 番 ICEBERG/GLACIER の行、`docs/research/glacier-iceberg-provenance.md` と同じ結論）
- de novo 候補は「届いても順位が付かない」恐れがある（こちらの analog-edit の候補生成は、リーク無しのロジットで天然物の正解に約 50% 届くが 1 位は約 3% で却下済み。記録は改善ループのスキルの参照文書 `casmi26-candidate-generation.md` にあり、リポジトリの Issue 番号はこの文書では確かめていない）。FRIGID の Top-1 は改めた値で NPLIB1 14%・MSG 9% で、heliosli の 2 位挿入の効果も分離されていない。GPU が要る（base 6.58 秒/スペクトル、Ada）。今週の候補にしない

### 3.4 S5 から見た fragnn・整列の位置づけ
- #109 の整列（凍結 DreaMS ＋ MSAlign 型の MLP）と fragnn（#120）は、どちらも**候補リストの InfoNCE（±10 ppm の窓から負例 127、0 番が正解）**で学習している（`origin/feat/109-align-local-train` の `scripts/train/align_train.py` の docstring 7 行目〈k_candidates 128〉・`negative_csr`〈PPM の窓〉・`sample_candidates`〈候補 0 = 正解〉、`origin/main` の `scripts/train/fragnn_train.py` の docstring 5〜10 行。**確認**）。S5 §5 の式 (3) の形では、負例の分布 q と正解の分布 p が違うと、スペクトルに依らない項 log p(m)/q(m) が点に入る
  - こちらの負例は fragnn で「train_structs の正準キーを持つ行」だけ（`fragnn_train.py` の docstring 9 行）。正解も train の構造なので、学習の中では p と q は近い。しかし推論では窓に COCONUT など train に無い構造が入るので、その候補への点はスペクトルと無関係に偏る恐れがある（heliosli の rr_v3 の docstring の「train の構造を約 0.34 SD 低く出す」と同じ型。#120 の本文 26 行目にも書いてある）
  - したがって、#120・#109 が採否の後の判断を blind の Δs（full − blind）で決める設計は S5 の推奨と合う。#109 の v3_c1 の Δs は +0.0413 [+0.0174, +0.0677]（#109 のコメント、11/15 夜）で、スペクトルの情報は入っている
  - 同じ #109 のコメントは、v3_c2n で full が対照より −0.041、blind が対照より −0.110 下がったことから「h 側に学習でよく正解になった分子ほど高い偏り（分子の事前分布）がある」と読んでいる。S5 §5 の「候補の目的関数がスペクトルに依らない事前分布を点にする」の、こちらの実例にあたる（読みは #109 のもので、この文書で確かめ直してはいない）
- S5 §4: 凍結 DreaMS の SSL の段は分離が弱く（silhouette +0.14）、凍結での整列は MRR が上がりにくい。#109 が v3 で不採用（v3_c1 +0.0125 [−0.005, +0.030]、v3_c2n −0.041）だったことと向きは合う（因果は確かめていない）
- S5 §6: 符号器ごと学習すると事前学習の差は消え、効くのは合成スペクトルで分子の範囲を広げること。この競技に当てはめると「GLACIER などで train に無い構造（COCONUT・PubChem 層）の合成スペクトルを作って学習に足す」になるが、GLACIER・ICEBERG は MassSpecGym で学習しており v3_c2n と重なるので、v3 で採否を決められない（§3.3 と同じ）。GPU も要る。今週は案にしない

## 4. 次に v3 で測る案（1 つ）: 自前で fold-safe に学習し直す同じ分子式の異性体 joint ranker（`isojoint` の段、2 位への挿入だけ）

### 4.1 中身
- S3 の ranker を**自前のコードで、v3 から完全に切り離して学習し直し**、縮小版 `scripts/local_eval/run.py` の最後に「2 位への挿入だけ」の段として足す。0.487 系統の上に載せる前の、縮小版での採否の判定
- 学習データ（fold-safe、#29 の rk_f{k} と同じ考え方）: identity fold 1〜4 の構造（S3 は fold 0。こちらは fold 0 を使わない）。fold k の分子のロジットは、fold k を学習から外した `data/ext/foldsafe/fpnet_fold{k}.pt` で作る（OOF）。v3 の 4 集合の正解（`data/holdouts/*/drop_keys.txt`、生のキーと正準キーの両方、それに写る生のキーも）と、パネルの天然物を学習分子から除く。test-like を先に取る規則・件数（3,000 / 5,000）・seed は S3 と同じにする
- 候補・特徴・LightGBM の設定・400 rounds・閾値の規則（OOF の precision 50% で insert）は S3 のまま固定する（新しい定数を作らない、別の値を試さない）。`swap` モードは使わない（1 位を動かさない）
- 採点: v3 の 4 集合は、fold 0 のロジット（`local_fold0_logits.py` の出力）で ranker の点を作る。対照 = `score,prior,frag`（#120 と同じ。#117 後の値 v3_c1 0.8535 / v3_c2 0.0000 / enveda_v1 0.8376 / v3_c2n 0.6284 と小数 4 桁で一致を先に確かめ、一致しなければ採点しない）。新 = `score,prior,frag,isojoint`
- PubChem 層の窓は v3_c2 では正解を含んでよい（クラス 2 の答えは PubChem にしか無いという集合の設計どおり）。v3_c1・enveda_v1・v3_c2n では正解はプールにあり、PubChem 層にもある見込み（#129 の方法の確認で、プールにある LOTUS 分子 100 のうち 95 が層にあった。v3 の正解そのものでは数えていない）
- blind 対照: 同じ重み・同じ窓で、FPNet の z だけを「同じ集合の、molecule_id の昇順で次の、同じ正負の分子」のスペクトルから作ったものに入れ替える（#120 本文 25 行目の規則をそのまま使う）。新 = `score,prior,frag,isojointblind`

### 4.2 採否の規則（C1 と C2 の条件、4 集合）
- 4 集合 = (答えの出所 × 参照の有無) の 4 つ: v3_c1（答えはプール、参照は他の装置のものが残る）、v3_c2（答えは PubChem 層だけ、参照なし）、enveda_v1（答えはプール、参照なし、enveda の問い合わせ）、v3_c2n（答えはプール〈COCONUT 収載〉、参照なし、問い合わせはほぼ公開ライブラリ）。各集合の作り方は `data/holdouts/*/hide_rule.json` と `HOLDOUT.json`
- 注意（レビューの指摘で書き直した）: v3_c2 の対照は 0.0000 で、2 位への挿入は v3_c2 では分子ごとの差を負にできない。そのため「v3_c2 の下限 > 0」だけなら、200 分子のうち約 4 分子で 2 位に正解が入れば満たされる（レビュー役の模擬: 0.5 の当たりが 3 分子までは下限 0.0、4 分子で下限 +0.0025・平均 +0.01）。これでは C2 でほとんど効かずに他の集合で −0.02 近く負けても採用になるので、効果の大きさの条件を足す（どの値も測る前のこの文書で決め、結果を見て変えない）
- **採用**（すべて満たすとき）: `bootstrap_diff`（分子の再抽出 2,000 回、seed 0）の「新 − 対照」で
  1. **4 集合すべてで 95% 下限 ≥ −0.02**（v3_c1・enveda_v1・v3_c2n では、挿入は正解が 2 位以下のときだけ順位を下げうる。分母は truth.csv の分子数で、v3_c2n は 195、ほかは 200）。平均の下限は C1（v3_c1）にだけ置く（下の 3）。enveda_v1・v3_c2n は C1・C2 の主の 2 集合ではなく、#120 などの既存の規則と同じく下限 ≥ −0.02 だけにそろえた。C2 の平均 ≥ +0.025 を求めるので、他の 2 集合で −0.02 近く負けたまま小さい C2 の上がりで採用になることは無い
  2. **C2（v3_c2）: 95% 下限 > 0 かつ平均 ≥ +0.025**。+0.025 は 200 分子の集合の MDE の下の端（下の行）で、v3_c2 では 10 分子で正解が 2 位に入るのにあたる（0.5 × 10 / 200）
  3. **C1（v3_c1）: 95% 下限 ≥ −0.02 かつ平均 ≥ −0.005**（1 位を動かさない段なので、1 位の正解は下がらない。下がるのは正解が 2 位以下の分子だけで、2 位 → 3 位の 1 分子が −0.0008〈(1/3 − 1/2) / 200〉。−0.005 は約 6 分子ぶんで、それより下がるなら挿入が正解の上に何度も入っている）
  4. 1 集合でも欠ければ不採用。判定のコードは `scripts/cv/frag_v3_check.py` の `judge` と同じ型で、上の 1〜3 を書いたものを新しく書く
- **0.487 系統へ進む条件**: 採用に加えて、v3_c2 で Δs（`isojoint` − `isojointblind`）の 95% 下限 > 0 **かつ** Δs の平均 ≥（新 − 対照）の平均の半分。つまり C2 の上がり分の少なくとも半分がスペクトルの情報から来ていること。文献の偏りだけで届いている（S5 §5 の形）なら Δs は 0 付近になり、0.487 系統には進めない
- 記録だけ（採否に使わない）: 集合ごとの挿入した分子の数、そのうち正解だった数、順位が下がった分子の数。OOF の閾値の値と OOF の正解率（学習の側の値で、v3 を見る前に Issue に書く）
- 測れる大きさ（MDE）: 200 分子の集合で約 0.025〜0.05（過去の対の bootstrap からの見積もり。改善ループのスキルの記録）。v3_c2 で条件 2 を満たすには、10 分子以上で 2 位に正解が入る必要がある（見込みは**未測定**）
- 結果が良くても悪くても規則どおりに記録し、定数・閾値の規則・特徴を変えて測り直さない

### 4.3 事前登録と手順（この PR ではしない）
1. 新しい Issue に §4.1・§4.2 を書き、独立レビューを通す（計算の前）
2. `scripts/ranker/` に学習（OOF 込み）と段のコード、合成データのテスト → PR → CI
3. 学習の前に、(a) 学習分子と v3 の 4 集合の重なりが 0（生・正準の両方のキー）、(b) fold k の学習分子が `fpnet_fold{k}.pt` の学習に入っていない（fold-safe の `split.parquet` で照合。identity fold と fold-safe の fold の対応は #29 の rk_f{k} にならうが、この文書では確かめていない）ことをスクリプトで assert
4. 1 回だけ採点し、#3 と Issue に C1・C2 を含む 4 集合の数値と区間を記録

### 4.4 必要な資源（見込み、**未計測**）
- 手元に無いもの: `casmi26-pubchem-popularity-prior` の `pc_ik14.npy`・`pc_np.npy`・`pc_lsid.npy`・`pc_lpmid.npy`（PubChem 層 1 億 600 万行と行が揃う配列。大きさは数えていないが、ik14 だけで 1 億 600 万 × 14 バイト ≈ 1.5 GB の見込み）。ダウンロードの前に関門のレビューと `df -h /`
- CPU: 候補ごとの指紋（最大 1,000 × 分子 5,000）を RDKit で作る。S3 は Kaggle で 5 時間の予算を置いている。S6 の説明では層の全構造の指紋がデータセットにあるが、`fp_bits` と同じビットかは**未確認**。先に 50 分子で時間と RSS を測る
- GPU 0（FPNet の推論は CPU で、`local_fold0_logits.py` と同じ）

### 4.5 なぜこの案か
- 自前で学習する部品（コードは S3 を参考に自前で書き、学習データを fold-safe に作り直す）で、0.487 系統にも 1 セルのオーバーレイ（2 位への挿入）として載せやすい。公開の S3 をそのまま写すのではなく、v3 から切り離した学習と blind 対照を足す点が違う
- fragnn（#120）とは役割が分かれる: fragnn はプールの窓の中で並べ替え、この案は PubChem 層の同じ分子式の候補から 2 位に 1 つ入れる。v3_c2（今の対照 0.0000）に届く部品はまだ無い
- 0.479 系統の PubChem チャネル（`lib_max < 0.9` のときだけ固定の枠に差し込む）とは別の入れ方で、LB の 748524 の報告（固定の枠への差し込みは LB で負け）とも違う形（文献と FPNet を学習で組み合わせ、margin の規則で挿入する）。ただし「LB で効くか」は v3 では分からない

## 5. 限界
- S1 の 0.495 の内訳はノートに無く、効果の分離はできない。S3 の点数・出力ファイル・OOF の値は読めていない。S4 は抄録と README だけで、本文は読んでいない。S5 は付録 D 以降を読んでいない
- §4 の案の効果の見込みは**未測定**。v3 は隠しテストの天然物の文献の分布を含まないので、v3 の結果は隠しテストでの効果の推定ではない
- 数値はすべて出典の著者の値か、こちらの過去の記録（#109・#120・`docs/design/lineage487-foldsafe-eval.md`）の写しで、この文書のために計算し直したものは無い

## 6. 再現の手順（読んだもの）
```
S=~/.hermes/profiles/massspec/cache/scratch/n1117d_research
kaggle kernels pull heliosli/casmi26-0-495 -p $S/h495 -m
# 0.494 demo は 11/16 夜に pull 済み（scratch n1116night_pub/heliosli_casmi26-0-494-public-demo）
for d in casmi26-analyticon-megx casmi26-m5-v2d casmi26-joint-ranker; do
  kaggle datasets metadata heliosli/$d -p $S/md_$d; kaggle datasets files heliosli/$d --page-size 50; done
kaggle datasets metadata dmitriigluzdov/casmi26-pubchem-popularity-prior -p $S/md_pop
kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --sort-by scoreDescending --page-size 30
# セルの比較: 2 つの ipynb の各セルの source の sha1 を並べ、違うセル（26・27）を difflib で比べた
```
