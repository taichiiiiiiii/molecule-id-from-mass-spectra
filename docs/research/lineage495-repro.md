# 0.495 系統の再現性の穴（FR 3-25 の学習の入力・extlib-gnps-full の組み直し）と heliosli 0.496 の部品、自前で学習し直す道・v3 で測る道

Issue #174。計画 2026-11-20 昼 項目 4（カード t_e44c0174）。親 #3、関連 #164（0.495 写しの床）。**読むだけ**で、学習・push・提出・GPU は 0。数えたのは採点を伴わない `scripts/cv/rank_lock_bound.py`（§5.3、既にある対照の CSV を読むだけ）だけ。ダウンロードは合計約 16 MB（下の出典 1〜4 の小さなファイルだけ。extlib の `ext_mz.npy`・`ext_in.npy`〈各 101.7 MB〉と heliosli の重みは落としていない）。

印の意味:
- **確認**: 原文（ファイル・CLI の出力）をこの調査で読んで確かめた
- **主張**: 出典にそう書いてあるが、こちらで再現していない
- **未確認**: 確かめられなかった、または推測

## 出典（読んだ範囲）
1. `taeyangg4/casmi26-fr3to25-clean`（https://www.kaggle.com/datasets/taeyangg4/casmi26-fr3to25-clean 、版 1、CC BY-NC-SA 4.0）: 10 ファイルを全部取得し `SHA256SUMS` で 9/9 OK（確認）。`fr_train_clean_v1.py`（795 行）・`train_provenance.json`・`train_result.json`・`fr_config.json`・`SOURCE.md` は全部読んだ。`train_log.jsonl` は 254 行のうち最初の数行・最後の 3 行と start/done の行を読んだ。`model.py` は `Store` と損失の部分だけ、`fragfeat.py` は `USABLE_ADDUCTS`・`featurize` の所在だけを見た
2. `agentili/casmi26-fr-reranker-blend`（https://www.kaggle.com/datasets/agentili/casmi26-fr-reranker-blend 、ライセンス表示 other、説明は空）: 8 ファイルを取得。`fragfeat.py`・`model.py` の sha256 を測り、`fr_config.json`・`fr_selftest.json` と `harmonized_fr_blend.py` の docstring を読んだ
3. `taeyangg4/casmi26-extlib-gnps-full`（https://www.kaggle.com/datasets/taeyangg4/casmi26-extlib-gnps-full 、版 1、CC BY-NC-SA 4.0）: `BUILD.json`・`SHA256SUMS`・`ext_meta.parquet` を取得（sha256 は SHA256SUMS と一致、確認）。`kaggle datasets metadata` の description を全文読んだ
4. `heliosli/casmi26-0-496-public-demo`（https://www.kaggle.com/code/heliosli/casmi26-0-496-public-demo 、`kaggle kernels pull -m`、ノート sha256 `87068208…`、30 セル〈コード 21〉）: kernel-metadata は全部、各セルは先頭のコメントと設定行を読んだ（セル 8 の 529 KB の埋め込みは読んでいない）。追加の 6 dataset は `kaggle datasets files`（全ページ）と metadata の description を読み、小さいファイル（`public_demo_manifest.json` 6 本、`joint-ranker/README.md`、`rt-np/README.md`、`m5-v2d/m5_manifest.json`、`rr-v3/rr3_config.json`）だけ取得
5. 0.495 原本 `taeyangg4/casmi26-lb-0-495-network-fr-p5-tail`（pull し直し、sha256 `82a14431…` で #164 と一致、確認）: kernel-metadata とセル 35（FR 3-25）の先頭のコメントと式の行
6. ディスカッション（`kaggle competitions topics show <slug>/<id> --format json`。この API は話題の本文を返さず、題とコメントだけ）: 757307（コメント 5 件を全部）、760376・760397（コメント 0）、745228（コメント 1）。話題の一覧 131 件のうち取得できたもの（§1.3）のコメントを taeyang・structdb・fr3to25・fr_train・extlib-gnps・pool_chemistry・MULTIPLEX で検索
7. この repo: `docs/research/final-repro-inventory.md`（R1〜R15 の主催者の答えの番号はここに合わせる）、`outputs/local_eval/net_mde.json`（`scripts/cv/net_mde.py`）、Issue #171 の採否の規則、今回の `scripts/cv/rank_lock_bound.py`

## 1. FR 3-25 の学習の入力（fr3to25-clean）

### 1.1 何をどう学習したか（確認: train_provenance.json・train_log.jsonl・スクリプト）
- 学習スクリプト `fr_train_clean_v1.py` 1.0.0（sha256 `ea363ad5…`、SHA256SUMS・provenance の `script_sha256` と一致）。5 段: `pool`（CPU）→ `queries`（CPU）→ `features`（CPU、6 プロセス）→ `verify` → `train`（GPU）
- 問い合わせ: train の構造ごとに 1 つ（metric InChIKey-14）、1 つの ingest_lib から k 本（k は見える test の分布 `K_P` で 1〜8）。train 254,971・val 10,812（構造の 4% を `sha256(seed:key)` で val）。候補 = 正解 + 同じ組成式の負例 16（Tanimoto 上位 8 + 乱択）+ ±0.02 Da の別組成式 4。train の候補 4,480,060・val 189,992、`val_masked_leak` 0
- 学習: FRNet（d 128、5 層、767,942 パラメータ）、listwise 損失、AdamW lr 5e-4・wd 1e-4、bq 64、12 epoch（47,808 step、warmup 500、cosine）、seed 20261113。best epoch 9、val group MRR 0.8892（最終 epoch は 0.8871）、最後の train_sub_mrr 0.924
- **GPU は Kaggle ではなく作者の手元の NVIDIA GeForce RTX 4070 Ti SUPER**（`info.gpu`、メモリの上限 5.5 GB に固定、ピーク 1.893 GB）。学習 3,182.5 s、ログの速さ 14〜23 step/s。特徴の store は 10,265,459,856 B（RAM、読み込み 146 s）
- 決定性: 「seeded … CUDA index_add_ atomics and TF32 make runs close but not bit-identical」（provenance の記載、主張）

### 1.2 入力の一覧（n = 6、スクリプトの `WHITELIST` で sha256 を assert する固定の入力。これ以外は読まないとスクリプトと SOURCE.md が書く）
1. `data/raw/train.parquet`（sha256 `9423f90e…`）: **確認**（競技の train。手元の `data/train.parquet` の sha256 が `9423f90ec08b941a9646b535bf0606183e86ff0c5a297978f5a70693d41bcd98` で一致）
2. `data/derived/structure_keys.parquet`（`9c68c4a3…`、列 normalized_smiles・inchikey14・metric_ik14・n_spectra）: **見つからない**。生成コードもファイルも、taeyangg4 の CASMI の dataset 2 件（fr3to25-clean・extlib-gnps-full）、CASMI のノート 1 件（0.495 原本）、agentili の dataset には無い。列から見て train の構造ごとの鍵の表（metric_ik14 は指標の InChIKey-14〈RDKit の互変異性の正準化の後〉と推測、未確認）なので、train だけから自前のコードで同じ意味の表は作れる見込み（バイト一致は未確認）
3. `data/derived/structdb_v1/entries.parquet`（`b976eff7…`、列 source・smiles・formula・mass・std_ik14）: **見つからない**（探した場所は 2 と同じ）。SOURCE.md は中身を「COCONUT, LOTUS, PubChemLite, ChEBI/LIPID MAPS, NPAtlas, KNApSAcK, NPASS, CMAUP, CMNPD and other public natural-product and metabolite lists」と書く（主張）。行数・各 DB の版・取得日・標準化の手順の記録は見つからない。KNApSAcK・NPASS・CMAUP・CMNPD のライセンスはこの調査で確かめていない（未確認）。主催者が名指しで可としたのは COCONUT（R10）・LOTUS と NPAtlas（R1）・PubChem（R12・R13）
4. `runs/20261104_h54_s2/pool_chem/pool_chemistry.json`（`d84cd40e…`、{smiles: [key, formula]}）: **見つからない**（同上）。SOURCE.md の「engine-1 candidate pool chemistry」から、v2 プール（`ahmedberatozer/casmi26-v2-pool`、TC に生成コードあり＝ final-repro-inventory の行 6）から作ったものと推測するが、作った手順は見つからない（未確認）
5. `runs/20261113_h111_s2_fr2to25/dataset/fragfeat.py`（`7b54023d…`）: **確認**（`agentili/casmi26-fr-reranker-blend` の `fragfeat.py` と sha256 一致）。規則ベースの特徴（学習した重み無し）。ただし agentili の dataset のライセンス表示は「other」で説明が空（ライセンスの中身は未確認）
6. `runs/20261113_h111_s2_fr2to25/dataset/model.py`（`b6276a28…`）: **確認**（同じく agentili と一致。アーキテクチャと損失だけ）

途中の生成物（`data/pool.parquet` `8d61ee36…`、`data/queries.parquet` `f84a0242…`、特徴 `train_000`〜`031.npz`・`val_000`〜`001.npz`）は sha256 だけが provenance にあり、ファイルは公開されていない（スクリプトから作り直すもの）。

### 1.3 探した場所（「無い」ではなく「ここには見つからない」）
- taeyangg4 の dataset 一覧（`kaggle datasets list --user taeyangg4`、10 件。CASMI は fr3to25-clean と extlib-gnps-full の 2 件だけ）、ノート一覧（`kaggle kernels list --user taeyangg4`、17 件。CASMI は 0.495 原本だけ）
- fr3to25-clean の 10 ファイル、0.495 原本のセル 35 のコメント（「Appended … by scripts/build_fr325_clean_v1.py (cell template fr3to25_cell_clean_v1.py 1.0.0)」とある。この 2 つのスクリプトも公開物には見つからない）
- `kaggle datasets list -s fr-reranker`（4 件: agentili・taeyangg4・boomvector の frozen assets・無関係の 1 件。boomvector は中身を見ていない）
- ディスカッション: 話題 131 件の一覧のうちコメントを取得して検索できたのは一部（API の 429 で残りを取り直した数は Issue #174 に書く）。taeyang の名のコメントと structdb・fr3to25・fr_train・pool_chemistry を含むコメントは**見つからない**。見つかった関係のあるものは 757307 の 기욱 이（2026-11-19）の質問（§2.3）だけ

### 1.4 判定
- **FR 3-25 の重みは「学習コードあり・入力 6 のうち 3 が公開物に見つからない」**。R7（756893、2026-11-16「public weights are fine to load only if the code and permitted data that produced them are available, so the weights can be regenerated」）に照らすと、入力 2〜4 が出るまで「重みを作り直せる」は満たされない読み。0.495 写しを最終の自前の枠に使わない（#164 のとおり）判断は変わらない
- **自前の学習し直しは成り立つ見込み**: スクリプトは公開（CC BY-NC-SA 4.0）で、入力 1・5・6 は取れる。2 は train から作れる。3・4 は「同じ物」は作れないので、許可されたデータ（v2 プール〈train + COCONUT〉・LOTUS・PubChem 層）から**自前の負例の表**を作り、スクリプトの `WHITELIST` を自前の版に書き換えた写しで回す。これは「rebuilt with own code（負例の出どころが違う）」で、原本の再現ではない。R14（745072「if you retrain equivalent models with your own published code and permitted data, that would satisfy the rule」）の形に当たる

## 2. extlib-gnps-full の組み直し

### 2.1 中身（確認: BUILD.json・ext_meta.parquet・description）
- 198,581 行・147,414 構造。列は inchikey14・normalized_smiles・adduct・ionization_mode・instrument_type・precursor_mz・source_library・source_id・licence の 9 つ
- **194,652 行が GNPS MULTIPLEX-SYNTHESIS-LIBRARY-ALL**（ext_meta では PARTITION-1〜6 に分かれ、合計が BUILD.json の 194,652 と一致）。licence は全行「GNPS library (CC0 for deposited spectra); propagated-type library」。(inchikey14, adduct) ごとに最大 1 行（確認。description の「one spectrum per InChIKey-14 and adduct」と合う）
- 残り 3,929 行は GNPS2 の各ライブラリ・MassBank・MS2DeepScore の訓練スペクトル・MSnLib・MoNA・MassSpecGym など。licence 列は全行「per source library (data/external/*/SOURCE.md)」で、**その SOURCE.md は dataset に無い**ので、この 3,929 行の行ごとのライセンスはファイルからは分からない
- source_id: CCMSLIB（GNPS の accession）197,466 行、MSBNK 1,004、MSnLibID 50、MoNA 30 ほか。空は 0。**同じ accession が 2 行に出るものが 1,680 個（3,360 行）**: 1,679 個は同じ構造・同じ precursor で adduct の書き方だけ違う（例 `[M+H-2H2O]+` と `[M-2H2O+H]+`）、1 個は `[M-H]-` と `[M+H]+`（gnps2:SUMNER と ms2deepscore の両方から入った行）
- 装置: Orbitrap 系がほとんど（LC-ESI-Orbitrap 196,075 行）。CASMI 2012 の記録が 1 行（ms2deepscore の正例 `MSBNK-CASMI_2012-SMI00142`）
- ピーク: `ext_mz`・`ext_in` は float32 [N,128]、m/z 昇順、sqrt(強度/最大)、0 詰め（BUILD.json の format）。**128 本の選び方（強度の上位か）と前処理は書かれていない**（未確認）

### 2.2 組み直す手順の見込み（未実行）
1. MULTIPLEX の 194,652 行: 行ごとの CCMSLIB accession があるので、GNPS2 から MULTIPLEX-SYNTHESIS-LIBRARY-ALL の 6 区画を取り、accession の一覧で行を選ぶ（選び方の規則を知らなくても、一覧そのものが選び方になる）。URL・入力の sha256・変換のスクリプトを自前で公開する
2. ピークの前処理は書かれていないので、自前の規則（例: 強度の上位 128 本 → m/z 昇順 → sqrt(I/max)）を決めて公開し、`ext_mz.npy`・`ext_in.npy` と一致するかを記録する（一致は要件にしない。R4 Q2 でビット一致は要らない）。照合には npy 2 本（203 MB）のダウンロードが要るので次の計画で
3. 残りの 3,929 行: MassBank の記録は記録ごとにライセンスが違う（CC BY または CC BY-NC-SA、description の主張）。行ごとのライセンスが dataset から分からないので、組み直すなら取り直すときに記録ごとのライセンスを付け直す。手間に比べて行は 2% なので、**最初の組み直しは MULTIPLEX の 194,652 行だけ**にする案（残りを足すかは別に決める）
4. 0.487 の棚卸しで「不可（生成コード無し）」とした `takumuhata/casmi26-extlib`（final-repro-inventory の表の行 21）の置き換えになる。どちらも生成コードは dataset に無いが、こちらは行ごとの accession があるので自前で組み直せる

### 2.3 GNPS-NIST14-MATCHES 117 行の扱い
- 117 行（116 構造、ESI-qTof 69・ESI-Ion Trap 42・ESI-QQQ 5・LC-ESI-qTof 1、adduct はほぼ `[M+H-H2O]+` 113）。description は「there are no spectra from the NIST library itself」と書く（主張）
- 主催者の答え: R11（757307、David Healey 2026-11-16 00:53 UTC）の本文は「GNPS-NIST14-MATCHES in GNPS is fine to use as」で**文が途中で切れている**（API が返す HTML も `<p>GNPS-NIST14-MATCHES in GNPS is fine to use as</p>` のまま、確認）。可の向きだが、何として可かは読めない
- 760376（Kurapapa 2026-11-19、題「Clarification on three external spectral libraries: licensing, reproducibility, and CASMI/NIST records」）はコメント 0（主催者の答え無し）。757307 の 기욱 이（2026-11-19 08:12 UTC）の「MULTIPLEX を URL・sha256・ビルドスクリプトを公開して使ってよいか。GNPS-NIST14-MATCHES は全部除く」にも答えが無い
- **案: 自前の組み直しでは 117 行（と CASMI 2012 の 1 行）を除く**。117 行は全体の 0.06% で、装置も競技の test（timsTOF）と違うので、除いて失うものは小さい見込み（未測定）。答えが「可」とはっきりしたら戻す

## 3. heliosli 0.496 の部品

### 3.1 0.495 原本との比較（確認: kernel-metadata）
- dataset_sources 13 個。0.495 原本（11 個）との共通は 7 個（sanpier の 2 個、dmitriigluzdov の 5 個）
- **0.495 にだけあるもの 4 個**: `taeyangg4/casmi26-extlib-gnps-full`、`taeyangg4/casmi26-fr3to25-clean`、`samartalwar/...-massbankharmonized`、`metric/rdkit-2026-3-3-wheel`。kernel_sources の `popwenman/casmi26-scorer-train-v80` も 0.496 には無い（空）。docker_image も違う
- **0.496 にだけあるもの 6 個**: すべて heliosli の dataset（下）
- つまり **0.496 は 0.495 の上に足したものではなく別の系統**。セル 0 の題は「LB 0.477 · Two 0.472 Fixes, Merged the Right Way」で、土台は busyaprime / Roman Tamrazov（Spectral Parallax）系の 0.477 ノート（v4g の資産の上）。その後ろに heliosli のセル 11・23〜29 が足されている

### 3.2 増えた部品の分類（6 個すべて「構造」。生成コードの有無は dataset のファイル一覧で判定）
**最重要: 6 個とも学習済みの重みが入っていない。** 各 dataset の `public_demo_manifest.json` は `"competition_weights": false`、`"note": "This version does not contain the model weights used for the 0.494 score."`、description は「Private learned parameters are replaced with initialized demo weights (the joint ranker uses a synthetic fit). Self-tests validate this demo version only」（確認、6 本とも同じ趣旨）。`rr3_config.json` は `"chosen": "public_demo_random_initialization"`。だから **今の公開版を写しても、heliosli の部品は乱数の初期値か合成の当てはめで動く**。題名の 0.496 がどの版・どの重みで出た値かは確かめられない（未確認）
1. `casmi26-rr-v3`（CC BY-NC-SA 4.0、200 MB）: 同じ組成式の異性体の並べ替え器（1〜2 結合切断の断片とピークの一致、グラフ網＋transformer、`fs0_lean_plain.pt` の符号化器）。セル 24、FREEZE（train の構造の候補は位置を動かさない）、lib_max ≥ 0.9 の分子は触らない。**構造（学習したモデル）**。生成コード: 推論・データ・モデルのコード（`rr_data.py`・`rr_model.py`・`rr_frag.py` ほか）はあるが、学習スクリプトは一覧に見つからない
2. `casmi26-frigid-e1`（other、1.6 GB）: FRIGID（de novo 生成、上流の `frigid_src/dlm`・`mist`）＋ E1 指紋、2 枚目の T4 で背景に回し、条件付きで 2 位に入れる（セル 11・28）。**構造（新しい候補源）**。生成コード: 上流のモデルコードと wheel はあるが、E1 の学習スクリプトは見つからない。`DLM.ckpt`（1.3 GB）は置き換え一覧に無い（上流の重みかは未確認）
3. `casmi26-joint-ranker`（other、8 KB）: LightGBM lambdarank（`joint_scn.txt`）で PubChem 層＋天然物の表の同組成式の候補を並べ、1 位を見直す（セル 26、CV-JOINT、margin ≥ 2.66 で 1 位を入れ替え）。**構造（学習したランカー）**。生成コード: 見つからない（モデルの txt と特徴の json だけ）
4. `casmi26-m5-v2d`（CC0-1.0、85 MB）: M2・M3（5 fold）・ICE-T（ms-pred の ICEBERG 系の fullT）でリストの中の異性体を判定（セル 27・29、M5_MODE swap）。**構造（学習したモデル）**。生成コード: **あり**（`isomer/train_m2.py`・`isomer/m3_fit_v2.py`、ms-pred の train スクリプト群）。ただし重みは demo
5. `casmi26-pxv2-deploy`（CC0-1.0、35 MB）: PX v2（joint の上位 20 を joint + 1.0 × z で再採点、セル 26 の中）。**構造（学習したモデル）**。生成コード: 推論・データ・モデルのコードだけ、学習スクリプトは見つからない
6. `casmi26-rt-np`（other、49 MB）: RT（スペクトル・構造の 2 塔の符号化器、`rt_best.pt`）と天然物の候補表 `npdb_np.parquet`（README: COCONUT 2026-11 CC0、LIPID MAPS LMSD 2026-11-01 CC BY 4.0、HMDB 5.0 非商用、主張）。**構造（学習したモデル＋候補表）**。生成コード: 見つからない（コードのファイルが無い）
- 定数の変更だけの部品は無い（セル 1 の VARIANT・CUSTOM は 0.477 の土台の切り替え）。CV-JOINT の 2.66・0.58、M5 の tau 3.376 などは部品の中の定数で、我々が写すと定数の写しになる
- 使える見込み: **低い**。重みが無く学習コードも 6 個中 1 個（m5）にしかないので、写しても 0.496 は出ない見込みで、自前で作り直すには学習データの作り方から作ることになる。今は「構造の着想（rank 1 の見直しを joint ranker で、de novo 候補を 2 位に）」として記録するだけにする

## 4. 自前で学習し直す道（見積もり、実行しない）

共通: スクリプトは固定の `WHITELIST` を assert するので、入力 2〜4 を自前の物に替えるには**自前の写し**（`WHITELIST` と入力の作り方だけを変え、`fragfeat.py`・`model.py` は sha256 固定のまま）を repo に置く。学習の手順（epoch 数など）を変えるかは、オーナーの「Loss が下がり切るまで」の規則との関係を事前登録で決める（原本は 12 epoch 固定・cosine・best val を保存。val は構造の 4% のハッシュで、学習データの中の内部の検証。v3 も LB も見ない）。

### (a) 提出用・全 train・ローカル MPS
- 入力: train.parquet（手元、sha 一致）、structure_keys（train から自前のコードで作る）、負例の表（v2 プール〈train + COCONUT〉＋ LOTUS ＋ PubChem 層の同じ組成式・±0.02 Da の構造。structdb_v1 の代わり）、fragfeat.py・model.py（agentili から取得済み）
- 時間の見込み（**どれも未検証**）:
  - pool・queries: RDKit の組成式を 27 万構造＋負例の表で数える。数十分の見込み
  - features: 約 26.6 万問い合わせ × 約 18 候補（4,670,052 候補）を fragfeat で特徴にする。1 候補あたりの時間の記録は公開物に無い（features の summary.json は provenance の sha256 だけ）。**1〜5 h（6 プロセス）と幅が広く、`--n-train 2000 --n-val 200` の試走で測る必要がある**
  - train: 作者の RTX 4070 Ti SUPER で 3,182 s（14〜23 step/s）。MPS の速さは分からない。以前の FPNet では M4 Pro の MPS が Kaggle T4 の 0.15 倍だったが、FR はモデルが小さく（768k パラメータ）index_add_ が多いので同じ比は当てはまらない見込み。**1〜10 h の幅で、短い実データの試走（1 epoch の一部、5 分）で測る必要がある**。GPU 3,182 s からの換算は未検証
- メモリの見込み（未検証）: 特徴の store が RAM に 10.3 GB。`model.Store` は shard を全部 `np.load` してから `np.concatenate` するので、読み込みのときに一時的に約 2 倍（〜20 GB）になる見込み。共用の Mac では、shard を memmap で持つ小さな包み（上流のファイルは変えず、Store を継承した自前のクラス）にする必要がある見込み。GPU 側は CUDA のピークが 1.893 GB（fp32 + TF32）なので、MPS でも数 GB の見込み。**features 段は `pq.read_table(TRAIN, columns=[ms2_mzs, ms2_normalized_intensities, …])` で train の全行のピークを読んでから `take` する**ので、そのままだと約 30 GB を超える見込み（`data/train.parquet` の全ピークは約 33 GB、casmi26.md）。row group ごとに読んで要る行だけ残す包みが要る
- 重いジョブの規則（memory_pressure の関門、1 本だけ、他のプロジェクトの GPU 使用の確認）に従う。M1（#167）の本番が終わってから

### (b) v3 で測る fold-safe 版
- 除くもの: fold-safe FPNet の `split.parquet` で fold 0（v3 の holdout の fold）の構造すべてと、v3 の 4 集合（v3_c1・v3_c2・enveda_v1・v3_c2n）の正解とその別名（raw key・正準の key・互変異性の別名、`dseen.alias_index` と同じ 1 段）を、**問い合わせ（正例）と負例の表（train の構造・負例の表のすべての出どころ）の両方から**除く。v3_c2 の正解は PubChem 層にいるので、PubChem 層由来の負例からも除く
- 除いた数を記録し、`verify` 段と同じ形の検査（val の正解・train の正解・候補の key と除く集合の重なりが 0）を、raw key と正準の key の両方で数える
- 大きさ: fold 0 はおよそ構造の 1/5 なので、問い合わせは約 21 万、時間とメモリは (a) の約 0.8 倍の見込み（未検証）
- 学習は 1 回だけ（seed・手順は事前登録で固定）。重みの選び方は内部の val（4%）だけで、v3 を見ない

## 5. v3 で測るなら（案。測定は入力の目処が立った後の計画で事前登録してから）

### 5.1 置き場所と合わせ方
- 0.495 原本の FR 3-25 は、最終リストの 1・2 位を固定し、3〜n 位を `key = −rank + W·z(fr)`（W 3.0、z はリスト全体で標準化）で並べ替える（セル 35 のコメント、確認）。W 3.0 はセル 35 の 23 行目のコメントで「H-A-111 s2c measured W above the author's 1.0 clamp locally」とあり、作者が測って選んだ値（我々が写すと定数の写し）
- v3 の縮小版では `run.py` の frag の後の上位 25 件に置き、1・2 位を固定、3〜25 位を **z(frag の後の点) + z(fr) の等重み**（rep・lfus と同じ既存の形、z は 3〜25 位の候補で ddof 0、点の無い候補は z(fr) = 0）で `argsort(kind='stable')` する案。W は使わない

### 5.2 採否の規則（事前登録の案。Issue #171 と同じ形）
- 対照 `score,prior,frag`（v3_c1 0.8535 / v3_c2 0.0000 / enveda_v1 0.8376 / v3_c2n 0.6284、n 200/200/200/195）。新しい並べ替え器なので、(答えの出所 × 参照の有無) の 4 集合 **すべて**で測る: v3_c1（正解はプールにある・参照は他の装置だけ）、v3_c2（正解がプールに無い）、enveda_v1（正解の参照を隠す）、v3_c2n（正解はプールにある・参照なし）
- **採用 ⇔ 4 集合すべてで「新 − 対照」の 95% 区間（分子ブートストラップ 2,000 回、seed 0）の下限 ≥ −0.02、かつ少なくとも 1 集合で下限 > 0**。集合が 1 つでも欠ければコードが採用を拒む。v3_c2 は並べ替えだけの段なので rr が全部 0 を assert。C1（v3_c1）と C2（v3_c2）を必ず書く
- 集合ごとの MDE（`outputs/local_eval/net_mde.json`、2.8·sd(Δrr)/√n、既にある変種の最小・中央値・最大）: v3_c1 0.0176 / 0.0324 / 0.0434、v3_c2 定義できない（対照が全部 0）、enveda_v1 0.0175 / 0.0321 / 0.0401、v3_c2n 0.0171 / 0.0428 / 0.0538

### 5.3 届く伸びの上限（`scripts/cv/rank_lock_bound.py`、確認）
1・2 位を固定して 3〜25 位だけを並べ替える段は、4 位以下の正解を最善でも 3 位に上げるだけ。対照の分子ごとの rr から、上限 = Σ max(0, 1/3 − rr) / n:
- v3_c1: 動かせる分子 5、上限 **0.0040**（MDE の最小 0.0176 より小さい）
- v3_c2: 0（正解が 25 件に無い、200/200）
- enveda_v1: 動かせる分子 8、上限 **0.0065**（MDE の最小 0.0175 より小さい）
- v3_c2n: 動かせる分子 32、上限 **0.0298**（MDE の最小 0.0171 と中央値 0.0428 の間）
- 照合: 同じスクリプトの `--lock 1` は `net_mde.json` の `gain_sum_lock1`（v3_c1 3.2917）、`--lock 0` は `gain_sum_unlocked`（26.2917）と一致（確認）

**読み**: 原本どおりの置き場所（3〜25 位）では、v3_c1・enveda_v1 は上限そのものが MDE より小さく、決まるのは v3_c2n だけ。採否の規則の「少なくとも 1 集合で下限 > 0」は v3_c2n でしか満たしようがない。0.495 の盤面では 3 位以下に正解がある分子がもっと多い可能性がある（縮小版の対照は 1・2 位に正解が多い）ので、**v3 で FR 3-25 を測っても「決まらない」になる見込みが高い**。次の計画の選択肢（どれを採るかはこの調査では決めない）:
- (i) FR を縮小版の別の置き場所（prior の後の上位 60 に z(fr) を等重みで足す＝ rep・lfus と同じ位置、1 位の固定なし）で測る。上限は lock 0 で v3_c1 0.1315・enveda_v1 0.1274・v3_c2n 0.2434 と大きい。ただしこれは 0.495 の FR 3-25 とは別の使い方（新しい構造）として事前登録する
- (ii) v3 では測らず、(a) で作り直した FR の重みを 0.495 床の FR の重みと入れ替える診断提出（帯は Public の SE ≈ 0.035 から、間は「決まらない」）で、再現性の置き換えとして読む

## 6. まとめ（受け入れ条件との対応）
- FR の学習の入力: 6 個。確認 3（train.parquet・fragfeat.py・model.py）、見つからない 3（structure_keys・structdb_v1・pool_chemistry。探した場所は §1.3）。自前の写しで入力 2〜4 を許可データから作り直す道は成り立つ見込み
- extlib: MULTIPLEX 194,652 行は accession で選び直せる。ピークの前処理と残り 3,929 行の行ごとのライセンスが穴。GNPS-NIST14-MATCHES 117 行は除く案（R11 は文が切れていて、760376 に答えが無い）
- heliosli 0.496: 0.495 の上ではなく 0.477 系の別系統。増えた 6 部品はすべて構造（学習したモデル 5・新しい候補源 1）で、**どれも競技で学習した重みが入っていない demo 版**。学習コードがあるのは m5 だけ
- 学習の見積もり: (a) features 1〜5 h・train 1〜10 h・RAM 10〜20 GB（どれも未検証、試走で測る）、(b) その約 0.8 倍
- v3 の 4 集合の規則と MDE は §5.2、原本の置き場所の上限は §5.3（v3_c1 0.0040・enveda_v1 0.0065・v3_c2n 0.0298・v3_c2 0）
