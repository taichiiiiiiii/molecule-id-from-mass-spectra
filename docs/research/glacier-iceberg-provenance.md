# 調査: GLACIER・ICEBERG（0.487 系統の順方向予測）の出どころと再現性、v3 で測る案（#86）

0.487 系統（最終候補）は、同じ組成式の候補を ICEBERG と GLACIER（coleygroup/ms-pred の順方向スペクトル予測）で並べ替える。この部品が主催者回答に照らして最終提出に残せるかを判断する材料として、出どころ・学習コード・学習データ・ライセンス・ハッシュの照合の可否を出典つきでまとめる。読んだのは README・LICENSE・設定・小さいファイルだけで、**重みはダウンロードしていない**（取得は合計約 18 MB。最大は ms-pred の `data/spec_datasets/msg/labels.tsv` 17.6 MB）。コードの変更は無い。

印の意味: **確認** = こちらで一次資料（ファイルの中身・コード・API の値）を見て確かめた。**主張** = 作者や第三者の記述だけで、こちらでは確かめていない。**未確認** = 資料が見つからない、または確かめる手段を今回は使っていない。

## 出典

1. coleygroup/ms-pred、commit `708148c2a8eb0c32120644436fefd2fe90cd2214`（2026-09-24、"Update GLACIER paper name"）。https://github.com/coleygroup/ms-pred/tree/708148c2a8eb0c32120644436fefd2fe90cd2214 。読んだファイル: `README.md`、`LICENSE`、`configs/glacier/joint_train_msg.yaml`（sha256 b5da82ad…）、`configs/glacier/joint_contr_finetune_msg.yaml`、`run_scripts/glacier/{01_train_joint.sh,run_all.sh,add_inten.sh}`、`src/ms_pred/glacier/{train_joint.py,train_contr_joint.py}`、`src/ms_pred/common/splitter.py`、`configs/iceberg/msg_all/*.yaml`（`dag_train_msg_all_iceberg.yaml` sha256 8f4b679a…、`dag_inten_contr_finetune_msg_all_iceberg.yaml` sha256 e5a32034…）、`configs/iceberg/msg_simulation/{dag_train_msg_simulation.yaml,dag_inten_contr_finetune_msg_simulation.yaml}`、`run_scripts/iceberg/msg_all/{run_all.sh,05_build_msg_all_iceberg_dataset.py,04_impute_missing_collision_energies.py,07_…,09_…}`、`data_scripts/create_msg_simulation_dataset.py`、`data/spec_datasets/msg/labels.tsv`（17,651,715 バイト、sha256 0b29f4b1…）
2. ms-pred のその後の commit（GitHub API `repos/coleygroup/ms-pred/commits`）: `bccbed684887`（2026-10-26 "Support all MassSpecGym dataset strategies from Hugging Face"）、`8dfe2d3ec7c6`（2026-11-03 "Enforce declared checkpoint adduct and instrument support"、README の GLACIER の重みのリンクを差し替え）、`e2fc3a95022b`（2026-11-03、README の引用と重みの申請先を修正。11/12 時点の main の先頭）
3. 上流の重みの配布ページ（Dropbox、ページの一覧だけを読んだ。11/12 23:40 JST 頃）
   - ICEBERG `msg_all`（Runzhong Wang 共有）: https://www.dropbox.com/scl/fo/kwm35ih8tlfnshfrcq8ot/AOeS4M0_v9MhqeEZys9sCRQ?rlkey=f1n6pbzx94g1k2el2wcbmee61 → `gen/best.ckpt` 39.64 MB（更新「9 days ago」）・`gen/hparams.yaml` 360 バイト（「4 months ago」）、`inten_contr/best.ckpt` 38.72 MB（「9 days ago」）・`inten_contr/hparams.yaml` 526 バイト（「4 months ago」）。hparams.yaml 2 つは取得した（sha256 b4866c3d… / 7b730add…）
   - GLACIER の旧リンク（commit 708148c2 の README が指す。Rui-Xi Wang 共有、フォルダ名 `GLACIER_checkpoint`）: https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui → `best.ckpt` 173.19 MB（「5 months ago」）
   - GLACIER の新リンク（commit 8dfe2d3 以降の README が指す。Runzhong Wang 共有、フォルダ名 `glacier_msg_simulation`）: https://www.dropbox.com/scl/fo/r82eluc6qhm48h9b4xgze/AK_SVKXUOcJlGk8TnC6na9I?rlkey=c4upk39c21h6go4rbzxfmd9jo → `best.ckpt` 173.18 MB（「9 days ago」）
4. Kaggle データセット `ahmedberatozer/casmi26-glacier`（ファイル作成 2026-09-28）と `ahmedberatozer/casmi26-iceberg`（同 2026-09-26）: `kaggle datasets files … --page-size 100`、`kaggle datasets metadata`、`README.md`・`LICENSE-NOTICE.txt`・`MANIFEST.json`・`gl_runner.py`（sha256 5c4f7431…）・`ice_runner.py`（sha256 dc0d5d1e…）・`src/ms_pred/LICENSE`、比較用に `src/ms_pred/{glacier/joint_model.py,glacier/dataset.py,iceberg/gen_model.py,iceberg/inten_model.py,iceberg/joint_model.py}`
5. Kaggle データセット `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets`（0.487 版の `dataset_sources` に入っている写し）: 最上位の `MANIFEST.json`（45,195 バイト）と `casmi26-glacier/MANIFEST.json`
6. 0.487 系統の凍結ノートブック（`k487_frozen_raw.bin`、59 セル、#77 で取得したもの）の F セル 26・41・45・49
7. GLACIER の論文: R.-X. Wang, R. Wang, C. W. Coley, "GLACIER: Rethinking Mass Spectrum Prediction as an Object Detection Problem", arXiv:2606.29161（v1 2026-06-28、v2 2026-10-05。論文のライセンスは CC BY-NC-ND 4.0）。https://arxiv.org/abs/2606.29161 （HTML の v1 本文を読んだ。v2 の HTML は取れなかった）
8. ICEBERG 2.0/2.1 の論文: Wang et al., "Neural Spectral Prediction for Structure Elucidation with Tandem Mass Spectrometry", bioRxiv, https://doi.org/10.1101/2025.05.28.656653 （ms-pred README の引用で確認。本文は読んでいない）
9. 主催者回答（`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/<id> --format json`、11/12 23:45 JST 頃に取得）: 742991、744338、745841、748882、757307、755373
10. 既存の調べ: `docs/research/mist-massspecgym.md`（#66、MassSpecGym のライセンスと重なり）、`outputs/research/msg_overlap.json`（`scripts/research/msg_overlap.py` の出力）、`docs/research/lineage-reproducibility.md`（部品表 13・14 行）

## 結論（先に）

- 規則の当てはめ: ICEBERG・GLACIER の MassSpecGym 版は主催者が名指しで承認している（744338「so is GLACIER」、745841「GLACIER/ICEBERG … are all fine」）。748882（10-13）の 1 番の条件「公開チェックポイント＋著者の元の学習コードと許可された学習データへの参照＋自前の微調整・推論コード」に当てはめると、**学習コードと学習データの参照は揃う（確認）**。ただし残り 2 つに穴がある: (a) Kaggle の重みは「公開チェックポイントそのもの」ではなく作者による保存し直しで、上流との一致はテンソル単位でもこちらでは**未確認**。(b) 推論コード（`gl_runner.py`・`ice_runner.py`・`fuse.py`・`gl_fuse.py`）は参加者（ahmedberatozer）のもので、ライセンスの記載が見つからない（**未確認**）。755373 は重みに触れていないので当てはまらない（確認）。
- よって「**規則上は残せる見込みだが、納品の条件を満たしたとはまだ言えない**」。最終に残すなら、提出前に (a) 上流の best.ckpt と Kaggle の .pt の state_dict の照合、(b) 推論コードのライセンスの確認（取れなければ ms-pred〈MIT〉の推論関数を使う自前のコードに置き換え）が要る。どちらも今回の範囲（重みを落とさない、50 MB 以下）では実行できない。
- v3 での測定: 事前に決めた規則（#66 と同じ、学習側 = MassSpecGym の train ∪ val）では v3_c2n の残りが 94 で 100 未満になり測定不能。4 集合が揃わないので、**今の v3 では採用の判定ができない（＝採用できない）**。

## 確かめたこと

### (i) GLACIER の学習コードのパスと設定

- 学習スクリプト: `src/ms_pred/glacier/train_joint.py`（ラベルつきの学習）と `src/ms_pred/glacier/train_contr_joint.py`（対照学習の微調整）。起動は `python launcher_scripts/run_from_config.py <yaml>`。**確認**（出典 1）
- MassSpecGym 用の設定: `configs/glacier/joint_train_msg.yaml`（`dataset-name: msg`、`split-name: split.tsv`、seed 1、batch 32、max-epochs 200、min-epochs 20、lr 0.0004、hidden 256、graphormer 5 層、`num-bins 150000`、`inten-loss-fn cosine`、保存先 `split_rnd1`）と `configs/glacier/joint_contr_finetune_msg.yaml`（`train-checkpoint: results/glacier_msg/split_rnd1/version_0/best.ckpt` から始め、`contr-loss-fn entropy`、`num-decoys 5`、PubChem の `pubchem_formulae_inchikey.hdf5` を使う）。**確認**（出典 1）
- ただし `run_scripts/glacier/01_train_joint.sh` は NIST20 の設定（`joint_train_nist20.yaml`）を呼ぶ形で、MSG 版を動かす起動スクリプトは無い（MSG 版は yaml を直接渡す）。`run_scripts/glacier/run_all.sh` は存在しない大文字のパス `run_scripts/GLACIER/…` を呼ぶ。**確認**（出典 1）
- MassSpecGym の学習データの置き場 `data/spec_datasets/msg/` には `labels.tsv` だけがあり、学習に必要な `splits/split.tsv`・`magma_outputs/magma_tree_with_inten.hdf5`・`spec_files.hdf5` はリポジトリに無い。commit 708148c2 の時点では `msg` 用の split.tsv を作るコードも見当たらない（`create_msg_simulation_dataset.py` は `msg_simulation` を作る）。**確認**（出典 1）
- 配布された旧 GLACIER チェックポイント（173.19 MB）が、どちらの yaml（対照学習の前か後か）・どの版のコードで作られたかの記録は見つからない。**未確認**
- 補足（Kaggle 側のコード）: `ahmedberatozer/casmi26-glacier` の `src/ms_pred/glacier/joint_model.py`・`dataset.py` は commit 708148c2 の同名ファイルと改行コード（CRLF）を除いて一致。**確認**（出典 1、4）

### (ii) 学習データの版と分割

- 学習時の分割の使い方: `train_joint.py` は `common.get_splits(spec_names, split_file)` で train / val / test に分け、train で学習、val の `val_loss` で best.ckpt を選び（`ModelCheckpoint(monitor='val_loss')`）、EarlyStopping も val で行い、最後に test を `trainer.test` で評価する。test は学習に使わない作り。**確認**（出典 1 のコード）
- 分割の中身: 論文 v1 §4.1 は「We adopt the official MassSpecGym data split, which contains 231,104 tandem mass spectra (MS/MS) spanning 31,602 unique molecules」と書く（MCES による分割）。**主張**（出典 7）。現在の上流 README の builder 説明は「The builder preserves the official `fold` assignments」。commit 708148c2 の `create_msg_simulation_dataset.py` も MSG の `fold` 列をそのまま train / val / test に写す（`normalize_fold`）。**確認**（コード。ただし旧 GLACIER チェックポイントの学習に使われた split.tsv そのものは**未確認**）
- 学習データの範囲の食い違い: 論文は MSG 全 231,104 スペクトルと書くが、commit 708148c2 の `data/spec_datasets/msg/labels.tsv` は **119,029 行・16,974 構造（InChIKey 14 文字）・付加イオンは [M+H]+ だけ・装置は Orbitrap 81,323 / QTOF 37,706・衝突エネルギーの欠けは 0 行**で、MSG の simulation challenge の部分集合の大きさ（上流 README「119,029 unique simulation-challenge spectra」）と一致する。このファイルの spec ID を `data/ext/msg/MassSpecGym.tsv` の `identifier` に当てると 119,029 行すべてが見つかり、fold は train 99,341 / val 9,734 / test 9,954、InChIKey 14 文字は全行一致。**確認**（出典 1、こちらで集計）。配布チェックポイントがこの 119,029 行で学習されたのか、論文の 231,104 行なのかは**未確認**
- 上流の新しいリンク（commit 8dfe2d3 以降）のフォルダ名は `glacier_msg_simulation` で、上流が今「MassSpecGym 版」として出している GLACIER は simulation 部分集合版と読める（フォルダ名による推測）。Kaggle 版は旧リンクのもの（`LICENSE-NOTICE.txt`: 「GLACIER_checkpoint.zip -> best.ckpt」）。旧と新はサイズが違う（173.19 MB / 173.18 MB）ので別のファイル。**確認**（出典 2、3、4）
- ICEBERG `msg_all`: `run_scripts/iceberg/msg_all/run_all.sh` は「衝突エネルギーが分かる部分（msg_known_ce）で学習 → その模型で欠けた衝突エネルギーを補う → 全 MSG で学習し直す → test_formula / test_mass で検索評価」の順。各段の yaml は `split-name: split.tsv`。`05_build_msg_all_iceberg_dataset.py` は元の `msg_all` の `splits` をそのまま写す（222 行目）。test で検索評価する作りなので test は学習に使わない設計。**確認**（コード）。ただし commit 708148c2 の時点で元の `data/spec_datasets/msg_all`（split.tsv を含む）を作るコードは無い（`run_all.sh` の冒頭「Assumes data/spec_datasets/msg_all already exists」）。builder が `msg_all` に対応したのは 10-26 の bccbed6。**確認**。配布された msg_all の重みの学習に使われた split.tsv そのものは**未確認**
- 衝突エネルギーの補完（`04_impute_missing_collision_energies.py`）は msg_known_ce で学習した模型で test の行の衝突エネルギーも補う作りかどうかは読み切れていない（**未確認**。補完は学習ではないが、test のスペクトルを見てラベルを作る可能性がある）

### (iii) ICEBERG の重みは MassSpecGym `msg_all` 版か

- 作者（ahmedberatozer）の申告: `LICENSE-NOTICE.txt`「ckpt/gen.pt, ckpt/inten.pt : ICEBERG msg_all (MassSpecGym-trained) weights published with ms-pred (README "open-source MassSpecGym-trained weights"), re-saved as {hyper_parameters, state_dict} only (optimizer / trainer state dropped; tensors bit-identical)」、`MANIFEST.json` の `model`「ICEBERG 2.1 (ms-pred, MassSpecGym msg_all checkpoint)」。**主張**（出典 4）
- 上流の配布物: commit 708148c2 の README は「pretrained ICEBERG 2.1 model weights for the MassSpecGym workflows: 1. `msg_all` 2. `msg_simulation`」として別々のリンクを出し、`msg_all` のリンクは "open-source MassSpecGym-trained weights for ICEBERG" と同じフォルダ（フォルダ名 `msg_all`）。中身は `gen/` と `inten_contr/`。**確認**（出典 1、3）。NIST 版は「ライセンス保持者がメールで申請」で、公開リンクには無い。**確認**（出典 1）
- 設定との照合: 上流の `gen/hparams.yaml` の値（hidden 512、layers 6、lr 0.000996、lr_decay 0.7214、dropout 0.2、GGNN、pe_embed_k 14、pool avg、set_layers 0、weight_decay 0 ほか）は `configs/iceberg/msg_all/dag_train_msg_all_iceberg.yaml` と一致する。`inten_contr/hparams.yaml`（hidden 256、gnn_layers 4、lr_decay 0.825、weight_decay 1e-7、frag_set_layers 3、contr_weight 1.0、ppm_tol 20 ほか）は `dag_inten_contr_finetune_msg_all_iceberg.yaml` と一致するが、学習率だけ 0.000736 で、これは対照学習の設定（0.0005）ではなく `dag_inten_train_msg_all_iceberg.yaml` の値。**確認**。ただし `msg_all`・`msg_known_ce`・`msg_simulation` の yaml は学習のハイパーパラメータが同じで（データセット名と保存先だけ違う）、hparams.yaml では msg_all 版かどうかを**区別できない**。msg_all 版である根拠は上流のフォルダ名と作者の申告だけ
- Kaggle のファイルが上流のこのフォルダの best.ckpt の保存し直しであること: **未確認**（(v) を参照）。上流の 2 つの best.ckpt は「9 days ago」（11/3 頃）に更新されていて、Kaggle 版（9/26）より新しい。11/3 の commit 8dfe2d3 は「metadata-only checkpoint migration tool」（`annotate_checkpoint_adducts.py`: 元のファイルの sha256 を記録し、全テンソルが変わらないことを確かめて対応付加イオン・装置の宣言を足す）を入れているので、テンソルは変えずに宣言を足した版に差し替えた可能性が高い（推測）。その場合ファイルの sha256 は変わる

### (iv) 重み・コードのライセンス

- ms-pred のコード: **MIT**（`LICENSE`「Copyright (c) 2023 Samuel Goldman」、GitHub API の `license.spdx_id = MIT`）。**確認**（出典 1）。Kaggle の `src/ms_pred/LICENSE`（1,092 バイト）は上流（1,071 バイト）と改行コードを除いて同じ文面。**確認**（出典 4）
- 上流の重み: Dropbox のフォルダには best.ckpt と hparams.yaml しか無く、重みだけのライセンス表記は無い。README は「open-source MassSpecGym-trained weights」とだけ書く。重みに MIT が及ぶかは書かれていない。**未確認**（作者の LICENSE-NOTICE は「The download carries no separate license file; used under the ms-pred repository's MIT License」＝**主張**）
- Kaggle データセットのライセンス欄: 2 つとも `other`。**確認**（出典 4）
- 推論の包み（`gl_runner.py`・`ice_runner.py`・`gl_fuse.py`・`fuse.py`）: ahmedberatozer の作。LICENSE-NOTICE は第三者部品（ms-pred、wheel）と `shim/`（「own code」）について書くが、これらの包みのライセンスは書いていない。**未確認**
- 学習データ MassSpecGym: **MIT**（HF の `cardData.license`、#66 で確認済み、出典 10）
- 同梱の wheel: rdkit 2025.3.6（BSD-3-Clause）、pytorch_lightning・torchmetrics・lightning_utilities（Apache-2.0）、einops（MIT）。**主張**（LICENSE-NOTICE。各 wheel の中のライセンスは読んでいない）
- GLACIER の論文自体は CC BY-NC-ND 4.0（論文の本文のライセンスで、コード・重みのものではない）。**確認**（出典 7）

### (v) こちらの Kaggle 入力の sha256 と上流の配布物の照合

- Kaggle 側の値（`MANIFEST.json`、**確認**）: `ckpt/glacier.pt` 60,531,887 バイト sha256 `9192e4c2eeafe83317d5a2c2641e1fc4d68934488757a4faafeb66f2ab397a12`（`ckpt_tensors` 264、`ckpt_params` 15,108,154、元の best.ckpt の sha256 として `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11` を記録）。`ckpt/gen.pt` 14,561,244 バイト `deef0451c1870d2125653bb212c9d95c5b5d9bf80156dfa9b22fedda7db49bea`、`ckpt/inten.pt` 13,793,037 バイト `c49e330e049d3a8b0095026ed694fcf70045c24305354853732aaa5f89d55068`
- 0.487 版の入力（v4g 写し、出典 5、**確認**）: 最上位の `MANIFEST.json` に ICEBERG の gen.pt・inten.pt は上と同じ sha256 で載る。GLACIER の重みの行は無い。v4g の中の `casmi26-glacier/MANIFEST.json` は ahmedberatozer 版と同じ sha256（53bf3f87…）で、`ckpt/glacier.pt` も同じバイト数（60,531,887）
- 0.487 版のノートブック（出典 6、**確認**）: F セル 41 は `N1_GL_RUNNER_SHA256 = '5c4f7431…'`（= `gl_runner.py` の MANIFEST の値）を持つが、添付の `gl_runner.py` の sha256 を計算してこの値と一致するかを print するだけで、assert はしない（`N1_CORE_SHA256`・`N1_DRIVER_SHA256` も同じ print の中）。F セル 26 の MANIFEST の assert は v4b の模型のファイルが対象で、ICEBERG・GLACIER は入らない。重み（glacier.pt・gen.pt・inten.pt）の sha256 を見る箇所は無い（凍結ノートブック 59 セルと包みのノートブック 31 セルを `9192e4c2`・`deef0451`・`c49e330e`・`5a47cecc`・`glacier.pt`・`gen.pt`・`inten.pt` で検索して 0 件）。部品表（`lineage-reproducibility.md` 55 行）の「F セル 41 で `N1_GL_RUNNER_SHA256` などランナーを固定」は、正しくは「照合して表示するだけ」
- 上流との照合の可否: **ファイルの sha256 では照合できない**。Kaggle の .pt は {hyper_parameters, state_dict} だけに保存し直したもので、上流の best.ckpt（optimizer などを含む。ICEBERG 39.64 MB + 38.72 MB、GLACIER 173.19 MB）とはバイトが違って当然。さらに上流の ICEBERG は 11/3 頃に差し替わっている。上流は sha256 を公開していない（Dropbox の一覧にハッシュは出ない）。照合できるのは **state_dict のテンソル単位**で、上流の best.ckpt と Kaggle の .pt を両方落として `torch.load` し、キーとテンソルが完全一致するかを見る方法（上流の新しい ICEBERG が 8dfe2d3 の移行ツールで作られていれば、ファイル内の `training_metadata.provenance.source_checkpoint_sha256` に元のファイルの sha256 も残っているはず）。GLACIER の旧リンクは今も開けるので、`5a47cecc…` との照合もそのファイルを落とせばできる。必要なダウンロードは上流 約 251 MB＋Kaggle 約 89 MB で、今回の上限（50 MB、重みは落とさない）を超えるので**実行していない（未確認）**

## 主催者回答への当てはめ

- 742991（David Healey、09-25）: 「we consider MassSpecGym and models trained on it and released with permissive licenses as fair game」。避けたいのは NIST のように重みの配布を制限するデータ。→ MSG 版の ICEBERG・GLACIER は「MSG で学習して公開された模型」に当たる。ただし「permissive licenses」の部分は、重み自体のライセンス表記が無い（(iv)）ので、ms-pred の MIT が及ぶという読みに頼る
- 744338（David Healey、09-29）: 質問「coleygroup/ms-pred の open-source MassSpecGym-trained GLACIER weights は使えるか」に「Yes, MassSpecGym is allowed, so is GLACIER」。→ GLACIER の MSG 版は名指しで可
- 745841（David Healey、10-05）: 「GLACIER/ICEBERG, DreaMS and ChEBI/LIPID MAPS are all fine」。質問者は「MassSpecGym-trained ICEBERG / GLACIER checkpoints from ms-pred (sha256 5a47cecca707d3ab…)」と sha256 まで書いている（質問者の主張）
- 748882（Marie Killian、10-13、A の 1）: 「For the approved MassSpecGym, GLACIER/ICEBERG, and DreaMS checkpoints, it is sufficient to deliver the released checkpoint, a reference to the authors' original training code and permitted training data, and your own complete fine-tuning and inference code … You do not need to re-run the pretraining.」
  - released checkpoint: Kaggle 版は保存し直しで、上流との一致は未確認（(v)）。GLACIER の旧リンクは今の README から外れている（commit 708148c2 の README には残る）→ **未充足**（照合すれば満たせる見込み）
  - 著者の元の学習コード: ms-pred commit 708148c2 の train_joint.py / msg 設定、msg_all の run_all.sh と設定 → **充足（確認）**。ただし学習データを作る一部（GLACIER の msg の split.tsv・MAGMa 出力、msg_all の元データ）は当時のリポジトリに無い
  - 許可された学習データ: MassSpecGym（MIT、757307 でも可）→ **充足（確認）**
  - 自前の推論コード: 今の推論の包みは参加者のもので、ライセンス表記が無い → **未充足**（748882 の 10-14 の 4 番「code shared by other competitors … allowed, provided they … comply with applicable licensing requirements」の条件を確かめられない）
- 748882（Marie Killian、10-14）: 「Third party weights that cannot be reproduced with the original training code and permitted training data are not allowed」。→ 学習コードと MSG はあるので「再現できない重み」には当たらないと読める。ただし配布チェックポイントの学習データの範囲（119,029 行か 231,104 行か）が未確認なので、厳密な再現の手順はまだ書けない
- 757307（David Healey、11-11）: 「MassSpecGym and MIST models are fine」。→ MSG を学習データに使うこと自体は可（MSG の作成で NIST を掃除にだけ使った点も含めた質問への答え）
- 755373（Zhenyu Zhang の投稿とコメント 5 件）: NIST の制限の実効性についての懸念で、ICEBERG・GLACIER の重みにも ms-pred にも触れていない（本文とコメントを読んで確認）。→ 当てはまる文は無い。NIST の禁止（投稿者の言い換え「prohibits NIST and its derivatives, including trained model weights」）に照らしても、使っているのは公開の MSG 版で、NIST 版（申請配布）ではない（(iii)）

→ 判断: **規則上は残せる見込み。ただし 748882 A1 の「released checkpoint」と「自前の推論コード」の 2 点が未充足で、最終に入れる前に片付ける必要がある**（下の「続き」）。

## v3 で測る案（書くだけ、実行しない）

- 形: 縮小版 `scripts/local_eval/run.py --stages score,prior,frag` の出力リストの上に、「同じ組成式の候補の並べ替え」の段を 1 つ足す。中身は 0.487 系統の F セル 41 と同じ: 上位 60 位（`TOPN = 60`、F セル 26）の中の同じ組成式のまとまりごとに `z(score) + ICE_LAM·z(ICEBERG) + GL_LAM·z(GLACIER)` で並べ替え、元の席に戻す（`gl_fuse.rerank_multi`、slot-preserving）。定数は 0.487 系統のまま `ICE_LAM = GL_LAM = 1.0`（F セル 26・41、確認）。**定数は動かさない**（ハイパラ調整の禁止）。案は 1 つだけで、ICE だけ・GL だけなどの別案は立てない
- 点の定義は Kaggle のランナーと同じ（`ice_ex_mean`／`gl_ex_mean`: 被覆されたスペクトル〈正イオンで [M+H]+ か [M+Na]+〉ごとの予測と問い合わせのエントロピー類似度の平均。装置の印は Orbitrap 系なら `Orbitrap`、それ以外〈timsTOF を含む〉は `QTOF`）
- リークの規則（#66 と同じものを先に書く）: 学習側 = MassSpecGym の train ∪ val（1.0 と 1.5 の和集合）。各 v3 集合から学習側と重なる正解構造を除いた部分集合で測る。残りが 100 分子未満の集合は「測定不能」。理由: GLACIER も ICEBERG も val でチェックポイントを選ぶので val は学習側。配布チェックポイントの学習範囲（GLACIER が 119,029 行の部分集合だけか）は未確認なので、MSG 全体の fold で数える
- 残りの数（`outputs/research/msg_overlap.json` の `remaining_rule` と `remaining_conservative`、確認）:
  - train ∪ val の規則: **v3_c1 199**（重なり 1）、**v3_c2 200**（0）、**enveda_v1 202**（0）、**v3_c2n 94**（重なり 106、測定不能）
  - 保守的な数え方（MSG 全行を学習側とみなす）: v3_c1 199、v3_c2 200、enveda_v1 202、**v3_c2n 83**（重なり 117）
- 採否の規則（設計 §5、#48 と同じ）: (答えの出所 × 参照の有無) の 4 集合 v3_c1・v3_c2・enveda_v1・v3_c2n のすべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、かつ 1 つ以上で下限 > 0。1 集合でも欠ければ不採用
- 帰結: **v3_c2n が 94（保守的に 83）で測定不能なので、今の v3 では規則上この段を採用できない**。進めるには、MSG と重ならない NP の集合を v3_c2n と同じ作り方で作り直すことが先に要る（別 Issue で、作り方を事前に決めてから）。#66（MIST）と同じ結論
- 参考（規則は変えない）: commit 708148c2 の `msg/labels.tsv`（119,029 行）の構造だけを学習側とみなすと、v3_c2n の重なりは train 46・val 4・test 2 で、train ∪ val の残りは 150 になる（こちらの集計）。しかし (1) 配布 GLACIER がこの部分集合で学習されたかは未確認、(2) ICEBERG `msg_all` は MSG 全体を使う、(3) 規則は数を見る前に決めたもの、の 3 点から、この数で規則を緩めない。GLACIER だけの段を測るなら、別の Issue でこの数を見たことを書いたうえで規則を事前登録し直す必要がある
- 費用の見込み（未測定）: 重みのダウンロード（Kaggle 版なら約 89 MB）、ms-pred の推論環境と RDKit 2025.3.6 の固定（ランナーは他の版を拒む）、CPU 推論の時間（上流 README は CPU 推論も可と書く。Kaggle の README の T4 の目安は 400 分子・上位 60 で 25〜30 分）。Kaggle GPU で回すなら 1 本。今日は実行しない

## 補足の発見

- 対応範囲の宣言（commit 8dfe2d3、確認）: 上流は今、学習 yaml ごとに対応する付加イオンと装置を宣言し、宣言外の入力では推論でエラーを出す。GLACIER の MSG 版は `supported-adducts: [[M+H]+]`、ICEBERG `msg_all` は `[[M+H]+, [M+Na]+]`、どちらも装置は `[Orbitrap, QTOF]`。一方 Kaggle の `gl_runner.py` は [M+Na]+ も被覆として GLACIER に渡す（52 行目 `COV = ('[M+H]+', '[M+Na]+')`）。Kaggle のコードは宣言の前の commit 708148c2 なのでエラーにならない。GLACIER の [M+Na]+ の予測は、上流の宣言の外で使っていることになる（部品表の「GLACIER は [M+H]+ だけ」は F セル 40 の説明文の記述で、ランナーのコードとは違う）
- `ms_pred` の推論部分のコード（joint_model.py ほか 5 ファイル）は commit 708148c2 と改行コードを除いて一致した（確認）。改変は shim（DGL・torch_scatter・LinSATNet の代替、作者の自作）とランナー側にある
- 上流の GLACIER の公開版は 11/3 に `glacier_msg_simulation` へ差し替わった。Kaggle 版（旧リンク）を最終に使うなら、参照先は commit 708148c2 の README と旧リンクになる。旧リンクがいつまで開けるかは分からない

## 続き（このカードでは実行しない）

1. 重みの照合（要: 約 340 MB のダウンロード、オーナーの了承）: 上流の旧 GLACIER best.ckpt と ICEBERG msg_all の gen・inten_contr best.ckpt、Kaggle の glacier.pt・gen.pt・inten.pt を落とし、sha256（GLACIER は 5a47cecc… と）と state_dict のキー・テンソルの完全一致を確かめるスクリプトを Issue→PR で入れる
2. 推論の包みのライセンス: ahmedberatozer に問い合わせるか、ms-pred の推論関数（MIT）を直接呼ぶ自前のランナーを書いて置き換える（最終に残す場合）
3. MSG と重ならない NP の集合（v3_c2n の作り直し）: 作り方を事前に決める Issue。これが無い限り、ICEBERG・GLACIER・MIST の段は v3 で採否を決められない
