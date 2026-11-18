# 最終 2 本の候補の部品ごとの再現性の棚卸し（11/16〜17 の主催者の回答に照らす）: 調査メモ（2026-11-19）

関連: #156、親 #3、元の表 `docs/research/lineage-reproducibility.md`（#44。以下「元の表」）、#82・#88・#152・#74・#77・#127

この日は**測っていない**。採点もしていないので、v3 の対照（v3_c1 0.8535 / v3_c2 0.0000）は変わらない。読んだものは、主催者と参加者のディスカッション、データセットのファイル一覧と小さいファイル、`ahmedberatozer/casmi26-training-code`（以下 TC）のコード 6 本、リポジトリの Issue と PR、ローカルの `fp_bits.npy` の形と値の範囲だけ。ビットの再計算はしていない（train.parquet を読むため）。

「確認」は、こちらで実物（コード・ファイル一覧・ハッシュ・投稿の本文）を読んだもの。「主張」は、カードや投稿に書かれているだけで、こちらで再現していないもの。

## 0. 対象の 2 本
- **(A) 0.487 写し**: 提出 58208946、Public 0.487（`kaggle competitions submissions` で 11/19 に確認）。ノートは `kernels/lb487/casmi26-lb487-copy.ipynb`（`origin/feat/77-lb487-owntables`・`origin/feat/127-align-0487-submit` にある 31 セル）。中の凍結コード（セル 12）は、0.479 版のセル 10 と 59 セルすべて一致する（元の表 §2 の 756893 の項）。だから元の表の行 1〜19 は (A) にもそのまま当てはまる。0.487 で足されたのは行 20〜22（v80・ExtLib・MassBank harmonized）。行 23（LOTUS）は添付されていない
  - `kernel-metadata.json`（確認）: `dataset_sources` は 10 件（samartalwar MassBank harmonized、sanpier/casmi-spectrum-model-v1、dmitriigluzdov の fold-safe-fpnet・natural-product-knowledge-table・natural-product-panel-benchmark・pubchem-popularity-prior・attributed-v4g-reproduction-assets、takumuhata/casmi26-extlib、sanpier/catalogue-experiment-saner、metric/rdkit-2026-3-3-wheel）。`kernel_sources` は `popwenman/casmi26-scorer-train-v80`、`model_sources` は空
- **(B) #127 版**: 提出 58492462、Public 0.480。(A) の外側セル 15 の `neural_score` の 1 行だけを変え、セル 5 の後に整列チャネルのセルを 1 つ足したもの（PR #135）
  - (A) との差（確認、`kernels/lb487_align/kernel-metadata.json`）: `dataset_sources` に `ahmedberatozer/casmi26-dreams-ssl-readiness-v1` を足し、`model_sources: ["taichiiiii/casmi26-align0487/PyTorch/foldsafe-v1/1"]` を足した。ほかは同じ
  - だから (B) の判定は「(A) のすべての行」＋「§5 の表 B の 4 行」になる。(A) に残る問題は (B) にもそのまま残る

## 1. 出典（すべて 2026-11-19 01:5x〜02:3x JST に取得）
1. ディスカッション。`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/<id> --format json` で取得。URL は `https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/<id>`
   - 748882、754610、756893、758233、759916、757307、757309、754292、745072、750330
   - 750330 は、取得した JSON のコメント 2 件の本文がどちらも空だった。本文は読めていない
2. TC（https://www.kaggle.com/datasets/ahmedberatozer/casmi26-training-code）。metadata の licenses は `other`、`LICENSE` は MIT（"Copyright (c) 2026 Ahmed Berat Özer"）。`kaggle datasets download -f` で 1 本ずつ取った。ファイルの sha256:
   - `README.md` 1c93dab9…0249
   - `LICENSE` 6f05399a…ddb7
   - `casmi/structs.py` 2c70df89…334a
   - `casmi/chem.py` 99474f23…69f1
   - `scripts/04_make_split.py` 634f4382…2e45
   - `scripts/06_train_fpnet.py` 20ae6ca4…3057
3. データセットのファイル一覧（`kaggle datasets files <id> --page-size 200 --csv`）と小さいファイル。対象は `dmitriigluzdov/casmi26-natural-product-knowledge-table`（106 ファイル、次のページ無し。`build_np.py` と `external_reference_spectra/INTEGRATION.json` を読んだ）、`hengck23/hengck23-dreams-enveda-casmi26`（9 ファイル）、`franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`（`MANIFEST.json`）、`sergiomalv/casmi-chebi-lmsd-structures`（`ATTRIBUTION.md`）、`prvsiyan/coconut-casmi26-candidates`（4 ファイル）、`megayak/casmi26-simulated-ranker-rows`
4. リポジトリ: 元の表（main abcc00c）、Issue #82・#88・#152・#74・#77・#127 の本文とコメント、PR #110・#128・#135 の本文、`kernels/lb487*/kernel-metadata.json`、`~/.hermes/profiles/massspec/workspace/casmi_daily/ref/frozen465.ipynb`（凍結コードの写し。`PANEL`・`TABLE` を使う場所を grep）
5. ローカル: `data/ext/v4g/casmi26-v2-pool/fp_bits.npy`（形・dtype・値の範囲・sha256 だけ）、`outputs/align_train/`（`train_log.jsonl` の行数と最後の行、`model_best.pt` の sha256）
6. GitHub のリポジトリの公開範囲: `gh repo view --json visibility` は `PUBLIC`（11/19 に確認）

## 2. 主催者の回答（判定に使うもの。原文の抜き書き）
略号は表の「当たる回答」の列で使う。日付・時刻はどれも JSON に入っている UTC。

- **R1** 748882 Marie Killian 2026-10-13: 「For the approved MassSpecGym, GLACIER/ICEBERG, and DreaMS checkpoints, it is sufficient to deliver the released checkpoint, a reference to the authors' original training code and permitted training data, and your own complete fine-tuning and inference code」「existing third-party weights may be used if the original training code and permitted training data needed to reproduce them are provided」。PubChem の CID-PMID・CID-Patent・CID-SID の件数は順位づけの特徴に使ってよい。LOTUS と NPAtlas は構造の出どころとして使ってよい
- **R2** 748882 Marie Killian 2026-10-14: 「Third party weights that cannot be reproduced with the original training code and permitted training data are not allowed」。微調整した重みは、元のコードと許可データが無ければ使えない。公開データセットの中のコードは使ってよい。745072（10-14）も同じ趣旨
- **R3** 748882 Marie Killian 2026-10-21: 「If the original training code and permissible training data is available for full reproducibility by the end of the competition, yes, those weights would be allowed」
- **R4** 748882 Marie Killian 2026-10-27（LeeDongHyuk 10-24 の質問への答え）
  - Q2「作者が学習・生成コードを公開していれば、チームが再実行せず、ビット一致しなくても十分か」→「Yes, that is sufficient」
  - Q3「候補表にも生成コードが要るか」→「If the candidate tables are modified in any way from the original download, you should provide code to document the steps taken」
- **R5** 754610 David Healey 2026-11-05: 「Officially released DreaMS checkpoints are fine to use」。最終提出は「reproducible end-to-end from the competition data and openly documented resources」でなければならない
- **R6** 754292 David Healey 2026-11-05: 「If a solution is using an intermediate or model that isn't reproducible it would be in violation」。756893 David Healey 2026-11-11: 「Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible」
- **R7** 756893 David Healey 2026-11-16: 「public weights are fine to load only if the code and permitted data that produced them are available, so the weights can be regenerated. A statement that they were trained on train.parquet isn't enough without the training code available」
- **R8** ビットの添字の表
  - 754610 Marie Killian 2026-11-17 22:49Z: 「A description of how the bit-index was chosen should be provided. That along with the list itself is sufficient」
  - 758233 Marie Killian 2026-11-17 18:26Z: 「There needs to be justification for the choice of indices」
- **R9** 754610 Marie Killian 2026-11-17 22:53Z: 「Yes, the official DreaMS ssl_model.ckpt is included in the officially released DreaMS checkpoints」
- **R10** 759916 Marie Killian 2026-11-17: 「COCONUT is allowed」
- **R11** 757307 David Healey 2026-11-11: 「MassSpecGym and MIST models are fine」。同 2026-11-16: 「GNPS-NIST14-MATCHES in GNPS is fine to use as」（取得した本文はここで切れている。原文のまま）
- **R12** 757309 David Healey 2026-11-11: PubChem の CID-SMILES と CC0 の COCONUT から作った候補索引を private 入力として付けることに「Those are fine to use」
- **R13** 745072 David Healey 2026-10-09: 「train.parquet should be considered open source」。PubChem と COCONUT も使ってよい
- **R14** 745072 Marie Killian 2026-10-27 21:20Z（id 3545813）: Truthseeking（10-21）が `ahmedberatozer/casmi26-fpnet-full1` と `prvsiyan/casmi26-ranker-features` を名指しして「元の学習コードが無いが使えるか、無理なら自前の公開コードと許可データで同等のモデルを学習し直せばよいか」と聞いたことへの答え。「Without the original training code, these components are not eligible. But if you retrain equivalent models with your own published code and permitted data, that would satisfy the rule」
  - 質問の時点で質問者は full1 の学習コードが無いと書いている。TC に `06_train_fpnet.py` があることには触れていない（TC のファイルの creationDate は 2026-10-19 08:25 で、質問より前。`kaggle datasets files` で確認）。だから、この答えが TC の公開の後の full1 にもそのまま当たるかは決まらない（Q-Y (1) の括弧書きが同じ点を問うている）
- **R15** 745072 Marie Killian 2026-10-27 21:23Z（id 3545817）: Truthseeking の 10-25 の質問（id 3544619、`casmi26-v3-models` の v3 の重みについて）への答えで、full1 への答えではない。後半は一般の原則として使う。「The winner's package would need to include the training script that reproduces the v3 weights. As we have stated, for any weights used, the original training code and access to permitted training data must be available」

**まだ答えの無い質問**（11/19 02:1x JST の取得時点。主催者の返答が見当たらない。なお 748882 には、本文も投稿者名も空のコメントが 3 件ある〈id 3538940・3550345・3548672〉。削除されたものとみられ、主催者の返答だったかどうかは分からない）
- **Q-L** 756893 LeeDongHyuk 2026-11-17: 次の (a)(b)(c) のとき、作者の元の重みを読んでよいか、自分で学習し直す必要があるか
  - (a) seed が記録されていない
  - (b) 起動コマンドが記録された引数から作り直したもの、または補助スクリプトが学習の後に書かれた
  - (c) 中間の入力ファイルの版がもう落とせないが、公開コードで同等のものを作れる
- **Q-A** 748882 Ashot Mirzoyan 2026-11-05: 学習の後に編集されたスクリプトで作った v4b の重みを、そのまま使ってよいか
- **Q-Y** 748882 yesterdays_rebel 2026-11-05: 3 つの場合を問う
  - (1) TC の重みと表をそのまま使ってよいか（括弧書きで「745072 の以前の返答は full1 に学習コードが無いとしていたが、TC には `06_train_fpnet.py` がある」と指摘している）
  - (2) fold-safe FPNet は同じ扱いか
  - (3) fe_D の元の DreaMS ckpt の config に NIST20+MoNA の probing 集合が書かれている点は問題か

## 3. 判定の決め方
- **可（コード＋許可データ）**: 生成コード（学習・生成のスクリプト）と許可データの両方が公開されていて、seed の欠落・コマンドの再構成・学習後の編集・入力の版の欠落のような実質の欠けが知られていない。R4 Q2 によりビット一致は要らない。承認済みのチェックポイント（R1・R5・R9・R11）もここに入れる
  - 但し書き: 実質の欠けを挙げずに「この部品は同じ扱いでよいか」を確かめるだけの未回答の質問（Q-Y (1) のうち欠けの無い部品、Q-Y (2)）は、判定を止めない。その場合は「当たる回答」の列にその質問を書き、印を付ける
- **未決（コードあり、Q-L 等に当たる）**: コードと許可データはあるが、seed の欠落、コマンドの再構成、学習後の編集、入力の版の欠落のどれかがある。R7 の原則には合う見込みが高い。ただし Q-L・Q-A・Q-Y に答えが無いので決められない
- **説明あり（コードは見つからない）**: 手順の説明はあるが、生成コードが見つからない。R4 Q3 と R6 により、候補表や中間物では足りない見込み（推測）。例外はビットの添字の表で、R8 により説明と表そのもので足りる
- **不可（生成コード無し・揃わない）**: 主催者の回答（R6）にそのまま当たる。生成コードが見つからない、または作者自身が「揃わない」と書いている
- **影響なし**: 提出の結果に使われない（表示だけ、または読まれない）
- **未確認**: このメモでは調べられていない

判定は「その部品を**今のまま**最終提出に入れてよいか」についてのもの。リーク無しで v3 で測れるかは元の表の列のとおりで、ここでは扱わない。

## 4. 表 A: (A) 0.487 写しの部品（(B) にも当てはまる）
行番号は元の表に合わせた。3 は A と B に分け、S1〜S6 は元の表の補足にあった部品と、今回新しく見つけた部品。

| # | 部品 | 生成コードの所在（出典 URL） | ライセンス | 当たる回答 | 判定 | 未確認の点 |
|---|---|---|---|---|---|---|
| 1 | Engine 1 ランカー `ranker_v4b.pkl` | TC `fe_v4/simulate_v1ctx.py` → `build_feats.py` → `train_v4_ranker.py`（https://www.kaggle.com/datasets/ahmedberatozer/casmi26-training-code 、README §2.3） | TC は MIT。データセット `ahmedberatozer/casmi26-v4b-models` は other | R4 Q2、R7、Q-L (b)、Q-A、Q-Y (1) | **未決**。README §5 によると、起動コマンドは再構成で、ブースターの学習ログが無い（記録は `ranker_v4b.json` だけ）。`simulate_v1ctx.py`・`ctxlib.py` は学習の後に編集されている。特徴量を作る FPNet B（3B）と fe_D（4）も未決なので、ここに連なる | Q-L・Q-A の答え |
| 2 | `fpnet_full1.pt` | TC `scripts/06_train_fpnet.py`（README §2.1 の「full1 (reconstructed)」の行 `--steps 40000 --holdout "" --val_every 40000`） | MIT / other | R7、R14、R15、Q-L (a)(b)、Q-Y (1) | **未決**。README §5: seed と正確な `--holdout` の文字列が記録されておらず、コマンドは再構成。Q-L の (a)(b) そのもの | Q-L の答え |
| 3A | FPNet A（`fpnet_0.pt` = `fe_A.pt`） | TC `06_train_fpnet.py`。README §2.1:「A (command from the script docstring; … seed 7). Best step 38,000」。ログは `logs/log_train_a.txt`（README §1 の表） | MIT / other | R4 Q2、R7、Q-Y (1)（確認の質問だけ） | **可**。seed 7 が記録され、コマンドはスクリプトの docstring にある。ビット一致はしないが、R4 Q2 でそれは足りる | ログの中身は読んでいない。Q-Y (1) の答え |
| 3B | FPNet B（`fpnet_1.pt` = `fe_B.pt`） | 同上。README §2.1:「B: same recipe, different seed (not recorded …)」 | MIT / other | R7、Q-L (a)、Q-Y (1) | **未決**（seed が記録されていない） | Q-L の答え |
| 4 | `fe_D.pt`（DreaMS の微調整） | TC `dreams_ft/review_fixes/train_dreams_fp.py` → `extract_step20k_ema.py`（README §2.2、seed 11） | MIT。DreaMS の port も MIT | R1、R2、R9、Q-L (b)、Q-Y (1)(3) | **未決**。README §5: 学習のコマンドは再構成。重みの取り出しは対話的に行い、`extract_step20k_ema.py` は後から書いたもので再実行していない。土台の DreaMS SSL は R9 で可だが、NIST の probing の点（Q-Y (3)）は未回答 | Q-L・Q-Y の答え |
| 5 | fe_v4 の表（`prior_tables.pkl`、`fragnet/stageA.npz`・`stageB.txt`、`analog_struct/*`、`frag2/*`） | TC `class3_v2/priors_build.py`、`fe_v4/res_fragnet.py` ほか（README §2.4〜2.6） | MIT / other | R4 Q2、R6、Q-L (b)、Q-A、Q-Y (1) | **未決**。README §5: `priors.py`・`res_fragnet.py`・`fams/fragnet.py`・`res_analog_struct.py` は出力の後に編集されている。fragnet の `sets`・`fitA` にはログが無い。`prior_tables.pkl` には作った時刻が入っている | Q-A の答え |
| 6 | v2 プール（`pool_meta`・`pool_fp`・`pool_frag_*`・`train_structs`・`train_fp_sel`） | TC `scripts/01_build_train_structs.py`・`02_build_coconut.py`・`05_build_pool.py`（README §3） | MIT / other。出どころは train.parquet と COCONUT | R4 Q3、R10、R12、R13、Q-Y (1)（確認の質問だけ） | **可**。入力は train.parquet と COCONUT（R10・R13）で、生成コードがある（R4 Q3 の「手順のコード」に当たる） | 再生成してバイト一致するか（README §3 は「学習に使ったファイルと同じ」と主張）。README の一覧に無い `pool_frag_*`・`train_fp_sel` をどのスクリプトが作るか（TC には `05b_build_frags.py` もあるが読んでいない） |
| 7 | `fp_bits.npy`（10,226 ビット） | TC `scripts/04_make_split.py` の末尾（46〜48 行目）→ `casmi/structs.select_bits`（§6 で確認）。README §3 は `fp_bits.npy` を 01/02/05 が作ると書くが、実際に書き出すのは 04 | MIT | R8、Q-Y (1)（確認の質問だけ） | **可**。選び方のコードと説明があり、表そのものも公開されている。R8 の「説明＋表」を超えて満たす | ビットの再計算（Vitalii の「ビット単位で再現」は主張のまま） |
| 8 | `rank_train.npz`（prvsiyan） | 一部だけ。cubres の回収した `make_ranker3.py` ほか（https://github.com/cubres/enveda-casmi-2026 ）、作者版 `prvsiyan/casmi26-generation-code`。作者自身が、FP ロジットのキャッシュの系統は「unresolved」で「not a full reproduction」と書く（元の表の行 8） | CC-BY-NC-SA-4.0 | R6（754292 はこの表を名指しした質問への答え） | **不可**（生成元が揃わない） | — |
| 9 | `sim_rank_rows_nofp.npz`（megayak） | **見つからない**。データセットは README・npz 3 個・pt 2 個の 6 ファイルだけ（https://www.kaggle.com/datasets/megayak/casmi26-simulated-ranker-rows 、11/19 にファイル一覧で再確認） | CC0-1.0 | R6（754292・756893 の名指し） | **不可**（生成コードが無い） | — |
| 10 | prvsiyan FP モデル（`fp_single_s2.pt`、`fp_merged_m1.pt`） | cubres に回収された `train_fp3.py`/`train_fp4.py` と起動引数。カードは「学習入力の版が固定されていない、seed は初期化の後に設定、公開重みとのバイト一致は未確立」 | CC-BY-NC-SA-4.0 | R7、Q-L (a)(c) | **未決**（学習入力の版が欠けている＝(c)、seed の扱い＝(a)） | Q-L の答え。#152 のローカルでの作り直しは関門 (b) で止まっている |
| 11a | Engine 2 の COCONUT バンク（`prvsiyan/coconut-casmi26-candidates` の `coco_fp`・`coco_mass`・`coco_meta`） | 説明だけ。ファイルは npy・pkl の 4 個で、README も .py も無い。データセットのカード（`kaggle datasets metadata` の description）の「How it was produced」に、出どころ（COCONUT 2.0）、10,407 列の指紋の並び（Morgan 半径 2・3 各 4,096、RDKit 経路 2,048 maxPath 6、MACCS 167）、`fp_bits.npy` の sha256 a61e372f… がある。カード自身が「The download date and generation runtime are not recorded」と書く（https://www.kaggle.com/datasets/prvsiyan/coconut-casmi26-candidates 、11/19 に確認） | CC-BY-NC-SA-4.0 | R4 Q3、R10、R12 | **説明あり**（COCONUT そのものは R10 で可。ただし R4 Q3 により、指紋を付けて変えた表には手順のコードが要る。それが見つからない。COCONUT のダウンロード日も記録が無い） | 作者の他のデータセット・カーネルに生成コードがあるか |
| 11b | ChEBI/LMSD バンク（`sergiomalv/casmi-chebi-lmsd-structures`） | 説明だけ。`ATTRIBUTION.md` に出どころ（ChEBI release 255、LMSD 2026-10-03）、標準化の手順、COCONUT との重複を除いたことが書かれている。コードは見つからない（ファイル 6 個に .py が無い） | CC BY 4.0 | R4 Q3 | **説明あり** | 作者の他の公開物にコードがあるか |
| 11c | PubChemLite の指紋（`franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`） | 説明だけ。`MANIFEST.json` に出どころ（`cosmicbe/casmi26-pubchemlite`、CC0）、入力 csv の sha256、`fp_bits` の sha256 a61e372f…、rdkit 2026.03.3、出力 3 ファイルの sha256 がある。コードは見つからない（ファイル 4 個〈MANIFEST.json と npy・pkl〉に .py が無い） | CC0-1.0 | R4 Q3、R12 | **説明あり** | 同上 |
| 11d | Engine 2 の 6,930 ビット（`fp_bits.npy` a61e372f…） | 作者版 `prvsiyan/casmi26-generation-code` の `prepare_legacy_exports.select_fingerprints(low=.005, high=.995)`。我々の #84 `scripts/engine2/select_fp_bits.py` で再現し、`array_equal` を確かめた（元の表の行 11） | Apache-2.0（作者版） | R8 | **可**（こちらで再現済み） | — |
| 12 | MetFrag-lite `frag_rescore.py` | ソースそのもの（学習しない）。凍結セル 51 のコメントに出どころ（ahmedberatozer/casmi26-v7c-e006-fraggap v1、Huseyin Emre Aksoy） | Apache-2.0（コメントの記載） | R5 | **可** | — |
| 13 | ICEBERG の重み | ms-pred の公開 MassSpecGym `msg_all` 版を保存し直したもの（https://github.com/coleygroup/ms-pred 、`ahmedberatozer/casmi26-iceberg` の LICENSE-NOTICE） | ms-pred は MIT | R1（承認済みのチェックポイント）、R11 | **可**。推論ランナー（`ice_runner.py`・`fuse.py`）のライセンス表記は未確認 | ランナーのライセンス |
| 14 | GLACIER の重み | 同上（`ahmedberatozer/casmi26-glacier`） | MIT | R1、R11 | **可**。ランナーのライセンスは 13 と同じく未確認 | 同上 |
| 15 | PubChem 層（`pc_mass`・`pc_off`・`pc_smiles`） | TC `scripts/02c_build_pubchem.py`（元の表の行 15 で docstring を確認） | MIT / other。PubChem は NCBI の方針 | R4 Q3、R12、R13、Q-Y (1)（確認の質問だけ） | **可** | 入力の PubChem の版 |
| 16 | 文献 prior（`pc_lsid` ほか、`pool_popularity.csv`） | データセットの `scripts/`（`build_pop.py`・`build_annot.py`・`build_delta.py`・`build_sources.py`・`download_sources.py` ほか、https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior ） | other（PubChem は NCBI、LOTUS は CC BY 4.0、NPAtlas は CC BY-NC 4.0） | R1（PubChem の件数、LOTUS・NPAtlas）、R4 Q3 | **可** | — |
| 17 | fold-safe FPNet（5 本） | `train_fivefold.py`・`train_fpnet.py`・`make_split.py`・学習ログ（https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet ） | other（スクリプトは Apache-2.0） | R7、Q-Y (2)（確認の質問だけ） | **可**。コード・ログ・許可データが揃い、R7 の原則に合う。実質の欠けは知られていない。個別の質問 Q-Y (2) には答えが無いので印を付ける（§3 の但し書き） | Q-Y (2) の答え |
| 18 | sanpier のスペクトル–候補ニューラル v1 | `train.py`・`prepare.py`・`config.json`・分割の manifest・学習履歴（https://www.kaggle.com/datasets/sanpier/casmi-spectrum-model-v1 ） | apache-2.0 | R7、R13 | **可**。全 train で学習しているが、それは規則上は問題にならない | — |
| 19 | 外部参照スペクトル（`external_reference_spectra/external_delta.parquet` ほか） | 説明だけ。`INTEGRATION.json`（pin の sha256、出どころの件数 MassBank 265・MoNA 258・GNPS 228、リンクの方針）・`DOWNLOADS.csv`・`SOURCE_MANIFESTS.json`・`LICENCES.md` がある。データセットの .py は `build_np.py`（LOTUS+NPAtlas の天然物の表を作る、外部スペクトルには触れない）と `sugar_and_acyl_units/sugarloss_units.py` だけ（https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-natural-product-knowledge-table 、11/19 に確認） | other（各レコードは CC BY 4.0 / CC0 と凍結セル 58） | R4 Q3、R6 | **説明あり**（外部参照を作るコードは見つからない） | `TECHNICAL_NOTES.md`（36 KB）は読んでいない |
| 20 | v80 スコアラー（`popwenman/casmi26-scorer-train-v80` の出力） | そのカーネル自体が学習コード（https://www.kaggle.com/code/popwenman/casmi26-scorer-train-v80 、元の表の行 20） | キットは apache-2.0。カーネルのライセンスは未確認 | R7、R13 | **可**。ただし `kernel_sources` は版を固定せず最新の出力を読む。最終提出のときに版が変わっていないかを確かめる必要がある | カーネルのライセンス |
| 21 | ExtLib（`takumuhata/casmi26-extlib`） | **見つからない**（元の表の行 21。データセット 6 ファイル、作者の他のデータセット 24 件・カーネル一覧を確認済み） | CC0-1.0 | R4 Q3、R6 | **不可**（生成コード無し） | — |
| 22 | MassBank harmonized（`samartalwar/...-massbankharmonized`） | 説明だけ（metadata の「Build pipeline (reproducible)」の 6 手順）。コードは見つからない | CC-BY-NC-SA-4.0 | R4 Q3、R6 | **説明あり** | — |
| 23 | LOTUS 候補（`qwertykeypadapple/casmi26-lotus-candidates`） | 説明だけ（README の手順） | CC0-1.0 | R1（LOTUS は可）、R4 Q3 | **影響なし**（(A) には添付されておらず、段は飛ばされる） | — |
| S1 | DreaMS の空き順位の埋め（`hengck23/hengck23-dreams-enveda-casmi26`） | ファイルは 9 個（11/19 に確認）。`dreams_minimal_model.py`・`run_demo.py`・`dreams_html_results_v2.py`・`embedding_model_state_dict.pt`（386 MB）・Enveda180 と MassSpecGym の埋め込み・索引・meta。埋め込みを作るコードが中にあるか、`embedding_model_state_dict.pt` が何の重みでどう作られたかは読んでいない | MIT（元の表の補足） | R1、R2、R7 | **未確認** | 3 本の .py の中身、state_dict の出どころ |
| S2 | natural-product panel（`dmitriigluzdov/casmi26-natural-product-panel-benchmark`） | 凍結セル 5 で `PANEL` を決め、セル 16 で `panel.parquet`・`prior_rule_summary.csv` の要約を表示するだけ（try/except の中）。凍結の全コードセルを grep し、`PANEL` が出てくるのはこの 2 セルだけと確認した | other（競技データの条件） | — | **影響なし**（表示だけ） | — |
| S3 | `sanpier/catalogue-experiment-saner` | (A) の credits の表（セル 30）に「attached as in Spectral Parallax; this run does not read it」とある | 作者のページのとおり | — | **影響なし**（作者の記述。こちらでは grep していない） | ノートの中で読まれていないことの grep |
| S4 | Engine 2 の HGB ランカー 12 本（ノートの中で学習する） | 学習コードはノートの中にある（`eng_casmi_engine.py` の `fit_rankers`・`fit_pv_rankers`） | 作者のコード | R6 | コード自体は可。ただし学習データが 8・9 なので、**8・9 が不可なら一緒に不可** | — |
| S5 | lb479 の補助モジュール 6 つと `consensus.py`・`outsiders.py` | ノートにソースごと埋め込まれ、SHA256 で固定（元の表の補足） | 作者のコード | R5 | **可**（コードそのもの） | — |
| S6 | RDKit の wheel（`metric/rdkit-2026-3-3-wheel`）と ICEBERG 用の rdkit 2025.3.6 | 環境（部品ではない） | RDKit は BSD-3 | — | **影響なし**（環境） | — |

### (A) のまとめ（表 A の 33 行の数え上げ。表の判定の列からスクリプトで数えて一致を確かめた）
- 可: **13 行**（3A・6・7・11d・12・13・14・15・16・17・18・20・S5）
- 未決: **6 行**（1・2・3B・4・5・10）。どれも Engine 1 と Engine 2 の学習・生成した部品
- 説明あり: **5 行**（11a・11b・11c・19・22）
- 不可: **3 行**（8・9・21）
- 影響なし: **4 行**（23・S2・S3・S6）
- 未確認: **1 行**（S1）
- 8・9 に従う: **1 行**（S4）
- 合計 13 + 6 + 5 + 3 + 4 + 1 + 1 = 33 行

**結論: (A) は今のままでは最終提出の要件を満たさない。** 理由は、主催者が名指しで「再現できない」とした Engine 2 の 2 表（8・9）と、生成コードの無い ExtLib（21）を読むため。説明だけの 5 行（11a・11b・11c・19・22）も、R4 Q3 に照らすと手順のコードが要る見込み。未決の 6 行（Engine 1 の 1・2・3B・4・5 と、Engine 2 の 10）は Q-L・Q-A・Q-Y の答えしだい。

## 5. 表 B: (B) #127 版で足した部品
| # | 部品 | 生成コードの所在（出典 URL） | ライセンス | 当たる回答 | 判定 | 未確認の点 |
|---|---|---|---|---|---|---|
| B1 | DreaMS `ssl_model.ckpt`（`ahmedberatozer/casmi26-dreams-ssl-readiness-v1` の `weights/ssl_model.ckpt`） | 公式の重み。#127 の本文（2026-11-18 夜のカード t_de2e4175 の確認）によると、バイト列が手元の写しと同じ（1,392,710,212 B、md5 4deee91d…）、md5 が Zenodo 10997887 の値と一致（https://zenodo.org/records/10997887 、コード https://github.com/pluskal-lab/DreaMS ） | Kaggle の表示は `other`。元のライセンスは、重みが CC-BY-4.0、コードが MIT（データセットの LICENSE-NOTICE.txt、#127 本文） | R5、R9 | **可** | 他人のデータセットなので、最終提出のときにまだ公開されているか |
| B2 | 整列モデル `model_best.pt`（private Kaggle Models `taichiiiii/casmi26-align0487/PyTorch/foldsafe-v1/1`、sha256 4e2a7cbe…cd51） | 自前のコード。PR #110 の `scripts/train/align_local_embed.py`（DreaMS の埋め込み）と `scripts/train/align_train.py`（学習）。止め方は #109 で事前登録。学習ログ `outputs/align_train/train_log.jsonl`（14 行、最良は step 24,000、step 28,000 で停止）と `align_config.json` はローカルにある（git の対象外）。リポジトリは PUBLIC（https://github.com/taichiiiiiiii/molecule-id-from-mass-spectra/pull/110 ） | 自前（Kaggle Models の licenseName は MIT） | R7、R9、R13 | **可（条件つき）**。入力は 3 種類: train.parquet のスペクトル、公式 DreaMS（B1）、v2 プールの `pool_meta`・`pool_fp`・`train_structs`・`fp_bits`（sha256 で照合）。分割には fold-safe の `split.parquet`（行 17 のデータセット）と自前の `kernels/sanpier_fold0_drop/sanpier_fold0_holdout_keys.json` を使う（`align_train.py` の 373〜415 行目付近）。だから B2 の判定は行 6・7・17 に依存し、それらが可である限り可。条件は 2 つ: (i) PR #110 は未マージで、コードは公開ブランチにしか無い。(ii) 学習ログと設定は公開していない。最終に使うなら、どちらも main に入れるか、提出物に同じものを付ける | — （`align_config.json` に `SEED` 42・`VAL_SEED` 0・`VAL_CAND_SEED` 1 が記録されている。バッチの順はステップ番号と seed から決まる〈PR #110〉） |
| B3 | 整列のセルと runner（`scripts/submit/align0487_runner.py`、`scripts/ship/strip_dreams_ckpt.py`、ノートの生成 `scripts/submit/make_lb487_align.py`） | 自前のコード。PR #128・#135（未マージ、公開ブランチ）。ノートにはソースが sha256 つきで埋め込まれている | 自前 | R5 | **可**（コードそのもの。B2 と同じく main には入っていない） | — |
| B4 | 候補の h の計算（`scripts/engine2/base_fs.py` の `fs_fp` と 10,226 ビット） | main にある自前のコード。ビットは表 A の 7 | 自前 / MIT | R8 | **可** | — |

**結論: (B) で足した 4 行は、コードを main に入れるという条件つきで可。** ただし (B) は (A) の全部を含むので、(A) と同じ理由で、今のままでは要件を満たさない。

## 6. Vitalii Neibauer の `select_bits` の主張（758233、2026-11-18 06:49Z）の出典の確認
主張（原文）:「ahmedberatozer's 10,226-bit list (casmi26-v2-pool fp_bits.npy). The same frequency band over a seeded sample of 200,000 train structures, applied to a 16,551-bit layout of ECFP, FCFP, RDK, AP, TT and MACCS. Source: ahmedberatozer/casmi26-training-code (MIT), casmi/structs.select_bits」「We regenerated both lists from train.parquet with RDKit and got them bit for bit」

こちらで確かめたこと（コードを読むだけ）:
- **コードの場所**: TC `casmi/structs.py` 56〜67 行目 `def select_bits(fp_packed, lo=0.005, hi=0.995, sample=200000, seed=0)`
  - `np.random.default_rng(seed)` で `min(sample, n)` 件を重複なしに選ぶ
  - 20,000 件ずつ `unpackbits` して、ビットごとの頻度を数える
  - `(freq >= lo) & (freq <= hi)` のビットの添字を `int32` で返す
- **呼び出しと引数**: TC `scripts/04_make_split.py` の最後で次のように呼ぶ。seed は渡していないので、既定の 0 になる

  ```python
  fp = np.load('work/train_fp_raw.npy')
  bits = structs.select_bits(fp, lo=0.005, hi=0.995, sample=200000)
  np.save('work/fp_bits.npy', bits)
  ```

  - 同じスクリプトの分割の乱数は `default_rng(2026)` で、ビットの選び方の乱数（seed 0）とは別
- **16,551 ビットの並び**: TC `casmi/chem.py` 264〜266 行目 `FP_BLOCKS = [("ecfp4", 4096), ("ecfp6", 4096), ("fcfp4", 2048), ("rdk6", 2048), ("ap", 2048), ("tt", 2048), ("maccs", 167)]`、`FP_RAW_BITS = sum(...)` = 16,551
  - 生成器（274〜280 行目と、MACCS は 295 行目）は次のとおり
    - Morgan 半径 2 と 3、各 4,096
    - FCFP は Morgan 半径 2 の特徴の不変量、2,048
    - RDKit の経路 maxPath 6、2,048
    - AtomPair 2,048
    - TopologicalTorsion 2,048
    - MACCS 167
  - Vitalii の「ECFP, FCFP, RDK, AP, TT and MACCS」と合う
- **手元の `fp_bits.npy` との整合**（確認）: `data/ext/v4g/casmi26-v2-pool/fp_bits.npy` は形 (10226,)、int32、値の範囲 2〜16,549、狭義単調増加、sha256 c3869e86…8677（元の表の値と同じ）
  - ブロックごとのビット数は ecfp4 1,464、ecfp6 3,278、fcfp4 792、rdk6 2,048、ap 1,678、tt 827、maccs 139
  - 範囲が 16,551 ビットの中に収まり、`np.where(...)` の返す昇順と矛盾しない
- **確かめていないこと**: ビットの再計算（train.parquet と `01_build_train_structs.py` の `train_fp_raw.npy` が要る）。だから「ビット単位で再現できる」は**主張のまま**。入力 `work/train_fp_raw.npy` の行（標準化と重複除去の後の構造）の作り方は `01_build_train_structs.py` にあるが、このメモでは読んでいない
- **規則との関係**: R8 の「選び方の説明＋表そのもの」に対して、TC にはコードそのものがある。だから表 A の 7 は「可」とした。Vitalii の投稿は出典の場所が正しいことの裏づけであって、判定の根拠ではない

## 7. 作り直しが要る学習・生成の優先順と GPU 時間の見込み
前提: Kaggle GPU は週 30 h で、他のプロファイルと共用。今週の残りは**未確認**。時間はどれも過去の Issue に記録された見込みで、このメモでは測っていない。「未測定」は、見込みの記録が無いもの。

1. **Engine 2 の 2 表（8・9）と、それで学習する HGB（S4）**（最優先。不可が確定していて、(A) と (B) の両方に入っている）
   - 経路 (i) 自前の 2 表に替える（#74・#77）: 生成は Kaggle の **CPU**。2 回はどちらも完走していて、約 5.6 h と約 6.3 h（#74 のコメント）。GPU は 0 h。0.487 写しに入れた診断は Public 0.435（Δ −0.052、事前登録の帯で「判定なし」、#77）
   - 経路 (ii) Engine 2 を外す: 0.479 版で外した診断は Public 0.378（提出 58215746、#62）
   - どちらも LB の下げが大きいので、最終の形をどうするかは方針の判断になる（このメモでは決めない）
2. **Engine 2 の FP モデル（10）と候補バンク（11a・11b・11c）**（Engine 2 を残す場合だけ要る）
   - FP モデル: #152 でローカル MPS の作り直しを測った。見込みは 7.62 h（関門 (a) 合格）。footprint が 17.44 GB で 14 GB を超え、関門 (b) で止まった。GPU は 0 h。候補の指紋を packbits のまま持つ修正で約 4.3 GB 減る見込み（未測定）
   - 候補バンク 3 つ（COCONUT 11a、ChEBI/LMSD 11b、PubChemLite 11c）: どれもカードの説明だけでコードが無い。出どころ（COCONUT、ChEBI release 255 と LMSD、`cosmicbe/casmi26-pubchemlite`）から指紋を作り直すコードを書く（CPU、6,930 ビットは 11d で再現済み）。**未測定**
3. **ExtLib（21）、外部参照（19）、MassBank harmonized（22）**（学習ではないが、外すか作り直す）
   - GNPS・MassBank・MoNA から自前で組み直す（CPU、ダウンロードあり）か、外す。**未測定**。21 を外したときの LB の差は測られていない（元の表の行 21）
4. **Engine 1 FPNet full1（2）**（未決。Q-L が「学習し直しが要る」なら最優先に上がる）
   - #82 の経路（fold-safe キット〈dmitriigluzdov、Apache-2.0〉の `train_fpnet.py` をそのまま自前のカーネルで回し、全 train を 82,855 ステップ）: 外挿は **14.36 h**（#82 の試走から）。週 30 h の約半分
     - キットの `train_fpnet.py` には途中保存からの再開が無く、1 セッション（11 h 以下）に収まらない。2 セッションに分けるには再開が要る。#82 の推奨は、PR #142（未マージ）で MPS 版に足した途中保存と再開を CUDA で使う案。CUDA で動くかは確かめておらず、事前登録が要る
     - この経路は R14 の後半（「retrain equivalent models with your own published code and permitted data」）に当たる見込み。ただしコードは第三者の公開キット（R2 により使ってよい）で、自作ではない。当たるかどうかは解釈で、確認されていない
   - ローカル MPS は同じ経路で 93.6 h の見込みで、関門で不合格（#82）
   - TC のレシピ（`06_train_fpnet.py --steps 40000`）での時間は**未測定**。1 ステップの速さが同じなら約半分になるが、これは推測
5. **Engine 1 ranker_v4b（1）と fe_v4 の表（5）**（未決。Q-L (b)・Q-A しだい）
   - ranker の段 1（模擬クエリ 15,701 件）は T4 のセッション時間の見込みで、3 カーネル × 2 シャードなら合計 12.1 h（補正すると 8.7 h）、6 カーネル × 1 シャードなら合計 20.3 h（補正すると 14.5 h）（#88 の試走から。T4 × 2 のセッションの週の枠の数え方は未確認）
   - 段 2（特徴）・段 3（学習）は**未測定**
   - 注意: ranker の特徴は FPNet A/B と fe_D の出力から作る。3B・4 を作り直すなら、その後で ranker も作り直す必要がある（依存の鎖）
6. **FPNet B（3B）と fe_D（4）**（未決。Q-L (a)(b) しだい）
   - FPNet B は TC の full1 と同じスクリプト（`--steps 40000`）。時間は**未測定**（#82 の 14.36 h はキットの 82,855 ステップの値で、そのままは使えない）
   - fe_D（DreaMS の微調整）は**未測定**
7. 作り直しは要らない: 3A・6・7・11d・12〜18・20・S5 と、表 B の全部（B2・B3 はコードを main に入れるだけ）。影響なしの 23・S2・S3・S6 は対象外
8. **保留**: S1（hengck23 の DreaMS の空き順位の埋め）は未確認なので、優先順を決めない。中の .py 3 本と `embedding_model_state_dict.pt` の出どころを読んでから決める

### 読み（判断の材料。決定ではない）
- 1・3 と、2 のうち候補バンク（11a・11b・11c）は、Q-L の答えを待たずに要る（Engine 2 を残す場合）。2 の FP モデル（10）は Q-L しだい。どれも GPU は要らない（CPU とローカル）が、LB の下げが大きい（#77 の −0.052、#62）。最終の 2 本を「規則に合う形」にするなら、Engine 2 と外部ライブラリの扱いを先に決める必要がある
- 4〜6 は Q-L・Q-A の答えしだい。全部やり直すと、GPU の見込みは T4 で少なくとも 14.36 h（full1 をキットの経路で）＋ 8.7〜20.3 h（ranker の段 1）＋ 未測定の分（B・fe_D・ranker の段 2〜3）になる。週 30 h の枠に 1 週では収まらない見込み。答えが「元の重みでよい」なら 0 h
- 締切は 12/14。答えが出ないまま 12 月に入るなら、どの形を最終にするかを、残りの週の GPU の枠と合わせてオーナーに聞く（お願いに回す）

## 8. 不明・未確認の一覧
- Q-L・Q-A・Q-Y への主催者の答え（11/19 02:1x JST の時点で無い）
- 750330（主催者の「許可される外部データ」の投稿）の本文。取得した JSON のコメント本文が空だった
- `fp_bits.npy` のビットの再計算（Vitalii の主張の中身）、TC `01_build_train_structs.py` の中身
- v2 プール・PubChem 層の再生成のバイト一致
- `hengck23/hengck23-dreams-enveda-casmi26` の 3 本の .py と `embedding_model_state_dict.pt` の出どころ（S1）
- `sanpier/catalogue-experiment-saner` がノートの中で読まれていないことの grep（S3）
- ICEBERG/GLACIER の推論ランナーのライセンス表記（13・14）、v80 のカーネルのライセンス（20）
- `prvsiyan/coconut-casmi26-candidates` の生成コードが作者の他の公開物にあるか（11a）。11b・11c も同じ
- natural-product-knowledge-table の `TECHNICAL_NOTES.md` に外部参照の生成手順があるか（19）
- GPU の今週の残り時間
