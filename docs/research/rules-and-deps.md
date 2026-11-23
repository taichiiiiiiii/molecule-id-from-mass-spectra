# 規則と依存の一覧（#195 A6）: 規則の 5 点と、最終候補の構成ごとの依存（2026-11-24）

Issue #214（親 #195 の A6、#3）。計画は `plan_2026-11-24_night.md` の項目 5（カード t_6de0c5ec）。**この作業では読むだけで、測定・学習・提出・GPU はどれも行っていない。**

このメモは、前の棚卸し `final-repro-inventory.md`（11/19。以下「棚卸し」。部品の番号 1〜23・S1〜S6・B1〜B4 と主催者の回答の略号 R1〜R15 はそちらに合わせる）と `lineage495-repro.md`（11/20）の後に出た、**11/21〜22 の主催者の回答**を取り込んで書き直したもの。

印の意味:
- **確認**: 原文（ページ・投稿・ファイル）をこの作業で読んで確かめた
- **主張**: 出典にそう書いてあるが、こちらでは再現していない
- **未確認**: 確かめていない、または推測

## 0. 読んだもの（2026-11-24 JST に取得）
1. 競技のページ 3 つ。どれも headless で読めた。web 抽出と、ブラウザの `document.body.innerText` の 2 通りで取得した。innerText の写しは scratch に置き、sha256 は次のとおり
   - Rules https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/rules （`9f75b372…c408`、35,204 字）
   - Overview https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview （`a1c94dbc…e171`）
   - Data https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data （`88ab271c…a537`）。正確な値は `rules-and-deps-sources.json` にある
2. ディスカッション。`kaggle competitions topics list` で 9 ページ・178 件の題を取った。**コメントが 1 件以上ある話題はすべて**（132 件）と、コメント 0 件の 3 件、合わせて 135 件を `topics show --format json --page-size 200` で取った。その中の主催者（David Healey・Marie Killian）のコメント 112 件はすべて読んだ（初版は 49 件・72 件だった。レビュー 1 回目の指摘で広げた）。取った話題の ID は `rules-and-deps-sources.json` にある。この JSON の API は話題の本文を返さない。そこで、本文が要る 741359（Welcome）・750330・756064・756893・760397・760683 などは、表の形の出力（`topics show` を `--format` なしで実行）で本文を読んだ
3. リポジトリの中のもの: 棚卸し、`lineage495-repro.md`、`src-audit.md`、`organizer-question-756893.md`、`kernels/lb495/`（ノートと kernel-metadata）、`scripts/train/fragnn_train.py`、`scripts/train/make_fpnet_full_kernel.py`、`scripts/engine2/build_tables.py`、Issue #74・#77・#82・#120・#195、PR #110
4. Kaggle のファイル一覧: `ahmedberatozer/casmi26-training-code`（以下 TC。`--page-size 200` で 2 ページ）、`ahmedberatozer/casmi26-spec-cache`（6 ファイルと metadata）

**読んでいないもの:** Code Competition FAQ、Kaggle の Terms of Service、コメント 0 件の話題 43 件の本文（JSON の API は話題の本文を返さないため。主催者の投稿した話題 741359・750330 の本文は表の形の出力で読んだ）。ただし、Kaggle の運営（inversion）のお知らせ 3 件（746954「Rerun test dataset has been updated」、746878「Metric Update (RemoveStereochemistry)」、749430「Rerun / Rescore Complete」）は本文を読んだ。電子の質量の修正、指標の立体の除去、採点のやり直しのお知らせで、(a)〜(e) に当たる文は無い。以下で「書いていない」と書くのは、**上の 3 ページと 135 件の話題の主催者のコメントの中に無い**、という意味である。

## 1. 規則の 5 点

### (a) 外部データ・事前学習済みモデル
**規則の本文（確認）**
- Overview の Code Requirements: 「Freely & publicly available external data is allowed, including pre-trained models」
- Rules 2.6.a: 「You may use data other than the Competition Data ("External Data") to develop and test your Submissions. However, you will ensure the External Data is either publicly available and equally accessible to use by all Participants of the Competition for purposes of the competition at no cost to the other Participants, or satisfies the Reasonableness criteria」
- Rules 2.6.b: 「The use of external data and models is acceptable unless specifically prohibited by the Host.」

**主催者の回答（確認）**
- 750330（David Healey、2026-10-17。題の投稿の本文）: 非商用のオープンなデータは使ってよい。「solutions can use external data that is open sourced with noncommercial provisions」「External data that is not open source, is not freely available to all, or is proprietary remains against the rules. … NIST is not permitted, as it is not open or freely available and the data including derivatives such as model weights cannot be redistributed.」
- 745841（David Healey、2026-10-05、id 3533343）: 「the train.parquet file can be treated effectively as open-source and any model trained on it is fine, redistributed artifacts are fine.」
- 重みの条件
  - 748882（Marie Killian、2026-10-13、id 3538202）: 「existing third-party weights may be used if the original training code and permitted training data needed to reproduce them are provided」
  - 750330（Marie Killian、2026-11-22、id 3558024）: 「You may use a publicly shared checkpoint, but you remain responsible for ensuring that the training code and permitted data are available and that the model and final solution are reproducible, with reasonable allowance for variability due to random seeds.」
- 名前を挙げて「可」とされたもの（すべて確認）
  - DreaMS の `ssl_model.ckpt`: 760397 id 3557921（2026-11-21）、748882 id 3558031。754610 id 3556278 は「the official DreaMS ssl_model.ckpt is included in the officially released DreaMS checkpoints」で、同じスレッドの「Officially released DreaMS checkpoints are fine to use」（id 3550242）と合わせた間接の承認
  - GLACIER/ICEBERG・DreaMS・ChEBI/LIPID MAPS: 745841 id 3533343
  - MassSpecGym と MIST: 757307 id 3553101
  - COCONUT: 759916 id 3556196
  - PubChem: 741857 id 3525543・3536677
  - LOTUS・NPAtlas、TabPFN、FRIGID の公式版: 748882 id 3538202
  - GNPS-NIST14-MATCHES（**11/21 に改めて可**）: 757307 id 3557906「GNPS-NIST14-MATCHES has been an approved source as well」、760376 id 3557917「Yes, data from GNPS-NIST14-MATCHES is permitted」
  - 過去の CASMI の記録: 760376 id 3557917「Historical CASMI records are permitted.」
  - 公開ライブラリだけで学習したモデル、化学言語・グラフのモデル、PubChemLite: 751373 id 3543071（David Healey、2026-10-22）「Models trained only on public libraries: correct, allowed」「Chemical language / graph models correct, allowed」「Yes PubChem Lite is fine」
  - ICEBERG の PubChem atlas（予測スペクトル）: 745134 id 3556263（2026-11-17）「ICEBERG PubChem atlas is permitted provided you include the code used to retrieve spectra.」
  - GNPS・MassBank・MoNA の新しい版、CC BY-SA の記録: 745228 id 3533256「Newer releases of public datasets are fine to use and CC BY-SA records are acceptable.」。750684 id 3545841「You are welcome to use more recent updates of the data sources used in train.parquet.」
  - Enveda-180 の Zenodo 版: 743569 id 3529232「Yes you can use the Zenodo release」
  - 名前を挙げられていないデータ: 748552 id 3538183「Competitors are not limited to the external datasets that have been explicitly approved by name. Other publicly available datasets may be used, provided they comply with the competition's external-data rules and applicable licensing requirements.」
- 名前を挙げて「不可」とされたもの（確認）
  - NIST: 750330 の本文、744470 id 3530300「any solution using or trained on NIST will be disqualified」
  - SIRIUS / CSI:FingerID: 751373 id 3543071「SIRIUS / CSI:FingerID: correct, not allowed」（質問の本文は理由として「the training set isn't fully public and may include NIST」と書いている）。#195 C5 の「SIRIUS / CSI:FingerID の持ち込み」は、この答えで見送りになる
  - METLIN と装置メーカーの商用ライブラリ: 742193 id 3527949「There are also fully commercial datasets like METLIN and those distributed by the mass spec companies (Thermo, Bruker, SciEx).  Also to avoid.」
  - PubChem の depositor の印のうち NIST MSDC を含むもの: 760328 id 3557924「Yes, this is allowed.」
- **棚卸しから変わった点**: `lineage495-repro.md` §2.3 では GNPS-NIST14-MATCHES の 117 行を除く案にしていた。上の 2 件で可が明らかになったので、除く理由はなくなった

### (b) test のスペクトルを学習や擬似ラベルに使ってよいか
**規則の本文（確認）**
- Rules 3.4.b: 「Submissions may not use or incorporate information from hand labeling or human prediction of the validation dataset or test data records.」
- Data: 「the file you see is comprised of examples from the training dataset; it will be replaced by the hidden test set during re-run, and will be approximately the same size.」
  - つまり、手元で見える test は train の写しで、隠れテストのスペクトルは再実行のときにだけノートに渡る
  - 主催者の確認（756064、David Healey 2026-11-11、id 3553096）: 「Correct, and the hidden test set has roughly similar size.」

**主催者の回答（確認）**
- 752297（Marie Killian、2026-10-26、id 3544964）: 「Yes, automated, unsupervised processing across test spectra at inference time is allowed.」
  - 質問の例（Vitalii Neibauer、id 3544112）: 「letting a confidently identified test molecule inform the ranking of candidates for a spectrally related test molecule」。答えはこの形を含む読みになる
  - 同じ形をはっきり問うた 751373 の XZXZZX（id 3544243「Is fully automated, label-free use of similarity between test spectra of different molecule_ids at inference time allowed …?」）にも、David Healey が id 3545025（2026-10-26）で「yes its allowed」と答えている
- Public LB は見える test.parquet ではなく、隠れテストの public の部分で計算される: 741404 id 3524688（David Healey）「the public leaderboard score is calculated not on the contents of the test.parquet downloadable dummy file … but rather on the public split of the Hidden test set provided to your submission notebook.」

**書いていないこと**
- 推論のときに test のスペクトルだけでモデルの重みや埋め込みを更新すること（test-time adaptation）
  - この点は 748882 の WEI ZIHE12（2026-10-30、id 3547211）が聞いている。取った 135 件の話題の主催者のコメント 112 件に、この点への答えは無い（adapt・pseudo などの語でも検索した）
  - だから「可」とは書かない。#195 C2 の条件（A6 で可と確かめられたときだけ候補にする）に照らすと、**今は候補にしない**
- 擬似ラベル（自分の予測を正解として学習すること）を名指しした文は、上の 3 ページにも 135 件の話題の主催者のコメントにも無い。「unsupervised processing」に入るかは**解釈になるので決めない**

### (c) 勝者に求められる再現性とライセンス
**規則の本文（確認）**
- 1.6 の見出し「WINNER LICENSE TYPE」の下に「Open Source - MIT」
- 2.5.a.1: 「You hereby license and will license your winning Submission and the source code used to generate the Submission under an Open Source Initiative-approved license … that in no event limits commercial use of such code or model containing or depending on such code.」
- 2.5.a.3: 「In the event that input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to grant an open source license in the preceding Section for that data and/or model(s).」
- 2.5.b: 「… a detailed description of methodology, where one must be able to reproduce the approach by reading the description … The description should also include a link to a code repository with complete and detailed instructions so that the results obtained can be reproduced.」
- 2.8.a.1: 「Deliver to the Competition Sponsor the final model's software code as used to generate the winning Submission … For avoidance of doubt, delivered software code should include training code, inference code, and a description of the required computational environment.」
- 3.6.c: 「if open source code is used in the model to generate the Submission, then you must only use open source code licensed under an Open Source Initiative-approved license … that in no event limits commercial use」
- 3.6.b: Kaggle で公開して共有したコードは「deemed to have licensed the shared code under an Open Source Initiative-approved license」
- 3.9.b: 規則に合わない提出は、失格にするか、通知から 1 週間以内の手直し（「the resolution of license conflicts …」）を求める

**主催者の回答（確認）**
- 中間物にも再現性が要る
  - 756893（David Healey、2026-11-11、id 3553098）: 「Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible.」
  - 753662（David Healey、2026-10-30、id 3547273、11/4 の追記）: 「Intermediates also need to be reproducible. All solutions need to be fully provenanced to their data.」
  - 760376（Marie Killian、2026-11-21、id 3557917）: 「Reproducibility is required. You must provide the data and code necessary to reproduce all parts of your solution, including intermediate artifacts.」「Participants remain responsible for documenting the sources of any derived datasets or artifacts」
- **ビット一致は要らない（11/22 に複数回）**
  - 748882 id 3558031（原文）: 「Bit-identical reproducibility is not required if the solution can be reproduced within a reasonable performance delta.」
  - 同じ趣旨の文が 748882 id 3558039、753662 id 3558032、753775 id 3558040、756893 id 3558044 にもある。言い回しは少しずつ違う（「reasonable delta」、3558044 は「if the model and final solution can be reproduced within a reasonable delta」）
  - 753900 id 3558038（2026-11-22）も同じ趣旨（「Bit-identical reproducibility is not required if the model and final solution can be reproduced within a reasonable delta.」）
  - **版の番号か取得日を書くこと**: 751373 id 3545025（David Healey、2026-10-26）「yes its sufficient to just make sure the code and permitted data are there; bit for bit reproducibility isn't required where versions are changing. However, we would ask that you provide a version number or download date so that it could be reproducible.」
  - 中間物の生成コード: 751373 id 3543071「We need the code that generates the intermediates as well」
  - 760397 id 3557921（2026-11-21）: 公開スクリプトで学習し直すのは可。公開の重みを直接使うのも、元の学習コードと許可データがあれば可。「with reasonable allowance for variability due to random seeds」
- 作り直した表も可
  - 756893（Marie Killian、2026-11-22、id 3558043）: 「Yes, you may rebuild the tables using permitted data, provided you supply the code and data necessary to reproduce them.」
  - 745072（Marie Killian、2026-10-27、id 3545813）: 「if you retrain equivalent models with your own published code and permitted data, that would satisfy the rule」
- 最終 2 本の片方が失格になっても、もう一方は残る
  - 753502 id 3558037: 「One submission being disqualified does not automatically disqualify the other. However, leaderboard position will reflect only eligible submissions.」
  - 756893 id 3558043、748882 id 3558039 も同じ趣旨
- 非商用のライセンスだけでは賞・メダルの対象外にならない。規則に合わない提出は順位から外されうる: 745615 id 3538192（Marie Killian、2026-10-13）「A noncommercial license does not automatically make a solution ineligible for prizes or medals.」「Submissions found to be noncompliant may be disqualified and removed from the leaderboard.」
- 自前で Kaggle の dataset に上げた PubChem のデータは、競技中は非公開でよいが、賞の対象の解には含める: 749209 id 3538739「The dataset would need to be included with a prize-eligible solution but does not need to be public during the competition.」
- 賞だけでなく順位からも外されうる
  - 752541 id 3545027（David Healey）: 「We reserve the right to remove submissions that are not reproducible from the leaderboard, regardless of ranking.」
  - 748882 id 3545819・3545820（Marie Killian）も同じ趣旨
- ビットの添字の表
  - 754610 id 3556275: 「A description of how the bit-index was chosen should be provided. That along with the list itself is sufficient.」
  - 758233 id 3556176: 「There needs to be justification for the choice of indices.」
- **個別の部品の可否を主催者は確かめていない**: 748882 id 3558031「I have not fully reviewed the checkpoints listed」、760376 id 3557917「I have not fully reviewed the datasets listed」。11/22 の答えは一般の規則で、特定の部品が「可」と認められたわけではない（確認）

**書いていないこと**
- 「reasonable delta」の大きさ（数値で書いた文は無い）
- 他の参加者が Kaggle の dataset に置いた、ライセンス表示の無いコード（例: TC の `other` 表示のコード）が 3.6.c の「OSI のライセンス」に当たるか
  - 748882 id 3538755 の原文: 「Use of public competition datasets and code shared by other competitors (including code files inside public Kaggle Datasets) is allowed, provided they come from permissible sources and comply with applicable licensing requirements.」。似た文が 742265 id 3538804（括弧の中は「including code files inside public models」）と 745072 id 3538730（括弧なし）にもある。Kaggle の dataset の中のコードをはっきり含めているのは 3538755 だけ
  - 3.6.b の「みなし」が dataset の中のコードにも及ぶかについての文は無い
  - 761272（2026-11-23、「Rule 6: helper code loaded from public datasets without a stated licence」）は、取得の時点でコメント 0 件だった

### (d) ホスト型 AI へデータを送ってよいか
**規則の本文（確認）**
- 2.4.b: 「You agree not to transmit, duplicate, publish, redistribute or otherwise provide or make available the Competition Data to any party not participating in the Competition.」
- 3.5.d: 「Privately sharing code or data outside of Teams is not permitted.」
- 2.6.b は LLM を「external data and models」の評価の対象として挙げている（「can exclude the use of the particular LLM, data set(s), or tool(s)」）

**主催者の回答（確認）**
- 760683（Marie Killian、2026-11-22、id 3558033）: 「Yes, the use of LLMs is allowed, including sharing competition data and code with them, provided participants continue to comply with the competition rules.」
- この作業の進め方（Claude にリポジトリのコードとログを読ませる）は、この答えの範囲に入る。条件は「規則に従い続ける」こと
- 書いていないこと: Baligh の追加の質問（760683 id 3557957。2.4.b・3.5.d・3.6.a との関係と、手元で動く AI との違い）には、取得の時点で答えが無い

### (e) test の各 molecule_id は互いに違う構造か
**書いていない。** 近い文は次のとおり（すべて確認）。
- Data の test.parquet の列の表で、`molecule_id` の説明: 「Anonymous compound identifier. Predictions are made per molecule_id.」
- Data: 「A molecule may have been measured several times at different collision energies or as different adducts. Predictions are made per molecule, not per spectrum」
- Overview: 「Each molecule has exactly one correct structure, so only your first correct guess counts.」
- Data: 「~1,500 spectra of ~400 molecules」

**読み（推測。事実としては扱わない）**: 「compound identifier」と「同じ分子の測り直しを 1 つの id にまとめる」書き方は、id ごとに違う化合物であることを思わせる。しかし「2 つの id が同じ構造を持つことは無い」とはどこにも書いていない。主催者への質問は 745758（類の内訳）・745715（見える test の id）にあるが、どちらもこの点を問うていない。

**影響**: UNIQ の前提（日報 Q3。同じ構造が複数の分子の 1 位にならない）は、規則からは確かめられない。だから UNIQ 型の規則は「規則が保証する性質」としては扱えない。手元の検証の集合は、作り方の上で構造が互いに違うので、この前提が崩れたときの損は測れない。

## 2. 最終候補の構成ごとの依存

### 2.1 判定の決め方（11/22 の回答に合わせて棚卸し §3 を改めた）
- **可**: 生成コードと許可データの両方が公開されていて、欠けているものが知られていない
- **可（条件つき）**: コードと許可データはあるが、seed が記録されていない、コマンドを後から組み直した、などの欠けがある
  - 11/22 の一般回答（ビット一致は要らない。reasonable delta の範囲で再現できればよい）に照らすと、可になる見込み
  - ただし主催者はその部品を個別に確かめていない（「I have not fully reviewed」）
  - reasonable delta の範囲に入ることを示す責任はチームの側にある（「you remain responsible」）。**こちらで作り直して差を確かめたものは無い**
  - 棚卸しで「未決」にした行（Q-L・Q-A・Q-Y の答え待ち）は、11/22 の答えでここに移した
- **危険（説明だけ）**: 手順の説明はあるが、生成コードが見つからない
  - 760376 id 3557917「You must provide the data and code necessary to reproduce all parts of your solution, including intermediate artifacts」に照らすと、説明だけでは足りない読み
  - 753662 id 3553097（David Healey、2026-11-11）も同じ向き: 「If we can't reproduce it because parts of the generating code is missing, the solution is open to being disqualified.  Use others' intermediate data artifacts at your own risk.」
  - 748882 id 3545819（Marie Killian、2026-10-27）: 「If the candidate tables are modified in any way from the original download, you should provide code to document the steps taken.」
  - 例外はビットの添字の表で、R8 により説明と表そのもので足りる
- **危険（不可）**: 生成コード、または学習の入力が、公開物に見つからない。質問で表が名指しされ、主催者が一般の規則（再現できない中間物は違反）で答えたもの（754292 id 3550245、756893 id 3553098）もここに入れる
- 「見つからない」は「X を探して見つからない」の意味で、「存在しない」とは言っていない（探した場所は棚卸しと `lineage495-repro.md` に書いてある）

### 2.2 共通の土台 F（凍結の 59 セル）
- 0.479・0.487・0.495 の写しは、どれも同じ凍結パイプラインをノートに埋め込んでいる
  - `kernels/lb495/notebook.ipynb` のセル 13 の `FROZEN_SHA256` と、`kernels/lb479` の同じ値は、どちらも `d1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88`
  - どちらも、埋め込みの b64 をデコードした中身の sha256 と一致した（確認）
  - デコードした 59 セルの中には、`rank_train`（4 か所）・`sim_rank_rows`（1）・`fpnet_full1`（2）・`casmi26-v4b`（4）・`coco_fp`（3）・`pubchemlite`（20）という名前が出てくる（確認。名前が出てくることを数えただけで、実行の経路はたどっていない）
- だから、棚卸しの表 A の行 1〜19 と S1〜S6 は、下の C1〜C4 のすべてに当てはまる

F の部品（表の「11/24 の判定」は 2.1 の決め方による。コードの所在とライセンスの出典 URL は棚卸しの表 A の同じ行にある）

| # | 部品 | 生成コードの所在 | ライセンス | 11/24 の判定（棚卸しからの変化） |
|---|---|---|---|---|
| 1 | Engine 1 ランカー `ranker_v4b.pkl` | TC `fe_v4/simulate_v1ctx.py` → `build_feats.py` → `train_v4_ranker.py` | TC は MIT、データセットは other | 可（条件つき）。未決から移した。コマンドの組み直し・学習後の編集あり（TC README §5、主張）。出力の後に手を入れた表には、手順のコードが要る（3545819） |
| 2 | `fpnet_full1.pt` | TC `scripts/06_train_fpnet.py` | MIT / other | 可（条件つき）。未決から移した。seed と `--holdout` が記録されていない |
| 3A | FPNet A | TC `06_train_fpnet.py`（seed 7） | MIT / other | 可（変化なし） |
| 3B | FPNet B | 同上（seed が記録されていない） | MIT / other | 可（条件つき）。未決から移した |
| 4 | `fe_D.pt`（DreaMS の微調整） | TC `dreams_ft/review_fixes/train_dreams_fp.py` → `extract_step20k_ema.py` | MIT | 可（条件つき）。未決から移した。土台の ssl_model.ckpt は 760397 id 3557921 で可。NIST の probing の点（Q-Y (3)）について: 748882 id 3558031 は「Also, the official DreaMS checkpoint ssl_model.ckpt is allowed.」と書く。probing の集合が config に書かれているのはこの ckpt なので、Q-Y (3) にも当たる、というのは**こちらの読み**（748882 のコメントは平らに並んでいて、3558031 が Q-Y への返事だとは示せない。同じコメントは「I have not fully reviewed the checkpoints listed」で始まる） |
| 5 | fe_v4 の表 | TC `class3_v2/priors_build.py`・`fe_v4/res_fragnet.py` ほか | MIT / other | 可（条件つき）。未決から移した。出力の後に編集されている。同上（3545819） |
| 6 | v2 プール | TC `01`・`02`・`05_build_pool.py` | MIT / other | 可（変化なし）。入力の版か取得日の記録が要る（3545025）（COCONUT の版） |
| 7 | `fp_bits.npy`（10,226） | TC `04_make_split.py` → `structs.select_bits` | MIT | 可（変化なし） |
| 8 | `rank_train.npz` | 一部だけ（cubres・`prvsiyan/casmi26-generation-code`。FP ロジットのキャッシュの系統は作者自身が「unresolved」と書く） | CC-BY-NC-SA-4.0 | **危険（不可）**（変化なし。754292 の質問で名指しされ、id 3550245 が一般の規則で答えた） |
| 9 | `sim_rank_rows_nofp.npz` | 見つからない（データセットの 6 ファイル） | CC0-1.0 | **危険（不可）**（変化なし。754292・756893 の質問で名指しされ、id 3550245・3553098 が一般の規則で答えた） |
| 10 | prvsiyan の FP モデル | cubres の `train_fp3.py`/`train_fp4.py`（Apache-2.0） | CC-BY-NC-SA-4.0 | 可（条件つき）。未決から移した。学習の入力の版が固定されていない。750330 id 3557002 の質問への答え（id 3558024）は一般論。入力の版か取得日の記録が要る（3545025） |
| 11a | COCONUT バンク（prvsiyan） | 説明だけ（データセットのカード） | CC-BY-NC-SA-4.0 | **危険（説明だけ）**（変化なし） |
| 11b | ChEBI/LMSD バンク | 説明だけ（`ATTRIBUTION.md`） | CC BY 4.0 | **危険（説明だけ）**（変化なし） |
| 11c | PubChemLite の指紋 | 説明だけ（`MANIFEST.json`） | CC0-1.0 | **危険（説明だけ）**（変化なし）。入力の版か取得日の記録が要る（3545025） |
| 11d | Engine 2 の 6,930 ビット | `prvsiyan/casmi26-generation-code`。こちらで再現済み（#84） | Apache-2.0 | 可（変化なし） |
| 12 | MetFrag-lite `frag_rescore.py` | ソースそのもの | Apache-2.0（コメントの記載） | 可（変化なし） |
| 13・14 | ICEBERG・GLACIER の重み | ms-pred の公開版 | MIT | 可（変化なし）。ランナーのライセンス表記は未確認 |
| 15 | PubChem 層 | TC `scripts/02c_build_pubchem.py`（ファイル一覧で確認） | MIT / other | 可（変化なし）。入力の版か取得日の記録が要る（3545025）（PubChem の版） |
| 16 | 文献 prior | データセットの `scripts/` | other（PubChem・LOTUS・NPAtlas） | 可（変化なし） |
| 17 | fold-safe FPNet 5 本 | dmitriigluzdov のキット | other（スクリプトは Apache-2.0） | 可（変化なし） |
| 18 | sanpier のスペクトル–候補ニューラル | データセットの `train.py` ほか | apache-2.0 | 可（変化なし） |
| 19 | 外部参照スペクトル（knowledge-table の `external_delta`） | 説明だけ（`INTEGRATION.json` ほか） | other | **危険（説明だけ）**（変化なし） |
| S1 | hengck23 の DreaMS の埋め | 中身を読んでいない | MIT | 未確認（変化なし） |
| S2・S3・S6 | 表示だけの panel・読まれない catalogue・RDKit の wheel | — | — | 影響なし |
| S4 | Engine 2 の HGB 12 本（ノートの中で学習） | ノートの中 | 作者のコード | 8・9 に従う（8・9 を替えれば可） |
| S5 | 補助モジュール・`consensus.py`・`outsiders.py` | ノートに埋め込み | 作者のコード | 可 |

F だけの数え上げ: 危険（不可）2（8・9）、危険（説明だけ）4（11a・11b・11c・19）、可（条件つき）6（1・2・3B・4・5・10）。**F を含む構成は、手を入れない限り、8・9 のために失格の危険がある。**

### 2.3 構成ごとの表
各構成の依存は「F ＋ 下の表の行」である。F の行のコードの所在とライセンスは 2.2 の表にある。

#### C1: 0.495 の写し（`taichiiiii/casmi26-lb495-floor`、#164・#212）
`kernels/lb495/kernel-metadata.json` の dataset_sources は 11 件、kernel_sources は `popwenman/casmi26-scorer-train-v80`（確認）。

| 依存 | 生成コードの所在 | ライセンス | 判定 |
|---|---|---|---|
| F（2.2 の全行） | 2.2 | 2.2 | 危険（8・9 は不可、11a〜c・19 は説明だけ） |
| 20 v80 スコアラー（kernel_sources） | そのカーネル自体が学習コード | キットは apache-2.0。カーネルのライセンスは未確認 | 可（条件つき）。版を固定していないので、最終のときに版を確かめて記録する（3545025） |
| 22 MassBank harmonized（samartalwar） | 説明だけ（metadata の 6 手順） | CC-BY-NC-SA-4.0 | **危険（説明だけ）** |
| extlib-gnps-full（taeyangg4、198,581 行） | 見つからない。ただし行ごとの GNPS の accession があるので、自前で組み直せる（`lineage495-repro.md` §2） | CC BY-NC-SA 4.0 | **危険（不可）**。GNPS-NIST14-MATCHES の 117 行は 11/21 に可になった |
| FR 3-25（`taeyangg4/casmi26-fr3to25-clean`） | 学習スクリプト `fr_train_clean_v1.py` はある。ただし入力 6 つのうち 3 つ（`structure_keys.parquet`・`structdb_v1/entries.parquet`・`pool_chemistry.json`）は公開物に見つからない（`lineage495-repro.md` §1） | CC BY-NC-SA 4.0。`fragfeat.py`・`model.py` の出どころ（agentili）は other で、説明は空 | **危険（不可）**。760376 id 3557917 の「the data and code necessary to reproduce」のうち、データの側が揃わない |
| `metric/rdkit-2026-3-3-wheel` | 環境 | BSD-3 | 影響なし |

**C1 の読み**: 失格の危険がある。8・9 に加えて、0.495 で足された 2 つ（extlib-gnps-full と FR 3-25）も「不可」の側にある。#195 B4 は「2 本目 = 床（0.495 の写し。ただし A6 で失格の危険が無いと確かめられた場合に限る）」と決めている。**この条件は満たされない。** B4 の規則どおり、2 本目をどうするかはオーナーの判断に回る（2.5 に書いた）。

#### C2: 0.487 の写し（提出 58208946、Public 0.487）
`kernels/lb487/` は main に無く、`origin/feat/77-lb487-owntables`・`origin/feat/127-align-0487-submit` にある（棚卸し §0）。

| 依存 | 生成コードの所在 | ライセンス | 判定 |
|---|---|---|---|
| F | 2.2 | 2.2 | 危険（8・9） |
| 20 v80 | 同上 | 同上 | 可（条件つき）。版を記録する（3545025） |
| 21 ExtLib（takumuhata） | 見つからない（データセットの 6 ファイル、作者の他のデータセット 24 件とカーネル一覧） | CC0-1.0 | **危険（不可）** |
| 22 MassBank harmonized | 説明だけ | CC-BY-NC-SA-4.0 | **危険（説明だけ）** |

**C2 の読み**: 失格の危険がある（8・9・21）。

#### C3: 0.487 ＋ 自前の Engine 2 の 2 表（#74・#77、提出 58411499、Public 0.435）
| 依存 | 生成コードの所在 | ライセンス | 判定 |
|---|---|---|---|
| F から 8・9 を除いたもの | 2.2 | 2.2 | 11a〜c・19 の危険（説明だけ）は残る |
| 自前の 2 表（8・9 の代わり） | 自前の `scripts/engine2/build_tables.py`（main）。Kaggle CPU で生成し、`taichiiiii/casmi26-engine2-build-code` で配った。2 表のファイル名を開くと例外にする仕掛けがある（#74） | 自前。**リポジトリに LICENSE が無い**（2.4） | コードは可。判定は入力に従う（11a〜c は説明だけ、folds の生成コードは未確認）。入力は train.parquet、fold-safe FPNet の fold 1〜4（行 17）、`dmitriigluzdov/casmi26-identity-disjoint-validation-folds`（**生成コードは未確認**）、v4g の中の候補バンク 11a〜c（説明だけ）。だから判定はこれらの入力に従う |
| S4 HGB | ノートの中で、自前の表から学習する | — | 可（自前の表に従う） |
| 20・21・22 | C2 と同じ | 同じ | 21 は不可、22 は説明だけ |

**C3 の読み**: 質問で名指しされた 8・9 の危険は消える。ただし 21（ExtLib）と、説明だけの 11a〜c・19・22 が残るので、今のままでは危険がある。

#### C4: 0.487 ＋ 自前で学習し直した FPNet（#82、S2 `fpnet_final.pt`、提出 58699228、Public 0.449）
| 依存 | 生成コードの所在 | ライセンス | 判定 |
|---|---|---|---|
| F から 2 を除いたもの | 2.2 | 2.2 | 8・9 の危険は残る |
| 自前の FPNet（2 の代わり） | 自前のカーネル生成 `scripts/train/make_fpnet_full_kernel.py`（main、PR #176 でマージ済み）。学習本体はキットの `train_fpnet.py`（dmitriigluzdov、Apache-2.0）で、TC の `casmi` パッケージ（`fpnet.py`・`traindata.py`・`spectra.py`・`chem.py`、sha256 で固定）を読む | 自前（LICENSE が無い）＋ Apache-2.0 ＋ MIT | 可。入力: train.parquet、v2 プール（行 6・7）、`ahmedberatozer/casmi26-spec-cache` |
| `ahmedberatozer/casmi26-spec-cache` | TC `scripts/03_build_spec_cache.py`（311 B）と `04_make_split.py`。TC のファイル一覧で存在を確認した。中身は読んでいない | other（説明では train.parquet だけから作った、と書いている。主張） | 可（条件つき）。コードがあることは確認したが、中身は未確認 |
| 20・21・22 | C2 と同じ | 同じ | 21 は不可、22 は説明だけ |

**C4 の読み**: 2 の「条件つき」は消えるが、8・9・21 の危険が残る。C3 と C4 を合わせても、21 と説明だけの行が残る。

#### C5: fragnn を載せた版（今は候補ではない。もし載せるならの依存）
fragnn は v1 が不採用（#120）、v2 も R1 で不採用（#196、11/24）。だから**今は最終の候補ではない**。次の版が採用されたときのために、依存だけを書く。

| 依存 | 生成コードの所在 | ライセンス | 判定 |
|---|---|---|---|
| 載せる先の系統（C1〜C4 のどれか） | 上の表 | 上の表 | 載せる先の危険をそのまま受け継ぐ |
| fragnn のコード | 自前の `scripts/train/fragnn_train.py`・`fragnn_bench.py`、`scripts/local_eval/fragnn_score.py`・`fragnn2_score.py`（main。テストあり） | 自前（LICENSE が無い） | 可 |
| fragnn の重み | ローカルの `outputs/fragnn_train/model_best.pt`（sha256 06341e23…651f、#120）。公開していない | 自前 | 学習コードは main にある。提出に使うなら Kaggle Model に上げる |
| DreaMS の埋め込み `outputs/align_embed` | `scripts/train/align_local_embed.py` は **main に無い**。ブランチ `feat/109-align-local-train`（PR #110 は 11/24 に閉じた。ブランチは残っている。`git ls-remote` で確認） | 自前 | 可（条件つき）。生成コードを main に入れることが条件 |
| 公式 DreaMS `ssl_model.ckpt`（B1） | 公式（Zenodo 10997887） | 重みは CC-BY-4.0、コードは MIT | 可（760397 id 3557921） |
| 断片 `data/ext/frag/frag_rescore.py`（行 12 と同じもの） | ソースそのもの | Apache-2.0（コメントの記載） | 可 |
| v2 プール・fold-safe の split（行 6・17） | 2.2 | 2.2 | 可 |

**C5 の読み**: fragnn そのものは自前のコードと許可データで作れる。危険は、載せる先の系統と、埋め込みの生成コードが main に無いことから来る。

### 2.4 構成に共通の欠け: リポジトリにライセンスのファイルが無い
- `gh repo view` の licenseInfo は none で、LICENSE ファイルも無い（確認）。リポジトリは PUBLIC
- 2.5.a.1 では、勝者は解とコードを OSI のライセンス（このコンペは MIT、1.6）で出す。賞を受け取る段になると要るもの。今の提出の可否には関係しない
- LICENSE を足すかどうかは、オーナーの判断にする（リポジトリの権利の扱いになるので、こちらでは足さない）

### 2.5 B4（最終 2 本）への影響（判断の材料。決定ではない）
- 今ある構成で、2.1 の決め方の「危険」が 1 つも無いものは無い。C1〜C4 はすべて F の 11a〜c・19 か、8・9・21 を持つ
- 危険を消すのに要る作業（どれも未着手、見積もりは棚卸し §7 のまま。測っていない）
  1. 8・9 → 自前の表（C3 で済んでいる）
  2. 21（ExtLib）と extlib-gnps-full → 外すか、自前で組み直す（`lineage495-repro.md` §2.2）
  3. 11a〜c → 出どころ（COCONUT・ChEBI 255 と LMSD・`cosmicbe/casmi26-pubchemlite`）から指紋を作る自前のコード
  4. 19・22 → 外すか、自前で組み直す
  5. FR 3-25（0.495 系統）→ 入力 2〜4 を自前の表に替えて学習し直す（`lineage495-repro.md` §1.4。カード t_3f8b180a）
- #195 B4 の 2 本目の条件（0.495 の写しは失格の危険が無いときだけ）は**満たされない**。B4 の規則どおり、次の 2 つをオーナーに聞く
  - 0.495 の写しを危険を承知で 2 本目に置くか
  - 置かないなら、手元の検証で次点の、1 本目と性格の違う自前の構成にするか
- t_3f8b180a（FR 3-25 の入力、blocked）: 上の 5 の作業になる。要るかどうかは、0.495 系統を最終に残すかどうか（上のオーナーの判断）で決まる。**ここでは決めない**

## 3. 756893 の質問の下書き（`organizer-question-756893.md`）の突き合わせ
- 下書きで聞きたいこと: 自前の生成コードを解と一緒に公開し、その生成器から作った表（元の表とは設計が違ってもよい）を読むなら、バイト一致しなくても再現可能と認められるか。言い換えると、要件は「解と一緒に公開したコードから作り直せること」で、「元のファイルと一致すること」ではないのか
- 突き合わせた答え（すべて確認）
  1. 「作り直した表でよいか」→ 756893 id 3558043（2026-11-22）「Yes, you may rebuild the tables using permitted data, provided you supply the code and data necessary to reproduce them.」。答えは「公開された設計に沿って」という条件を付けていない
  2. 「バイト一致は要らないか」→ 11/22 の 5 件（2.1 に挙げた id 3558031・3558039・3558032・3558040・3558044）「Bit-identical reproducibility is not required」
  3. 「自前のコードで同等のものを作ればよいか」→ 745072 id 3545813「if you retrain equivalent models with your own published code and permitted data, that would satisfy the rule」
  4. 「要件は再現できることか」→ 760376 id 3557917「You must provide the data and code necessary to reproduce all parts of your solution, including intermediate artifacts」、753662 id 3547273「All solutions need to be fully provenanced to their data」
- 注意（確認）: 1 の 3558043 は、Shishir の 3557241（「following the published design」と書いた質問）への答えである。下書きはこれを「自前の設計」に広げている（下書きの要旨に「言い換え（承知の上）」とある）。だから**自前の設計の場合を直接に答えた文は無い**。自前の設計の表が 4 の一般の規則で扱われる、というのは**こちらの読み（解釈）**である
- **判断: 投稿しない。** 自前の設計で作った表は、元の表の写しではなく「自前の中間物」になる。中間物は 4 の一般の規則（コードとデータで作り直せること）で扱われる、と読む。この読みが外れる危険は残るが、それを承知で投稿しない。理由は次の 2 つ
  - 11/22 の答えは、個別の場合の可否を言わずに一般の規則を繰り返す形になっている（3558031・3557917 の「I have not fully reviewed」）。同じ形の答えしか返ってこない見込みが高い（推測）
  - 下書きの論点（ビット一致が要らないこと、作り直しが可であること）は 1〜3 で答えられている
- 下書きのファイルは記録として残す。投稿しない理由は #195 にコメントする

## 4. 未確認・書いていない点の一覧
- (b) test-time adaptation（test のスペクトルだけで重みを更新する）と、擬似ラベル: 書いていない
- (e) test の id ごとに構造が違うか: 書いていない
- reasonable delta の大きさ: 書いていない
- ライセンス表示の無い、Kaggle の dataset の中のコードに、3.6.b の「みなし」が及ぶか: 書いていない（761272 はコメント 0 件）
- 760683 の Baligh の追加の質問への答え: 取得の時点で無い
- 部品ごとに主催者が確かめたかどうか: していない、と主催者自身が書いている（3558031・3557917）
- `dmitriigluzdov/casmi26-identity-disjoint-validation-folds` の生成コード（C3 の入力）、spec-cache の `03_build_spec_cache.py` の中身、v80 カーネルのライセンス、ICEBERG/GLACIER のランナーのライセンス、S1: いずれも未確認
- Code Competition FAQ と Kaggle の Terms of Service は読んでいない
