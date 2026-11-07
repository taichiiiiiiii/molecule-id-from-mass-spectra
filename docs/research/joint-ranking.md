# 合同候補（プール＋PubChem）のランキング学習: 調査メモ（2026-11-08）

関連: #24（方針 11/7〜11/14）の 3、ランカー自前学習 Issue（本メモと同じ PR）

## 出典
1. Kaggle ディスカッション 756504「Open-source pipeline: identity search + analog propagation + LightGBM ranker (with notes on validation)」Yohann Roux, 2026-11-06。
   `kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/756504`
2. GitHub リポジトリ https://github.com/Majyles/kaggle-casmi-2026 （2026-11-08 に shallow clone して README・`casmi/ranking.py`・`scripts/train_ranker.py`・`scripts/build_ranker_set.py` を読んだ。ライセンス MIT）
3. Kaggle ディスカッション 748524「Validation stragegy」Bull, 2026-10-12。
   `kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/748524`
4. MassSpecGym: Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", NeurIPS 2024, arXiv:2410.23326, DOI 10.48550/arXiv.2410.23326（arXiv HTML 版の検索要約で確認。本文全体は未読）
5. Kaggle ディスカッション 754292（主催者 David Healey の回答 2026-11-05）、755039（k256.dev、private 側のクラス比についての推測）

## 要点（出典に書かれていること。こちらで再現していないものは「主張」と書く）
### 756504 / Majyles リポジトリ（公開 LB 0.313）
- 検証: enveda-np-examples（timsTOF の天然物、250）を subset A、COCONUT に載る公開ライブラリの天然物 1,000 を subset B。
  クラスは参照スペクトルを隠して模擬する: class 1（同じライブラリ・装置系の参照を隠す）、class 1x（さらに問い合わせと同じ装置系の参照を全部隠す＝テストの timsTOF 対 他装置）、class 2（その分子の参照を全部隠す）。
- 主張: 同じ装置の class 1 で学習したランカーは identity 類似度 ~0.99 を前提に学び、テストの他装置一致（中央値 ~0.90）を信用しなくなる。class 1x で学習し、正解の参照が 1 本も残らないグループ（`reachable_only`）を class 1 の学習から外すと直る。
  → こちらの ref_last の失敗（LB −0.036）と同じ現象。
- ランカー: LightGBM `LGBMRanker`（lambdarank）、特徴量 20（analog 伝播スコア 3 種と相対値・順位、identity 最大値・一致スペクトル数、類似アナログとの Tanimoto、`has_visible_spectra`、`in_coconut`、NP-likeness、質量誤差、候補数、問い合わせスペクトル数、グループ内の順位・相対値）。
  `has_visible_spectra` は「模擬条件で見えてよい参照」から作る（全 train から作ると class 2 の正解が「参照あり」になりリークする、とコードのコメントに明記）。
- 学習データ: 検証と重ならない天然物 2,000（COCONUT 収載、enveda-180・drug_plus に無い）。クラス重み class1 0.16 / class2 0.45 を sample_weight に使う（「クラス重みは identical と analog の事前分布として働く」）。
- 主張: ランカーは手調整の式より A で +0.018（95% CI [+0.008, +0.029]）、B で +0.037。LB 0.295 → 0.313。改善は主に class 2。
- 主張: 「答えの周りの同じ質量の異性体（主に合成品）」が主な誤り要因で、候補を天然物（COCONUT）に絞ると改善（LB +0.025）。

### 748524（Bull）
- 主張: 公開 LB は約 130 分子、1 分子の圏外→1 位で約 0.008。
- 主張: クラス比を LB から推定して class 1 ≈28%、class 2 ≈20%、class 3 ≈52%。
- 主張: **PubChem 候補を固定順位（5 位、10 位…）に差し込むと、ローカル +0.01〜0.02 だったが LB は 0.295 → 0.288。「PubChem にしか無いテスト分子はほとんど無いようだ」**。
- 比較は同じ検証分子での差を構造単位のブートストラップで見る。再学習を伴う変更は seed を変えて複数回学習し平均で比べる（seed だけで class 2 MRR が ~0.02 動いた、という主張）。

### MassSpecGym の retrieval 設定（4）
- 候補は同じ質量の分子を最大 256 件（多ければ無作為抽出）。分子集合 3 種から順に詰める。bonus 版は同じ組成式で絞る。
- 基準法の 1 つは Fingerprint FFN（指紋を予測し候補をコサイン類似度で並べる）。こちらの f·z と同じ系統。
- この競技との違い: 候補数 256 固定・無作為抽出なので、「プールか PubChem か」の出所の区別は扱っていない。設定値として持ち込めるものは無い。

### 754292 / 755039（再現性とクラス比）
- 主催者の回答（754292）: 「解は再現可能でなければならない。再現できない中間物やモデルを使う解は違反で、取り除く権利を留保する」。
  → 自前データから生成コードつきで学習するランカー（本 Issue）はこの要件に合う方向。公開系統の共有ランカー特徴量（生成コード無し）への依存を減らせる。
- 755039 は「private 側に class 3 が多いのでは」という推測で、根拠となるデータは無いと本人が書いている。計画には使わない。

## この競技で使えるか
- コード: MIT なので、著作権表示を残せば部品の流用は可。ただし今回の計画（項目 4）は自前実装で足り、外部コードは取り込まない。設計上の知見（class 1x、reachable_only、模擬条件に沿った参照特徴）だけ使う。
- 再現性: Majyles の手順は train.parquet と COCONUT CSV から再生成できる（README の 6 段）。こちらのランカーも同じく自前データからの生成コードを持つ形にする（organiser の再現性要件に合う）。
- 計画への反映（v3 に合わせて選んだ値ではない）:
  - 項目 2 の C1 型は v3 と同じ「enveda 系の参照だけ隠す」規則（= class 1x 相当、問い合わせは timsTOF）。reachable_only に当たる除外は実装していない。
    代わりに実測で確かめた: 学習グループ C1（500 分子）も v3_c1（200）も、隠した後に公開参照が 0 本になる分子は 0（1 分子あたりの残り参照 最小 5 / 中央値 9〜11 本、v3_c1 は最小 2 / 中央値 10）。C1 の定義が公開スペクトルを要求するため。
  - 項目 4 の参照特徴（ライブラリ類似度・lib_max）は各グループの hide_rule / drop_keys を当てて作る（計画どおり）。
  - クラス重み: 計画は C1:C2 = 1:1 の件数で揃え、sample_weight は使わない（固定）。

## 計画への懸念（判断待ちとして日報に回す）
- v3_c2 は「enveda-180 にしか無い構造」で、答えはプールから消すので PubChem 層からしか届かない。
  「ほぼ合成品、COCONUT ~0%、PubChem 99%」という値は以前のセッションのメモにあるだけで、このリポジトリに出力が無い（未確認）。
  今回の実測: pool_meta.parquet の src 列で、v3_c2 の正解がプールに載る 197 キーはすべて src=1、v3_c1 の 198 キーもすべて src=1。
  src=2（436,515 行、学習の sid を持たない行。COCONUT 由来と見られるが src の定義は未確認）の行を持つ正解は両方とも 0。
  一方 756504 の表と 748524 は、テストの class 2 は「主に COCONUT に載る天然物」で、PubChem だけにある分子はほとんど無いと主張している（748524 は LB で −0.007 の実測つき。ただし差は約 1 分子分で、ノイズの範囲とも言える）。
  0.479 版の PubChem 差し込みも LB では効いていない可能性がある。
- もしこの主張が正しければ、v3_c2 はテストの class 2 を表しておらず、C2 での向上は LB に移らない（逆に、PubChem 候補を上げるランカーはプール内の答えを押し下げて損をする）。
- 追加の実測（pool_meta.parquet の src、fold 0 の hold 55,102 構造）: キーごとの src は 1 / 2 / 3 のどれか 1 つ（1: 248,440、2: 436,515、3: 25,746 キー）。
  src=3 は学習 sid を持つので「学習構造かつ COCONUT」と見られる（定義は未確認）。
  enveda-180 を含む hold 構造は src=1 が 36,386、src=3 が 9（約 0.02%）。enveda-180 を含まない hold 構造は src=1 が 12,966、src=3 が 5,186。
  src=3 = COCONUT という読みが正しければ、v3_c2 型（enveda-180 だけの構造）は COCONUT にほぼ載らず、フォーラムの言う「COCONUT に載る class 2」とは別物。
- 測る案（検証 v3 の追加、次回以降）: **v3_c2n** = fold 0 の hold で src=3（COCONUT に載ると見られる）の公開ライブラリ構造から 200（seed 固定）、参照を全部隠してプールには残す。
  ただしこれらの構造には timsTOF の enveda スペクトルが無いので、問い合わせは公開ライブラリのスペクトル（例: QTOF）になり、テストの装置とはずれる（Majyles の subset B と同じ妥協）。ランカーと 0.479 版規則と joint を v3_c1 / v3_c2 / v3_c2n で比べる。
  採否の条件に「C2n で 0.479 版規則に対して下がらない（差の 95% 区間の下限 ≥ −0.02）」を足すかはオーナー判断。
