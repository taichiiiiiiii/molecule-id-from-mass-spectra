# 調査: 断片の説明で同じ分子式の異性体を区別する方法（#124）

計画 2026-11-16 夜 項目 5。親 #3、#108。前段 #118（`docs/research/isomer-rerank.md`、PR #119）、#120（自前の断片の並べ替え器 fragnn の事前登録、PR #121）。**読むだけ**: GPU 0、学習 0、提出 0、v3 の採点 0。2026-11-16 03:00〜04:30 JST に読んだ。

印の意味（#118 と同じ）: **確認** = 原文・コード・API の応答を読んだ。**主張** = 著者の申告で、こちらでは再現していない。**未確認** = 調べたが分からない（理由を書く）。**推定** = こちらの計算や読みで、測っていない。

## なぜ調べるか
- v3_c1（score+prior+frag、MRR@25 0.8535）の 1 位の外れ 49 のうち 44 は同じ分子式の異性体（#118）。上位 25 はほぼ同じ式の候補なので、分子式の絞り込みでは直せない
- #120 の fragnn は「候補を 1〜2 本の結合で切った断片の質量」と観測ピークの一致を学習で重みづけする。公開の方法が、同じ式の候補の中でどれだけ当てているか、指紋の方法と比べてどうかを数値で確かめ、設計の根拠にする
- 既存の数（クローンの `outputs/local_eval/<集合>/result_*.json`、#117 の後、11/15 23:22 ごろの出力。確認）: 学習なしの MetFrag-lite の frag 段（上位 60 の同じ式のグループの中だけを並べ替える、`gl_fuse.rerank_multi`、λ 0.5）は score+prior に対して v3_c1 0.8219 → 0.8535（+0.032）、enveda_v1 0.8230 → 0.8376（+0.015）、v3_c2n 0.5873 → 0.6284（+0.041）。各集合 1 回の値で、区間は出していない

## 出典
1. MassSpecGym: Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", NeurIPS 2024 Datasets & Benchmarks, arXiv:2410.23326v3 https://arxiv.org/abs/2410.23326 。PDF の本文と表 3（molecule retrieval）・表 4（spectrum simulation）を読んだ。コード https://github.com/pluskal-lab/MassSpecGym （GitHub API の license = MIT、最終 push 2026-05-08）
2. MetFrag 2.2: Ruttkies, Schymanski, Wolf, Hollender, Neumann, "MetFrag relaunched: incorporating strategies beyond in silico fragmentation", J. Cheminform. 8:3 (2016), doi:10.1186/s13321-016-0115-9 （PMC4732001、CC BY 4.0）。全文を読んだ
3. MetFrag の統計学習: Ruttkies, Neumann, Posch, "Improving MetFrag with statistical learning of fragment annotations", BMC Bioinformatics (2019), doi:10.1186/s12859-019-2954-7 （PMC6612146）。要旨と方法の前半を読んだ（PMC 版の取得は途中で切れ、結果の表は読めていない）。コード https://github.com/ipb-halle/MetFragRelaunched （GitHub API に license の項が無く、リポジトリの直下にも LICENSE のファイルが見当たらない）
4. MAGMa: Ridder ほか（著者の全員は読んでいない）, "Automatic Compound Annotation from Mass Spectrometry Data Using MAGMa", Mass Spectrometry (Tokyo) 誌の CASMI2013 の特集（年・巻・頁は読んでいない）、PMC4321337 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321337 。CASMI 2013 の節を読んだ。コード https://github.com/NLeSC/MAGMa （license = Apache-2.0、最終 push 2022-10-30）
5. CFM-ID 4.0: Wang, Allen, Tian, Oler, Gautam, Greiner ほか（Wishart を含む）, "CFM-ID 4.0 – a web server for accurate MS-based metabolite identification", Nucleic Acids Res. 50 (W1) (2022)、PMC9252813 https://pmc.ncbi.nlm.nih.gov/articles/PMC9252813/ （CC BY 4.0）。全文を読んだ。アルゴリズムの論文 Wang et al., Anal. Chem. 2021, doi:10.1021/acs.analchem.1c01465 は 403 で読めなかった（未確認）。学習データの補足として、CFM-ID 4.0 を学習に無いスペクトルで測った論文 PMC11521105 https://pmc.ncbi.nlm.nih.gov/articles/PMC11521105/ の要旨と序論を読んだ
6. ICEBERG: Goldman, Li, Coley, "Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks", arXiv:2304.13136v2 https://arxiv.org/abs/2304.13136 。PDF の retrieval の段を読み直した（#118 の出典 2 と同じ）。コード https://github.com/samgoldman97/ms-pred （MIT、最終 push 2026-11-03）。README に MassSpecGym で学習した ICEBERG・GLACIER の重みの配布がある（確認）
7. FraGNNet: Young et al., "FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction", arXiv:2404.02360v2、TMLR 08/2025。数値は #118 の出典 3 で読んだもの（今回は読み直していない）。コード https://github.com/FraGNNet/fragnnet （BSD-2-Clause、最終 push 2026-11-13）。README に重みの配布・MassSpecGym の語は無い（`gh api …/readme` の grep で 0 件、確認）
8. MIST: Goldman et al., "Annotating metabolite mass spectra with domain-inspired chemical formula transformers", Nat. Mach. Intell. (2023), doi:10.1038/s42256-023-00708-3 。読んだのは bioRxiv 版の PDF https://www.biorxiv.org/content/10.1101/2022.12.30.522318v1 の retrieval の段（取得は 50,000 字で切れ、表 S2 は読めていない）。コード https://github.com/samgoldman97/mist （MIT）。評価の設定の補足として、Coley 研の再現の報告 https://github.com/coleygroup/Audit_ML_MS_analysis/blob/main/DETAILED_RESULTS.md を読んだ
9. 競技の規則: 外部データの規則 750330（David Healey、2026-10-17、`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/750330`、今回 03:0x JST に取得）。主催者の回答 757307（MassSpecGym と MIST）・748882（第三者の重みの条件）は `docs/research/mist-massspecgym.md`・`isomer-rerank.md` の原文の記録を使った（今回は取り直していない）

## 規則（出典 9）
- 確認 750330: 非商用のライセンス（CC BY-NC など）のオープンなデータは使ってよい。オープンでない・誰でも自由に手に入らない・独占のデータは不可。**NIST は不可で、モデルの重みなど派生物も含む**（"NIST is not permitted ... the data including derivatives such as model weights cannot be redistributed"）
- 確認 757307（記録から）: 「MassSpecGym and MIST models are fine」
- 確認 748882（記録から）: 再現できない第三者の重みは不可。承認済みのチェックポイントは、学習コードと許可された学習データへの参照と、自前の微調整・推論の全コードを渡せば足りる
- したがって、各方法の判断は「(1) 重みが NIST 由来か、(2) 学習コードが公開か、(3) 許可されたデータ（競技の train、MassSpecGym など）で作り直せるか」で決まる

## 方法ごとの要点

### MassSpecGym の同じ分子式の設定（出典 1）
- 確認（本文）: 検索の候補は正解と同じ質量の分子で 1 スペクトル最大 256。"bonus chemical formulae challenge" は、分子式を与えて候補を**同じ分子式の分子だけ**にした設定
- 確認（表 3、分子の検索、hit rate @1 / @5 / @20、%、99.9% の bootstrap 区間は省略）:
  - 質量の候補: Random 0.37 / 2.01 / 8.22、DeepSets 1.47 / 6.21 / 19.23、Fingerprint FFN 2.54 / 7.59 / 20.00、DeepSets + Fourier 5.24 / 12.58 / 28.21、**MIST 14.64 / 34.87 / 59.15**
  - **同じ分子式の候補（bonus）**: Random 3.06 / 11.35 / 27.74、DeepSets 4.42 / 14.46 / 30.76、Fingerprint FFN 5.09 / 14.69 / 31.97、DeepSets + Fourier 6.56 / 16.46 / 33.46、**MIST 9.57 / 22.11 / 41.12**
- 確認（表 4、スペクトルの模擬。予測スペクトルとの類似で候補を並べた hit rate @1 / @5 / @20、%）:
  - 質量の候補: FFN Fingerprint 8.44 / 21.43 / 38.57、GNN 3.95 / 11.92 / 26.27、**FraGNNet 46.64 / 72.56 / 83.58**
  - **同じ分子式の候補（bonus）**: Precursor m/z 2.09 / 8.52 / 22.65、FFN Fingerprint 7.62 / 22.70 / 44.12、GNN 3.63 / 13.55 / 33.77、**FraGNNet 31.93 / 63.20 / 82.70**
- 確認（本文）: スペクトルの模擬の課題は [M+H]+ だけ、装置は QTOF・Orbitrap に絞る。表 3 の検索は [M+H]+ と [M+Na]+ を含む。**表 3 と表 4 は評価のスペクトルの集合が違う**ので、MIST 9.57 と FraGNNet 31.93 を同じ条件の比較とは言えない（推定。両表の差の大きさから、断片を列挙する方法が同じ式の候補で指紋の方法より強い、という向きは読める）
- 読み（推定）: 候補を同じ式に絞ると、Random は 0.37 → 3.06 と上がる（候補が減る）のに MIST は 14.64 → 9.57 と下がる。指紋の予測は、式の違う候補を落とす力の一部を「式」から得ていて、同じ式の中では弱くなる。FraGNNet（断片の確率分布 → 正確な質量のスペクトル）は 46.64 → 31.93 で、同じ式でも 3 割を 1 位にする
- 学習データ: MassSpecGym（MoNA・MassBank・GNPS・MSnLib 由来、MIT）。この競技で可（757307）
- この競技で使えるか: 方法の比べ方（同じ式に絞った順位）は使える（下の「v3_c1 で測る案」）。MassSpecGym で学習した模型を v3 で測ると、v3_c2n の正解の 58.5% が MassSpecGym にある（#66、`mist-massspecgym.md`）のでリークになる

### MetFrag 2.2（出典 2）
- 確認（要旨・表 1）: 473 スペクトル（Eawag の HR-MS/MS を統合）、候補は PubChem の質量検索（5 ppm）。**断片の説明だけ**で 1 位は MetFrag2.2 30/473（6.3%）、CFM-ID 43/473（9.1%）。上位 5 は 145 対 170、上位 10 は 226 対 232、順位の中央値 12 対 11。文献の数（references）と保持時間を足すと PubChem で 336/473（71%）、ChemSpider で 420/473（89%）。CFM-ID と組むと PubChem で 343
- 確認（本文）: 3 つの別のデータ（合計 824 スペクトル）でも、断片だけの 1 位は 2〜9%、等重みで全部入れると 64〜82%
- 確認（本文）: 立体異性体は InChIKey の第 1 ブロックで 1 つにまとめる（「MS/MS では普通は区別できない」）。terbutylazine と secbutylazine（同じ式の異性体）は断片がよく似る例として挙げ、保持時間などで分ける
- 異性体の区別の数値: **同じ式に絞った数値なし**。候補は質量の窓（5 ppm）で、同じ式の割合は書かれていない（未確認）
- 学習データ: 2.2 の断片の点は学習なし（結合の切断の組み合わせ）。重みは 3 つの点の組み合わせの比で、データで選んでいる（本文の三角図）
- ライセンス: 論文 CC BY 4.0。コードのライセンスは**未確認**（出典 3 の欄）
- この競技で使えるか: 考え方は縮小版の frag 段（MetFrag-lite `frag_rescore.py`、学習なし）として既に入っている。Java の本体は使わない（重ねる意味が無く、ライセンスも未確認）

### MetFrag の統計学習（出典 3）— fragnn にいちばん近い
- 確認（要旨）: 観測ピークの m/z（と中性損失）と、それを説明した断片の部分構造の組の頻度を学習データから数え、候補の点にする（擬似カウントと点の重みは「正しく 1 位の数」で選ぶ、と方法の節にある）。CASMI 2016 の 87 スペクトル（正・負イオン、学習に使った構造を参加者の間で外したもの）で、**1 位が 5 → 21、上位 10 が 39 → 55**（MetFrag の元の断片だけの提出との比較）。CASMI 2016 の勝者 CSI:IOKR より多い、負イオンでは参加者の中で最良、と主張
- 確認（本文）: 学習のスペクトルは MoNA と GNPS から、参加者が出した InChIKey で集めた
- 異性体の区別の数値: **同じ式に絞った数値なし**。CASMI 2016 の候補は主催者が配ったリスト（質量の窓と読める。同じ式の割合は未確認）
- ライセンス: 論文は BMC（CC BY と読めるが、ページの表記は読んでいない＝未確認）。コードは MetFragRelaunched の中（ライセンス未確認）
- この競技で使えるか: 重みも頻度の表も配布物を使わない。**同じ考え方（どのピーク・断片の説明を信じるかを学習で決める）を競技の train だけで自前に学習するのが fragnn**。数の大きさ（断片だけで 1 位 5 → 21/87）は、学習で断片の説明の重みを決めると、学習なしの断片の点より大きく伸びうることを示す（主張。データも候補の作りも違うので、v3 での大きさは分からない）
- 違い: MetFrag は頻度の表（ピークと部分構造の組の出現）で、新しい部分構造に弱い（推定）。fragnn は部分構造ではなく「何本切ったか・水素の移動・強度・質量の誤差」というピークごとの特徴を学習するので、表に無い組にも点が付く（設計の違い。効くかは未測定）

### MAGMa（出典 4）
- 確認（本文）: 候補を結合の切断で部分構造に分け、観測ピークを部分構造で説明できるかの罰則の和で並べる（学習なし）。CASMI 2013 のカテゴリー 2（構造）に 12 問を出し、**6 問で正解が 1 位、3 問で上位 5、3 問はそれより下**。同点は PubChem の物質数（refscore）で並べた。候補は PubChem の質量検索（10 ppm と 0.002 Da、問題によって 4 ppm）、一部の問題では与えられた分子式で取得
- 確認（本文）: 苦手な例は、断片のでき方に転位（rearrangement）が入るもの。許容 0.002 Da が厳しすぎて説明できないピークが残った例（後で 0.005 Da にして直った）
- 異性体の区別の数値: **同じ式に絞った数値なし**（12 問の順位だけ）
- 学習データ: なし（規則と罰則）
- ライセンス: コード Apache-2.0（確認）。論文は Mass Spectrometry 誌の開放の記事（ライセンス表記は読んでいない＝未確認）
- この競技で使えるか: 規則上は使える（学習なし、外部データなし）。ただし縮小版の MetFrag-lite と同じ型の点で、足しても証拠の種類は増えない（推定）。fragnn の入力の設計には「許容 0.005 Da」と「転位で説明できないピークが残る」ことの 2 点を引き継ぐ（fragnn は `frag_rescore` の許容 0.005 Da と水素の移動 −2〜+3 をそのまま使う設計で、#120 と一致）

### CFM-ID 4.0（出典 5）
- 確認（出典 5 の本文）: CASMI 2016 カテゴリー 3（Orbitrap のスペクトル）で「204 問のうち 162 を同定」。どの順位までを同定と数えたかは、この論文の本文に書かれていない（Anal. Chem. の論文は 403 で読めず、**未確認**）
- 確認（出典 5 の表 1 の本文）: HMDB 5.0 の 401（正）・237（負）化合物、候補は HMDB の予測スペクトル（同じ親イオン質量の候補が平均 11 個超）で、**1 位 48.1%（正）・38.4%（負）、上位 10 で 95.8%・93.7%**。PubChem の一部（約 210 万、候補は平均 4,096（正）・1,624（負））では **1 位 7.0%・6.3%**。学習の化合物は評価から外した、とある
- 異性体の区別の数値: **同じ式に絞った数値なし**（候補は同じ親イオン質量）
- 学習データ: CFM-ID の既定の学習は QTOF の [M+H]+ 12,165、[M−H]− 6,120 スペクトル（出典 5 の補足論文 PMC11521105 の序論、確認）。その出所が NIST17・METLIN を含むかは**未確認**（補足論文は「NIST17 の後に NIST20 で足されたスペクトル」を学習に無いものとして評価しているので、NIST17 が学習に入っていると読めるが、原論文で確かめていない）
- ライセンス: CFM-ID のコードは LGPL-2.1 と読める（フォークの LICENSE.md を検索結果で見ただけ＝**未確認**）。論文 CC BY 4.0
- この競技で使えるか: 配布の重みは、NIST 由来なら 750330 で不可（NIST 由来かは未確認なので、**使わない側に倒す**）。学習コードで競技の train から作り直すことは規則上は可だが、CFM の学習（EM による断片木の確率の学習）の時間はこの Mac で見積もっていない（未確認）。fragnn の方が小さく、同じ断片の列挙を共有できるので、CFM の作り直しは今は計画しない

### ICEBERG（出典 6）
- 確認（PDF の retrieval の段）: テストのスペクトルごとに、PubChem から正解と Tanimoto が最も高い「decoy」の**異性体**を最大 49 個取り、予測スペクトルとの類似で並べる。**同じ式の候補の中の順位**にあたる設定
- 主張: NPLIB1 の top-1 で次点に約 9 ポイント（相対で約 46%）の差。NIST20 の top-1 は MassFormer と同等。図の値そのものは読めていない（9 / 0.46 から、ICEBERG ≈ 29%、次点 ≈ 20% と逆算できるが**推定**）。CASMI22 の正イオン 303 で top-1 12.9%（#118 で読んだ付録 S1.1）
- 学習データ: 論文は NIST20 と NPLIB1。MassSpecGym で学習した重みの配布あり（確認、README）
- ライセンス: コード MIT、論文 CC BY-NC-SA 4.0（#118 の確認）
- この競技で使えるか: MassSpecGym 版は可（主催者が名指しで承認、`glacier-iceberg-provenance.md`）。0.487 系統にすでに入っていて、縮小版（v3）には入っていない。NIST20 で学習した重みは 750330 で不可

### FraGNNet（出典 7）
- 主張（#118 で読んだ表 2、NIST20、5 seed）: 正解と Tanimoto の高い PubChem の 49 分子を候補にした top-1 は、InChIKey 分割で FraGNNet-D4 36.0、ICEBERG 36.0、MassFormer 24.2。Scaffold 分割で 30.3、27.9、20.4。この候補が同じ式に限られるかは本文のその段に書かれていない（#118 の確認）
- 確認（出典 1 の表 4）: MassSpecGym で学習した FraGNNet は、同じ式の候補で hit@1 **31.93%**（上の MassSpecGym の欄）。**同じ式の数値がある断片の方法は、今回読んだ中でこれだけ**
- 確認（#118）: 学習コードに「分子式ごとの難しい負例のサンプラー」（`formula_hard_negative_dynamic_sampler`）がある
- 学習データ: 論文は NIST20、MassSpecGym 版は MassSpecGym の論文の中で学習されたもの。MassSpecGym 版の重みの配布は**未確認**（FraGNNet の README に記述なし）
- ライセンス: コード BSD-2-Clause、論文 CC BY 4.0
- この競技で使えるか: NIST20 の重みは 750330 で不可。MassSpecGym か競技の train で作り直せば規則上は可。ただし GPU の学習で、週 30 時間の枠と他の項目の兼ね合いがある。今の計画では fragnn（ローカル MPS、#120 の外挿 6.69 時間）を先にする

### MIST（出典 8）— 指紋の方法の比べ相手
- 確認（bioRxiv の本文）: 対照学習の MIST で top-1 37.39%、予測した指紋で直接引くと 31.70%、CSI:FingerID 38.24%（「retrospective dataset」、PubChem から候補）。候補が同じ式かは、Coley 研の再現の報告（出典 8 の補足）に「NPLIB1 の候補は PubChem の同じ分子式の構造」とある。論文の表 S2 そのものの設定は読めていない（**未確認**）
- 確認（出典 1 の表 3）: MassSpecGym の同じ式の候補で MIST は hit@1 9.57%
- 主張（出典 8 の補足）: 公開の MIST は検証の損失で選んだ早い epoch の重みで、指紋の精度の指標と検証の損失が一緒に動かない（BCE で scaffold 分割は epoch 5〜10 を選ぶ）
- 学習データ: MassSpecGym 版（`mist_msg_noaug.ckpt`）は MassSpecGym の実スペクトルだけ（`mist-massspecgym.md`）
- ライセンス: MIT。規則: 757307 で可
- この競技で使えるか: 規則上は可。ただし指紋の方法で、縮小版の FPNet と同じ種類の証拠。同じ式の中を分ける力は MassSpecGym の数で弱い（9.57%）。**fragnn と重ならない証拠を足す、という #120 の理由づけの裏付け**として使う

## 表にしない要約（同じ式の候補での 1 位）
- 同じ式に絞った数値がある: MassSpecGym bonus の Fingerprint FFN 5.09%、DeepSets 4.42%、MIST 9.57%（以上、指紋・スペクトルの埋め込み）、FraGNNet 31.93%（断片、別のスペクトルの集合）、ICEBERG（NPLIB1、図の値は未読、次点に +9 ポイント）
- 同じ式に絞った数値なし（質量の窓の候補）: MetFrag 2.2（断片だけで 6.3%）、MetFrag の統計学習（CASMI 2016 で 5 → 21/87）、MAGMa（CASMI 2013 で 6/12）、CFM-ID 4.0（HMDB で 48.1%、PubChem で 7.0%）
- 読み（推定）: (1) 同じ式の中では指紋の方法が弱く、断片を列挙して質量で合わせる方法が強い向きが、MassSpecGym の 2 つの表の両方で出る。(2) 学習なしの断片の点（MetFrag 2.2、MAGMa、縮小版の frag 段）は単独では弱いが、学習で重みを決めると大きく伸びた例がある（MetFrag の統計学習）。(3) v3_c1 の (c) 44 件は FPNet の段から 2 位以下が 39 件（#118）で、指紋が同じ式の中を分けられない、という (1) と同じ形をしている
- この 3 点は fragnn の設計（断片の質量だけで点、ピークごとの特徴を学習、FPNet に等重みで足す）と矛盾しない。ただし文献の設定（候補の数、装置、データ）は v3 と違い、**v3 での効果の大きさは文献からは決まらない**

## この競技で使えるかのまとめ
- 使う（考え方だけ、自前で作る）: MetFrag の統計学習の考え方（どの説明を信じるかを学習）→ fragnn（#120）。MassSpecGym の bonus の測り方（同じ式に絞った順位）→ 下の案
- 使える・すでに入っている: ICEBERG・GLACIER の MassSpecGym 版（0.487 系統）、MetFrag-lite（縮小版の frag 段）
- 規則上は作り直せば可、今は計画しない: FraGNNet（GPU の学習）、CFM-ID（学習時間が未確認、重みの出所が未確認）
- 使わない: 配布の重みが NIST 由来のもの（FraGNNet の NIST20 版、ICEBERG の NIST20 版、出所が未確認の CFM-ID の既定の重み）。MetFrag の Java 本体（ライセンス未確認、重ねる意味なし）
- v3 での測り方の注意: MassSpecGym で学習した模型は v3_c2n で正解の 58.5% を見ている（#66）ので、v3 の採点には使えない

## v3_c1 で測る案（#120 の事前登録に足す形。定数の掃引ではない）

**案: 同じ分子式に絞った順位（MassSpecGym の bonus と同じ形）で、fragnn の点そのものが異性体を分けるかを読む**

- 何のためか: #120 の採否は最終のリスト（score → fragnn → prior → frag）の MRR@25 で決まる。そこでは prior・frag 段・z 合成が混ざり、fragnn の点が**同じ式の異性体を分けたのか**、別の理由（式の違う候補を動かした、など）で動いたのかが分からない。文献の主張（同じ式の中で断片が指紋に勝つ）を v3_c1 で直接確かめる
- いつ・何を使うか: #120 の本番の学習のあと、#120 の v3 の採点と**同じ回に 1 回だけ**。#120 で作る点（fragnn・fragnnblind の窓の全候補の点）と、score 段の FPNet の点 s、既存の MetFrag-lite の `frag_rescore.score_molecule` の点 f（学習なし、`DEFAULT_CFG` のまま）だけを使う。新しい定数は入れない
- 定義:
  - 分子ごとに、score 段で点の付いた ±10 ppm 窓の全候補（InChIKey14 で重複を除く。正準化は `scripts/validate/mrr25.py` の `to_key`）のうち、**正解と同じ分子式（RDKit `CalcMolFormula`）の候補だけ**を集合 G とする。正解が窓に無い分子、|G| < 2 の分子は数から外し、外した数を記録する
  - 4 つの点で G の中を並べ、正解の順位を出す: (A) FPNet の s、(B) MetFrag-lite の f、(C) fragnn の n（スペクトルごとの logit の平均。#120 と同じ）、(D) fragnnblind の n。同点は悲観的（正解を同点の最後）に置く。点の付かない候補は最下位
  - 指標: 同じ式の中の逆数順位の平均（上限なし）と 1 位の割合。|G| の中央値も出す
  - 差: (C − A)、(C − B)、(C − D) を、分子の再抽出 2,000 回（seed 0、`bootstrap_diff` と同じ）の 95% 区間で
  - 追加の数え上げ: 対照（score+prior+frag）の (c) 44 件のうち新（score+fragnn+prior+frag）で 1 位になった数と、対照で 1 位だった 151 件のうち新で 1 位でなくなった数（正味の件数。#118 の「正味 12 件ほどで初めて見える」の確認）
- 読み方（結果の前に固定。**#120 の採否の規則と 0.487 系統へ進む条件は変えない**。この案は記録と次の設計のためだけ）:
  - (C − D) の下限 ≤ 0: fragnn の同じ式の中の区別は、スペクトルから来ていると言えない（#120 の blind の条件を、同じ式に絞って見たもの）
  - (C − B) の下限 > 0: 学習が、固定の MetFrag-lite より同じ式の中を分ける（MetFrag の統計学習と同じ向き）。上限 < 0: 学習の付加価値が無い。間は「判定なし」
  - (C − A) の下限 > 0: 文献と同じ向き（同じ式の中で断片が指紋に勝つ）。間・負は「判定なし」「逆」とそのまま書く
  - どの結果でも、重み・λ・候補の数を動かさない。次の設計（例: FPNet と等重みで足すのをやめて同じ式のグループの中だけに当てる、のような構造の変更）を考えるときの材料にだけ使い、それをするなら新しい Issue で v3 の別の回の前に事前登録する
- 測れる大きさ（推定）: 200 分子の 1 集合で、逆数順位の差の検出の目安は約 0.03〜0.05（#118 の MDE の見積もりと同じ型。同じ式に絞ると候補が減るので、ばらつきは別に出す）
- 必要な作業（#120 の採点のカードでする）: `scripts/local_eval/iso_rank_diag.py`（仮の名前）と合成データのテスト（同点・|G| < 2・正解が窓に無い・点の付かない候補）。PR で入れてから使う。費用は MetFrag-lite を窓の全候補で回す分（今の frag 段は上位 60 だけ。窓の候補の数は数えていない＝未測定）
- #120 への入れ方: この案を #120 のコメントに「追加の記録（採否に使わない）」として書き、#120 の採点の前に独立レビューで通す。#120 の本文の採否・止め方・帯は書き換えない

## 独立レビュー
（PR のコメントに記録する）

## 再現のコマンド
```
S=~/.hermes/profiles/massspec/cache/scratch
kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/750330
for r in NLeSC/MAGMa ipb-halle/MetFragRelaunched samgoldman97/mist coleygroup/MIST samgoldman97/ms-pred FraGNNet/fragnnet pluskal-lab/MassSpecGym; do
  gh api repos/$r -q '.license.spdx_id+" "+.pushed_at'
done
gh api repos/FraGNNet/fragnnet/readme -q .content | base64 -d | grep -in "massspecgym\|checkpoint\|weights"
# 論文: web で取得（arXiv 2410.23326 の PDF、PMC4732001、PMC6612146 と BMC の版、PMC4321337、PMC9252813、PMC11521105、arXiv 2304.13136v2 の PDF、bioRxiv 2022.12.30.522318 の PDF）
cat outputs/local_eval/{v3_c1,enveda_v1,v3_c2n}/result_score+prior.json outputs/local_eval/{v3_c1,enveda_v1,v3_c2n}/result_score+prior+frag.json
```
