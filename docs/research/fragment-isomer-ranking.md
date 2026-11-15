# 調査: 断片の説明で同じ分子式の異性体を区別する方法（#124）

計画 2026-11-16 夜 項目 5。親 #3、#108。前段 #118（`docs/research/isomer-rerank.md`、PR #119）、#120（自前の断片の並べ替え器 fragnn の事前登録、PR #121）。**読むだけ**: GPU 0、学習 0、提出 0、v3 の採点 0。2026-11-16 03:00〜04:30 JST に読んだ。

印の意味（#118 と同じ）: **確認** = 原文・コード・API の応答を読んだ。**主張** = 著者の申告で、こちらでは再現していない。**未確認** = 調べたが分からない（理由を書く）。**推定** = こちらの計算や読みで、測っていない。

## なぜ調べるか
- v3_c1（score+prior+frag、MRR@25 0.8535）の 1 位の外れ 49 のうち 44 は同じ分子式の異性体（#118）。上位 25 はほぼ同じ式の候補なので、分子式の絞り込みでは直せない
- #120 の fragnn は「候補を 1〜2 本の結合で切った断片の質量」と観測ピークの一致を学習で重みづけする。公開の方法が、同じ式の候補の中でどれだけ当てているか、指紋の方法と比べてどうかを数値で確かめ、設計の根拠にする
- 既存の数（クローンの `outputs/local_eval/<集合>/result_*.json`、確認）: 学習なしの MetFrag-lite の frag 段（上位 60 の同じ式のグループの中だけを並べ替える、`gl_fuse.rerank_multi`、λ 0.5）を足すと、score+prior → score+prior+frag は v3_c1 0.8219 → 0.8535、enveda_v1 0.8230 → 0.8376、v3_c2n 0.5873 → 0.6284。**ただし score+prior の値は 11/10 00:03〜00:05 の出力（#117 の drop_keys の作り直しの前）で、+frag の値は 11/15 23:22〜23:24（#117 の後）**。同じ除外での差ではないので、差（+0.032 / +0.015 / +0.041）は参考にとどめる。各集合 1 回の値で、区間は出していない

## 出典
1. MassSpecGym: Bushuiev et al., "MassSpecGym: A benchmark for the discovery and identification of molecules", NeurIPS 2024 Datasets & Benchmarks, arXiv:2410.23326v3 https://arxiv.org/abs/2410.23326 。PDF の本文と表 3（molecule retrieval）・表 4（spectrum simulation）を読んだ。コード https://github.com/pluskal-lab/MassSpecGym （GitHub API の license = MIT、最終 push 2026-05-08）
2. MetFrag 2.2: Ruttkies, Schymanski, Wolf, Hollender, Neumann, "MetFrag relaunched: incorporating strategies beyond in silico fragmentation", J. Cheminform. 8:3 (2016), doi:10.1186/s13321-016-0115-9 （PMC4732001、CC BY 4.0）。全文と表 1・表 2 を読んだ
3. MetFrag の統計学習: Ruttkies, Neumann, Posch, "Improving MetFrag with statistical learning of fragment annotations", BMC Bioinformatics (2019), doi:10.1186/s12859-019-2954-7 （PMC6612146、CC BY 4.0）。私は要旨と方法の前半を読んだ（PMC 版の取得は途中で切れた）。表 2 とライセンスは独立レビューが Europe PMC の全文で読んだもの（下の欄に「レビューの確認」と書く）。コード https://github.com/ipb-halle/MetFragRelaunched （GitHub API に license の項が無く、リポジトリの直下にも LICENSE のファイルが見当たらない）
4. MAGMa: Ridder, van der Hooft, Verhoeven, "Automatic Compound Annotation from Mass Spectrometry Data Using MAGMa", Mass Spectrom. (Tokyo) 2014; 3 (Spec Iss 2): S0033, doi:10.5702/massspectrometry.S0033 （書誌は独立レビューが本文の冒頭で確認）、PMC4321337 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4321337 。CASMI 2013 の節を読んだ。コード https://github.com/NLeSC/MAGMa （license = Apache-2.0、最終 push 2022-10-30）
5. CFM-ID 4.0: Wang, Allen, Tian, Oler, Gautam, Greiner ほか（Wishart を含む）, "CFM-ID 4.0 – a web server for accurate MS-based metabolite identification", Nucleic Acids Res. 50 (W1) (2022)、PMC9252813 https://pmc.ncbi.nlm.nih.gov/articles/PMC9252813/ （CC BY 4.0）。全文を読んだ。アルゴリズムの論文 Wang et al., Anal. Chem. 2021, doi:10.1021/acs.analchem.1c01465 は 403 で読めなかった（未確認）。学習データの補足として、CFM-ID 4.0 を学習に無いスペクトルで測った論文 PMC11521105 https://pmc.ncbi.nlm.nih.gov/articles/PMC11521105/ の要旨と序論を読んだ
6. ICEBERG: Goldman, Li, Coley, "Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks", arXiv:2304.13136v2 https://arxiv.org/abs/2304.13136 。PDF の retrieval の段を読み直した（#118 の出典 2 と同じ）。コード https://github.com/samgoldman97/ms-pred （MIT、最終 push 2026-11-03）。README に MassSpecGym で学習した ICEBERG・GLACIER の重みの配布がある（確認）
7. FraGNNet: Young et al., "FraGNNet: A Deep Probabilistic Model for Tandem Mass Spectrum Prediction", arXiv:2404.02360v2、TMLR 08/2025。数値は #118 の出典 3 で読んだもの（今回は読み直していない）。コード https://github.com/FraGNNet/fragnnet （BSD-2-Clause、最終 push 2026-11-13）。README に重みの配布・MassSpecGym の語は無い（`gh api …/readme` の grep で 0 件、確認）
8. MIST: Goldman et al., "Annotating metabolite mass spectra with domain-inspired chemical formula transformers", Nat. Mach. Intell. (2023), doi:10.1038/s42256-023-00708-3 。私が読んだのは bioRxiv 版の PDF https://www.biorxiv.org/content/10.1101/2022.12.30.522318v1 の retrieval の段（取得は 50,000 字で切れた）。9.6 節（Fingerprint retrieval）と表 S2 は独立レビューが PDF で読んだもの（下の欄に「レビューの確認」と書く）。コード https://github.com/samgoldman97/mist （MIT）。評価の設定の補足として、Coley 研の再現の報告 https://github.com/coleygroup/Audit_ML_MS_analysis/blob/main/DETAILED_RESULTS.md を読んだ
9. 競技の規則: 外部データの規則 750330（David Healey、2026-10-17、`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/750330`、今回 03:0x JST に取得）。主催者の回答 757307（MassSpecGym と MIST）・748882（第三者の重みの条件）は `docs/research/mist-massspecgym.md`・`isomer-rerank.md` の原文の記録を使った（今回は取り直していない）

## 規則（出典 9）
- 確認 750330: 非商用のライセンス（CC BY-NC など）のオープンなデータは使ってよい。オープンでない・誰でも自由に手に入らない・独占のデータは不可。**NIST は不可で、モデルの重みなど派生物も含む**（"NIST is not permitted ... the data including derivatives such as model weights cannot be redistributed"）
- 確認 757307（記録から）: 「MassSpecGym and MIST models are fine」
- 確認 748882（記録から）: 再現できない第三者の重みは不可。承認済みのチェックポイントは、学習コードと許可された学習データへの参照と、自前の微調整・推論の全コードを渡せば足りる
- したがって、各方法の判断は「(1) 重みが NIST 由来か、(2) 学習コードが公開か、(3) 許可されたデータ（競技の train、MassSpecGym など）で作り直せるか」で決まる

## 方法ごとの要点

### MassSpecGym の同じ分子式の設定（出典 1）
- 確認（本文）: 検索の候補は正解と同じ質量の分子で 1 スペクトル最大 256（多ければ無作為に取る）。"bonus chemical formulae challenge" は、分子式を与えて候補を**同じ分子式の分子だけ**にした設定（候補の取り方は同じ手順で、式が同じものから取る）
- 確認（表 3、分子の検索 = スペクトルから指紋などを予測して候補を引く方法。hit rate @1 / @5 / @20、%、99.9% の bootstrap 区間は省略）:
  - 質量の候補: Random 0.37 / 2.01 / 8.22、DeepSets 1.47 / 6.21 / 19.23、Fingerprint FFN 2.54 / 7.59 / 20.00、DeepSets + Fourier 5.24 / 12.58 / 28.21、**MIST 14.64 / 34.87 / 59.15**
  - **同じ分子式の候補（bonus）**: Random 3.06 / 11.35 / 27.74、DeepSets 4.42 / 14.46 / 30.76、Fingerprint FFN 5.09 / 14.69 / 31.97、DeepSets + Fourier 6.56 / 16.46 / 33.46、**MIST 9.57 / 22.11 / 41.12**
- 確認（表 4、スペクトルの模擬 = 候補の分子からスペクトルを予測し、観測との類似で並べる方法。hit rate @1 / @5 / @20、%）:
  - 質量の候補: Precursor m/z 0.38 / 1.72 / 7.17、FFN Fingerprint 8.44 / 21.43 / 38.57、GNN 3.95 / 11.92 / 26.27、**FraGNNet 46.64 / 72.56 / 83.58**
  - **同じ分子式の候補（bonus）**: Precursor m/z 2.09 / 8.52 / 22.65、FFN Fingerprint 7.62 / 22.70 / 44.12、GNN 3.63 / 13.55 / 33.77、**FraGNNet 31.93 / 63.20 / 82.70**
- 確認（本文）: スペクトルの模擬の課題は [M+H]+ だけ、装置は QTOF・Orbitrap に絞る。表 3 の検索は [M+H]+ と [M+Na]+ を含む。**表 3 と表 4 は評価のスペクトルの集合が違う**ので、表 3 の MIST 9.57 と表 4 の FraGNNet 31.93 は同じ条件の比較ではない
- 同じ表の中で言えること（確認した数から）: 表 4 の中では、同じ式の候補で、断片を列挙する FraGNNet（31.93）が、分子の指紋から模擬する FFN Fingerprint（7.62）や GNN（3.63）より大きく上。表 3 の中では、同じ式に絞ると Random は 0.37 → 3.06 と上がる（候補が減る）のに、MIST は 14.64 → 9.57 と下がる（推定: MIST の検索の力の一部は、式の違う候補を落とす力）
- 言えないこと: 表 3 には断片の方法が無いので、「指紋を予測して引く方法」と「断片の方法」の優劣は、この論文の中では同じ条件で比べられない
- 学習データ: MassSpecGym（MoNA・MassBank・GNPS・MSnLib 由来、MIT）。この競技で可（757307）
- この競技で使えるか: 方法の比べ方（同じ式に絞った順位）は使える（下の「v3_c1 で測る案」）。MassSpecGym で学習した模型を v3 で測ると、v3_c2n の正解の 58.5% が MassSpecGym にある（#66、`mist-massspecgym.md`）のでリークになる

### MetFrag 2.2（出典 2）
- 確認（本文・表 1「Comparison of in silico fragmentation results for 473 Eawag Orbitrap spectra (formula search)」）: 473 スペクトル（[M+H]+ 319、[M−H]− 154）。**候補は正しい分子式で取った**（"The correct molecular formula was used to retrieve candidates"）＝**同じ分子式の候補の中の順位**。立体異性体は InChIKey の第 1 ブロックで 1 つにまとめ、同点は悲観的な順位（正解を同点の最後）で数える
- 確認（表 1、断片の説明だけ、1 位の数）: MetFrag2010（ChemSpider）73（15%）、**MetFrag2.2 は ChemSpider で 105（22%）、PubChem で 30（6%）**、CFM-ID（PubChem）43（9%）、MetFrag2.2 + CFM-ID（PubChem）62（13%）。PubChem の上位 5 は MetFrag2.2 145、CFM-ID 170。候補の数は PubChem の方が多い（推定: 1 位の割合の差の大半は候補の数の差）
- 確認（本文）: 文献の数（references）と保持時間を足すと 1 位は PubChem 336（71%）、ChemSpider 420（89%）、CFM-ID と組むと PubChem 343。ただしこの重みは同じデータの上で組み合わせを選んだ最良値（楽観的）で、3 つの別のデータ（合計 824 スペクトル）での確かめでは、断片だけの 1 位 2〜9%、等重みで全部入れて 64〜82%
- 確認（本文）: terbutylazine と secbutylazine（同じ式の異性体）は断片がよく似る例として挙げ、保持時間などで分ける
- 異性体の区別の数値: **あり**（上の表 1。同じ式の候補、断片だけで 1 位 6〜22%）
- 学習データ: 断片の点は学習なし（結合の切断の組み合わせ）
- ライセンス: 論文 CC BY 4.0。コードのライセンスは**未確認**（出典 3 の欄）
- この競技で使えるか: 考え方は縮小版の frag 段（MetFrag-lite `frag_rescore.py`、学習なし）として既に入っている。Java の本体は使わない（重ねる意味が無く、ライセンスも未確認）
- 読み（推定）: 学習なしの断片の点だけでは、同じ式の候補の中の 1 位は 1〜2 割にとどまる。縮小版の frag 段が FPNet の点に足す形（z 合成、上位 60 の同じ式のグループの中）で使っているのは、この弱さに合っている

### MetFrag の統計学習（出典 3）— fragnn にいちばん近い
- 確認（要旨）: 観測ピークの m/z（と中性損失）と、それを説明した断片の部分構造の組の頻度を学習データから数え、候補の点に足す（擬似カウントと点の重みは「正しく 1 位の数」で選ぶ、と方法の節にある）。CASMI 2016 の 87 スペクトル（正・負イオン、学習に使った構造を参加者の間で外したもの）で、**1 位が 5 → 21、上位 10 が 39 → 55**（MetFrag の元の提出との比較）。負イオンでは参加者の中で最良、と主張
- レビューの確認（表 2、同じ 87 スペクトル・主催者の候補リスト、1 位の数）: MetFrag 2.4.5 21、MetFrag 2.3 5、CFM-ID 15、CSI:IOKR 16、MAGMa 6、MS-Finder 10。21 は MetFrag の断片の点に統計の 2 項（ピーク・損失）を足した**合成の点**で、統計の項だけの点ではない
- 確認（本文）: 学習のスペクトルは MoNA と GNPS から、参加者が出した InChIKey で集めた
- 異性体の区別の数値: **同じ式に絞った数値なし**。CASMI 2016 の候補は主催者が配ったリスト（同じ式の割合は未確認）
- ライセンス: 論文 CC BY 4.0（レビューの確認）。コードは MetFragRelaunched の中（ライセンス未確認）
- この競技で使えるか: 重みも頻度の表も配布物を使わない。**同じ考え方（どのピーク・断片の説明を信じるかを学習で決める）を競技の train だけで自前に学習するのが fragnn**。数の大きさ（同じ 87 で 5 → 21、ほかの方法 6〜16）は、断片の説明に学習した項を足すと伸びうることを示す（主張。データも候補の作りも違うので、v3 での大きさは分からない）
- 違い: MetFrag は頻度の表（ピークと部分構造の組の出現）で、表に無い部分構造には点が付かない（推定）。fragnn は部分構造ではなく「何本切ったか・水素の移動・強度・質量の誤差」というピークごとの特徴を学習する（設計の違い。効くかは未測定）

### MAGMa（出典 4）
- 確認（本文）: 候補を結合の切断で部分構造に分け、観測ピークを部分構造で説明できるかの罰則の和で並べる（学習なし）。CASMI 2013 のカテゴリー 2（構造）に 12 問を出し、**6 問で正解が 1 位、3 問で上位 5、3 問はそれより下**
- 確認（本文）: 候補は PubChem のローカルの写し（2013-12-02、1,200 Da まで、C・H・N・O・P・S）から取った。ピークの m/z の一致の許容は 10 ppm と 0.002 Da の大きい方で、課題 4〜6 は候補を取る質量の許容を 4 ppm にした。課題 7・8 は与えられた元素組成で PubChem から取った。課題 1・2・14 は PubChem の物質数（refscore）5 以上に絞り、同点は断片の点と refscore の「refined ranking」で並べた
- 確認（本文）: 苦手な例は、断片のでき方に転位（rearrangement）が入るもの。許容 0.002 Da が厳しすぎて説明できないピークが残った例（後で 0.005 Da にして直った）
- 異性体の区別の数値: **同じ式に絞った数値なし**（12 問の順位だけ）
- 学習データ: なし（規則と罰則）
- ライセンス: コード Apache-2.0（確認）。論文のライセンス表記は読んでいない（未確認）
- この競技で使えるか: 規則上は使える（学習なし、外部データなし）。ただし縮小版の MetFrag-lite と同じ型の点で、足しても証拠の種類は増えない（推定）。fragnn の入力の設計には「許容 0.005 Da」と「転位で説明できないピークが残る」ことの 2 点を引き継ぐ（fragnn は `frag_rescore` の許容 0.005 Da と水素の移動 −2〜+3 をそのまま使う設計で、#120 と一致）

### CFM-ID 4.0（出典 5）
- 確認（出典 5 の本文）: CASMI 2016 カテゴリー 3（Orbitrap のスペクトル）で「204 問のうち 162 を同定」。どの順位までを同定と数えたかは、この論文の本文に書かれていない（Anal. Chem. の論文は 403 で読めず、**未確認**）
- 確認（出典 5 の表 1 の本文）: HMDB 5.0 の 401（正）・237（負）化合物、候補は HMDB の予測スペクトル（同じ親イオン質量の候補が平均 11 個超）で、**1 位 48.1%（正）・38.4%（負）、上位 10 で 95.8%・93.7%**。PubChem の一部（約 210 万、候補は平均 4,096（正）・1,624（負））では **1 位 7.0%・6.3%**。学習の化合物は評価から外した、とある
- レビューの確認（出典 5 の表 1）: HMDB での 1 位の候補の分子式が正しい割合は 98.2%（正）・97.8%（負）。つまり HMDB での 1 位の外れは、ほぼ同じ式の中で起きている（推定）
- 確認（出典 2 の表 1）: 同じ式の候補（PubChem、473 スペクトル）で CFM-ID は 1 位 43（9%）。この CFM-ID は MetFrag 2.2 の論文の時点の版（4.0 ではない。版の番号は未確認）
- 異性体の区別の数値: 4.0 では**同じ式に絞った数値なし**（候補は同じ親イオン質量）。前の版は上の 43/473
- 学習データ: CFM-ID の既定の学習は QTOF の [M+H]+ 12,165、[M−H]− 6,120 スペクトル（出典 5 の補足論文 PMC11521105 の序論、確認）。その出所が NIST17・METLIN を含むかは**未確認**（補足論文は「NIST17 の後に NIST20 で足されたスペクトル」を学習に無いものとして評価しているので、NIST17 が学習に入っていると読めるが、原論文で確かめていない）
- ライセンス: CFM-ID のコードは LGPL-2.1 と読める（フォークの LICENSE.md を検索結果で見ただけ＝**未確認**）。論文 CC BY 4.0
- この競技で使えるか: 配布の重みは、NIST 由来なら 750330 で不可（NIST 由来かは未確認なので、**使わない側に倒す**）。学習コードで競技の train から作り直すことは規則上は可だが、CFM の学習（EM による断片木の確率の学習）の時間はこの Mac で見積もっていない（未確認）。fragnn の方が小さく、同じ断片の列挙を共有できるので、CFM の作り直しは今は計画しない

### ICEBERG（出典 6）
- 確認（PDF の retrieval の段）: テストのスペクトルごとに、PubChem から正解と Tanimoto が最も高い「decoy」の**異性体**を最大 49 個取り、予測スペクトルとの類似で並べる。**同じ式の候補の中の順位**にあたる設定
- 主張（要旨）: NPLIB1 の top-1 で **29% 対 次点 20%**（相対で 46% の改善）。NIST20 の top-1 は MassFormer と同等。CASMI22 の正イオン 303 で top-1 12.9%（#118 で読んだ付録 S1.1）
- 学習データ: 論文は NIST20 と NPLIB1。MassSpecGym で学習した重みの配布あり（確認、README）
- ライセンス: コード MIT、論文 CC BY-NC-SA 4.0（#118 の確認）
- この競技で使えるか: MassSpecGym 版は可（主催者が名指しで承認、`glacier-iceberg-provenance.md`）。0.487 系統にすでに入っていて、縮小版（v3）には入っていない。NIST20 で学習した重みは 750330 で不可

### FraGNNet（出典 7）
- 主張（#118 で読んだ表 2、NIST20、5 seed）: 正解と Tanimoto の高い PubChem の 49 分子を候補にした top-1 は、InChIKey 分割で FraGNNet-D4 36.0、ICEBERG 36.0、MassFormer 24.2。Scaffold 分割で 30.3、27.9、20.4。この候補が同じ式に限られるかは本文のその段に書かれていない（#118 の確認）
- 確認（出典 1 の表 4）: MassSpecGym で学習した FraGNNet は、同じ式の候補で hit@1 **31.93%**（上の MassSpecGym の欄、[M+H]+ だけの集合）
- 確認（#118）: 学習コードに「分子式ごとの難しい負例のサンプラー」（`formula_hard_negative_dynamic_sampler`）がある
- 学習データ: 論文は NIST20、MassSpecGym 版は MassSpecGym の論文の中で学習されたもの。MassSpecGym 版の重みの配布は**未確認**（FraGNNet の README に記述なし）
- ライセンス: コード BSD-2-Clause、論文 CC BY 4.0
- この競技で使えるか: NIST20 の重みは 750330 で不可。MassSpecGym か競技の train で作り直せば規則上は可。ただし GPU の学習で、週 30 時間の枠と他の項目の兼ね合いがある。今の計画では fragnn（ローカル MPS、#120 の外挿 6.69 時間）を先にする

### MIST（出典 8）— 指紋の方法の比べ相手
- 確認（bioRxiv の本文）: top-1 は MIST の対照学習で 37.39%、予測した指紋で直接引くと 31.70%、CSI:FingerID 38.24%（「retrospective dataset」）
- レビューの確認（bioRxiv の 9.6 節・表 S2）: 検索は「スペクトルの分子式に合う PubChem の全異性体」を cosine で並べる＝**同じ式の候補の中の順位**。37.39 は MIST (5x ensemble) + Contrastive（単体の contrastive は 34.45）、31.70 は MIST (5x) + Cosine。同点は楽観的に扱う。評価のデータは NIST・MoNA・GNPS から作った 31,145 スペクトルを 3 つの構造の分割で平均したもの（NPLIB1 ではない）
- 確認（出典 1 の表 3）: MassSpecGym の同じ式の候補で MIST は hit@1 9.57%
- 読み（推定）: 同じ指紋の方法でも、同じ式の中の 1 位は 9.57%（MassSpecGym、同点の扱いは MassSpecGym の定義）から 37.39%（論文のデータ、5x ensemble、楽観的な同点、NIST を含む）まで、データと評価の作りで大きく違う。**指紋の方法が同じ式の中で一律に弱い、とは言えない**
- 主張（出典 8 の補足）: 公開の MIST は検証の損失で選んだ早い epoch の重みで、指紋の精度の指標と検証の損失が一緒に動かない（BCE で scaffold 分割は epoch 5〜10 を選ぶ）
- 学習データ: 論文の重みは NIST を含むデータ。MassSpecGym 版（`mist_msg_noaug.ckpt`）は MassSpecGym の実スペクトルだけ（`mist-massspecgym.md`）
- ライセンス: MIT。規則: MassSpecGym 版は 757307 で可。論文の重み（NIST を含む）は 750330 で不可
- この競技で使えるか: MassSpecGym 版は規則上は可。ただし指紋の方法で、縮小版の FPNet と同じ種類の証拠。fragnn の「FPNet と重ならない証拠」という理由づけは、MIST の数ではなく、証拠の種類（指紋か断片の質量か）の違いに置く

## 表にしない要約（同じ式の候補での 1 位）
- 同じ式に絞った数値がある:
  - 指紋・埋め込みで引く方法: MassSpecGym bonus の Fingerprint FFN 5.09%、DeepSets 4.42%、MIST 9.57%。MIST の論文のデータでは MIST (5x) + Contrastive 37.39%、CSI:FingerID 38.24%（楽観的な同点）
  - 分子から模擬して比べる方法: MassSpecGym bonus（[M+H]+ だけの別の集合）の FraGNNet 31.93%、FFN Fingerprint 7.62%、GNN 3.63%。ICEBERG は NPLIB1 の 49 異性体で 29%（次点 20%）
  - 学習なしの断片の点: MetFrag 2.2 は 473 スペクトルで PubChem 30（6%）、ChemSpider 105（22%）。同じ表で CFM-ID（前の版）43（9%）
- 同じ式に絞った数値なし（質量の窓や主催者の候補）: MetFrag の統計学習（CASMI 2016 で 5 → 21/87、同じ 87 でほかの方法 6〜16）、MAGMa（CASMI 2013 で 6/12）、CFM-ID 4.0（HMDB で 48.1%、PubChem で 7.0%）
- 読み（推定）:
  1. 同じスペクトルの集合の中で比べられる表 4（MassSpecGym）では、断片を列挙して正確な質量で合わせる FraGNNet が、分子の指紋や GNN から模擬する方法より同じ式の候補で大きく上。ただし「指紋を予測して引く方法」と「断片の方法」の優劣は、文献からは同じ条件で決まらない（MIST は 9.57〜37.39 と設定で大きく違う）
  2. 学習なしの断片の点（MetFrag 2.2、MAGMa、縮小版の frag 段）は単独では同じ式の中で 1〜2 割の 1 位。断片の説明に学習した項を足すと大きく伸びた例がある（MetFrag の統計学習、5 → 21/87）
  3. v3_c1 の (c) 44 件は FPNet の段から 2 位以下が 39 件（#118）で、FPNet の点が同じ式の中を分けられていない
- fragnn の設計（断片の質量だけで点、ピークごとの特徴を学習、FPNet に等重みで足す）との関係: 2 と 3 は設計と合う。1 からは「断片の方が指紋より強い」とまでは言えないので、**fragnn の根拠は「指紋と種類の違う証拠を、学習で重みづけして足す」ことに置き、断片の方が強いとは前提にしない**。文献の設定（候補の数、装置、データ、同点の扱い）は v3 と違い、v3 での効果の大きさは文献からは決まらない

## この競技で使えるかのまとめ
- 使う（考え方だけ、自前で作る）: MetFrag の統計学習の考え方（どの説明を信じるかを学習）→ fragnn（#120）。MassSpecGym の bonus・MetFrag 2.2 の表 1 の測り方（同じ式に絞った順位）→ 下の案
- 使える・すでに入っている: ICEBERG・GLACIER の MassSpecGym 版（0.487 系統）、MetFrag-lite（縮小版の frag 段）
- 規則上は作り直せば可、今は計画しない: FraGNNet（GPU の学習）、CFM-ID（学習時間が未確認、重みの出所が未確認）、MIST の MassSpecGym 版（指紋の方法で FPNet と同じ種類）
- 使わない: 配布の重みが NIST 由来のもの（FraGNNet・ICEBERG の NIST20 版、MIST の論文の重み、出所が未確認の CFM-ID の既定の重み）。MetFrag の Java 本体（ライセンス未確認、重ねる意味なし）
- v3 での測り方の注意: MassSpecGym で学習した模型は v3_c2n で正解の 58.5% を見ている（#66）ので、v3 の採点には使えない

## v3_c1 で測る案（#120 の事前登録に足す形。定数の掃引ではない）

**案: 同じ分子式に絞った順位（MassSpecGym の bonus・MetFrag 2.2 の表 1 と同じ形）で、fragnn の点そのものが異性体を分けるかを読む**

- 何のためか: #120 の採否は最終のリスト（score → fragnn → prior → frag）の MRR@25 で決まる。そこでは prior・frag 段・z 合成が混ざり、fragnn の点が**同じ式の異性体を分けたのか**、別の理由（式の違う候補を動かした、など）で動いたのかが分からない。文献の読み（学習した断片の項が同じ式の中で効く）を v3_c1 で直接確かめる
- いつ・何を使うか: #120 の本番の学習のあと、#120 の v3 の採点と**同じ回に 1 回だけ**。#120 で作る点（fragnn・fragnnblind の窓の全候補の点）と、score 段の FPNet の点 s、既存の MetFrag-lite の `frag_rescore.score_molecule` の点 f（学習なし、`DEFAULT_CFG` のまま）だけを使う。どれも v3_c1 の正解を学習で見ていない（FPNet は fold 0 の fold-safe、MetFrag-lite は学習なし、fragnn は #120 の DROP で v3 の正解を外す）。新しい定数は入れない
- 測る集合: **v3_c1 だけ**で読む。ほかの 3 集合（v3_c2・enveda_v1・v3_c2n）は同じ定義で数を記録するだけで、読み方には使わない
- 定義:
  - 分子ごとに、score 段で点の付いた ±10 ppm 窓の全候補（InChIKey14 で重複を除く。正準化は `scripts/validate/mrr25.py` の `to_key`）のうち、**正解と同じ分子式（RDKit `CalcMolFormula`）の候補だけ**を集合 G とする。分子式が作れない候補（RDKit で読めない）は G から外し、その数を記録する。電荷を持つ候補も式のとおりに扱う（式が正解と同じなら G に入る）
  - 正解が窓に無い分子、|G| < 2 の分子は数から外し、外した数を記録する
  - 4 つの点で G の中を並べ、正解の順位を出す: (A) FPNet の s、(B) MetFrag-lite の f、(C) fragnn の n（スペクトルごとの logit の平均。#120 と同じ）、(D) fragnnblind の n。同点は悲観的（正解を同点の最後）。点の付かない候補（RDKit で読めない・使えるスペクトルが無い）は点のある候補の下に置き、その中も悲観的に扱う（正解に点が付かなければ G の最下位）
  - **主の指標**: G の中の逆数順位の平均（上限なし）。1 位の割合と |G| の中央値は記録だけ
  - 差: (C − A)、(C − B)、(C − D) の主の指標の差を、分子の再抽出 2,000 回（seed 0、`bootstrap_diff` と同じ）の 95% 区間で。3 つは個別の区間で、多重性の補正はしない（どれも採否に使わないため）
  - 追加の数え上げ: 対照（score+prior+frag）の (c) 44 件のうち新（score+fragnn+prior+frag）で 1 位になった数と、対照で 1 位だった 151 件のうち新で 1 位でなくなった数（正味の件数。#118 の「正味 12 件ほどで初めて見える」の確認）
- 読み方（結果の前に固定。**#120 の採否の規則と 0.487 系統へ進む条件は変えない**。この案は記録と次の設計のためだけ）:
  - (C − D) の下限 ≤ 0: fragnn の同じ式の中の区別は、スペクトルから来ていると言えない（#120 の blind の条件を、同じ式に絞って見たもの）
  - (C − B) の下限 > 0: 学習が、固定の MetFrag-lite より同じ式の中を分ける（MetFrag の統計学習と同じ向き）。上限 < 0: 学習の付加価値が無い。間は「判定なし」
  - (C − A) の下限 > 0: 断片の学習した点が、同じ式の中で FPNet より分ける。上限 < 0: FPNet の方が分ける。間は「判定なし」。文献は優劣を決めていない（上の要約の 1）ので、どちらの結果も「予想どおり」とは書かない
  - どの結果でも、重み・λ・候補の数を動かさない。次の設計（例: FPNet と等重みで足すのをやめて同じ式のグループの中だけに当てる、のような構造の変更）を考えるときの材料にだけ使い、それをするなら新しい Issue で v3 の別の回の前に事前登録する
- 測れる大きさ（推定）: 200 分子の 1 集合で、逆数順位の差の検出の目安は約 0.03〜0.05（#118 の MDE の見積もりと同じ型。同じ式に絞ると候補が減るので、ばらつきは別に出す）
- 必要な作業（#120 の採点のカードでする）: `scripts/local_eval/iso_rank_diag.py`（仮の名前）と合成データのテスト（同点・|G| < 2・正解が窓に無い・点の付かない候補・分子式が作れない候補）。PR で入れてから使う。費用は MetFrag-lite を窓の全候補で回す分（今の frag 段は上位 60 だけ。窓の候補の数は数えていない＝未測定）
- #120 への入れ方: この案を #120 のコメントに「追加の記録（採否に使わない）の案」として書き、#120 の採点の前に独立レビューで通す。#120 の本文の採否・止め方・帯は書き換えない

## 独立レビュー
- 1 回目（Opus、作業に関わっていない delegate、読むだけ）: **FAIL**（major 2、minor 6、info 2）。MassSpecGym の表 3・4、MetFrag の統計学習の 5 → 21、MAGMa の 6/12、CFM-ID 4.0 の数、ICEBERG の設定、MIST の 37.39 / 31.70 / 38.24、手元の MRR、750330 の原文、各リポジトリのライセンスは一致と確認
  - major: MetFrag 2.2 の表 1 は同じ分子式の候補（formula search）だった → 「同じ式の数値あり」に移し、ChemSpider 105/473 と CFM-ID 43/473 を足した。5 ppm は候補の窓ではなかった
  - major: 「同じ式の中では指紋が弱く断片が強い」は言い過ぎ（表 3 に断片の方法が無い、MIST の論文の検索も同じ式で 37.39%）→ 読みを表 4 の中の比較に弱め、MIST の同じ式の設定を足し、fragnn の根拠を「種類の違う証拠を学習で足す」に置き直した
  - minor: ICEBERG の 29% 対 20% は要旨にある → 直した／MIST の 5x ensemble・データ（NIST を含む）・同点 → 足した／MetFrag の統計学習の表 2 とライセンス、21 が合成の点 → 足した／MAGMa の許容の読み違いと書誌 → 直した／score+prior の値は #117 の前の出力 → 明記した／測る案の主の指標・多重性・集合・分子式の作れない候補・点の無い候補の同点 → 固定した／表 4 の Precursor m/z の行・候補の上限 256 → 足した
  - info: CFM-ID 4.0 の表 1 で 1 位の分子式が正しい割合 98.2% / 97.8% → 足した


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
