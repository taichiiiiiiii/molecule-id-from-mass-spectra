# 調査: 候補 B（BDE で重みをつけた断片の点）に使う BDE 表の出典・ライセンス・被覆

- Issue: #184（計画 2026-11-22 夜 項目 4、カード t_34c05267）。親 #3、関連 #178（`train-independent-evidence.md` §4.2・§5）
- 書いた日: 2026-11-22（JST）。**MRR の採点はしていない**。v3 の採否はこの文書では決めない。GPU 0、提出 0
- 印: 【未確認】= 出典か手元の数で確かめていない主張。【推測】= 自分の解釈。【未読】= 読んでいない部分
- ライセンスの読みは法律の助言ではない。下の「転載の条件」は各ライセンスの本文の要約で、迷う点は【判断待ち】と書いた

## 0. なぜ調べるか

候補 B（#178 §4.2）は MetFrag 2.2 の式 2 の形で、合った断片の切った結合の BDE（結合解離エネルギー）の和で点を割る。点は候補の構造・問い合わせのスペクトル・BDE の表だけから出るが、**表の出典とライセンスが決まらないので事前登録できない**（#178 §4.2 の「足りないもの (1)(2)」）。MetFrag の `BondEnergies.java` は、リポジトリにライセンスのファイルが無いので写さない。この文書は (1) 表の候補を 5 つ調べ、(2) v3 の 4 集合の窓の候補の構造で「表に値のある結合の割合」を測り、(3) 推奨の表と表に無い結合の規則と、候補 B の事前登録の下書きを書く。

## 1. 候補の表（この回に読んだもの）

取り方: Wikipedia は `action=raw` のウィキテキストを、OpenStax と LibreTexts は HTML を curl で落とし、表の `<td>` をスクリプトで読んだ（scratch に置き、リポジトリには入れない）。MetFrag は GitHub API で `BondEnergies.java` を読んだ。BDE-db2 は figshare から csv.gz を落とした。**スクリプト `scripts/research/bde_coverage.py` には表のキー（元素の組 × 次数）だけを入れ、BDE の値は入れていない**。キーは落とした原本と照合し、T1・T3・T4・T5 は一致を確かめた（T5 は実行時に `--bdedb2` で毎回照合）。T2 は 13 項目を目で数えた。

### T1: OpenStax *Chemistry*（初版 2015）Table 7.2 "Bond Energies (kJ/mol)"
- URL: https://openstax.org/books/chemistry/pages/7-5-strengths-of-ionic-and-covalent-bonds （著者 Flowers, Robinson, Langley, Theopold、OpenStax、2015-03-11）。同じ表が 2e（https://openstax.org/books/chemistry-2e/pages/7-5-strengths-of-ionic-and-covalent-bonds 、2019）と、LibreTexts の写し「5.8: Bond Energies」（https://chem.libretexts.org/Courses/University_of_Toronto/Chemistry:_Physical_Principles/05:_Thermochemistry/5.08:_Bond_Energies 、2026-10-26 更新）にもある。初版と 2e の表は 68 項目のキーと値がすべて同じ（原本の 2 つの HTML を照合）
- ライセンス（確認）:
  - 初版: ページの書籍のメタデータが `http://creativecommons.org/licenses/by/4.0/`、本文の脚注が「Textbook content produced by OpenStax is licensed under a Creative Commons Attribution License」。**CC BY 4.0**
  - 2e: メタデータが `by-nc-sa/4.0`、脚注も CC BY-NC-SA。**CC BY-NC-SA 4.0**（使わない）
  - LibreTexts 5.8: 「shared under a CC BY 4.0 license and was authored, remixed, and/or curated by OpenStax」（タグ `license:ccby`、`licenseversion:40`）
  - **注意（確認）**: OpenStax の初版・2e のページには「This book may not be used in the training of large language models or otherwise be ingested into large language models or generative AI offerings without OpenStax's prior written permission.」とある。LibreTexts 5.8 のページにはこの文は見当たらなかった。この回の作業では OpenStax のページをスクリプトで読み、表のキーと値がエージェント（LLM）の文脈に入った。CC BY 4.0 の許諾とこの文がどう両立するかは【未確認】【判断待ち】（§4.3）
- 転載の条件（CC BY 4.0）: 表示（作者・題名・URL・ライセンスへのリンク・変更したならその旨）だけ。継承（SA）も非営利（NC）も無い。リポジトリに値のファイルを入れるなら、そのファイルの先頭に出典とライセンスを書けばよい
- 収録の範囲: 68 項目、すべて単・二重・三重の区別つき、芳香族の項目は無い。重原子どうしで二重・三重があるのは C=C・C≡C・C=N・C≡N・C=O・C≡O・N=N・N≡N・O=O だけ。**S=O・P=O・N=O・C=S は無い**、N–S・O–S（単）も無い。H–X は 11
- 元の数値の出典はページに書かれていない【未確認】。値は「平均の結合エネルギー」（多くの分子の平均）で、特定の結合の BDE ではない（本文）

### T2: Wikipedia "Bond dissociation energy" §Representative bond enthalpies
- URL: https://en.wikipedia.org/wiki/Bond_dissociation_energy （revid 1368968041、2026-08-12T03:24:24Z）。"Bond energy" の記事（2026-11-19 版）には表が無い
- ライセンス: Wikipedia の本文は **CC BY-SA 4.0**（Wikipedia の利用規約の一般の表示。このページ固有には確かめていない【未確認】）
- 転載の条件（CC BY-SA 4.0）: 表示に加えて、**表を改変して配布するもの（値のファイル）は CC BY-SA 4.0（か互換のライセンス）で出す**（継承）。同じファイルに別の制限を足せない。値のファイルをリポジトリに入れると、そのファイルは BY-SA になる。コード本体まで BY-SA になるかは組み合わせ方による【未確認】
- 収録の範囲: 2 つの表で 13 キー。重原子どうしは C–C・C–F・C–O・C≡O・C=O（CO2 と ホルムアルデヒド）・O=O・N≡N・C=C・C≡C の 9 つ。**C–N も C=N も無い**。値は特定の分子の BDE（例 C–O はメタノール）で、平均ではない

### T3: Huheey, Keiter & Keiter, *Inorganic Chemistry* 4th ed. (1993) の値（LibreTexts の Tro の Map 9.10 Table 1 経由）
- URL: https://chem.libretexts.org/Bookshelves/General_Chemistry/Map:_A_Molecular_Approach_(Tro)/09:_Chemical_Bonding_I-_Lewis_Structures_and_Determining_Molecular_Shapes/9.10:_Bond_Energies_and_Bond_Lengths 。表の下に「Source: Data from J. E. Huheey, E. A. Keiter, and R. L. Keiter, Inorganic Chemistry, 4th ed. (1993).」。原本の教科書は読んでいない【未読】
- ライセンス: LibreTexts のページは **CC BY-NC-SA 4.0**（「shared under a CC BY-NC-SA 4.0 license」）。元の値は市販の教科書（著作権あり）から
- 転載の条件: 非営利に限る・同じライセンスで継承・表示。賞金のある Kaggle の競技の提出物が「非営利」に当たるかは【未確認】。競技の勝者のライセンスの条件はこのリポジトリで確かめていない【未確認】。使わない側に倒す
- 収録の範囲: 58 キー。T1 に無い **N=O・S=O** がある。P=O・C=S・N–S・O–S は無い。値には「≈」つきのものがある

### T4: MetFrag（MetFragRelaunched）`BondEnergies.java` の既定の表 — 参照だけ、写さない
- URL: https://github.com/ipb-halle/MetFragRelaunched/blob/master/MetFragLib/src/main/java/de/ipbhalle/metfraglib/additionals/BondEnergies.java （最後の変更 commit 659ca873、2016-11-10）
- ライセンス: GitHub API でリポジトリのライセンスは 404（ファイルが無い）、README.md・pom.xml にもライセンスの記述が無い（確認）。MetFrag 2.2 の論文（doi:10.1186/s13321-016-0115-9）は「GNU LGPL version 2.1 or later」と書く（#178 §1.3）が、このリポジトリに及ぶかは【未確認】。ファイルに値の出典の注記は無い。表に無い結合には `Constants.DEFAULT_BOND_ENERGY`（C–C の値、とコメントにある）を返す
- 扱い: **値は写さない**。被覆の比較のためにキーの集合（どの結合に値があるか）だけをスクリプトに入れた（値は持たない）。キーの集合が著作物に当たるかは【推測】で、当たらないと読んだ
- 収録の範囲: 77 行の登録で 76 キー（三重は '~'）。T1 に無い P=O・P=S・C=S・S=O・N=O・S=S のほか、B・Ge・Sn・Pb・As・Se などの元素がある。N–S・O–S（単）は無い。芳香族の項目は無い。CDK の芳香族の結合をどの次数で引くかは読んでいない【未読】

### T5: BDE-db2（DFT で計算した単結合の BDE の集まり）— 表は自分で作る
- 出典: Shree Sowndarya S. V., Y. Kim, S. Kim, P. C. St. John, R. S. Paton, "Expansion of bond dissociation prediction with machine learning to medicinally and environmentally relevant chemical space", *Digital Discovery* 2, 1900–1910 (2023)、doi:10.1039/D3DD00169E（著者・巻・頁は RSC と NREL の書誌の検索結果で確認）。論文は CC BY 3.0（NREL の PDF の検索結果の抜粋の表示。PDF の本文は取得できず【未読】）。M06-2X/def2-TZVP の 531,244 の BDE・BDFE（同じ抜粋）。データ: figshare doi:10.6084/m9.figshare.19367051（`bde-db2.csv.gz` 26,047,039 B、sha256 `e4fab4eb9b3808f3f5b9b9fffa1317f8e9c057854e02e1e766716737cf838ada`）、**figshare の license は CC BY 4.0**（API の表示、確認）。GitHub `patonlab/bde-db2` は API のライセンスが null（表示なし）
- 中身（確認、csv を読んだ）: 831,871 行・65,540 分子、列 `molecule, bond_index, fragment1, fragment2, bde, bdfe, bdscfe, bond_type, set`。`bde` の単位は本文で確かめていない【未確認】。値の大きさ（C–C の例の行で 81.6）からは kcal/mol と読める【推測】（前の版の ALFABET は誤差を kcal/mol で書く〈PyPI の説明〉）。無作為の 3,000 行を RDKit で読み、**すべて単結合・環に入らない結合**だった。`bond_type` に出る元素の組は 24（C–H・C–C・C–N・H–N・C–O・H–O・C–S・C–F・N–O・N–N・N–S・C–Cl・H–S・C–I・O–P・C–P・O–O・O–S・H–P・N–P・Br–C・Cl–N・F–N・F–S）
- 転載の条件（CC BY 4.0）: 表示だけ。**組ごとの代表値（中央値など）を自分のコードで作る**なら、生成の手順（コード）と出典を一緒に出せるので、756893（中間物を再現できること）にも合う
- 収録の範囲: 単結合だけ。**二重・三重・芳香族は無い**。値は環に入らない結合だけから出ているので、環の結合に当てはめるのは外挿【推測】
- 関連: 前の版 BDE-db（figshare doi:10.6084/m9.figshare.10248932、**MIT**、C・H・N・O だけ）と、それで学習した ALFABET（St. John et al., *Nat. Commun.* 11 (2020)、doi:10.1038/s41467-020-16201-z。PyPI の表示は MIT、GitHub `NatLabRockies/alfabet` の LICENSE は BSD 型の文〈冒頭だけ読んだ〉）。ALFABET は「単結合・環に入らない結合」だけを予測する学習済みの GNN で、学習した重みを読むので「表」ではない。候補にはしない（重みの再現の手間と、学習した点を入れないという #178 の方針のため）

### 1.6 調べたが候補にしなかったもの
- CRC Handbook・Luo *Comprehensive Handbook of Chemical Bond Energies*（CRC 2007）: 有料の本で、本文を読んでいない【未読】。転載の条件が商用の出版社の規約になる
- NIST の Chemistry WebBook などの生成熱から自分で BDE を計算する案: 今回は手をつけていない（§4.4 の次の手）

## 2. ライセンスのまとめ（転載の条件）

- T1 OpenStax 初版 / LibreTexts 5.8: CC BY 4.0。表示だけ。ただし OpenStax のページには LLM への取り込みを断る文がある（§1 T1、§4.3）
- T2 Wikipedia: CC BY-SA 4.0。値のファイルは BY-SA で出す（継承）
- T3 Huheey（LibreTexts Tro）: CC BY-NC-SA 4.0、元は市販の教科書。非営利に限る → 使わない
- T4 MetFrag: リポジトリにライセンスの表示なし → 値は写さない（キーの集合だけ比較に使った）
- T5 BDE-db2: CC BY 4.0（figshare）。表示だけ。代表値は自分のコードで作る

## 3. 被覆の測定（採点はしない）

### 3.1 定義（1 回目の前に commit d5d3592〈committer 2026-11-22T00:28:27+09:00〉で固定。スクリプトの docstring）
- 集合: v3_c1・v3_c2・enveda_v1・v3_c2n。窓の候補は run.py と同じ ±10 ppm の窓（`run.visible_window`。pool=remove の集合では drop_keys を除く）。目標の質量は `pc_logits.npz` の `__targets__`。集合ごとに pool の key でまとめた**一意の構造**を数える
- 結合: `Chem.MolFromSmiles` の分子の全結合（H は暗黙）＝ MetFrag-lite（`data/ext/frag/frag_rescore.py`）が切る重原子のグラフの辺
- 結合のキー: (元素 a, 元素 b, 次数)。主は **Kekulé 形**（`Chem.Kekulize(clearAromaticFlags=True)` の後の 1・2・3）、副は芳香族のまま（AROMATIC はどの表にも無い）
- 規則（値は機械的に決まり、v3 を見て選ばない）: R0 = (a, b, 次数) が表にある。R1 = R0 で無ければ同じ元素の組の単結合 (a, b, 1) の値。R2 = R1 でも無ければその表の C–C 単結合の値（R2 の被覆は定義から 100% なので数えない）
- 量: 結合の割合（値のある結合 / 全結合）と、全結合に値がある構造の割合。参考に正解の構造（truth.csv の smiles）も同じ量を数えた（採否に使わない）

### 3.2 2 回目（T1+T5 を足した）
1 回目の結果を見てから、T1 に T5 の単結合の組を足した組み合わせ「T1+T5」（値は T1 を先に引き、T1 に無いキーだけ T5。足されるキーは N–S・O–S の 2 つ）を commit cd63d46 で足して測り直した。**この組み合わせは 1 回目の後に作った**。1 回目の 5 つの表の数値は 2 回目でもすべて同じだった（json を照合）。

### 3.3 結果
`/usr/bin/time -l .venv/bin/python scripts/research/bde_coverage.py --bdedb2 <scratch>/bde-db2.csv.gz` → `outputs/local_eval/bde_coverage.json`（sha256 `39c0875dde87e2b81b0424943ad6910d755b23ceaf16b2c3fd05a1fcc62b78dd`、git_head cd63d46、script sha256 `34bd1ed8c35c…`、`bdedb2_keys_match` true、RDKit 2026.09.1）。16.16 s、**ピーク RSS 773,636,096 B（0.77 GB）**。1 回目（d5d3592）は 15.38 s・770,572,288 B。下は Kekulé 形の値（小数 4 桁、json の `bond_frac`・`struct_frac`）。「結合」= 値のある結合の割合、「構造」= 全結合に値がある構造の割合。

- **v3_c1**（窓の延べ 27,504 行、一意の構造 24,343、結合 610,406、うち芳香族 238,128、読めない構造 0）
  - T1 OpenStax: R0 結合 0.9898・構造 0.9022 / R1 結合 0.9902・構造 0.9108
  - T2 Wikipedia: R0 0.7220・0.0797 / R1 0.7220・0.0797
  - T3 Huheey: R0 0.9964・0.9227 / R1 0.9965・0.9250
  - T4 MetFrag（キーのみ）: R0 0.9968・0.9265 / R1 0.9968・0.9265
  - T5 BDE-db2 単結合: R0 0.7612・0.0023 / R1 0.9999・0.9991
  - T1+T5: R0 0.9929・0.9080 / R1 1.0000・0.9996
- **v3_c2**（26,601 行、23,776、結合 603,281、芳香族 236,724、読めない 0）
  - T1: R0 0.9904・0.9072 / R1 0.9908・0.9154
  - T2: R0 0.7281・0.1050 / R1 0.7281・0.1050
  - T3: R0 0.9966・0.9269 / R1 0.9967・0.9297
  - T4: R0 0.9970・0.9316 / R1 0.9970・0.9316
  - T5: R0 0.7610・0.0011 / R1 0.9999・0.9989
  - T1+T5: R0 0.9933・0.9131 / R1 1.0000・0.9998
- **enveda_v1**（26,315 行、24,412、結合 628,758、芳香族 238,402、読めない 0）
  - T1: R0 0.9884・0.8843 / R1 0.9887・0.8930
  - T2: R0 0.7387・0.0713 / R1 0.7387・0.0713
  - T3: R0 0.9963・0.9148 / R1 0.9964・0.9176
  - T4: R0 0.9967・0.9191 / R1 0.9967・0.9191
  - T5: R0 0.7637・0.0017 / R1 0.9999・0.9991
  - T1+T5: R0 0.9917・0.8894 / R1 1.0000・0.9999
- **v3_c2n**（15,788 行、15,220、結合 445,401、芳香族 117,046、読めない 0）
  - T1: R0 0.9961・0.9537 / R1 0.9965・0.9646
  - T2: R0 0.8831・0.4919 / R1 0.8831・0.4919
  - T3: R0 0.9981・0.9627 / R1 0.9981・0.9647
  - T4: R0 0.9988・0.9694 / R1 0.9988・0.9694
  - T5: R0 0.7955・0.0035 / R1 0.9999・0.9989
  - T1+T5: R0 0.9973・0.9561 / R1 1.0000・0.9999

「1.0000」は 4 桁に丸めた値で、R1 でも値の無い結合が残る（T1+T5 の R1 で v3_c1 27 本・v3_c2 15・enveda_v1 6・v3_c2n 8。B–O・B–C・N–Si・C–Se・As–C・Mg–N など）。

芳香族のまま数えると（副）、芳香族の結合（v3_c1 で全結合の 39%）はどの表にも無いので、R0 の結合の割合は T1 で 0.5982〜0.7334、構造の割合は 0.0451〜0.3141 に落ちる（json の `aromatic_R0`）。どの表も芳香族の値を持たないので、**芳香族の結合をどう引くかの規則が要る**（§5 の R_ar）。

値の無い結合（Kekulé 形・R0、v3_c1 の上位。json の `top_missing`）:
- T1: O=S 4,045・N–S 1,835・N=O 143・O–S 84・C=S 60・O=P 37 …（スルホニル・スルホンアミド・ニトロ・リン酸）
- T1+T5: O=S 4,045・N=O 143・C=S 60・O=P 37 …（N–S・O–S は T5 で埋まる）
- T2: C–N 121,284・C=N 22,172・C–S 7,369・N–N 6,927 …（使えない）
- T3: N–S 1,835・O–P 87・O–S 84・C=S 60・O=P 37 …
- T4: N–S 1,835・O–S 84・O–Si 10 …
- T5: C=C 87,593・C=O 30,007・C=N 22,172・O=S 4,045 …（多重結合が全部無い）

参考（採否に使わない）: 正解の構造で全結合に値がある割合（Kekulé・R0）
- v3_c1（n 200）: T1 0.8500、T2 0、T3 0.8800、T4 0.8800、T5 0、T1+T5 0.8500
- v3_c2（n 200）: T1 0.8450、T2 0、T3 0.8750、T4 0.8750、T5 0、T1+T5 0.8600
- enveda_v1（n 200）: T1 0.8350、T2 0、T3 0.8850、T4 0.8850、T5 0.0050、T1+T5 0.8350
- v3_c2n（n 195）: T1 0.9128、T2 0.4154、T3 0.9333、T4 0.9641、T5 0.0154、T1+T5 0.9179

読み: 正解の構造は窓の候補（v3_c1 で T1 0.9022）より値の無い結合を含む割合が高い（0.8500）。表に無い結合の規則（R1・R2）が**正解に多めに当たる**ので、その規則の値の偏りが正解とおとりに違う向きで効きうる【推測】。値を v3 を見て選ばないことは変えず、§5 の記録の項目に入れた。

## 4. 推奨

### 4.1 推奨: T1+T5（OpenStax 初版 Table 7.2 を主、N–S・O–S の単結合だけ BDE-db2 から自分で作る）＋ 規則 R1 → R2 ＋ 芳香族は R_ar
- 理由
  - 使える条件の表で被覆がいちばん広い: CC BY だけ（継承も非営利も無い）の組み合わせで、Kekulé・R0 の結合の割合 0.9917〜0.9973（4 集合）、R1 で 4 桁で 1.0000。ライセンスの条件の悪い T3（0.9963〜0.9981）・T4（0.9967〜0.9988）との差は R0 の結合の割合で 0.08〜0.50 ポイント（T3: 0.08〜0.46、T4: 0.15〜0.50）
  - T2 は C–N が無く結合の割合 0.72〜0.88 で使えない。T5 だけでは多重結合が無い（0.76〜0.80）
  - 値の作り方を全部コードと出典で出せる（T1 は値の転記〈出典・版・取得日つき〉、T5 は csv からの代表値の計算）。756893 の再現性に合う
- 弱いところ（隠さない）
  - T1 は平均の結合エネルギーで、BDE（特定の結合の切断）ではない。T5 は DFT の BDE。**2 つの尺度を混ぜる**（N–S・O–S の 2 キーだけ）
  - S=O・P=O・N=O・C=S は T1+T5 にも無く、R1 で単結合の値を使う。単結合の値は二重結合の値より小さいので、スルホニル・リン酸・ニトロの切断が「安く」見える向きになる【推測】
  - 正解の構造は値の無い結合を含む割合が高い（§3.3 の参考）

### 4.2 表に無い結合の規則（候補 B で使う案。値は機械的に決まり、v3 を見て選ばない）
- R0: (a, b, 次数) が T1 にあれば T1 の値。T1 に無く T5 にある単結合（N–S・O–S）は T5 の代表値
- T5 の代表値: `bde-db2.csv.gz` の全行（train・valid・test の区別なし）で、`bond_type` ごとの `bde` の**中央値**を kJ/mol に直したもの（×4.184。単位は kcal/mol と読んだが【未確認】なので、実装の前に論文の本文で確かめる）
- R1: R0 で無ければ同じ元素の組の単結合の値（T1 → T5 の順）
- R2: R1 でも無ければ T1 の C–C 単結合の値（MetFrag の既定値と同じ考え）
- R_ar（芳香族の結合）: Kekulé 形の次数は RDKit の割り当て次第で同じ環の中でも単・二重が入れ替わるので、芳香族の結合は (a, b, 1) と (a, b, 2) の値の平均を使い、どちらかが無ければ R1 → R2。**R_ar での被覆はこの回は測っていない**【未測定】（Kekulé 形の R0 を目安にした）

### 4.3 判断待ち（ユーザーへ）
- OpenStax のページには「LLM の学習や LLM・生成 AI への取り込みには事前の書面の許可が要る」という文がある。CC BY 4.0 の表示と両立するのか、LibreTexts の写し（CC BY 4.0、この文は無い）を出典にすればよいのかは【判断待ち】。この回はページをスクリプトで読み、値がエージェントの文脈に入った。**推奨の案: 出典は LibreTexts 5.8（CC BY 4.0、作者 OpenStax）とし、値のファイルを実装のカードでリポジトリに入れる前にユーザーに確かめる**。この文書では値をリポジトリに入れていない
- 認めない場合の次の手: §4.4

### 4.4 使えない場合の次の手
- T5 だけで単結合の代表値を作り、多重結合は「単結合の値 × 次数」（定数を選ばない規則）にする案。尺度は DFT の BDE にそろうが、二重結合の値は物理的に正しくない【推測】
- 小さい分子の生成熱（NIST WebBook・ATcT など。ライセンスは【未確認】）から自分で平均の結合エネルギーを計算する案。手間が大きい

## 5. 候補 B の事前登録の下書き（この文書では採点しない。実装と採点は別の Issue で、独立レビューの後）

- 変更: frag 段（MetFrag-lite、`data/ext/frag/frag_rescore.py` の点 f）を、MetFrag 2.2 の式 2 の形の点に差し替える: SC = Σ_p RelMass_p^α · RelInt_p^β / (Σ_b BDE_b)^γ。α = 1.84・β = 0.59・γ = 0.47 は論文の値（著者が別の部分集合で決めたもの）をそのまま使い、選ばない。BDE は §4.1 の T1+T5 と §4.2 の規則 R0・R1・R2・R_ar。frag 段の他（上位 60、同じ分子式の群、λ = 0.5、z の合わせ方）は変えない。新しい重みを足さない
- 実装の前の注意: 今の frag_rescore.py は質量ごとに「切断の最少の回数」しか持たず、**どの結合を切ったかを持たない**。BDE の和には断片ごとの切った結合が要るので、列挙を変える（同じ質量に複数の切り方があるときは BDE の和の最小を取る、と先に決める）。MetFrag-lite の他の設定（許容 0.005 Da、H のずれ −2〜+3、切断 2 回まで）は替えない
- 作る前の条件（#178 §2.3・§2.4）: Δ_dom = d_dom(BDE の点) − d_dom(MetFrag-lite) を rep_dseen.py と同じ定義（対の bootstrap 2,000 回・seed 0、sd0 の判定は 2 つの点の組）で測り、**v3_c1 と e180_c2k のどちらでも higher_train でない**こと。higher_train なら採点せずに止める
- 採点: 1 回だけ。`run.py --holdout v3_c1|v3_c2|enveda_v1|v3_c2n` で、新 = `score,prior,frag`（f を差し替えた版）、対照 = 今の `score,prior,frag`（v3_c1 0.8535・v3_c2 0.0000・enveda_v1 0.8376・v3_c2n 0.6284、#178 §5）
- **採用規則（共通の決まり）: 4 集合（C1 = v3_c1、C2 = v3_c2、enveda_v1、v3_c2n）すべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、かつ 1 つ以上で下限 > 0**（分子の bootstrap 2,000 回・seed 0、下限は 4 桁に丸めた値）。集合が 1 つでも欠ければ不採用。上がっても下がっても規則どおりに決める
- **C1・C2 の両方を必ず報告する**。C2（v3_c2）は窓に正解が無い（`iso_rank_diag.json` の excluded.truth_absent 200/200）ので、frag 段の差し替えでは**定義から差 0 の見込み**。それでも 4 集合の 1 つとして測り、0 であることを確かめる
- MDE（`net_mde.json`、#178 §5）: v3_c1 0.0176〜0.0434、enveda_v1 0.0175〜0.0401、v3_c2n 0.0171〜0.0538、v3_c2 なし。効きが 0.02〜0.03 より小さければ判定できない
- 記録だけ（採否に使わない）: 同じ分子式の中の順位（`iso_rank_diag.py` の定義）で BDE の点と MetFrag-lite を比べる。正解の構造のうち R1・R2・R_ar の規則に当たる結合を切った断片で説明されたピークの割合（§3.3 の正解の偏りの確認）
- 替えないこと: 表（T1+T5）、規則（R0〜R2・R_ar）、α・β・γ、frag 段の他の定数。v3 や LB を見て選び直さない

## 6. 未確認のまま残っていること
- OpenStax の LLM の取り込みを断る文と CC BY 4.0 の関係（§4.3、判断待ち）。T1 の元の数値の出典
- Wikipedia のこのページ固有のライセンスの表示（一般の規約は CC BY-SA 4.0）
- 競技の勝者のライセンスの条件（NC・SA が許されるか）
- BDE-db2 の `bde` の単位（kcal/mol と読んだ）と論文の本文（PDF を取得できなかった）
- MetFrag が芳香族の結合をどの次数で引くか（CDK の側。読んでいない）
- R_ar の規則での被覆（測っていない）

## 7. 再現のコマンド
```
# 被覆（採点なし、約 16 s、ピーク RSS 0.77 GB）。--bdedb2 は figshare 10.6084/m9.figshare.19367051 の bde-db2.csv.gz
/usr/bin/time -l .venv/bin/python scripts/research/bde_coverage.py --bdedb2 <scratch>/bde-db2.csv.gz
.venv/bin/python -m pytest -q tests/test_bde_coverage.py
```
