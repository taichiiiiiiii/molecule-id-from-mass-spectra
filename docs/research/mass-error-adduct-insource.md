# 前駆体の質量誤差・付加イオンの誤り・in-source 断片（#203、A11・A12）

対象: #195 の A11（質量の誤差と候補の絞り込み）と A12（付加イオンの誤りと in-source 断片）。A4（伸びしろの分解）の 1 区分として使う。
事前登録: Issue #203 の本文（2026-11-22T17:29Z）。数え上げのコードは 31c4a62 でコミットしてから動かした（その後の修正は下の「手順の記録」）。

## 結論

- **timsTOF では、±10 ppm の窓から正解が外れる分子は 0 だった。** 対象は v3_c1・v3_c2・enveda_v1・見える test で、隠れテストと同じ装置。train の timsTOF 1,154,969 スペクトルでも、窓の外は 2 本（2×10⁻⁶）だけ。|ppm| の 99% 点は 2.8〜6.0 ppm（装置 × 付加イオンの組ごと）。系統的なずれがあり、[M+H]+ は +2.2 ppm（中央値）。
- **v3_c2n では 195 分子中 18 分子（9.2%）で、正解が窓に入らない。** 18 分子とも、窓の中心を決めるスペクトルが timsTOF 以外の公開ライブラリのもの（qTof・Ion Trap・LC-ESI-QQ など）。既知の差（事前に固定した表）に当たった分子は 0/18。v3_c2n の対照（score+prior+frag）でもこの 18 分子の rr はすべて 0。したがって v3_c2n の上限は、窓の中で並べ替えるどんな変更でも 177/195 = 0.9077 になる。
- **事前登録の読み方に従った判断**: timsTOF の集合はどれも窓の外が 1% 未満。v3_c2n は 1% を超えるが、既知の差には当たらない。そこで、別の付加イオンの仮説で候補を取る新しい候補源は**作らず、測らない**。窓の幅を 99% 点（2.8〜6.0 ppm）に狭める案も、効くのは組成式の違う候補だけで、その上限が MDE より小さい（§5）。こちらも**測らない**。
- **断片の誤差**（train の標本 525 スペクトル、12,908 ピーク）: TOL 0.005 Da 以内で一致したピークでは、|誤差| の中央値が 0.00067 Da（timsTOF は 0.00082 Da）。0.005 Da の外側にも、背景を引いた一致の過剰が全体の約 21% ある（timsTOF 20.8%）。ただし背景は平らではない。0.0126・0.0238・0.0364 Da に山があり、これは N/CH2・O/NH2・O/CH4 の組成の差と同じ値である。したがって 21% は「TOL の外にある本当の一致」の上限で、別の組成の断片との偶然の近さも含む。TOL は frag_rescore の既存の定数で、変えれば定数の変更になる。この調査では変更を提案しない。

## 1. 出典

文献の本文の確認は、作業に関わっていない調査役（読むだけ）がした。確認したものは［確認］、要旨しか読んでいないものは［未確認］と書く。数値は調査役が本文で確かめた値で、このカードでの読み直しはしていない。

- ［確認］Vasilopoulou CG et al. 2020, "Trapped ion mobility spectrometry and PASEF enable in-depth lipidomics from minimal sample amounts", *Nat Commun* 11:331. https://doi.org/10.1038/s41467-019-14044-x
  - timsTOF Pro で脂質を測った。前駆体の質量誤差の絶対値は中央値 1.06 ppm。照合の許容幅は前駆体 5 ppm、MS/MS 10 ppm。
  - 対象は脂質で、天然物ではない。CC BY 4.0。ppm の窓を考える根拠として使える。
- ［確認］Knolhoff AM, Callahan JH, Croley TR 2014, "Mass Accuracy and Isotopic Abundance Measurements for HR-MS Instrumentation", *J Am Soc Mass Spectrom* 25:1285–1294. https://doi.org/10.1007/s13361-014-0880-5
  - Orbitrap は 1.06±0.76 ppm、QTOF（maXis、ロックマス補正あり）は 1.62±1.88 ppm。個々の測定では QTOF が 10 ppm を超えることがある。
  - 組成式を正しく決めるには 3 ppm 以内が要る、と先行研究を引いている。
  - 読んだのは出版社版ではなく写し。数値の引用だけに使う。
- ［確認］Thermo Fisher 2016, PITTCON 発表資料 "Challenges in Untargeted Metabolomics"（スライド 14）. https://assets.thermofisher.com/TFS-Assets/CMD/Reference-Materials/PP-LC-MS-Untargeted-Metabolomics-Orbitrap-PITTCON2016-EN.pdf
  - 12 スキャンの誤差の幅は QTOF 17.65 ppm、Q Exactive 1.55 ppm。メーカーの資料なので補助としてだけ使う。
- ［未確認］Rajkumar P et al. 2026, "TIMS-Bench", bioRxiv. https://www.biorxiv.org/content/10.64898/2026.05.23.724673v1.full
  - 確かめたのは Abstract だけ。「前駆体が 20 ppm 以上ずれたスペクトルを除いた」という記述は検索の要旨で見ただけ。
- ［確認］Giera M, Aisporna A, Uritboonthai W, Siuzdak G 2024, "The hidden impact of in-source fragmentation in metabolic and chemical mass spectrometry data interpretation", *Nat Metab* 6:1647–1648. https://doi.org/10.1038/s42255-024-01076-x
  - METLIN の 0 eV のスペクトルから、in-source 断片が LC-MS/MS のピークの 70% 超を説明しうると推定した。
  - 実試料の特徴を数えたものではない。METLIN はこの競技（インターネット無し）では使えない。
- ［確認］Xu YF, Lu W, Rabinowitz JD 2015, "Avoiding Misannotation of In-Source Fragmentation Products as Cellular Metabolites…", *Anal Chem* 87:2273–2281. https://doi.org/10.1021/ac504118y
  - 代謝物に見える in-source 断片を 20 件超報告した。例は ATP→ADP、malate−H2O→fumarate など。
  - 割合ではなく件数の報告。
- ［確認］Houriet J et al. 2025, "Multilaboratory Untargeted Mass Spectrometry Metabolomics Collaboration…", *Anal Chem* 97(30):16110–16122. https://doi.org/10.1021/acs.analchem.4c05577
  - 天然物（ashwagandha）を Orbitrap で測り、8 チームが注釈した。610 特徴の 28.2% が付け直された。
  - チームは 59.3% を [M+H]+ としたが、合議の結果では 26.1%。取り違えの内訳は [M+H+NH3]+ 31 件、[M+H−H2O]+ 29 件、[M+Na]+ 17 件、in-source 断片 69 件。
  - Open Access。ただしライセンスの種別は未確認。
- ［確認：抄録のみ］Liu B, Tang Z, Huan T 2025, "Adduct-Induced Variability in Tandem Mass Spectrometry", *Anal Chem* 97(31):17058–17066. https://doi.org/10.1021/acs.analchem.5c02792
  - NIST 20 では、[M+Na]+ と [M+K]+ のスペクトルは [M+H]+ と似ていない。NIST は商用なので使えない。
- ［確認］Bushuiev R et al. 2024, "MassSpecGym", NeurIPS Datasets & Benchmarks, arXiv:2410.23326（Supplementary Algorithm 1）
  - 質量の課題では、候補を中性質量 ±10 ppm（最大 256 個）で取る。組成式の課題は完全一致で取る。付加イオンは [M+H]+ と [M+Na]+ だけで、装置は 98% が Orbitrap か QTOF。
  - 前駆体の質量が記載の値・SMILES と合うスペクトルだけを残している。MIT。
- MS-DIAL・SIRIUS/ZODIAC・CAMERA の付加イオンの推定の誤り率を数えた一次資料は、今回見つからなかった（「無い」とは言っていない）。

この競技での読み: ±10 ppm の窓は MassSpecGym と同じ幅。timsTOF の精度の報告（約 1 ppm）とも矛盾しない。天然物では付加イオン・脱水・in-source 断片の取り違えが多い（Houriet 2025）。ただしこれは LC-MS の特徴に注釈を付けるときの誤りの話である。この競技では付加イオンが各スペクトルに付いて渡される。下の数え上げでは、timsTOF の付加イオンの記載と質量はすべて正解と整合した。

## 2. 方法（事前登録どおり）

- コード: `scripts/eda/mass_error_adduct.py`。純粋な関数は `tests/test_mass_error_adduct.py` でテストしている。
- 理論の中性質量 M
  - train: `molecular_formula` 列を RDKit の最多同位体の質量で単同位体質量に直した。末尾の電荷 + / − は無視した。読めなかった式は 50 行。
  - v3: truth.csv の SMILES に `CalcMolFormula` を当て、同じ解析で M を出した。
  - 見える test: `outputs/test_library_match.parquet` の `best_key`（cos の最小値は 1.0）で train の式を引いた、推定の正解。
- 観測の中性質量: `casmi.chem.neutral_mass`（窓の中心を作る `local_fold0_logits.py` と同じ式）。表に無い付加イオンは train の 339 行で、件数だけ数えた。
- 窓: `scripts/local_eval/run.py` の `PPM = 10.0`。中心は分子のスペクトルの中性質量の中央値。
- 断片の誤差の標本: row group ごとに `default_rng(20261123 + rg)` で 25 行を引き、計 525 スペクトル。
  - 断片は `frag_rescore.fragment_levels`（既定の設定）、ずらしは `fragnn_bench.shift_table`、ピークは強度の上位 32 本。
  - 各ピークについて、最も近い「断片 + ずらし」との差を符号つきで取った。
  - 背景は 0.02〜0.05 Da の平均の密度とした。
- ピーク RSS（`/usr/bin/time -l`）と時間: train 1.90 GB・43 s、frag 0.15 GB・0.7 s、holdouts 0.61 GB・4.4 s。

## 3. 前駆体の ppm 誤差（A11）

### train、timsTOF（隠れテストと同じ装置）

| 装置 | 付加イオン | 件数 | 中央値 ppm | \|ppm\| 99% 点 | ±10 ppm の外 |
|---|---|---|---|---|---|
| timsTOF | [M+H]+ | 589,076 | +2.22 | 4.13 | 0 |
| timsTOF | [2M+Na]+ | 188,158 | +0.80 | 4.72 | 0 |
| timsTOF | [M-H]- | 185,023 | +0.80 | 2.76 | 1 |
| timsTOF | [2M+H]+ | 89,487 | +0.69 | 4.76 | 0 |
| timsTOF | [2M-H]- | 52,933 | +0.01 | 4.46 | 0 |
| timsTOF | [M+CH2O2-H]- | 24,661 | +0.35 | 4.34 | 0 |
| timsTOF | [M+Na]+ | 16,037 | +1.03 | 4.85 | 0 |
| timsTOF | [M+NH4]+ | 4,800 | +1.49 | 5.18 | 0 |
| timsTOF | [M+Cl]- | 2,414 | −0.15 | 4.85 | 0 |
| timsTOF | [M+K]+ | 1,100 | +0.26 | 5.53 | 1 |
| timsTOF | [2M+CH2O2-H]- | 659 | −0.29 | 5.24 | 0 |
| timsTOF | [M+C2H4O2-H]- | 376 | +0.39 | 5.99 | 0 |
| timsTOF | [2M+C2H4O2-H]- | 187 | +0.20 | 5.01 | 0 |
| timsTOF | その他（< 100 件） | 58 | +1.51 | 5.53 | 0 |

timsTOF 全体は 1,154,969 本で、中央値 +1.58 ppm、99% 点 4.32 ppm、窓の外 2 本。

### train、公開ライブラリ（件数の多い組の一部。全 265 組は `mass-error-1123/train.json`）

| 装置 | 付加イオン | 件数 | 中央値 ppm | \|ppm\| 99% 点 | ±10 ppm の外 |
|---|---|---|---|---|---|
| Orbitrap | [M+H]+ | 411,477 | +0.01 | 37.7 | 4.64% |
| Orbitrap | [M-H]- | 153,793 | +0.01 | 2.5 | 0.08% |
| QTOF | [M+H]+ | 64,466 | −2.40 | 123,522 | 30.85% |
| Orbitrap | [M+Na]+ | 64,177 | −0.03 | 94,379 | 26.53% |
| QTOF | [M+Na]+ | 61,470 | −2.68 | 143,394 | 30.55% |
| LC-ESI-QTOF | [M+H]+ | 58,364 | +0.02 | 12.0 | 1.15% |
| Orbitrap | [M+CH2O2-H]- | 53,865 | +0.01 | 1.3 | 0.00% |
| LC-ESI-QFT | [M+H]+ | 35,146 | −0.00 | 3.2 | 0.30% |
| qTof | [M+H]+ | 21,335 | +0.51 | 1,066,386 | 18.02% |

- train 全体（2,539,234 本）の窓の外は 4.91%（124,644 本）。ほとんどが公開ライブラリの行である。
- 事前登録外の参考として、窓の外の行が既知の差に当たるかも見た（`train.json` の `a12_train_reference`）。当たったのは 6,412 本（5.1%）だけ。
  - 目立つのは ToF の [M+NH4]+・[M−H2O+H]+・[M−2H2O+H]+ で、Δ = −13C（1.00336 Da）が大半だった（それぞれ 701/1,084、829/1,036、494/535）。前駆体として同位体ピークの値が記載されている。
  - Orbitrap [M+H]+ では −H2O 504、+Na−H 292、+NH3 246。
- 主催者の `precursor_error_ppm` 列は自前の値と一致しない（差の中央値 1.6 ppm）。主催者の列の中央値は 1.512 ppm。定義は未確認なので、この調査では使わない。

### v3 の 4 集合と見える test（分子 = 窓の中心と正解の M の差）

| 集合 | 分子 | 中央値 ppm | \|ppm\| 99% 点 | 窓の外の分子 | 窓の外のスペクトル |
|---|---|---|---|---|---|
| v3_c1 | 200 | +1.73 | 4.02 | 0 | 0 / 1,252 |
| v3_c2 | 200 | +1.79 | 3.72 | 0 | 0 / 1,215 |
| enveda_v1 | 200 | +1.71 | 4.12 | 0 | 0 / 614 |
| v3_c2n | 195 | +0.02 | 814.5 | **18（9.2%）** | 152 / 1,675 |
| 見える test（推定の正解） | 400 | +2.09 | 4.22 | 0 | 0 / 1,213 |

- v3_c1・v3_c2・enveda_v1・見える test のスペクトルはすべて timsTOF だった。
- v3_c2n の問い合わせのスペクトルは公開ライブラリ由来で、装置は 28 種（`HOLDOUT.json` の `instruments_of_queries`）。
- v3_c2n の窓の外の 18 分子は、どれも timsTOF のスペクトルを持たない。その 18 分子では、正解はプールにあるが窓には入らない（`n_in_pool_not_window` = 18）。対照の rr はすべて 0 だった（`submission_score+prior+frag.csv` と `mrr25` で確認。MRR 0.6284 は既存の値と一致）。

## 4. 付加イオンの誤り・in-source 断片（A12）

事前に固定した既知の差の表は Issue #203 にある（H2O・2H2O・NH3・Na−H・K−H・CO2・HCOOH・CH3OH・CH3CN・H2・ヘキソース・デオキシヘキソース・ペントース・グルクロン酸・13C、±両符号、±10 ppm × M、多量体 2M・M/2）。

- v3_c1・v3_c2・enveda_v1・見える test: 窓の外の分子もスペクトルも 0 なので、当たりの数も 0。
- v3_c2n、分子: 18 分子中、当たりは 0。
  - Δ は +0.004〜+0.014 Da（12〜23 ppm）の小さなずれが 9 分子。
  - 0.04〜0.29 Da のずれが 7 分子。Ion Trap・LC-ESI-QQ などの低分解能の装置で、整数に近い精度の前駆体値と見られるが、未確認。
  - −1.0098 Da が 1 分子（m_00075）。H 1 個分（1.0078 Da）の差に近いが、H は表に無いので数えていない。
  - +24.0002 Da が 1 分子（m_00131）。
- v3_c2n、スペクトル: 窓の外の 152 本のうち、当たりは 4 本（+NH3 2、−(Na−H) 1、2M 1）。装置は qTof 38、QTOF 37、Orbitrap 24、Linear Ion Trap 17、nan 11、LC-ESI-QQ 9、そのほか。
- 同じ `molecule_id` で付加イオンが違うスペクトルが同じ中性質量を指すか（±10 ppm）:
  - v3_c1 133/133、v3_c2 117/117、enveda_v1 86/86、見える test 108/108（すべて一致）。
  - v3_c2n は 90/102（88.2%）。ずれた付加イオンは [M+H]+ 7、[M−H]− 4、[M+Na]+ 3 だった。

## 5. この競技での読み・v3 で測る案

1. **新しい候補源（別の付加イオンの仮説で候補を取る）は作らない。測らない。** 理由は次のとおり。
   - timsTOF の集合（隠れテストと同じ装置）では、付加イオンの記載と質量がすべて正解と整合した。窓の外は 0/600 分子、見える test も 0/400。
   - 失点しうるのは v3_c2n だけで、原因は既知の差ではない（0/18）。v3_c2n のクエリが timsTOF 以外の公開スペクトルであることから来る。
   - 事前登録の読み方（各集合で窓の外が 1% 未満なら無視できる区分、1% 以上でも既知の差に当たらなければ新しい候補源の根拠にならない）のとおりである。
2. **A4 への入力**: 窓・付加イオンが原因の失点は、v3_c1・v3_c2・enveda_v1 で 0、v3_c2n で 18/195。v3_c2n の対照 0.6284 と上限 1 の差 0.3716 のうち 0.0923 は、窓の中の並べ替えでは取れない（上限 0.9077）。A4 の文書に「v3_c2n の取れない分」として書く。
3. **窓を 99% 点に狭める案（#195 A11 の固定の幅）は測らない。**
   - timsTOF の 99% 点は 2.8〜6.0 ppm。狭めると候補から消えるのは、正解と組成式の違う候補だけである（同じ組成式の異性体は同じ質量）。
   - #178 の神託の上限（正解より上の組成式違いの候補をすべて正解の下へ）は、v3_c1 +0.0053、enveda_v1 +0.0036 で、MDE より小さい（MDE は `outputs/local_eval/net_mde.json` の v3_c1 の変種で 0.018〜0.036）。狭めた窓で取れるのはこの上限以下である。
   - したがって 1 回測っても判定できない。
4. **v3_c2n の作り方について**（集合は変えない）: v3_c2n では、窓の中心に低分解能・公開データのスペクトルが混じる。そのため「プールにあり参照なし」型の比較では、9% の分子がどの変更でも 0 になる。今の規則（値を見てから集合を選び直さない）を守り、この集合は変えない。次に集合を作るときに、「窓の中心は高分解能のスペクトルだけで取る」を事前登録で決めるかどうかは、計画の側で判断する（このカードでは決めない）。
5. **断片の TOL 0.005 Da**: 一致の核は 0.001 Da より内側にある（timsTOF の中央値 0.00082 Da）。TOL の外側の過剰（約 21%）には組成の差の山（0.0126・0.0238・0.0364 Da）が含まれ、本当の測定誤差とは分けられない。TOL は frag_rescore（0.479 版の v7c と同じ）の定数で、変えれば定数の変更になる。ハイパラ調整にあたるので提案しない。

## 手順の記録

- 31c4a62 で数え上げのコードとテストをコミットしてから、本番の数え上げを動かした。
- その後、コードを 3 回直して全段を動かし直した。設定や定数は変えていない。
  1. 装置の名前が None の行が装置別の集計から漏れていたので、文字列に揃えた。
  2. 正解を「隠した」かどうかの判定を `library.hide_rule`/`drop_keys`（run.py と同じ規則）に揃えた。
  3. 事前登録外の参考の集計を足した（train の窓の外の既知の差、断片の timsTOF の刻みの分布、断片の ppm の分位点）。
- 最後の実行が上の数値である。v3 の各集合の窓の外の分子の数（v3_c2n 18、ほかは 0）は、煙の試し・2 回目・最後の実行で同じだった。
- 数値の JSON: `docs/research/mass-error-1123/{train,frag,holdouts}.json`（`outputs/local_eval/mass_error_1123/` の写し）。
  - train.json: sha256 3f049355…
  - frag.json: sha256 c0c999f4…
  - holdouts.json: sha256 b0117b7e…
  - RDKit 2026.09.1。

再実行: `.venv/bin/python scripts/eda/mass_error_adduct.py train && … frag && … holdouts`（計約 50 s）
