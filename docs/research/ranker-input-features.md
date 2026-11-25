# 学習した並べ替えの手法が使う入力の一覧と、自前の特徴表（#244）への当てはめ

Issue #251（2026-11-26 夜 計画 項目 5、F2・C3 #195）。**この文書では何も測っていない**（G1 未承認。dev の正解ラベルでの当てはめ・MRR の計算をしていない、holdout の値を見ていない）。下の「効いた／効かなかった」はすべて各論文の著者の数字で、こちらの測定ではない。

- 目的: 自前のランカー（#247 の LambdaRank、入力は #244 の特徴表）に足す特徴群の候補を、出典つきで決める
- こちらの特徴表の列は #244 の本文の定義（`fz`・`tani_pred`・`lib_max_sim`・`has_ref`・`is_tier`・`prior_unverified_pop`・`ctrl_rank`・`q_rank`・`metfrag_lite`・`has_frag`・`mf_*` 26 列・`np_score`・`has_np`・5 つの点のチャネルの式 (8) の相対化 `<v>_dmean / _z / _dmax / _softmax / _sd`）。「既にある」はこの列名で書く

## 0. 出典と読んだ範囲

印: **【開いて確かめた】** = 本文（と書いた範囲の補足）を自分で取得して読んだ。**【要約のみ】** = 本文の要約・他の文書の記述しか見ていない部分。

1. **MS2Query** — de Jonge NF, Louwen JJR, Chekmeneva E, et al. "MS2Query: reliable and scalable MS2 mass spectra-based analogue search". Nat Commun 14, 1752 (2023). doi:10.1038/s41467-023-37446-4 、https://www.nature.com/articles/s41467-023-37446-4
   - **【開いて確かめた】** 本文の HTML（Results、Discussion、Methods の Workflow・Training・Benchmarking）と補足 PDF（41467_2023_37446_MOESM1_ESM.pdf）の Supplementary Note 2・3（Supplementary Table 1・2 と、重みづけの比較の文）。2026-11-26 に取得
   - 読んでいない: 補足の Note 1・4〜9 の大部分、図
   - ライセンス: 論文は CC BY（ページの permissions のリンクに `oa=CC BY`）。ソフトウェア iomega/ms2query は Apache-2.0（GitHub API の license で 2026-11-26 に確かめた）
2. **MetFrag 2.2** — Ruttkies C, Schymanski EL, Wolf S, Hollender J, Neumann S. "MetFrag relaunched: incorporating strategies beyond in silico fragmentation". J Cheminform 8, 3 (2016). doi:10.1186/s13321-016-0115-9 、https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0115-9
   - **【開いて確かめた】** 本文の HTML（式 1、各点の項の節、Results の Table 1〜3 を説明する段落、Availability）。2026-11-26 に取得。表そのものは HTML に無く（「Full size table」のリンク）、数字は本文の文から取った
   - 読んでいない: Additional file 1〜5、表の画像
   - ライセンス: 論文は CC BY 4.0（本文の Rights and permissions）。ソフトウェアは本文の Availability に「GNU LGPL version 2.1 or later」。今のリポジトリ MetFragRelaunched のライセンスは【未確認】（`train-independent-evidence.md` §1.3 と同じ）
3. **CSI:FingerID ＋ COSMIC** — Hoffmann MA, Nothias LF, Ludwig M, et al. "High-confidence structural annotation of metabolites absent from spectral libraries". Nat Biotechnol 40, 411–421 (2022). doi:10.1038/s41587-021-01045-9 、https://www.nature.com/articles/s41587-021-01045-9
   - **【開いて確かめた】** 本文の HTML（COSMIC の確信度の説明、Method evaluation、Online Methods の Confidence score computation の冒頭、評価の分け方）と補足 PDF（41587_2021_1045_MOESM1_ESM.pdf）の **Supplementary Table 1（確信度の SVM の 22 特徴と重み）**。2026-11-26 に取得
   - **【要約のみ】** CSI:FingerID の点そのもの（Modified Platt score: Dührkop ら PNAS 112:12580, 2015、Covariance score: Ludwig ら Bioinformatics 34:i333, 2018）。両論文は開いていない。COSMIC の補足に名前と出典があるだけ
   - ライセンス: 論文は CC BY 4.0（本文の Rights and permissions）。COSMIC は SIRIUS の機能で、**SIRIUS はこのコンペで使えない**（主催者の回答。`selective-prediction.md` §1.2 に引用）。使うのは考え方だけ
   - 注意: COSMIC の SVM は **並べ替えをしない**（本文「COSMIC neither changes annotations (re-ranks structure candidates) nor discards any annotations」）。1 位の候補が正しいかの確信度で、入力は 1 位の候補と分子ごとの量。ランカーの特徴にするには候補ごとの量に読み替える（§3 の表に書いた）
4. **arXiv:2611.28096 の Stage 2** — Xue J, Wu Y, Zhang Q. "Auditing two-stage MS/MS molecular retrieval on MassSpecGym v1.5: Reranking gains, candidate priors and spectrum dependence". arXiv:2611.28096v1 (2026-11-17). https://arxiv.org/abs/2611.28096 、doi:10.48550/arXiv.2611.28096
   - **【開いて確かめた】** HTML 版 https://arxiv.org/html/2611.28096v1 を `curl -sL -A Mozilla` で取り、タグを除いて読んだ（166,539 字、2026-11-26）。読んだ: §2.2 の Stage 1 の末尾、§2.3 の全部、§3.2（Table 3）、§3.6（Table 6）、§3.7（Table 7）。Table 1 と式 (6)(8) は `mass-feasibility-features.md` で既に読んでいる
   - 読んでいない: 補足 S6.3 以外の S6、S7 の大部分（この文書のためには）
   - ライセンス: arXiv nonexclusive-distrib 1.0（`two-stage-retrieval-audit.md`）。コードは MIT と書かれているが URL は未確認。使うのは定義だけ
5. **MassSpecGym** — Bushuiev R, et al. "MassSpecGym: A benchmark for the discovery and identification of molecules". arXiv:2410.23326（読んだのは v3 の HTML、https://arxiv.org/html/2410.23326v3 、2026-11-26）
   - **【開いて確かめた】** §3.2 の検索の課題の定義（候補 ≤ 256、質量か分子式で揃える bonus）、Table 5（検索の基準の結果）と基準の手法の説明の段落
   - 読んでいない: 付録、データ作りの節の細部
   - ライセンス: 論文は arXiv nonexclusive-distrib 1.0（abs ページ）。リポジトリ pluskal-lab/MassSpecGym は MIT（GitHub API、2026-11-26）。データは使わない（外部データの規則は確かめていない）

## 1. 各手法の並べ替えの入力と ablation（著者の数字）

### 1.1 MS2Query（ランダムフォレスト、ライブラリのスペクトルの並べ替え）【開いて確かめた】
- 流れ: 問い合わせとライブラリの全スペクトルの MS2Deepscore → 上位 2,000 件 → 5 特徴のランダムフォレスト（scikit-learn、250 本、深さ 5、MSE）が「問い合わせとライブラリの構造の Tanimoto」を予測し、その値で並べる
- 入力 5 個（Methods の番号）:
  1. Spec2Vec の類似度（問い合わせ × ライブラリのスペクトル）
  2. 問い合わせの precursor m/z
  3. precursor m/z の差（問い合わせ − ライブラリ）
  4. **化学的に近いライブラリ分子 10 個の MS2Deepscore の平均**: 候補のライブラリ構造に Tanimoto（RDKit daylight 2,048 bit）で最も近い 10 構造を選び、各構造の全スペクトルと問い合わせの MS2Deepscore を構造ごとに平均してから 10 個を平均
  5. その 10 構造と候補の構造の Tanimoto の平均
- ablation（補足 Note 2、検証の MSE、小さいほど良い）: 全部 0.0255。抜いたとき: 4 を抜く 0.0331、3 を抜く 0.0283、1 を抜く 0.0272、2 を抜く 0.0265、5 を抜く 0.0259。Gini 重要度は 4: 0.62、3: 0.18、2: 0.13、1: 0.05、5: 0.02。4 を「1 本のスペクトルの MS2Deepscore」に替えると 0.0337
- 試して入れなかったもの（補足 Note 3）: cosine と modified cosine は重要度が常に 0（MS2Deepscore と Spec2Vec があるとき、深さ 5 で）。4 の平均の取り方: 構造ごとに平均してから平均（検証 0.0257）が全スペクトルの平均（0.0286）より良かった。Tanimoto で重みをつける案も試した（表 S3、数字は読んでいない）
- こちらとの違い: MS2Query の候補はライブラリにあるスペクトル（類似物の検索）で、こちらの候補は構造（参照の無い候補が多い）。4 の考え方は「候補の構造に近いライブラリ構造の参照スペクトルが問い合わせに似ているか」なので、**参照を持たない候補にも値が付く**（下の §3 の M4・M5、§4 の群 N1）

### 1.2 MetFrag 2.2（点の線形結合）【開いて確かめた】
- 式 1: S_final = ω_Frag·S_Frag + ω_RT·S_RT + ω_Refs·S_Refs + ω_Incl·S_Incl + ω_Excl·S_Excl + ω_Suspects·S_Suspects + …（CFM-ID の項も足せる）。**各項は候補リストの中の最大値で割って正規化**（本文「normalized to the maximum value within the candidate result list」）
- 項: 断片の点（in silico の断片化）、保持時間（log P と RT の線形モデル。利用者の標準物質が要る）、参照の数（PubChem の特許数・PubMed 数、ChemSpider の参照数など。組み合わせて 1 項にもできる）、SMARTS の部分構造の含む／除く、suspect list（InChIKey の第 1 ブロックで他の DB にあるか）、CFM-ID の確率
- ablation（本文の数字、473 本の Eawag Orbitrap、分子式で候補を取る。`fragment-isomer-ranking.md` と同じ）: 断片の点だけで 1 位 PubChem 30（6.3%）・ChemSpider 105（22%）。断片＋RT＋参照で重みを乱数で 1,000 通り引いた最良が PubChem 336（71%）・ChemSpider 420（89%）。断片＋CFM-ID は重み 0.67/0.33 の最良で 30 → 61。**重みは同じデータで最良を選んだ値**（著者がそう書いている）なので、この増え方は楽観的
- こちらへの意味: 参照の数（人気度）の項が一番効いた。これは `prior_unverified_pop` と同じ種類の情報で、PubChem の層の候補（Q）では今 0 になっている

### 1.3 COSMIC の確信度（線形 SVM、向きの強制）【開いて確かめた（補足 Table 1）／CSI:FingerID の点そのものは要約のみ】
- 入力 22 個（補足 Table 1。1〜19 は PubChem と生体分子の DB の両方、20〜22 は生体分子の DB だけ）。向き（P = 正の重みに限る、N = 負に限る）を前もって決めた線形 SVM、Platt scaling で確率に。候補 1 個の場合は別の SVM
  - 1・2: 1 位と 2 位の点の差（log と生、Modified Platt）、3・4: 同じく Covariance score
  - 5・6: 1 位の Modified Platt / Covariance の点、7・8: その較正した点（PubChem の候補の点の分布からの E 値）
  - 9: 断片化の木が説明した強度の和、11: 断片化の木の点
  - 10: 候補リストの大きさの log
  - 12: 予測した指紋の「質」Σ max(1 − p_i, p_i)
  - 13: 1 位と 2 位の候補の Tanimoto、14: 予測の指紋と 1 位の候補の Tanimoto
  - 15: 予測の指紋のビット数（p ≥ 0.5）、16: 1 位の候補の指紋のビット数
  - 17〜19: 別の点の付け方で付け直した点（Covariance の 1 位を Modified Platt で、その逆、その較正値）
  - 20・21: 生体分子の DB の 1 位と PubChem の 1 位の点の差（2 つの点の付け方で）、22: PubChem の候補リストの大きさの log
- 効いたもの（本文と補足 Table 1 の重み、merged spectra の bio2+ の列）: 本文は「(較正した) CSI:FingerID の点、1 位と 2 位の差、候補の数がとても重要」「指紋の質や断片化の木の点は重みがほぼ 0」。表の重みでも 12・11 は 0、22（PubChem の候補の数の log）は −0.1119（bio1 では −0.4195）、1 は 0.1452、3 は 0.1104
- こちらへの意味: 20〜22 は「プール側の 1 位と PubChem 側の 1 位のどちらを信じるか」の量で、こちらの C1/C2 の振り分け（`is_tier` の候補を上げるか）にそのまま当たる。ただし COSMIC は 1 位の確信度で、並べ替えの効果は測っていない

### 1.4 arXiv:2611.28096 の Stage 2（式 (6)(8) 以外の入力）【開いて確かめた】
- **pair reranker**（MLP 4,107 → 256 → 64 → 1、残差を Stage 1 の点に足す、式 (4)）の入力: 問い合わせのベクトル u と候補のベクトル v_j の [u, v_j, |u − v_j|, u ⊙ v_j]（Stage 1 の対照学習の埋め込み、1,024 次元）＋ shortlist の中の相対の点 11 個（3 つの checkpoint のそれぞれの標準化したロジットと最大との差で 6、checkpoint の間の標準化ロジットの平均・標準偏差・範囲で 3、ensemble の点の標準化と最大との差で 2）。**候補の位置・候補の ID・正解の分子式は入力にしない**と明記
- **coverage の点 c_j（式 6）と anchor a_j = p_j + 0.25·z(c)_j（式 7）**: 他の多くの候補も説明できるピークは寄与が小さく、少数の候補だけが説明できるピークが差になる
- **木（LightGBM LambdaRank）の 36 特徴**: 式 (8) を 2 つの点のチャネル s（Stage 1）と g = s + 0.25 z(c) に当てた 10 個 ＋ Table 1 の 13 測定とその shortlist 平均との差 26 個。anchor を初期の点にして残差を学ぶ。木は out-of-fold の上流の出力で学ぶ（S6.3）
- **support の規則**: 付加イオンが [M+H]+ か [M+Na]+、ピークが有効、shortlist の全候補がつながっていて同位体でなく重原子 ≤ 60。満たさなければ木の残差 0（anchor の順）。付加イオンは質量の計算と support にだけ入り、装置の種類と衝突エネルギーは使わない
- **質量の事前分布（§3.6、凍結した系の外の補足の行）**: 質量の誤差の Gaussian の点 3 個（σ = 0.01・0.1・1 ppm）を、検証で当てた重みで足すと、どの手法も test の R@1 が上がった（最終系 31.05 → 37.88、+6.82 [5.64, 8.07]。質量の点だけで 1 位 8.11%）
- ablation（Table 3、公式 test、top-20、R@1）: pair reranker 22.84、pair + coverage 24.19、coverage ×2 の対照 25.75、score だけの木 27.86、feasibility の木 30.37、support で切り替える最終 31.05。**26 個の mass-feasibility の特徴は score だけの木より +2.52（top-5）、検証の top-20 で +2.59 [2.13, 3.08]**
- 効かなかったもの（Table 7、§3.7）: 学習した peak–fragment の類似度は定数の coverage と同じ程度（+0.93 vs +1.02、順位の 99.1% が同じ）。**候補の記述子を feasibility の特徴に足す: +0.25 [−0.59, 1.08]**。同じ分子式で coverage が等しい 531 問い合わせでは feasibility の木は score だけの木を上回らない（−0.38 [−2.13, 1.38]）。特徴群を 4 つ抜く ablation もどれも改善なし。著者は「mass-feasibility の証拠をうまく統合したことによる利得で、構造を見分ける学習は確かめられていない」とまとめている

### 1.5 MassSpecGym の検索の基準【開いて確かめた】
- 検索の課題: 候補は質量が同じ（許容誤差の中）分子を最大 256 個、bonus は同じ分子式の分子に絞る（§3.2）
- 基準の手法（Table 5）: Random、DeepSets、Fingerprint FFN（予測した Morgan 指紋と候補の cosine）、DeepSets + Fourier features、MIST（予測した指紋の cosine）。**どれも 1 つの点で並べるだけで、学習した再ランキングの基準は無い**。この文書の目的（ランカーの入力）には、入力の一覧として足すものは無い
- 数字（Table 5、hit rate @1、質量で揃えた候補）: Random 0.37、DeepSets 1.47、Fingerprint FFN 2.54、DeepSets + Fourier 5.24、MIST 14.64。同じ分子式の bonus: Random 3.06、Fingerprint FFN 5.09、MIST 9.57（表のとおり。MIST は bonus の方が低い。理由は読んだ範囲に書かれていない）
- こちらへの意味: 予測した指紋と候補の指紋の似かた 1 つ（こちらの `fz`・`tani_pred`）が基準の形。2611.28096 はこの上に Stage 2 を足して R@1 を上げている（§1.4）

## 2. 入力をまとめた見方（読んだ範囲から）

- どの手法も **候補リストの中で相対化した量**を入れる: MetFrag は最大で割る、COSMIC は 1 位と 2 位の差、2611.28096 は平均との差・z・最大との差・softmax・σ（式 8）と coverage の中央値との差（式 6）。こちらの `<v>_dmean / _z / _dmax / _softmax / _sd` は 5 チャネルに既にある
- **候補の周りの既知の参照を使う量**（MS2Query の 4・5）は、こちらの表に無い。こちらの `lib_max_sim` は候補そのものの参照だけで、参照の無い候補は 0（`has_ref` = 0）
- **プール側と PubChem 側の比較**（COSMIC の 20〜22）は、こちらの表では `is_tier` と各チャネルの値から木が学ぶしかない（群ごとの比較の量は無い）
- **質量の誤差**（MS2Query の 3、2611.28096 §3.6）は、こちらの表に無い
- 効かなかった側の証拠: 候補だけの記述子（2611.28096 Table 7 の +0.25 [−0.59, 1.08]）、cosine の追加（MS2Query、MS2Deepscore があるとき）、断片化の木の点・指紋の質（COSMIC の重み 0）

## 3. 入力の一覧と、こちらの特徴表での扱い

扱い: **既にある**（#244 の列名）／**作れる**（必要なデータ・作り方、オフラインで動くか）／**作れない**（理由）。「出どころ」は playbook §4.3 の区分（自前 / 公開の部品 / 主催者が可とした重み）。

| # | 出典の入力 | こちらでの扱い | 列・作り方 | 必要なデータ・ライセンス・オフライン |
|---|---|---|---|---|
| M1 | MS2Query 1: Spec2Vec の類似度 | 作れない（そのものは）／近いものは既にある | 近いもの: `lib_max_sim`（エントロピー類似度）。Spec2Vec は GNPS で学習した外部のモデル | Spec2Vec の重みを使うには外部データ・モデルの規則の確認が要る（`rules-and-deps.md` の手順で未確認）。自前で学ぶなら別の項目 |
| M2 | MS2Query 2: 問い合わせの precursor m/z | 作れる（群で一定） | `q_prec_mz` = test の `precursor_mz` | test.parquet にある。群の中で一定なので、ランカーには他の列との組み合わせでだけ効く |
| M3 | MS2Query 3: precursor m/z の差 | 作れる | `ppm_err` = (候補の中性の単一同位体の質量 + 付加イオンのずれ − `precursor_mz`) / `precursor_mz` × 1e6（付加イオンのずれは `mass_feasibility.parse_adduct` と同じ）、`abs_ppm_err` | オフライン。候補は ±10 ppm の窓の中なので値は −10〜10。同じ分子式の候補では同じ値なので、異性体は分けない（誤りの多くは同じ式の異性体、`fragment-isomer-ranking.md`）。窓は #203 の結論で変えない |
| M4 | MS2Query 4: 構造の近いライブラリ分子 10 個のスペクトルとの類似度の平均 | 作れる（新しい群 N1、§4） | `nbr_lib_sim_mean` = 候補の構造に Tanimoto で近い参照の構造（参照のある構造だけ、隠す規則の後）上位 10 個のそれぞれで、問い合わせとその参照スペクトルのエントロピー類似度の最大を取り、10 個を平均（MS2Query の「構造ごとに平均してから平均」の形。類似度は MS2Deepscore の代わりに `library.lib_scores` と同じエントロピー類似度）。候補自身の構造は近傍から除く | train.parquet の参照（`normalized_smiles` あり）と `library.load_refs` の隠す規則。オフラインで動く。参照の構造は数万の見込みなので、近傍の表は 1 回だけ作る（時間は未測定）。**漏れの点検**: 近傍の参照にも dev の正解の構造・その互変異性の別名の参照を隠す規則を同じく当てる（§4.2 (b)、raw と canonical の両方の鍵） |
| M5 | MS2Query 5: その 10 構造との Tanimoto の平均 | 作れる（A1 と一緒） | `nbr_lib_tani_mean`（A1 で選んだ 10 構造と候補の Tanimoto の平均）、`nbr_lib_tani_max` | M4 と同じ |
| F1 | MetFrag: 断片の点（リストの最大で正規化） | 既にある | `metfrag_lite`・`has_frag`（MetFrag-lite v7c、学習なし）と `metfrag_lite_dmax` など | — |
| F2 | MetFrag: 保持時間（log P–RT の線形モデル） | 作れない | — | このコンペのデータに保持時間の列が無い（test.parquet・train.parquet の列を 2026-11-26 に確かめた） |
| F3 | MetFrag: 参照の数（PubChem の特許数・PubMed 数） | プールの候補は既にある／PubChem の層の候補は作れない（今は） | `prior_unverified_pop`（プール L だけ、Q は 0） | Q の候補の特許数・PubMed 数はローカルに無い（`pool_popularity.csv` はプールの鍵だけ）。PubChem から取るには外部データの規則と取得の手順の確認が要る【未確認】。`build_pop.py` で train 由来でないと確かめるまで `prior_unverified` の印は外さない（#244） |
| F4 | MetFrag: SMARTS の部分構造の含む／除く | 作れない | — | 部分構造の情報の源（MS1 の同位体、別の分類器）がこのコンペの入力に無い |
| F5 | MetFrag: suspect list（他の DB にあるか） | 作れない（今の規則では） | — | 例えば COCONUT の所属は、プールの行では由来の印 `src` と同じ情報になる（#244 で `np_coconut`・`src` を入れない列にした理由と同じ） |
| F6 | MetFrag: CFM-ID の確率（予測スペクトルの点） | 作れる（GLACIER の予測ができた後） | `glacier_cos` など（予測スペクトルと問い合わせの一致を CE ごと） | 項目 2 のジョブ（#242・draft PR #246、L60 の 544,886 本を手元 CPU で生成中）。Q の候補は L60 の外なので値が無い（欠けの印が要る）。漏れの規則は #242 の msg_overlap |
| C1 | COSMIC 1〜4: 1 位と 2 位の点の差 | 既にある（候補ごとの形で） | `fz_dmax`・`tani_pred_dmax`（最大との差）。1 位の候補にとっての「2 位との差」は `_dmax` だけでは分からない | 作れる: `fz_gap_next` = 候補の点 − その候補を除いた群の最大（1 位では 2 位との差、他では最大との差と同じ）。オフライン |
| C2 | COSMIC 5〜8: 1 位の点・較正した点（PubChem の候補の点の分布の E 値） | 一部は既にある／作れる | `fz`・`fz_z`（群の中の z）。較正の形: `fz_tier_pct` = PubChem の層の窓の全候補（上限 60 に切る前）の `fz` の分布の中での候補の分位 | `pubchem.probe` は窓の全候補に `sc1` を計算している（ただし選んだビットのうち 4096 未満の部分だけの f·z で、`fz` そのものではない。2026-11-26 にコードで確かめた）。分位は `sc1` の分布で取るか、窓の全候補の `fz` を計算し直すかを先に決める（後者は時間が延びる、未測定）。オフライン |
| C3 | COSMIC 9・11: 断片化の木が説明した強度・木の点 | 近いものは既にある | `mf_int_cov`（Table 1 の intensity coverage）、`metfrag_lite` | SIRIUS は使えない（主催者）。COSMIC では重みがほぼ 0（11 は 0） |
| C4 | COSMIC 10・22: 候補リストの大きさの log（PubChem 側も） | 作れる（群で一定） | `log_n_pool_window`（±10 ppm のプールの窓の候補数、隠す前でなく見える候補）、`log_n_tier_window`（PubChem の層の窓の大きさ、上限 60 に切る前） | オフライン。`pubchem.run_holdout` が窓の大きさを返す変更が要る。COSMIC では 22 の重みが大きい（−0.1119） |
| C5 | COSMIC 12・15: 予測の指紋の質・ビット数 | 作れる（群で一定） | `fp_quality` = Σ max(1 − p, p)、`fp_len_pred` = Σ[p ≥ 0.5]（p = σ(z)、選んだビットで） | オフライン。COSMIC では 12 の重みは 0、15 は 0.0178 |
| C6 | COSMIC 13: 1 位と 2 位の候補の Tanimoto | 作れる | `tani_to_top` = 候補と、対照の 1 位（`ctrl_rank` = 1）の候補の Tanimoto（選んだビットで）。1 位自身は 2 位との値 | オフライン |
| C7 | COSMIC 14: 予測の指紋と候補の Tanimoto | 既にある | `tani_pred` | — |
| C8 | COSMIC 16: 候補の指紋のビット数 | 作れる | `fp_len_cand` = 候補の選んだビットの数 | オフライン。COSMIC の bio2+ の重み 0.0569 |
| C9 | COSMIC 17〜19: 別の点の付け方で付け直した点 | 作れる | `fz_loglik` = Σ_i [f_i log σ(z_i) + (1 − f_i) log(1 − σ(z_i))]（Modified Platt の考え方に近い独立のベルヌーイの対数尤度。Modified Platt・Covariance の正確な式は【要約のみ】で読んでいないので、同じ式だとは言わない） | 同じ FPNet の出力の上の自前の式。オフライン |
| C10 | COSMIC 20・21: プール側の 1 位と PubChem 側の 1 位の点の差 | 作れる（新しい群 N2、§4） | `pool_vs_tier_gap` = max_{L} fz − max_{Q} fz、`tani_pred` でも同じ。候補ごとには `fz − max_{反対側}` | オフライン。C1/C2 の振り分けに当たる量。**前例（playbook §4.1）: 振り分けを学習したランカー model_c2p は手元で C1 0.869・C2 0.442 と勝ち、LB は 0.246 で score+prior（0.282）に負けた**ので、測るときは型ごとの条件（§4）を必ず通す |
| A1 | 2611.28096: pair reranker の [u, v, \|u − v\|, u ⊙ v] | 作れる（学習が要る。今の木の特徴にはしない） | 近いもの: p = σ(z) と候補のビット f で [p, f, \|p − f\|, p ⊙ f] を入れる小さな MLP（自前の学習モデル）の出力を 1 列の特徴に | fold-safe の学習が要る（dev の分子の fold の外で学ぶ）。10,226 bit 級の入力なので木には直接入れない。GPU 0 で手元で学べるかは未測定 |
| A2 | 2611.28096: checkpoint ごとの点と、その間の平均・標準偏差・範囲 | 作れない（今は） | — | dev の分子 1 つにつき、その分子を見ていない FPNet は自分の fold のモデル 1 つだけ（他の fold のモデルはその分子で学んでいる。§4.2 (a)）。複数の seed の fold-safe モデルを作れば作れる（学習が要る） |
| A3 | 2611.28096 式 (6): 候補の相対の coverage c_j | 作れる | `mf_cov_rel` = Σ_p w_p [u_jp − median_k u_kp]_+ / max(W, 1e-8)（κ・τ_c = 0.1 は論文のまま、群 = L ∪ Q、偶数のときは下の中央値）。`mass_feasibility.measurements` の h・w・W からそのまま出る | オフライン。今の `mf_*` 26 列に式 (6) は入っていない（#244 の列の定義を確かめた） |
| A4 | 2611.28096 式 (8): g = s + 0.25 z(c) のチャネルの相対化 | 作れる（A3 の後） | `g_*`（`fz` と A3 から、0.25 は論文の定数のまま） | 定数は論文のまま使い、こちらで選ばない |
| A5 | 2611.28096 Table 1 の 13 測定と平均との差 | 既にある | `mf_*` 26 列（#239） | — |
| A6 | 2611.28096 support の規則（付加イオン・重原子 ≤ 60・つながり） | 作れる（印として） | `mf_supported`（群の全候補がつながり・同位体でない・重原子 ≤ 60、付加イオンが [M+H]+ / [M+Na]+） | オフライン。論文は規則で木を切るが、こちらは印の列にして木に任せる（規則で切ると新しい分岐になる） |
| A7 | 2611.28096 §3.6: 質量の誤差の Gaussian の点（σ = 0.01・0.1・1 ppm） | 作れる（M3 の形で） | `ppm_err`・`abs_ppm_err`（M3）。σ の 3 個と重みは検証で当てたものなので写さず、生の誤差を木に入れる | オフライン。MassSpecGym の候補は同じ質量（窓）で、このコンペの precursor の値が測った値か理論値かは【未確認】（train には `precursor_error_ppm` の列がある） |
| G1 | MassSpecGym: 予測の指紋と候補の cosine | 既にある（近いもの） | `fz`（ビットの内積）、`tani_pred` | — |

## 4. 作れる群と測る案（今夜は測らない）

順番は playbook §5 の 11/28〜の群の順（相対化 → 予測スペクトル → 断片の説明率・中性損失 → NN の点）に、この文書の新しい群を差し込む案。どれも **既定のハイパラの LambdaRank（#247、`lambdarank_oof.py` の既定 + seed）で、群を足す／足さないの 1 行**。

- 群 N1（近傍の参照の照合、M4・M5）: 参照を持たない候補（C2 型・プールにあり参照なしの型）に値が付く唯一の照合の量。MS2Query で一番効いた入力（抜くと検証の MSE 0.0255 → 0.0331）
- 群 N2（プール側と PubChem 側の比較、C10・C4・C2 の `fz_tier_pct`）: C1/C2 の振り分けの量。前例（model_c2p）があるので型ごとの条件を厳しく見る
- 群 N3（coverage と g、A3・A4・A6）: 2611.28096 の Stage 2 の残り。Table 3 では coverage の足し方より 26 測定の木の方が効いた（著者の数字）ので、増え方は小さい見込み
- 群 N4（質量の誤差、M2・M3・A7）: 同じ式の異性体は分けない。式の違う候補が上にいる分子だけに効く
- 群 N5（指紋の付け方の別の点、C1・C5・C6・C8・C9）: 同じ FPNet の出力の読み直し。COSMIC では一部の重みが大きい（1・3・16・17）が、それは確信度の分類で、並べ替えの効果は分からない
- 群 N6（予測スペクトル、F6）: GLACIER の生成（#246）の後。playbook §5 の 2 番目の群
- 学習が要るもの（A1 の pair reranker の形、A2 の複数 seed）: 自前の学習モデルとして別の Issue にする（今夜は作らない）

測り方（各群で同じ。**G1 の承認の前は dev でも測らない**）:
1. **dev**: G1 承認後、台帳 `experiments.csv` に 1 行（群を足した版 − 足さない版）。B8 の 20 集合（3,901 分子、fold 1〜4、leave-one-fold-out）で、**5 つの型（a〜e）を型ごとに**報告し、まとめた値を使うなら #249 の型の重み（`docs/research/type-weights.md`、draft PR #250、main には未マージ）で。dev の実験の上限 40 本（playbook §4.1）に数える
2. **holdout**: 12/6 の凍結版に入れる場合だけ（入れない群は holdout で測らない）。v3 の 4 集合を **型ごとに**報告する: C1 型 = `v3_c1u_t10`・`enveda_v1u_t10`（プールにあり参照なし）、C2 型 = `v3_c2_t10`・`v3_c2nu_t10`。条件は playbook §4.1: **どの型も、同じ holdout の score+prior の基準より −0.02 以上悪くない**（C1 も C2 も）。holdout は 12/6 と 12/11 の 2 回だけで、中間の結果で群を足し引きしない
3. 共通: ハイパラは既定のまま、群の中の定数は出典のまま（MS2Query の 10 個、2611.28096 の 0.25・τ_c = 0.1・κ）で、dev の値を見て変えない。漏れの点検 §4.2 (a)〜(e) を群ごとに（N1 は (b) の隠す規則が近傍の参照にも効いているか、N2・C4 は PubChem の層の窓が hold の正解を隠した後の窓か）

## 5. 言えないこと

- 上の効いた／効かなかったはすべて別のデータ（GNPS、Eawag、MassSpecGym）の著者の数字で、このコンペで同じだけ効くとは言えない
- COSMIC の入力は 1 位の確信度のためのもので、並べ替えの特徴にしたときの効果はどの出典も測っていない
- CSI:FingerID の Modified Platt・Covariance の式は読んでいない（§0 の 3）
- MetFrag の重みは同じデータで選んだ最良で、楽観的（§1.2）
