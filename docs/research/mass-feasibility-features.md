# 質量で説明できるか（mass-feasibility）の 26 特徴 — 定義・自前実装・測り方

Issue #238（2026-11-25 昼 計画 項目 3、F2 #195）。**この文書とコードでは特徴の値を一度も読んでいない**（G1 承認前。速さ・件数・NaN/inf の数だけを測った）。

## 出典

- Jiaqi Xue, Yutong Wu, Qiang Zhang. "Auditing two-stage MS/MS molecular retrieval on MassSpecGym v1.5: Reranking gains, candidate priors and spectrum dependence". arXiv:2611.28096v1、2026-11-17
- URL: https://arxiv.org/abs/2611.28096 （読んだのは HTML 版 https://arxiv.org/html/2611.28096v1 、2026-11-25 に取得）、DOI: https://doi.org/10.48550/arXiv.2611.28096
- 2 段の検索の監査としての読み（候補集合の定義・近傍の近さ・介入）は `docs/research/two-stage-retrieval-audit.md`（#232、PR #233）。そちらで「読んでいない」としていた §2.3 の残りと Table 1 を、この文書で読んだ

## 読んだ範囲（この文書のため）

- 読んだ: §2.3 の全部（pair reranker、Mass-feasibility edges and the role of adduct、Coverage score and anchor score、Score and mass-feasibility features、Table 1、Feasibility tree、Top-5/top-20、Support と fallback、Matched controls）、§3.2 の Table 3 を説明する段落、§3.7 の最初の 2 段落、補足の Table S3（test の 15 行）、S6.3（tree の OOF 学習データ）、S6.4（提案の族の監査）、S7.9 の Table S27 まで
- **読んでいない**: §3.7 の Table 7 の中身と Fig. 5 の説明の後半、S7.9 の Table S28、S6.5〜S6.7、S7.1（validation での開発の経緯）、S7.10 以降、参考文献。図は見ていない

## 論文の定義（§2.3、Table 1）と自前のコードの対応

コード: `scripts/features/mass_feasibility.py`。

| 論文の言葉 | 自前のコード |
|---|---|
| 断片の提案 f_jr: 決定的な atom-mask の列挙（環でない単結合 1 本、または 1 つの環の中の 2 本を切る）、最大 64 個 | `proposals()`: 重原子のグラフで環でない単結合 1 本 / RDKit の `RingInfo.BondRings()` の各環の 2 本を切り、端の原子を含むつながった部分を原子集合で重複除去。切っても離れない部分（元のつながった部分そのもの）は除く。質量 = 原子の単一同位体の質量 + その原子に付く水素 |
| M_j: 候補の中性の単一同位体の質量 | 同じ数え方の全原子の和（電荷は数えない） |
| 付加イオンのずれ Δ_a = m(H) − m_e（[M+H]+）、m(Na) − m_e（[M+Na]+）、RDKit の同位体の質量 | `parse_adduct()` の Δ_断片。[M+H]+・[M+Na]+ では論文と同じ値 |
| h ∈ {−2,…,2}、d = f + Δ_a + h·m(H)、ℓ = M − f + h·m(H) | `H_SHIFTS`、`edge_counts()` の `d`・`ell` |
| 辺: 直接のイオン（x_p と d）か中性損失（m0 − x_p と ℓ）の絶対誤差が 20 ppm 未満（分母は提案の質量）。経路と h は OR、誤差そのものは特徴にしない | `edge_counts()`: `|x − d| < 20e-6·d`、`|loss_obs − ℓ| < 20e-6·ℓ`（d, ℓ > 0 のときだけ）。e_jp = 辺のある提案 r の数 |
| e_jp、h_jp = 1[e_jp > 0]、w_p = √I_p、W = Σ w_p、b_jp = h_jp·1[Σ_k h_kp = 1] | `measurements()` の `e`・`h`・`w`・`W`・`b` |
| Table 1 の 13 測定（下）、数の分母は 1、強度の分母は 1e-12 で下を止める | `measurements()`（`MEASUREMENTS` の順） |
| 各測定から「候補リストの平均」を引いたもので計 26 | `group_features()` の後ろ 13 列（`*_dmean`） |

Table 1 の 13 測定（P = 残したピークの数、n_j = Σ_p h_jp）:

1. Matched peak count: n_j → `mf_n_matched`
2. Matched peak fraction: n_j / P → `mf_frac_matched`
3. Log feasible-edge count: log(1 + Σ_p e_jp) → `mf_log_edges`
4. Proposals per matched peak: Σ_p e_jp / n_j → `mf_props_per_matched`
5. Unambiguous matched fraction: Σ_p 1[e_jp = 1] / n_j → `mf_unambig_frac`
6. Intensity coverage: Σ_p w_p h_jp / W → `mf_int_cov`
7. Exclusive intensity coverage: Σ_p w_p b_jp / W → `mf_excl_int_cov`
8. Exclusive peak fraction: Σ_p b_jp / P → `mf_excl_peak_frac`
9. Mean matched peak mass: Σ_p x_p h_jp / n_j → `mf_mean_matched_mz`
10. Low-m/z intensity coverage: x_p/m0 < 1/3 → `mf_low_int_cov`
11. Middle-m/z: 1/3 ≤ x_p/m0 < 2/3 → `mf_mid_int_cov`
12. High-m/z: x_p/m0 ≥ 2/3 → `mf_high_int_cov`
13. Near-precursor: x_p/m0 ≥ 0.9 → `mf_nearprec_int_cov`（12 と重なる。論文も「重なることがある」と書く）

## 固定の定数（論文の値。検証に合わせて動かさない）

- 20 ppm（`PPM`、分母は提案の質量）、提案は最大 64 個（`MAX_PROPOSALS`）、h ∈ {−2,…,2}（`H_SHIFTS`）、重み √I
- テスト `test_constants_fixed_from_paper` で値を固定している。変えるなら新しい Issue で理由を書き、検証の値を見て選ばない

## 自前の決めごと（論文に書かれていない点。値を見る前に固定、2026-11-25）

1. **残すピーク**: 論文は「retained observed peaks」とだけ書き、Stage 2 の残し方を読んだ範囲に見つけられなかった。こちらは「有限・I > 0・m/z ≤ m0 + 0.5 のピークのうち強度の上位 100 本」（`MAX_PEAKS`・`PREC_MARGIN`）。上位 100 本は同じ論文の DreaMS・EmbCos・MAGMa の監査が使う数に合わせた。このコンペのスペクトルは数百本の小さなピークを持つ（b8_f1_a_t10 の 896 本で、ピーク数の中央値 226.5、最大 2,766、最初のスペクトルは 425 本。すべて timsTOF）
2. **64 個への間引き**: 論文は「deterministically subsampled to a fixed cap」とだけ書く（S6.4）。こちらは重複を除いた原子集合を（原子数, 原子番号の並び）で並べ、i·L/64（i = 0…63）番目を取る。原子の番号は canonical SMILES を読み直した順なので、同じ構造は書き方によらず同じ提案になる（テストあり）
3. **付加イオンの一般化**（論文は [M+H]+・[M+Na]+ だけを supported とし、それ以外は tree の補正を 0 にする）: metadata の adduct 文字列から前駆体のずれ Δ_前駆体（式の質量の和 − 電荷 × m_e）を計算する。断片のイオンの担い手 Δ_断片 は、正イオンで Na（+Na があるとき）> K > H、負イオンで Cl（+Cl があるとき）> 脱プロトン。中性損失は (m0 − Δ_前駆体) − (x_p − Δ_断片) と ℓ を比べる。[M+H]+・[M+Na]+・[M−H]− では Δ_前駆体 = Δ_断片 なので論文の m0 − x_p と同じ（テストあり）。2M などの多量体では中性損失の経路を使わない（直接のイオンだけ）。2 価以上と読めない文字列のスペクトルは使わない
4. **1 分子に複数のスペクトル**（論文はスペクトルごとの問い合わせ）: スペクトルごとに 13 測定を出し、使えるスペクトル（付加イオンが読めて、残すピークが 1 本以上）の単純平均を候補の値にする。平均からの差は平均の後に取る（差の平均と同じ）。使えるスペクトルが無い分子は 26 列すべて 0。スペクトルの並びは `run.py` の `query_spectra`（1 分子 16 本まで）と同じものを渡す
5. **候補群** = 呼び出し側が渡す候補の並び。論文は shortlist 20（または 5）の中で排他・平均を計算する。こちらでは dev のランカーの群と同じ並び（`run.py` の上位 60）を使う予定で、排他（b_jp）は 60 候補の中で数えることになる。論文の値とは直接比べない
6. 論文の support の条件（重原子 60 以下・つながっている・同位体なし）は特徴の計算の条件にしない（ランカーの入力にするので全候補に値を出す。結合の無い候補・読めない SMILES は提案 0 個で 0 の行になる）

## 論文の ablation（数値と条件）

- 公式 test 17,556 スペクトル、R@1（クエリ単位）。同じ Stage 1 の shortlist を並べ替えるので R@20 は全行 72.62%（Table 3 / S3）
- top-5: Stage 1 ensemble 21.01 → pair 23.00 → pair + coverage 23.88 → coverage ×2 対照 24.85 → **score-only tree 26.77 → feasibility tree 29.29（+2.52）**
- top-20: score-only tree 27.86 → feasibility tree 30.37（+2.51）。validation の top-20 で +2.59 [2.13, 3.08]
- score-only tree は 10 個の score 特徴（coverage を含む g の 5 個を含む）だけ。差の +2.5 は「26 個の細かい測定」の寄与（同じ anchor・容量・学習データ）
- tree は LightGBM 4.6.0 の LambdaRank、anchor を初期値にした残差、3 seed × 50 本（depth 4 など固定、本数の探索なし）、上流は OOF（S6.3: 8,000 connectivity を 2 fold、外の較正 1,024）
- feature-channel 対照（Stage 2 の特徴だけに別の分子のスペクトルを入れる）で R@1 −3.29〜−3.43。特徴はクエリのスペクトルを使っている（§3.3）
- **注意（§3.7）**: 「同じ分子式のクエリで shortlist の候補の coverage が等しいもの」では feasibility tree は score-only tree を上回らなかった。提案の族は MAGMa が割り当てたピークの 26.9%（強度 34.8%）しか原子集合として再現せず、supported の候補の質量で説明できるピークの中央値は 1 本。著者は利得を「質量で説明できる証拠のよりよい統合」と書き、構造の識別の学習は確立していないとする
- この競技への読み: 期待している「同じ分子式の異性体の取り違え」に効くかは論文からは言えない（むしろ等しい coverage の同じ分子式では差が出なかった）。効くとすれば「候補ごとに説明できるピークの数・位置・排他の違い」がある場合で、それはこちらの dev で型ごとに測るまで分からない

## ライセンスと再現の条件

- 論文: arXiv nonexclusive-distrib 1.0（#232 で確認済み）。使うのは**定義だけ**。本文・図・作者のコード（MIT と書かれている、URL は #232 で見つけられなかった）は使わない・再配布しない
- 自前のコードの依存は RDKit・numpy のみ（ベンチは pandas・pyarrow と `scripts/local_eval/run.py` の関数を読むだけ）。外部データ・学習済みの重みは使わない。決定的（乱数なし）で、同じ RDKit の版なら同じ値になる
- 最終の提出に入れる場合も、生成のコードはこのリポジトリにあり、組み直せる

## この競技との違い

- 装置: 論文は MassSpecGym v1.5（Orbitrap・QTOF などの混在）。隠れ test は Enveda の timsTOF が中心と見ている（未確認の部分は `rules-and-deps.md` などの既存の記録による）。20 ppm は timsTOF の MS2 の精度に対して狭すぎる・広すぎるかは未確認。**それでも 20 ppm は動かさない**（前駆体の質量誤差の調査は #203 にあるが、MS2 のピークの誤差はこの Issue では数えていない。誤差に合わせて定数を選ぶのはハイパラ調整になる）
- 付加イオン: 論文は [M+H]+・[M+Na]+ だけ。こちらは [M+NH4]+・[M−H2O+H]+・[M+K]+・負イオンなども入る → 上の「自前の決めごと 3」
- 1 分子に複数のスペクトル（衝突エネルギー違い）→「自前の決めごと 4」
- 候補群の大きさ 60（論文は 20 / 5）→「自前の決めごと 5」

## 速さ（値は読まずに測った。b8_f1_a_t10、dev の 1 集合）

- `scripts/features/mass_feasibility_bench.py`（`/usr/bin/time -l`、memguard MIN_FREE_PCT=15、RDKit 2026.09.1）。記録: `docs/research/mass-feasibility/bench_b8_f1_a_t10.json`
- 193 分子・896 スペクトル・候補 10,518 行（run.py の score,prior の上位 60、分子によっては 60 未満）。26 列の NaN 0・inf 0（スクリプトの assert）。使えなかったスペクトル 0、読めない SMILES 0
- 特徴の計算 10.14 s = **0.0525 s/分子**（中央値 0.0555、最大 0.0905）、スペクトル × 候補あたり 0.208 ms。候補の並びを作る（pool の読み込み・採点）30.7 s は別。全体 41.6 s、**peak RSS 1.77 GB**（大半は pool の読み込み）
- 特徴表（parquet 1.4 MB）は scratch に書き、記録の後に消した
- **dev 全体の外挿**（1 プロセス）: dev 220,443 構造（PR #237 の一覧）× 0.0525 s ≈ 11,600 s ≈ **3.2 h**。うち B8 に使った 3,901 構造だけなら ≈ 205 s。前提: 1 分子あたりのスペクトル数（ここ 4.64 本）と候補数（54.5）が b8_f1_a_t10 と同じ（dev 全体のスペクトル数は数えていない）。提案の列挙はキャッシュ（ここでヒット 559 / ミス 9,959）が効くので、集合をまとめて回すと少し速くなる見込み。並列にすれば割った分だけ短くなる

## v3 / dev での測り方（G1 承認後。この Issue では測らない）

- 候補: playbook §5 の 11/27（run.py の `--dump-features` で候補ごとの特徴表を書き出す、別のカード）が先に要る
- 測り方の案（事前登録は別の Issue で、値を見る前に）: dev の LambdaRank（既定のハイパラ）に、この 26 列を**1 つの群として 1 行**足すか外すかの比較だけ。型（(a)〜(e)・型なし）ごとに別々に報告し、型の重みは先に決める。holdout（v3）は凍結したコードで最後に 1〜2 回だけ読む
- 26 列の中から列を選ぶ・定数を変える、は行わない（ハイパラ調整の禁止）
- 合わせて記録するもの（採否には使わない）: 同じ分子式の候補が上位にある分子だけの値（§3.7 の注意を確かめるため）
