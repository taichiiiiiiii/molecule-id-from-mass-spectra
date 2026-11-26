# 誤り分析と特徴群を外した行（2026-11-26 昼、dev）

Issue #257（親 #255・#228・#195 B9）、計画 plan_2026-11-26_noon.md 項目 2。

**dev の値は開発用で、ここから採否もハイパラも決めない。** 採否は 12/6 の holdout で 1 回（#250 の規則）。holdout の特徴表は読んでいない。群の定義・誤り分析の定義・抜き出しの規則は値を見る前に #257 の本文で固定した。

## 1. 特徴群を 1 つずつ外した行（exp004〜exp006）

土台 exp003（dev_b8_cc.parquet、rows=all、64 列、LightGBM 4.7.0 lambdarank 既定・num_boost_round=100・seed 0、leave-one-fold-out）。各行は seed 0 の 1 回。

- 群 R（式 (8) の相対化 25 列＋cc_z）、外した列の一覧の sha256 `57cb7535…668f275`
- 群 M（`mf_*` 13 列＋`mf_*_dmean` 13 列）、sha256 `c5f6d497…3a45`
- 群 N（`np_score`・`has_np`）、sha256 `6c2f959f…a76db`。`np_score_*` の相対化 5 列は群 R なので exp006 には残る（生の値 2 列の効果）
- 「exp003 − cc・cc_z」は exp002 と特徴・行・seed が同じなので回さず、exp002 を使う
- 群と引数: `scripts/ranker/feature_groups.py`（`--check` で重なり 0・exp003 の列に含まれることを確かめる）、当てはめ `scripts/ranker/run_group_ablation.sh`（memguard の下で 1 本ずつ）。当てはめは commit 10eceda、評価は 3af07b4（exp004・exp005）と e07fc38（exp006）、どれも dirty なし。当てはめのコード（lambdarank_oof.py）は 3 つの commit で同じ
- 資源: 当てはめ 15.6 / 15.5 / 15.6 分、peak RSS 1.71 / 1.74 / 2.04 GB、swap 6.68 → 6.56 GB（開始 → 終了）

台帳 `docs/experiments/experiments.csv` の値（対照 = score,prior,frag の出力。12/6 の基準 score,prior とは別物。型 c・e の対照はプールだけの並びなので 0）:

- exp004（−R、38 列）: a 0.8980（対照との差 +0.0018 [−0.0130, +0.0172]、n 789）/ b 0.7408（+0.1114 [+0.0968, +0.1281]、n 787）/ c 0.4335（+0.4335 [+0.4055, +0.4619]、n 788）/ d 0.9688（+0.1164 [+0.1004, +0.1334]、n 745）/ e 0.4108（+0.4108 [+0.3826, +0.4392]、n 792）
- exp005（−M、38 列）: a 0.8968（+0.0006 [−0.0141, +0.0154]）/ b 0.7326（+0.1033 [+0.0874, +0.1197]）/ c 0.4550（+0.4550 [+0.4255, +0.4836]）/ d 0.9736（+0.1213 [+0.1045, +0.1388]）/ e 0.3708（+0.3708 [+0.3425, +0.3986]）
- exp006（−N、62 列）: a 0.8997（+0.0035 [−0.0118, +0.0182]）/ b 0.7384（+0.1090 [+0.0930, +0.1265]）/ c 0.4555（+0.4555 [+0.4275, +0.4837]）/ d 0.9705（+0.1181 [+0.1019, +0.1347]）/ e 0.4055（+0.4055 [+0.3768, +0.4337]）

土台 exp003 との差（行 − exp003、型ごと、分子単位の対応のある bootstrap 2,000 回 seed 0、`feature_groups.py --compare`、`runs_2026-11-26_noon/group_compare_257.json`）。記述だけ:

- −R（exp004）: a −0.0033 [−0.0129, +0.0061] / b +0.0005 [−0.0101, +0.0110] / **c −0.0317 [−0.0468, −0.0160]** / d −0.0037 [−0.0094, +0.0018] / e +0.0071 [−0.0070, +0.0215]
- −M（exp005）: a −0.0044 [−0.0140, +0.0048] / b −0.0077 [−0.0172, +0.0026] / c −0.0102 [−0.0235, +0.0025] / d +0.0012 [−0.0045, +0.0074] / **e −0.0329 [−0.0453, −0.0202]**
- −N（exp006）: a −0.0016 [−0.0085, +0.0050] / b −0.0019 [−0.0107, +0.0072] / c −0.0096 [−0.0210, +0.0019] / d −0.0020 [−0.0063, +0.0025] / e +0.0019 [−0.0095, +0.0126]
- −cc・cc_z（exp002）: a −0.0067 [−0.0156, +0.0021] / b −0.0077 [−0.0169, +0.0013] / **c −0.0148 [−0.0268, −0.0038]** / d −0.0007 [−0.0058, +0.0043] / e −0.0012 [−0.0116, +0.0095]

読み方（事実）: 区間が 0 をまたがないのは 3 つで、どれも正解が PubChem 層にある型（c・e）: 相対化を外すと c が下がり、質量の 26 特徴を外すと e が下がり、cc を外すと c が下がる。プールの型（a・b・d）では、どの群を外しても区間が 0 をまたぐ。1 回の seed で、seed の揺れは測っていない。

漏れの点検 §4.2 (a)〜(e): 3 行とも all_ok（`runs_2026-11-26_noon/leak_check_257.json`）。(d) は #256 の minor 2 に従い、meta の `exclude_features` と求めていない登録済みの列（exp004 の cc_z）から外した列を読み、学習の列に無いこと・期待の一覧と一致することを確かめるよう直した。

## 2. exp003 の誤り分析

定義（#257）: 正解の順位 = exp003 は OOF の点の降順で cand_key の重複を除いた並びでの位置、対照は `ctrl_rank`（score,prior,frag の上位 60）、補助は `q_rank`（PubChem 層の f·z 順）。並びに居なければ ∞。コードは `scripts/ranker/error_analysis.py`、出力は `runs_2026-11-26_noon/error_analysis_exp003.json`。ラベルは当てはめと同じ定義で作り直し、正解の行の数 3,398 が当てはめの n_pos と一致した。実行 549 秒、peak RSS 1.92 GB。

### 2.1 型ごとの 下がった / 上がった / 変わらない（対称に数えた）

対照（ctrl_rank）との比べ:
- a: 下がった 65 / 上がった 70 / 変わらない 654（計 789）
- b: 47 / 239 / 501（計 787）
- c: 0 / 587 / 201（計 788）— 対照の並びに正解が無いので「下がった」は 0（定義どおり）
- d: 3 / 168 / 574（計 745）
- e: 0 / 563 / 229（計 792）— c と同じ

補助（q_rank、PubChem 層の f·z 順）との比べ（c・e の抜き出しに使う）:
- c: 下がった 143 / 上がった 251 / 変わらない 394（計 788）
- e: 186 / 186 / 420（計 792）
- （a 0/786/3、b 0/722/65、d 0/740/5。プールの正解は q_rank に居ないので参考のみ）

下がった分子全体の内訳（抜き出しの基準と同じ比べ。a・b・d は対照、c・e は q_rank）:
- a 65 件: 1 位と正解の分子式が同じ 62、1 位が PubChem 層 34、1 位に参照あり 21、正解に参照あり 65、正解が 25 位の外 5
- b 47 件: 同じ分子式 43、1 位が PubChem 層 3、1 位に参照あり 1、正解に参照あり 0、25 位の外 9
- c 143 件: 同じ分子式 138、1 位が PubChem 層 58（残り 85 はプール）、1 位に参照あり 80、正解に参照あり 0、25 位の外 7
- d 3 件: 同じ分子式 2、1 位が PubChem 層 1、1 位に参照あり 1、正解に参照あり 0、25 位の外 2
- e 186 件: 同じ分子式 184、1 位が PubChem 層 78、1 位に参照あり 95、正解に参照あり 185、25 位の外 7
- 合計 444 件のうち同じ分子式 429（96.6%）

### 2.2 抜き出した 19 件

規則: 型ごとに「下がった」分子を molecule_id の sha1 の昇順で先頭 4 件。**型 d は下がった分子が 3 件しかないので 3 件で、計 19 件**（計画は 20 件。規則どおり他の型から補っていない）。各件の「特徴の差」は (1 位 − 正解) を dev の表の全 430,979 行の標準偏差で割った絶対値の上位 5 列（exp003 の 64 列）、「pred_contrib の差」はその分子の fold のモデルの寄与の差（補助）。候補源 L = プール（対照の上位 60）、Q = PubChem 層。

- **a mf1a_00148**（sha1 00eebb9b）: 正解の順位 exp003 2 / 対照 1。分子式 同じ（1 位 C17H20ClN3O2、正解 C17H20ClN3O2）。候補源 1 位 L・has_ref 1 / 正解 L・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 3.37e-06 − 0.549（6.44 sd）、np_score_z 0.877 − 2.1（1.22 sd）、mf_mean_matched_mz_dmean 41.5 − -1.5（1.04 sd）、mf_mean_matched_mz 186 − 143（0.66 sd）、mf_n_matched_dmean 1.72 − -0.777（0.66 sd）
  - pred_contrib の差（補助、上位 5）: fz_softmax -0.45、ctrl_rank -0.30、mf_mean_matched_mz_dmean +0.26、lib_max_sim_dmax +0.21、lib_max_sim +0.19
- **a mf3a_00181**（sha1 049038c5）: 正解の順位 exp003 5 / 対照 1。分子式 同じ（1 位 C12H21N5O3S、正解 C12H21N5O3S）。候補源 1 位 Q・has_ref 0 / 正解 L・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 0.999 − 1.67e-06（11.72 sd）、lib_max_sim_z -0.47 − 3.44（3.97 sd）、lib_max_sim_dmean -0.0585 − 0.427（3.29 sd）、lib_max_sim 0 − 0.486（3.11 sd）、q_rank 1 − 61（3.02 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank -2.34、has_ref +1.05、fz_softmax +0.73、lib_max_sim +0.50、metfrag_lite_z +0.35
- **a mf3a_00100**（sha1 08425ca1）: 正解の順位 exp003 2 / 対照 1。分子式 同じ（1 位 C21H26F2N2O4、正解 C21H26F2N2O4）。候補源 1 位 Q・has_ref 0 / 正解 L・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 0.995 − 1.38e-07（11.67 sd）、lib_max_sim_z -0.269 − 5.44（5.79 sd）、lib_max_sim_dmean -0.0278 − 0.563（4.01 sd）、lib_max_sim 0 − 0.591（3.78 sd）、lib_max_sim_softmax 0.0109 − 0.0196（3.72 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank -3.87、has_ref +0.79、metfrag_lite_z +0.63、fz_softmax +0.62、fz_z +0.54
- **a mf1a_00124**（sha1 114f0b70）: 正解の順位 exp003 3 / 対照 2。分子式 同じ（1 位 C17H19F2N5O、正解 C17H19F2N5O）。候補源 1 位 Q・has_ref 0 / 正解 L・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: lib_max_sim_dmean -0.129 − 0.722（5.78 sd）、lib_max_sim 0 − 0.852（5.45 sd）、lib_max_sim_softmax 0.00835 − 0.0196（4.78 sd）、lib_max_sim_z -0.657 − 3.67（4.40 sd）、lib_max_sim_dmax -0.942 − -0.0901（3.24 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank -3.38、fz_softmax +1.73、lib_max_sim -1.12、fz_z +1.05、fz +0.68
- **b mf1b_00109**（sha1 0219ae59）: 正解の順位 exp003 19 / 対照 10。分子式 同じ（1 位 C20H24O4、正解 C20H24O4）。候補源 1 位 L・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: tani_pred_z 1.9 − -0.44（2.61 sd）、tani_pred_dmean 0.0381 − -0.0088（2.25 sd）、mf_high_int_cov_dmean 0.0427 − -0.0206（1.94 sd）、tani_pred_dmax -0.0144 − -0.0613（1.87 sd）、metfrag_lite_z -1.34 − 0.314（1.65 sd）
  - pred_contrib の差（補助、上位 5）: mf_high_int_cov_dmean +0.48、np_score +0.46、tani_pred_softmax +0.43、tani_pred +0.39、tani_pred_dmean +0.33
- **b mf1b_00092**（sha1 047cdb95）: 正解の順位 exp003 15 / 対照 6。分子式 同じ（1 位 C9H14O5、正解 C9H14O5）。候補源 1 位 L・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: mf_nearprec_int_cov_dmean 0.00335 − -0.0447（4.25 sd）、mf_frac_matched_dmean 0.0317 − -0.156（3.59 sd）、mf_high_int_cov_dmean 0.0207 − -0.0868（3.30 sd）、mf_frac_matched 0.282 − 0.094（2.69 sd）、mf_nearprec_int_cov 0.0481 − 0（2.16 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +2.94、metfrag_lite_dmax +1.29、has_ref +0.84、cc -0.44、mf_high_int_cov_dmean +0.36
- **b mf1b_00164**（sha1 19110cd6）: 正解の順位 exp003 30 / 対照 21。分子式 同じ（1 位 C26H36O10、正解 C26H36O10）。候補源 1 位 L・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: mf_excl_int_cov_dmean 0.00787 − -0.00114（2.21 sd）、mf_excl_int_cov 0.00901 − 0（2.19 sd）、mf_excl_peak_frac_dmean 0.00512 − -0.0013（1.88 sd）、mf_excl_peak_frac 0.00642 − 0（1.86 sd）、cc_z 0.676 − -0.731（1.42 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +3.38、has_ref +0.95、metfrag_lite_dmax +0.41、lib_max_sim_z +0.22、fz_softmax +0.20
- **b mf4b_00155**（sha1 1936038e）: 正解の順位 exp003 9 / 対照 8。分子式 同じ（1 位 C38H62O12、正解 C38H62O12）。候補源 1 位 L・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_z 1.58 − 0.392（1.19 sd）、cc_z 0.716 − -0.401（1.13 sd）、metfrag_lite_z 1.69 − 0.834（0.86 sd）、cc 0.17 − 0.101（0.63 sd）、fz_dmean 20.2 − 5.02（0.49 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +2.30、np_score -0.53、fz_z +0.52、metfrag_lite_dmax +0.48、has_ref +0.40
- **c mf2c_00170**（sha1 0462020f）: 正解の順位 exp003 16 / q_rank 8。分子式 同じ（1 位 C12H13BrN4O、正解 C12H13BrN4O）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: lib_max_sim_z 4.73 − -0.493（5.30 sd）、lib_max_sim_dmean 0.6 − -0.0626（4.50 sd）、lib_max_sim 0.663 − 0（4.24 sd）、lib_max_sim_softmax 0.0203 − 0.0105（4.19 sd）、mf_low_int_cov_dmean -0.0876 − 0.0691（3.78 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +4.69、metfrag_lite_dmax -1.46、has_ref -0.73、lib_max_sim_dmean +0.30、fz_dmean -0.28
- **c mf2c_00126**（sha1 05899eb3）: 正解の順位 exp003 6 / q_rank 5。分子式 同じ（1 位 C16H11F2N3O、正解 C16H11F2N3O）。候補源 1 位 Q・has_ref 0 / 正解 Q・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 0.364 − 8.17e-06（4.26 sd）、mf_unambig_frac_dmean -0.0274 − 0.137（0.70 sd）、mf_unambig_frac 0.72 − 0.885（0.58 sd）、mf_props_per_matched_dmean 0.0974 − -0.0672（0.49 sd）、mf_n_matched_dmean 3.2 − 4.63（0.38 sd）
  - pred_contrib の差（補助、上位 5）: fz_softmax +0.32、np_score +0.28、mf_unambig_frac_dmean +0.19、fz +0.18、fz_dmax +0.15
- **c mf1c_00136**（sha1 058a686a）: 正解の順位 exp003 5 / q_rank 1。分子式 同じ（1 位 C19H26N2O2、正解 C19H26N2O2）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 3.51e-07 − 1（11.72 sd）、lib_max_sim_dmean 0.654 − -0.186（5.70 sd）、lib_max_sim 0.84 − 0（5.38 sd）、lib_max_sim_softmax 0.0156 − 0.00672（3.77 sd）、lib_max_sim_z 2.8 − -0.795（3.65 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +4.44、fz_softmax -0.99、lib_max_sim +0.75、has_ref -0.62、fz_z -0.39
- **c mf1c_00153**（sha1 05a18736）: 正解の順位 exp003 2 / q_rank 1。分子式 同じ（1 位 C16H12N6O2、正解 C16H12N6O2）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: lib_max_sim_z 4.2 − -0.691（4.97 sd）、lib_max_sim_dmean 0.604 − -0.0993（4.77 sd）、lib_max_sim 0.703 − 0（4.51 sd）、fz_softmax 0.658 − 0.342（3.71 sd）、lib_max_sim_softmax 0.0151 − 0.00746（3.24 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +3.64、has_ref -0.59、fz_softmax -0.58、fz_z -0.54、lib_max_sim +0.49
- **d mf1d_01a13c**（sha1 2bf67f04）: 正解の順位 exp003 28 / 対照 11。分子式 同じ（1 位 C17H32N2O2、正解 C17H32N2O2）。候補源 1 位 Q・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: metfrag_lite_z 2.06 − -1.26（3.33 sd）、metfrag_lite_dmean 0.324 − -0.198（2.98 sd）、mf_low_int_cov_dmean 0.0895 − -0.0314（2.91 sd）、metfrag_lite_softmax 0.0161 − 0.00954（2.68 sd）、ctrl_rank 61 − 11（2.51 sd）
  - pred_contrib の差（補助、上位 5）: metfrag_lite_dmax +2.70、ctrl_rank -1.83、fz_softmax +1.14、has_ref -0.45、metfrag_lite_z +0.42
- **d mf3d_25e6f5**（sha1 df580b24）: 正解の順位 exp003 5 / 対照 3。分子式 同じ（1 位 C17H26N2O4、正解 C17H26N2O4）。候補源 1 位 L・has_ref 1 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: lib_max_sim_z 4.51 − -0.595（5.18 sd）、lib_max_sim_dmean 0.33 − -0.0435（2.53 sd）、lib_max_sim 0.373 − 0（2.39 sd）、has_ref 1 − 0（2.11 sd）、lib_max_sim_softmax 0.0116 − 0.00796（1.53 sd）
  - pred_contrib の差（補助、上位 5）: has_ref -1.70、fz_softmax +1.33、fz_dmean +0.82、fz +0.73、fz_dmax +0.69
- **d mf1d_36fbb2**（sha1 f2abdcab）: 正解の順位 exp003 86 / 対照 51。分子式 違う（1 位 C21H21N5、正解 C18H22FN5O）。候補源 1 位 L・has_ref 0 / 正解 L・has_ref 0
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: ctrl_rank 1 − 51（2.51 sd）、fz_z 0.789 − -1.27（2.06 sd）、np_score_z 0.669 − -0.726（1.40 sd）、metfrag_lite_z 0.243 − -0.962（1.21 sd）、mf_mean_matched_mz_dmean 19 − -30.5（1.20 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +5.47、fz_softmax +1.10、has_ref +1.07、np_score +0.29、fz_dmax +0.28
- **e mf2e_00130**（sha1 02ab39d3）: 正解の順位 exp003 8 / q_rank 1。分子式 同じ（1 位 C16H20N4O、正解 C16H20N4O）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: fz_softmax 6.21e-09 − 0.999（11.71 sd）、q_rank 61 − 1（3.02 sd）、ctrl_rank 1 − 61（3.01 sd）、is_tier 0 − 1（2.01 sd）、metfrag_lite_z 1.36 − -0.413（1.78 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +3.84、fz_softmax -1.18、metfrag_lite_dmax +0.84、metfrag_lite_z +0.59、has_ref -0.53
- **e mf4e_00100**（sha1 03d5e630）: 正解の順位 exp003 18 / q_rank 7。分子式 同じ（1 位 C13H20N4O3、正解 C13H20N4O3）。候補源 1 位 Q・has_ref 0 / 正解 Q・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: lib_max_sim_dmean -0.112 − 0.742（5.80 sd）、lib_max_sim 0 − 0.854（5.47 sd）、lib_max_sim_z -0.667 − 4.4（5.14 sd）、lib_max_sim_softmax 0.00752 − 0.0177（4.32 sd）、lib_max_sim_dmax -0.919 − -0.0646（3.25 sd）
  - pred_contrib の差（補助、上位 5）: has_ref +0.31、lib_max_sim +0.19、lib_max_sim_dmax -0.17、ctrl_rank -0.16、mf_high_int_cov +0.14
- **e mf3e_00183**（sha1 03d84d4c）: 正解の順位 exp003 47 / q_rank 46。分子式 同じ（1 位 C15H17N3O、正解 C15H17N3O）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: ctrl_rank 1 − 61（3.01 sd）、is_tier 0 − 1（2.01 sd）、q_rank 61 − 46（0.76 sd）、lib_max_sim_softmax 0.0158 − 0.0173（0.62 sd）、lib_max_sim_dmean 0.654 − 0.742（0.60 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +4.91、lib_max_sim +0.59、has_ref -0.32、fz_softmax +0.26、fz_dmax +0.25
- **e mf4e_00052**（sha1 0519fe9d）: 正解の順位 exp003 34 / q_rank 22。分子式 違う（1 位 C10H10F3NO4、正解 C13H12FNO2S）。候補源 1 位 L・has_ref 1 / 正解 Q・has_ref 1
  - 特徴の差（1 位 − 正解、|差|/sd の上位 5）: ctrl_rank 1 − 61（3.01 sd）、is_tier 0 − 1（2.01 sd）、q_rank 61 − 22（1.97 sd）、fz_dmean -38.7 − 13.8（1.69 sd）、fz_z -1.16 − 0.414（1.58 sd）
  - pred_contrib の差（補助、上位 5）: ctrl_rank +4.11、fz_dmax -0.57、lib_max_sim_dmean +0.43、fz_softmax -0.30、has_ref -0.29

19 件の pred_contrib の差の 1 位の列: ctrl_rank 13 件、fz_softmax 2、has_ref 2、mf_high_int_cov_dmean 1、metfrag_lite_dmax 1。上位 5 に ctrl_rank が入るのは 16 件。

### 2.3 仮説（次の特徴の候補、3 つ。検証していない）

事実と分けて書く。以下は 19 件と 2.1 の内訳から立てた**仮説**で、どれも dev で測っていない。

1. **仮説 H1（プールと PubChem 層を同じ物差しで比べる入力が欠けている）**: 19 件中 13 件で寄与の差の 1 位が ctrl_rank で、ctrl_rank はプールの候補にしか値が無い（PubChem 層は 61）。対照の点（score＋prior＋frag）にあたる量が PubChem 層の候補には無いため、ランカーは「対照の並びの順位」で源の違いを代用している可能性がある。候補: 対照と同じ式の点（prior・frag の項を含む）を L・Q のすべての候補に計算した列、または L∪Q を通した 1 つの順位
2. **仮説 H2（参照の一致の強さが「同じ化合物の別装置の参照」か「異性体の参照」かを分ける入力が欠けている）**: 型 a で下がった 65 件の正解はすべて参照を持つが、34 件では参照の無い PubChem 層の候補が 1 位に来た（抜き出しの a 3 件で正解の lib_max_sim は 0.49〜0.85）。型 c では 143 件中 80 件で参照を持つ候補が参照の無い正解の上に来た（抜き出しの c 3 件で 1 位の lib_max_sim 0.66〜0.84）。参照の装置・付加イオン・衝突エネルギーが問い合わせと合うか、2 番目に近い参照との差、などの入力が無い。候補: 参照の装置・付加イオンの一致の印、同じ分子式の候補の中での lib_max_sim の差
3. **仮説 H3（同じ分子式の異性体を分ける断片の証拠が弱い）**: 下がった 444 件のうち 429 件（96.6%）で 1 位と正解の分子式が同じ。型 b の 4 件では差の上位が `mf_*`・`tani_pred` の 1〜4 sd 程度の小さな差に分かれている。候補: 異性体ごとに予測した MS/MS とのスペクトルの一致（項目 3 の GLACIER のような順方向の予測）、ピークごとの説明の一致

これらの採否は決めない。次の特徴として入れるかは計画（②）で決め、入れる場合は定義を値を見る前に Issue に固定する。
