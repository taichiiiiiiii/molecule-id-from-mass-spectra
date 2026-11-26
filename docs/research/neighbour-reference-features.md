# 群 N1: 近傍の参照の照合の特徴（MS2Query の入力 4・5 の形）

Issue #261（2026-11-26 昼 計画 項目 4、F2・N1、親 #195）。前段は #252（`ranker-input-features.md` の M4・M5・§4 の群 N1）。

**この文書では特徴の値を見ていない。** MRR も正解のラベルも作っていない（G1 のとおり、dev の台帳の行は別の回）。測ったのは dev 1 集合での秒数とメモリだけ。下の「効いた」はすべて MS2Query の著者の数字で、こちらの測定ではない。

## 0. 出典と読んだ範囲

印: **【開いて確かめた】** = 本文・コードを自分で取得して読んだ。**【要約のみ】** = 他の文書の記述しか見ていない。

1. **MS2Query** — de Jonge NF, Louwen JJR, Chekmeneva E, Camuzeaux S, Vermeir FJ, Jansen RS, Huber F, van der Hooft JJJ. "MS2Query: reliable and scalable MS2 mass spectra-based analogue search". *Nat Commun* 14, 1752 (2023). doi:10.1038/s41467-023-37446-4 、https://www.nature.com/articles/s41467-023-37446-4
   - **【開いて確かめた】** 本文の HTML（Results の手法の概要、Methods の 5 特徴・「Tanimoto scores as structural similarity label」・SQLite ライブラリの段落・Speed performance）と補足 PDF（41467_2023_37446_MOESM1_ESM.pdf）の **Supplementary Note 2（Table S1・S2）と Note 3（平均の取り方、Table S3、装置の種類の特徴）**。2026-11-26 に取得
   - **【開いて確かめた】** コード iomega/ms2query（GitHub、main の commit 8e6e410a）: `ms2query/ms2library.py` の `_calculate_average_multiple_library_structures`、`ms2query/create_new_library/create_sqlite_database.py` の 204 行目、`ms2query/create_new_library/calculate_tanimoto_scores.py` の `get_fingerprint`・`calculate_highest_tanimoto_score`
   - 読んでいない: 補足の Note 1・4〜9、図（Fig. 4 は本文の説明文だけ読んだ）
   - ライセンス: 論文は CC BY 4.0（本文の末尾「Creative Commons Attribution 4.0 International License」）。コードは Apache-2.0（GitHub API の license、2026-11-26）。**こちらは MS2Query のコードを使わず、式（定義）だけを自分で書いた**（`scripts/features/neighbour_ref.py`）。出典はこの文書と docstring に書く

## 1. MS2Query の近傍の入力の定義（出典の文）

- 流れ: 問い合わせとライブラリの全スペクトルの MS2Deepscore → 上位 2,000 本 → 5 特徴のランダムフォレストが「問い合わせとライブラリの構造の Tanimoto」を予測して並べる（本文 Methods、#252 §1.1 と同じ）
- **入力 4**（本文 Methods）: 「the average MS2Deepscore between a query spectrum and all spectra of 10 chemical similar library molecules」。「These 10 library molecules are selected based on the known chemical structures of the spectra in the library, by selecting the library structures with the highest Tanimoto score. For each of the 10 library molecules all corresponding library spectra are selected.」
- **入力 5**（同）: 「the average of the Tanimoto score between the starting library structure and the 10 library structures」
- **平均の取り方**（補足 Note 3「Weighting of the average of multiple library structures」）: 構造ごとに平均してから 10 個を平均（検証 MSE 0.0257）が、全スペクトルをまとめて平均（0.0286）より良かったので前者を採った。コード `_calculate_average_multiple_library_structures` も構造ごとの平均（`average_inchikey_scores`）を 10 個で割る
- **重み**（補足 Note 3「Weighting based on Tanimoto score」、Table S3）: Tanimoto の 1〜3 乗の重みは改善しなかった（検証 MSE 0.0257 / 0.0260 / 0.0262 / 0.0265）ので **重みなし**
- **構造の近さ**（本文「Tanimoto scores as structural similarity label」）: RDKit の daylight 指紋 2,048 ビット、構造の単位は InChIKey の先頭 14 文字。コード `get_fingerprint` は `Chem.RDKFingerprint(mol, fpSize=2048)`
- **10 個の選び方に自分自身が入るか**: 本文には書いていない。コード `create_sqlite_database.py` 204 行目は `calculate_highest_tanimoto_score(list_of_spectra, list_of_spectra, 10)` で、ライブラリの全構造どうしの Tanimoto の上位 10 を取り、自分を除く処理は無い（自分との Tanimoto は 1 なので通常は 1 位に入る）。→ **自分自身を含む**（コードを読んだうえでの判断。本文の文では確かめていない）
- 効き（補足 Note 2、著者の数字）: 全 5 特徴で検証 MSE 0.0255。入力 4 を抜くと 0.0331（Table S2、最大の悪化）、入力 5 を抜くと 0.0259。Gini 重要度は入力 4 が 0.62、入力 5 が 0.02（Table S1）。入力 4 を「1 本のスペクトルの MS2Deepscore」に替えると 0.0337（Note 2 の本文）
- 前もって用意する量（本文 Methods の SQLite の段落）: 「This top 10 list of most related InChIKeys is pre-calculated for every unique InChIKey in the library」

## 2. この競技での形（定義。#261 のコメントで、測る前に固定した）

候補ごとの 3 列（`scripts/features/neighbour_ref.py`）:
- `nbr_ms_mean`: 候補の構造に Tanimoto で近い参照の構造（隠す規則の後に参照スペクトルが 1 本以上残る train の構造）上位 10 個。構造ごとに「残った参照スペクトルそれぞれについて、問い合わせとのエントロピー類似度（問い合わせが複数なら最大）」を平均し、10 個を重みなしで平均
- `nbr_tani_mean`: その 10 個と候補の Tanimoto の平均
- `nbr_n`: 使えた近傍の数（参照のある構造が 10 未満のときだけ 10 未満）
- 近傍 0 件 → `nbr_ms_mean`・`nbr_tani_mean` は NaN。問い合わせのスペクトルが無い分子は `nbr_ms_mean` だけ NaN（単体テスト `test_zero_neighbours_nan`）

定数と出典の場所:
- k = 10: 本文 Methods「10 chemically similar library molecules」、コード `calculate_highest_tanimoto_score(..., 10)`
- 構造ごとに平均してから平均: 補足 Note 3（MSE 0.0257 vs 0.0286）
- 重みなし: 補足 Note 3・Table S3
- 自分自身を含む: コード `create_sqlite_database.py` 204 行目（上の §1）
- 近傍は質量で絞らずライブラリの全構造から: 本文の「selecting the library structures with the highest Tanimoto score」とコード（全構造どうし）
- dev の値を見てこれらを動かさない（playbook §4、ハイパラ調整の禁止）

論文から置き換えたところ（計画 #261 で決めた置き換え。dev の値では選ばない）:
- 類似度: MS2Deepscore → `library.lib_scores`（`lib_max_sim`）と同じエントロピー類似度（`casmi.spectra.entropy_sim`、`SearchCfg` の既定 tol 0.01・ppm_tol 20）。MS2Deepscore は GNPS で学習した外部のモデルで、規則の確認をしていない
- 指紋: RDKit daylight 2,048 ビット → 既存の指紋（`fp_bits.npy` の 10,226 ビット。プールの候補と参照の構造は `pool_fp.npy` の行、PubChem 層の候補は `casmi.chem.raw_fingerprint(smiles)[fp_bits]`。300 構造で両者が一致することを確かめた）
- 構造の単位: inchikey14 → `train_structs.parquet` の正準の鍵（`key`。互変異性の別名 275,810 → 274,186 構造）
- 問い合わせが複数（分子ごとに先頭 16 本、`run.query_spectra` と同じ）: 参照 1 本ごとに問い合わせの最大を取る（`lib_scores` と同じ形。論文は問い合わせ 1 本なので、ここは論文に無い決め事）
- 同点: Tanimoto が同じなら正準の鍵の名前順で先のもの
- #252 の M4 は「候補自身の構造は近傍から除く」と書いていたが、MS2Query のコードは自分を除かないので、論文の側に合わせた（#252 の記述の訂正）

## 3. 漏れの点検（playbook §4.2）

- **(b) 隠す規則**: 集合の `hide_rule.json`（無い集合は `rule=all`、`library.hide_rule` と同じ）と `drop_keys.txt` を当てる。`lib_max_sim` と同じ規則だが、**raw の inchikey14 と正準の鍵の両方で隠す**（正解の互変異性の別名の参照も隠す。`library.load_refs` は raw の鍵だけなので、こちらの方が厳しい）。`rule=enveda_libs` の型（a・e）は隠す構造の enveda 系の行だけを除き、他の装置の行は残る。単体テスト `test_hidden_refs_not_used` で、(1) `rule=all` では隠す構造が近傍に入らず、その行（別名の行も）を使わない、(2) `enveda_libs` では enveda の行だけを使わない、(3) 参照の残らない構造が近傍に入ったら止まる、を確かめた。実データの実行でも「近傍に入った構造はすべて参照が残る構造」を assert している
- **(c) 「近傍の特徴は dev の分子を除いた train から作る」との関係**: 型 a・e は「正解の参照が他の装置にある」型として作った集合なので、`hide_rule` のままでは正解の構造（他の装置の参照）が近傍に入る。これは `lib_max_sim` が正解自身の他の装置の参照を照合するのと同じ扱いで、隠れテストの C1 型（正解の参照がライブラリにある）を写した形。ただし §4.2(c) を文字どおりに読むと「dev の分子を除く」なので、読み方が分かれる
  - ★案（計画の指定）: `--mode hide_rule`（`lib_max_sim` とそろえる）
  - 別の案: `--mode all_truth`（型 a・e でも集合の正解の構造の参照をすべて隠す）
  - **未決**: #252 §4 は「★案を採るなら playbook §4.2(c) の改訂と独立レビューが要り、漏れの点検を緩める変更なのでオーナーの承認も要りうる」と書いた。この回は時間の測定だけなので決めない。**台帳の行を作る前に Issue で決める**（決めるまで dev の当てはめに N1 を入れない）。時間はどちらの mode でも同じ（隠す行が増えるだけ）
- (a)(d)(e): N1 は FPNet のロジット・予測スペクトルを使わないので (a)(d) は当たらない。(e) はランカー側（`lambdarank_oof.py`）の分子単位の GroupKFold のまま
- holdout（v3 の 4 集合）では特徴を作っていない（今日は holdout を読まない）

## 4. 時間とメモリ（dev 1 集合）

対象: `b8_f1_a_t10`（型 a、`rule=enveda_libs`、193 分子・22,098 行・SMILES で一意な候補 21,463）。入力の表 `outputs/local_eval/features/dev_b8.parquet`（sha256 609de988…e6e6c7）。M4 Pro、`memguard_run.sh`（MIN_FREE_PCT 15）の下で `/usr/bin/time -l`。1 回目の開始前: memory_pressure 空き 83%、vm_stat（free + inactive + speculative）22.8 GB、swap 使用 6,182 MB。3 回目の開始前: memory_pressure 空き 81%、swap 6,182 MB、終了時も swap 6,182 MB。

1 回目（参照スペクトルを全部読んでから計算）は peak が大きかったので、流し読みに直して測り直した。3 回とも記録する:
- 1 回目（全部読む版）: 134.7 s、max RSS 9.34 GB、peak footprint 12.29 GB
- 2 回目（参照を train.parquet から 8,192 行ずつ流し読み、候補の指紋を uint8 のまま持つ）: 143.9 s、max RSS 6.02 GB、peak footprint 5.82 GB
- **3 回目（2 回目 ＋ 行列の塊を小さく: 参照 8,192 × 候補 1,024）: 142.6 s、max RSS 3.64 GB、peak footprint 3.46 GB**（これを既定にした）
- 出力は 2 回目と 3 回目で sha256 が同じ（7b08ccd9…d66282）。1 回目とは行・`nbr_n`・`nbr_tani_mean` が完全に一致し、`nbr_ms_mean` の差は最大 1.1e-16（平均の足し方の順の違い）。比べたのは差だけで、値そのものは見ていない
- 3 回目の段ごとの秒数: ライブラリの読み込み 0.4 s（キャッシュあり。初回の作成も 2.0 s）、候補の指紋 13.6 s、Tanimoto の上位 10（候補 21,463 × 構造 274,186、10,226 ビット、float32 の行列積）97.5 s、参照の流し読みと類似度 29.5 s、特徴 0.2 s
- 量: 近傍に入った構造 111,850、使った参照スペクトル 948,666 本、(分子, 構造) の組 174,061。隠した行 1,250（正準の鍵 200）。NaN は 0（近傍が 10 未満の候補 0）

外挿（どれも推定。測っていない）:
- **dev 全体（B8 の 20 集合、430,979 行、集合ごとの一意な候補の合計 425,565）**: 時間の大半（Tanimoto の上位 10）は一意な候補の数に比例するので 142.2 s × 425,565 / 21,461 ≈ **約 2,820 s（約 47 分）**、1 プロセスで集合を順に。peak は集合ごとなので約 3.5 GB のまま
- **holdout（v3 の 4 集合、1,358 分子）**: 分子あたりの候補の数が dev と同じと置くと 142.2 s × 1,358 / 193 ≈ **約 1,000 s（約 17 分）**
- Kaggle の 9 時間の実行に載せる場合: 隠れテストの分子数と Kaggle の CPU の行列積の速さで決まり、**未測定**。ライブラリの構造は 274,186 で固定なので、候補あたり 274,186 × 10,226 ビットの積が要る（ここでは 21,463 候補で 97.5 s）。12/1 の Kaggle での時間測定の対象に入れるかは別に決める
- 速くする余地（未測定、定義は変えない）: 集合をまたいで同じ候補の上位を共有する（20 集合で一意な候補は 370,644。隠す構造は集合ごとに高々 200 なので、上位 210 を 1 回求めてから集合ごとに外せば同じ結果）

## 5. 測る案（今日は測らない）

- **dev**: §3 (c) を Issue で決めた後に、台帳 `experiments.csv` に 1 行（N1 の 3 列を足した版 − 足さない版、`lambdarank_oof.py` の既定のハイパラ・seed）。B8 の 20 集合で **5 つの型（a〜e）を型ごとに** n・MRR・対照・Δ・95% 区間を出す（#247 の書式）。型をまとめた値は出さない。dev の実験の上限 40 本（playbook §4.1）に数える。漏れの点検は §4.2 (a)〜(e) を群ごとに
- **holdout**: 12/6 の凍結版に N1 を入れる場合だけ、v3 の 4 集合（C1 型 `v3_c1u_t10`・`enveda_v1u_t10`、C2 型 `v3_c2_t10`・`v3_c2nu_t10`）で特徴を作り、型ごとに報告する。条件は計画の C1・C2: **型すべてで (新 − score+prior の基準) の 95% CI 下限 ≥ −0.02、かつ 1 つ以上で下限 > 0**（オーナー 11/26 承認 4、#250 の書式）。C1 と C2 を別々に報告する。holdout は 12/6 と 12/11 の 2 回だけで、中間の結果で群を足し引きしない
- 定数（k = 10、平均の取り方、重みなし、自分を含む）は §2 のまま動かさない

## 6. 言えないこと

- MS2Query の効き（入力 4 を抜くと MSE 0.0255 → 0.0331）は、ライブラリのスペクトルを候補とする類似物の検索での数字。こちらの候補は構造で、多くは参照を持たない。このコンペで同じだけ効くとは言えない
- MS2Deepscore をエントロピー類似度に、daylight 2,048 ビットを既存の 10,226 ビットに替えた影響は測っていない
- 型 a・e の近傍に正解の構造が入る扱い（§3 (c)）が隠れテストの C1 型の分布と合うかは確かめていない
- 外挿の時間（§4）は 1 集合からの比例の推定。型 b〜e（`rule=all`）の集合では測っていない
