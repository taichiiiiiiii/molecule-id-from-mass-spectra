# 構造で分けた検証の分布のずれ（Critical Mass）と v3 の保留側の分子の大きさ・質量

Issue #222（#195 の分布の観点、B10 の前段）。計画 `plan_2026-11-24_noon.md` 項目 3。

**結論（記述だけ。事前登録どおり、この値で検証の規則・集合・採否は何も変えない）**

- timsTOF の問い合わせの 3 集合（v3_c1u_t10・v3_c2_t10・enveda_v1u_t10）と見える test は、fold-0 の学習側（G5）より質量の幅が狭く、少しだけ軽い。precursor_mz の四分位幅は 294〜351 などで約 45〜57 Da、G5 は 302〜397 で 95 Da。重原子数の中央値は 4 群とも G5 と同じ 23。P(群 > G5) は 0.41〜0.46
- 他の装置の問い合わせの v3_c2nu_t10 だけが重い側に寄る。重原子数の中央値は 27（G5 は 23）、第 3 四分位は 38（G5 は 26）、precursor_mz の中央値は 384（G5 は 339）。P(群 > G5) は 0.61〜0.63、KS の D は重原子数で 0.30
- Critical Mass が示した「距離のしきい値で分けた保留側は重くなる」仕組みは、私たちの fold には当てはまらない。identity-disjoint の fold は距離のしきい値を使わず、同じ identity の group を種つきの shuffle で 5 つに振り分けているからである（§1.3）。測ったずれは、保留集合の作り方、特に問い合わせの装置の選び方と同じ向きに出ている（§3.2。これは解釈で、原因はまだ確かめていない）
- 見える test は train の写しなので（best_cos ≥ 0.99 が 400/400）、隠れテストの分布の代わりにはならない。見える test が timsTOF の 3 集合に近いことは、隠れテストについて何も言わない

## 1. 出典と split の定義

### 1.1 Oga ほか「Critical Mass: Structure-Disjoint Splits Drift With the Libraries They Are Built On」

- arXiv:2610.12694v1（2026-10-08）、DOI https://doi.org/10.48550/arXiv.2610.12694 。NeurIPS 2026 の workshop「AI Data Readiness for Scientific Discovery」。本文は HTML 版 https://arxiv.org/html/2610.12694v1 で読んだ（本文と付録の見出しまで読み、付録 B〜F は一部だけ）
- split の定義（§1 Introduction、§2 Methods の「Problem formulation」）: 分子の組ごとの距離を出し、しきい値 T より近い組を辺にして、連結成分ごとに 1 つの fold に入れる（single-linkage）。距離は Kretschmer ほかの MCES 距離で、重原子のグラフの辺の編集距離（付録 A.1 の式 (1)(2)、結合の重みは単 1・芳香 1.5・二重 2・三重 3）。MassSpecGym の MCES-10（T = 10）に合わせている。myopic MCES は MassSpecGym が使った計算法で、この論文は admissible な下界と union-find の skip で同じ分割を厳密に速く求める方法を足している（§2「Exploiting the partition」）
- 偏りの指標（§2「Experimental design」の最後の段落、Figure 2 の見出し）: 「mass AUC」は、一様に引いた保留側の化合物が一様に引いた学習側の化合物より重い確率。0.5 が偏りなし
- 要点（§3、Figure 2・3 の見出し）:
  - 公開ライブラリの和集合（227,690 構造）で MCES-10 をかけると、最大の連結成分が 95.5% を占め、保留側は最大 4.5% しか取れない（Enveda-180 だけだと 0.84%）
  - mass AUC は和集合で 0.881、Enveda-180 で 0.836（Figure 2 の見出し）。N = 10^4 で約 0.70、N = 10^5 で約 0.88 と、ライブラリが大きいほど上がる（§3 本文）
  - Enveda-180 は質量の幅が狭い（四分位幅 44 Da）が、それでも学習側の中央値 329.2 Da に対して保留側は 384.3 Da（§3 本文）
  - 仕組み（§3「An absolute threshold is a relative one」）: 10 の編集は小さい分子には大きく、大きい分子には小さい。小さい分子ほど最大の成分につながりやすく、大きい分子が小さな成分に残って保留側に入る
  - T = 5 にすると 80/10/10 の split が作れ、質量の偏りはほぼ消える（Figure 6 の見出し: 中央値 331.2 Da 対 343.1 Da）。ただし保留側の新しさは下がる
- この競技への関係: 私たちの fold は MCES の single-linkage ではないので、この仕組みはそのままは当てはまらない。使えるのは「保留側と学習側の質量・大きさを比べて、検証の読みの偏りを確かめる」という観点と、mass AUC という指標の形（§2 の補助の指標に使った）

### 1.2 Bushuiev ほか「MassSpecGym: A benchmark for the discovery and identification of molecules」

- NeurIPS 2024 Datasets and Benchmarks Track。arXiv:2410.23326（v3、2025-02-14）、https://arxiv.org/abs/2410.23326 。HTML 版 https://arxiv.org/html/2410.23326 で §2・§3.4 と Figure 3 の見出しを読んだ（付録は読んでいない）
- split の定義（§3.4「Dataset splitting」）: 29 千の分子を、MCES を距離にした agglomerative clustering（linkage は最小距離 = single-linkage）で train・validation・test に分ける。linkage の距離のしきい値は 10 で、fold をまたぐ分子の組はどれも編集距離 10 以上になる。さらに装置の種類・衝突エネルギー・付加イオン・分子の出現回数でスペクトルを層別し、メタデータが fold の間で釣り合うようにしている
- Figure 3 の見出し: 従来の 2D InChIKey で分ける split では、Tanimoto > 0.85 の近い分子が test 側に漏れる。MCES の split はそれを防ぐ
- この競技への関係: 私たちの identity-disjoint の fold は、この論文が「漏れる」とする 2D InChIKey 型（正確には scorer key の group）にあたる。近い類縁体は学習側と保留側の両方にあり得る（fold-safe FPNet の README にも「Molecules with related ring systems can still be on both sides」とある）。一方で、MCES 型のような質量の偏りが仕組みから生まれることはない

### 1.3 Gluzdov「CASMI26 Identity-Disjoint Validation Folds」と「CASMI26 Fold-Safe FPNet」（Kaggle dataset）

- https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-identity-disjoint-validation-folds 。手元の写し `data/ext/panel/casmi26-identity-disjoint-validation-folds/README.md`（Kaggle のページの本文と同じ）の「How it was made」と「What is inside」の表から取った
- split の定義: identity は競技の採点と同じ「RDKit で標準の互変異性体にした後の InChIKey の先頭 14 文字」。保存されたキーか scorer key を共有する構造を 1 つの group にまとめ（274,184 group）、種つきの shuffle で group ごとに 5 つの fold（各 55,512〜55,514 構造）に振り分ける。Enveda が測った分子の割合は fold の間で似せてある。`fold_scaffold` は環の骨格も一緒にする、より厳しい分け方
- https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet （手元 `data/ext/foldsafe/README.md`）の「What is inside」の表: `split.parquet` は fold-0 の FPNet から外した 55,367 構造で、理由は fold 0（55,102）か panel の天然物（265）
- 私たちの G5（fold-0 の学習側）の作り方: `structures.parquet` の一意の inchikey14（275,810）から `split.parquet` の 55,367 を除いた 220,443 構造。G5 に `fold_identity` = 0 の構造は無い（数え直した: fold 1〜4 が 55,106・55,118・55,119・55,100）
- この fold は距離のしきい値を使わないので、Critical Mass の言う percolation（大きな連結成分）も、それに伴う質量の偏りも、仕組みとしては起きない。ただし v3 の保留集合は fold 0 の中から型ごとに選んで作っているので（`scripts/cv/build_holdout_v3.py`・`scripts/cv/holdout_expand.py`）、選び方によるずれは起こり得る。それを §3 で測った

## 2. 数え方（Issue #222 に数える前に固定。1 回だけ数えた）

- 群:
  - G1〜G4: v3 の `_t10` 4 集合の正解（`outputs/local_eval/<h>/truth.csv`）。v3_c1u_t10（394 分子）・v3_c2_t10（197）・enveda_v1u_t10（373）・v3_c2nu_t10（394）
  - G5: fold-0 の学習側の構造（§1.3、220,443）
  - G6: 見える test（400 分子・1,213 本）。**見える test は train の写し**で（`outputs/test_library_match.parquet` の best_cos ≥ 0.99 が 400/400、#195 A5）、構造は best_key（train の inchikey14）で引いた
- 指標:
  - precursor_mz（単位はスペクトルの行）。G1〜G4 は各 holdout の `test.parquet` の全行。G5 は train.parquet の行のうち inchikey14 が G5 にあり、付加イオンが t10 の 10 種に入る 1,692,436 行（G5 の全付加イオンは 1,982,405 行）。G6 は `data/test.parquet` の全行（付加イオンはすべて t10 の中）
  - 重原子数（単位は構造）。RDKit 2026.09.1 の `GetNumHeavyAtoms()`。G5 の値は `structures.parquet` の `n_heavy` と 220,443 件すべて一致した
  - 件数・中央値・第 1/第 3 四分位（numpy linear）、G5 との 2 標本 KS の D と p 値（scipy 1.18.1 `ks_2samp`。ここで書いた numpy の D と 1e-12 以内で一致することを assert した）、補助に P(群 > G5)（Mann–Whitney の U/(n1·n2)、同値は 0.5。Critical Mass の mass AUC と同じ形）
- コード: `scripts/research/split_drift.py`（純粋な関数は `tests/test_split_drift.py`）。実行は 1 回、16 秒、peak RSS 0.73 GB = 0.68 GiB（`/usr/bin/time -l` の maximum resident set size 729,088,000 B。JSON の `maxrss_gb` は GiB で 0.68）。出力 `outputs/local_eval/split_drift/split_drift.json`（写しは `docs/research/split-drift/split_drift.json`、sha256 e25aa00b…）。ログ `logs/split_drift_222.log`
- 漏れの確認: G1〜G4 の正解のうち G5 に入るものは 0（4 集合とも）。見える test の構造は 317/400 が G5 に入る（残りの 83 はすべて split.parquet の hold = fold 0 の構造）

## 3. 結果

### 3.1 表（6 群 × 指標）

precursor_mz（スペクトルの行）

| 群 | n | 第 1 四分位 | 中央値 | 第 3 四分位 | KS D（対 G5） | KS p | P(群 > G5) |
|---|---|---|---|---|---|---|---|
| G1 v3_c1u_t10 | 1,836 | 294.12 | 325.13 | 351.15 | 0.2003 | 5.5e-65 | 0.411 |
| G2 v3_c2_t10 | 896 | 305.15 | 334.18 | 350.43 | 0.2088 | 1.0e-34 | 0.438 |
| G3 enveda_v1u_t10 | 949 | 305.19 | 332.01 | 351.10 | 0.2074 | 3.2e-36 | 0.438 |
| G4 v3_c2nu_t10 | 3,309 | 308.17 | 384.12 | 528.31 | 0.2315 | 3.9e-156 | 0.606 |
| G5 fold-0 の学習側 | 1,692,436 | 302.10 | 339.18 | 397.20 | — | — | — |
| G6 見える test（train の写し） | 1,213 | 300.14 | 329.20 | 347.16 | 0.2165 | 2.5e-50 | 0.421 |

重原子数（構造）

| 群 | n | 第 1 四分位 | 中央値 | 第 3 四分位 | KS D（対 G5） | KS p | P(群 > G5) |
|---|---|---|---|---|---|---|---|
| G1 v3_c1u_t10 | 394 | 20 | 23 | 25 | 0.1177 | 3.3e-05 | 0.439 |
| G2 v3_c2_t10 | 197 | 21 | 23 | 25 | 0.1379 | 9.9e-04 | 0.449 |
| G3 enveda_v1u_t10 | 373 | 21 | 23 | 25 | 0.1151 | 9.5e-05 | 0.458 |
| G4 v3_c2nu_t10 | 394 | 22 | 27 | 38 | 0.3026 | 2.0e-32 | 0.631 |
| G5 fold-0 の学習側 | 220,443 | 21 | 23 | 26 | — | — | — |
| G6 見える test（train の写し） | 400 | 21 | 23 | 25 | 0.1490 | 3.3e-08 | 0.442 |

### 3.2 読み（記述だけ）

- KS の p 値は小さく出るが、precursor_mz はスペクトルの行が単位で、同じ分子の行は独立ではない（v3_c2nu_t10 は 394 分子に 3,309 行）。p 値は「独立な標本なら」の値で、差の大きさは D と P(群 > G5) で読む
- timsTOF の 3 集合と見える test は、中央値が G5 より 5〜14 Da 低く、四分位幅が G5（95 Da）の半分ほど（45〜57 Da）。重原子数の中央値は G5 と同じで、第 3 四分位が 1 小さい。分布の上の裾（400 Da 以上、重原子 27 以上）が薄い、という形である
- v3_c2nu_t10 は逆に重い側に裾が長い（precursor_mz の第 3 四分位 528、重原子数の第 3 四分位 38）
- 解釈（未確認）: timsTOF の集合は Enveda が測った分子から選ばれている。Critical Mass は Enveda-180 の質量の幅が狭い（四分位幅 44 Da）と書いており、timsTOF の 3 集合の幅（45〜57 Da）はそれに近い。v3_c2nu_t10 はほぼ timsTOF 以外の装置の問い合わせなので（3,309 行のうち timsTOF は 15 行）、他の公開ライブラリの、より重い分子が入る。どちらも fold の分け方ではなく保留集合の選び方から来ている、という説明と矛盾しない。ただし、v3 の選び方の各段（装置・付加イオン・参照の有無）でどれだけずれるかは数えていない
- 検証の読みへの意味（記述）: v3_c2nu_t10 の差は、大きい分子での差を他の集合より重く含む。timsTOF の 3 集合は、学習側の上の裾（大きい分子）をほとんど測っていない。隠れテストの分布は分からないので（見える test は train の写し）、どちらが隠れテストに近いかはここからは言えない

## 4. v3 で測る案（オーナーの判断に回す。KS の値によらない）

- 案: 採否の報告（#196 の R1 で読む 4 集合の差）に、**重原子数の帯ごとの差を記述として添える**。判定の規則（R1）は変えない。帯の差は採否に使わない
- 帯: G5 の重原子数の第 1・第 3 四分位で 3 つに分ける（≤ 21 / 22〜25 / ≥ 26）。境界は学習側だけで決まり、保留側の値は使わない。計画の文言は「質量の帯」だが、ここでは**質量の代わりに重原子数で帯を切った**。理由: precursor_mz は付加イオンで変わり、1 つの分子に複数の値がある（帯は分子の単位で決める必要がある）。重原子数は構造ごとに 1 つで、分子の大きさの指標としても §3 の表で質量と同じ向きに動いている。中性の単同位体質量（`structures.parquet` の `mono_mass`）で切る形もあり得る。どちらにするかもオーナーの判断に含める（`scripts/research/split_drift_bands.py`、`outputs/local_eval/split_drift/bands.json`、写しは `docs/research/split-drift/bands.json`）
- 帯ごとの分子数: v3_c1u_t10 138 / 179 / 77、v3_c2_t10 58 / 106 / 33、enveda_v1u_t10 104 / 199 / 70、v3_c2nu_t10 97 / 82 / 215
- MDE: 集合の MDE は 2.8/1.96 × fragnn v1 − 対照の 95% bootstrap 区間の半幅（2.8 = 1.96 + 0.84 で、両側 5%・検出力 80%。対になった差 Δrr の区間を使うので対の設計の分は入っている。半幅 ∝ 1/√n）（`outputs/local_eval/a0i_1123/a0i_result.json`、#195 A0 (i)）で、v3_c1u_t10 0.0249・enveda_v1u_t10 0.0279・v3_c2nu_t10 0.0279（#199 の v3_c1u_t10 の値は 0.0230 で、見積もり方が違う）。帯の MDE は MDE_集合 × √(n/n_帯) で粗く見積もると 0.037〜0.064（帯の間で sd(Δrr) が同じという仮定つき、未確認）。v3_c2_t10 は手元の対照が 0（正解が手元の候補に無い）なので MDE は出さない。**帯の差は MDE が集合の 1.5〜2.3 倍で、どの帯でも、これまでの段の差の大きさ（fragnn v1 − 対照: v3_c1u_t10 +0.0033・enveda_v1u_t10 +0.0058・v3_c2nu_t10 +0.014、`a0i_result.json`）は見分けられない。** 使えるのは「集合の差が 1 つの帯だけから来ていないか」の目安だけである
- 費用: 報告のスクリプトに帯の集計（既にある submission CSV・truth.csv・重原子数を読むだけ、CPU で数秒）と、そのテストを足す。GPU 0、提出 0、Kaggle 0。差の帯ごとの区間は既存の bootstrap の関数を帯ごとに呼ぶだけ
- やらない理由になり得ること: 帯の差を見ると、それで採否を変えたくなる（選択の危険）。足すなら「帯の差は記述だけ、採否は R1 だけ」と報告の雛形に書いておく必要がある

## 5. ライセンス・再現性

- Critical Mass: arXiv の perpetual non-exclusive license。読んだ範囲（本文・付録の見出し・参考文献）でコードの公開先は見つからなかった（「無い」ではなく「読んだ範囲で見つからない」）。この調査では論文のコードもデータも使っていない。使ったのは指標の形（mass AUC）だけ
- MassSpecGym: arXiv の perpetual non-exclusive license。データとコードは https://github.com/pluskal-lab/MassSpecGym で公開と本文にある。ライセンスの種類は本文の読んだ範囲では確認していない。この調査ではデータもコードも使っていない
- identity-disjoint folds・fold-safe FPNet: 構造と識別子は競技の training file から来ており、競技のデータの規約に従う（identity-disjoint folds の README の「Licence and credits」は「competition data terms」とだけ書く）。fold-safe FPNet の README の「Licence and credits」は、checkpoint と競技由来のファイルを「non-commercial use with attribution」とし、学習・評価のスクリプトを Apache-2.0 としている。ここでは `structures.parquet`（fold の割り当て・SMILES・n_heavy）と `split.parquet` を読んだだけで、競技の外へは出していない
- 再現: `.venv/bin/python scripts/research/split_drift.py`（16 秒、peak 0.73 GB = 0.68 GiB）と `.venv/bin/python scripts/research/split_drift_bands.py`（数秒）。入力は手元の train.parquet・test.parquet・4 集合の truth.csv と test.parquet・`outputs/test_library_match.parquet`・上の 2 つの parquet。乱数は使わない。重原子数は RDKit 2026.09.1、KS の p 値は scipy 1.18.1

## 6. まだ分からないこと・次の段

- v3 の選び方の各段（装置・付加イオン・参照の有無）で、どの段でずれが生まれるか（数えていない）
- 隠れテストの分布（見える test は train の写しなので測れない）
- B10（型の比率を変えたときの感度、train と見える test の敵対的検証）はこの調査の次の段で、ここではしていない
