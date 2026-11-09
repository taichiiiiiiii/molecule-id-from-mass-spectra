# 外部ライブラリ（ExtLib・MassBank harmonized）型の検証集合を作れるか（Issue #52）

調べた日: 2026-11-10。件数はすべて `scripts/cv/extlib_coverage.py`（メタデータ列だけを読む）の出力から写した。スペクトル本体は読んでいない。

## 1. 出典
- ExtLib: https://www.kaggle.com/datasets/takumuhata/casmi26-extlib（副題「External MS2 spectral library (GNPS+MassBank+MoNA) merged」、説明文は空、Kaggle 上の licence 表示 CC0-1.0）。ファイル: `ext_meta.parquet`（スペクトル 200,724 行: inchikey14, normalized_smiles, adduct, ionization_mode, instrument_type, precursor_mz, formula）、`ext_mass.parquet` / `ext_fp_keys.parquet`（各 144,924 構造）、`ext_mz.npy` / `ext_in.npy`（各約 103 MB、今回は取得していない）、`coconut_index.parquet`
- MassBank harmonized: https://www.kaggle.com/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized（Kaggle 上の licence 表示 CC-BY-NC-SA-4.0）。説明文あり: MassBank/MassBank-data（https://github.com/MassBank/MassBank-data）を解析した派生物で、「Build pipeline (reproducible)」として 6 段の手順（clone → flat-file の解析 → RDKit で SMILES 正規化・InChIKey 作成 → ND licence の 149 件を除く → InChIKey で構造の重複を除き Morgan/MACCS を計算 → Parquet）を文章で書いている。利用条件は「research and competition use」、元の機関の出典表示を求めている。`spectra.parquet` 139,744 スペクトル（source はすべて MassBank、行ごとの `license` 列あり）、`structures_with_fingerprints.parquet` 20,204 行
- 使っている公開ノートブック（上位、2026-11-09 時点の一覧）: https://www.kaggle.com/code/alexchilton/casmi26-lb-0-484-peaks-to-structures-extlib 、https://www.kaggle.com/code/ozertuu/0-485-sota-casmi26-from-peaks-to-structures 。上乗せの規則（10 ppm 以内・cosine ≥ TAU・一致ピーク 4 本以上で 1 位に上げる、自前のライブラリ一致 ≥ 0.90 の分子は除く）は 11/9 の日報の読み取りによる

## 2. 件数（train 275,810 構造、fold-safe hold 55,102 構造）

ExtLib:
- 構造 151,338（スペクトル 200,724）。train にある 6,414、train に無い 144,924（`ext_mass.parquet` / `ext_fp_keys.parquet` の構造数と同じ。レビュー役の再計算では、この 2 つの構造表には train の構造も hold の構造も 0 件。作者は構造表から train の構造を除いているらしい。train の構造は `ext_meta` のスペクトル行（8,701 行）にだけ残る）
- (i) train に無い構造で外部スペクトルを持つ数: 144,924
- (ii) そのうち timsTOF の問い合わせを作れる数: **0**（timsTOF のスペクトルは train の enveda 系にしか無く、train に無い構造には問い合わせが無い）。ExtLib 自体にも装置名に "tims" を含むスペクトルは 0
- (iii) 装置: LC-ESI-Orbitrap 196,207（97.7%）、LC-ESI-qTof 1,272、ESI-Orbitrap 706、LC-ESI-QTOF 645、空欄 359 ほか
- (iv) hold 構造との重なり 1,306。そのうち train に timsTOF スペクトルがある構造は **2**、train に公開ライブラリの（test と同じ付加イオンの）スペクトルがある構造は 1,143
- train にある ExtLib 構造の train スペクトルの出どころ（行数）: riken 59,993、gnps 45,756、massbank 17,730、pluskal_ms2 17,001、mona 15,454、msdial 8,681 ほか

MassBank harmonized:
- 構造 18,316（InChIKey 先頭 14 文字）。train にある 10,401、無い 7,915。timsTOF の問い合わせを作れる「train に無い構造」は 0、"tims" を含むスペクトル 0
- 装置: LC-ESI-QTOF 53,555、LC-ESI-QFT 26,190、LC-ESI-ITFT 22,677、EI-B 11,202 ほか
- hold との重なり 2,056、そのうち timsTOF あり **0**、公開ライブラリの問い合わせあり 1,916
- 行ごとの licence: CC BY 49,317、CC BY-NC-SA 32,754、CC BY-SA 21,630、dl-de/by-2-0 20,658、CC BY-NC 8,434、CC0 6,951
- ExtLib と重なる構造 1,999

既存の検証集合の正解との重なり（truth.csv の inchikey14）: v3_c1 0/200、v3_c2 0/200、enveda_v1 0/200、v3_c2n ExtLib 18/195・MassBank 44/195。11/9 の日報は v3_c2n の ExtLib を「21 件」と書いていたが、この定義では 18。差の原因は未確認（日報側の数え方が残っていない）。

## 3. 結論
- **隠れテストと同じ timsTOF の問い合わせで、外部ライブラリにだけ参照がある正解を持つ集合は作れない**（ExtLib 2 構造、MassBank 0 構造）。class 2 型（train に無い構造）は問い合わせが無いので、定義上測れない
- 作れるのは、装置がテストとずれた問い合わせ（公開ライブラリのスペクトル）の集合だけ（ExtLib 1,143 構造、MassBank 1,916 構造）。これで測れるのは「外部の参照しか無い正解を、別の公開装置のスペクトルから拾えるか」で、隠れテストで効く性質（timsTOF の問い合わせと Orbitrap 主体の外部参照の一致の強さ）は測れない

## 4. licence と再現性
- ExtLib: Kaggle 上は CC0 と表示。ただし中身は GNPS・MassBank・MoNA を合わせたもので、元のスペクトルには NC や SA 条件のものがある（MassBank harmonized の行ごとの licence を参考にした。ExtLib の行ごとの licence は無い）。作り方のコードは、データセットの説明・ファイル一覧からは見当たらない（作者の他のデータセットや議論は今回は探していない）。競技の外部データの規則に合うかは未確認
- MassBank harmonized: Kaggle 上は CC-BY-NC-SA-4.0、説明文の利用条件は「research and competition use」、行ごとの licence（NC・SA を含む）を保持。作り方は説明文に 6 段の手順として書かれているが、コードそのものはデータセットからは見当たらない（手順は MassBank-data から自前で書き直せる粒度）
- どちらも最終提出で使うなら、作り方（取得元・結合のコード）を自前で書くか、作者のコードを見つける必要がある

## 5. 測るなら（案。今日は実行しない）
案 E1（装置ずれ版、弱い代理）:
- 集合 `v3e`: hold ∩ ExtLib の 1,143 構造から、test と同じ付加イオンの公開ライブラリのスペクトルを問い合わせにして 200 分子を選ぶ（`build_holdout_v3.py` の公開ライブラリ側の選び方と同じ）。正解の train 参照は全部隠し、正解はプールに残す（v3_c2n と同じ hide_rule）。外部参照は ExtLib のスペクトルだけ
- 前提として必ず確かめること: 問い合わせのスペクトルと ExtLib のスペクトルが同じ記録の写し（GNPS・MassBank・MoNA は train にもある）でないこと。`ext_mz.npy` / `ext_in.npy`（約 205 MB）を取り、問い合わせと同じ構造の ExtLib スペクトルとの cosine が 0.99 以上のものは同じ記録とみなして除く。除いた件数を先に記録する
- コマンド（予定。`run.py` に外部ライブラリ段は無いので、段の追加が先に要る）: `for h in v3_c1 v3_c2 enveda_v1 v3_c2n v3e; do .venv/bin/python scripts/local_eval/run.py --holdout $h --stages score,prior,frag[,extlib]; done`
- 採否（事前に固定する形）: 分子ブートストラップ 2,000 回・seed 0 の「新 − 対照」95% 区間で、v3_c1・v3_c2・enveda_v1・v3_c2n すべて下限 ≥ −0.02、かつ v3e で下限 > 0。TAU などの定数は公開ノートブックの値を写さず、事前に 1 つに決めて動かさない
- 前提 2: 構造表（`ext_mass` / `ext_fp_keys`）には hold の構造が 0 件なので、v3e の正解は ExtLib のスペクトル行（`ext_meta`）にしか現れない。公開ノートブックの上乗せがスペクトル行での一致だけで 1 位を決めるなら v3e で再現できるが、構造表を候補源にも使う部分（train に無い 144,924 構造）は v3e では測れない。実装前に、公開ノートブックのどちらの使い方かをコードで確かめる
- 限界: 装置がずれているので、v3e で勝っても隠れテストで勝つ根拠は弱い。v3_c1 型でクラス 1 を壊さないか（スキルの記録には、素朴な「ライブラリ一致 ≥ τ を 1 位」が手元で装置をまたぐクラス 1 を −0.07 下げた、とある。リポジトリの docs・Issue では出典を見つけられず、未確認）を見る対照としての意味の方が大きい

案 E2: v3_c2n の ExtLib 重なり 18 分子だけで測る → 分子数が少なく、SE が大きすぎるので採らない。

## 6. 取得物
取得したのはメタデータ 3 ファイル（ExtLib）と 3 ファイル（MassBank）で合計約 65 MB。scratch（`~/.hermes/profiles/massspec/cache/scratch`）に置き、件数を記録してレビューを通したあと削除する。
