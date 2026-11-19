# 監査: v2 プールの `src` の意味と、検証 v3 の正解・おとりの `src` 内訳（Issue #162）

読むだけの監査（採点しない）。`scripts/local_eval/run.py` に触れないので v3 の対照は変わらない。lfus（#150、v3_c2n −0.0387 で不採用）の判定は、この結果がどうでも選び直さない（Issue #162 の事前登録）。

- コード: `scripts/cv/src_audit.py`（テスト `tests/test_src_audit.py`）。事前登録どおりの表 1・表 2 と判定は ee1543a（実行前にコミット）、追加の確認 `--coco-canon` は ed50070（1 回目の実行の後、2 回目の前にコミット）
- 出力（git の対象外）: `outputs/local_eval/src_audit/src_audit.json`（3 回目、b926ae4。pubchem の列を直した後）、`src_audit_run1_pubchem_bug.json`（1 回目の写し）、`src_audit_cococanon.json`（2 回目、追加の確認込み）。pubchem の列の修正は b926ae4（§8）

## 1. `src` を作ったコード（確認）

`ahmedberatozer/casmi26-training-code`（Kaggle データセット、2026-10-19 の版、README は MIT と表記、Kaggle の表示は "other"）の `scripts/05_build_pool.py`（sha256 `faa560b5…f8c888f`）:

- 3 行目（docstring）: `pool_meta.parquet: key, smiles, formula, mass, n_heavy, src (1=train, 2=coconut, 3=both), train_sid`
- 21〜28 行目: train 側を score key（互変異性体を正準化した InChIKey14）で重複除去し、COCONUT 側は `~co.key.isin(tr_keys)`（train に無いものだけ）を残す
- 29〜30 行目: `both = set(co.key) & tr_keys`、`src_tr = np.where(trk.key.isin(both), 3, 1)`
- 37〜38 行目: `src=np.concatenate([src_tr, np.full(len(cok), 2, np.int8)])`、`train_sid` は train 側だけ sid、COCONUT 側は −1
- COCONUT 側の入力は `scripts/02_build_coconut.py`（COCONUT の CSV lite 2026-09 → 最大の断片・電荷を外す・立体を落とす → score key で重複除去、100〜1500 Da、重原子 5〜150）

つまりコードの上では **1 = train 構造で COCONUT に無い、2 = COCONUT だけ（train に無い）、3 = train 構造で COCONUT にもある**。

注記: 手元の `pool_meta.parquet`（710,701 行）の列は key・smiles・formula・mass・n_heavy・src・train_sid・np_likeness・pid で、今の 05_build_pool.py が書く tier・in_pcl・pubmed・patent・anno が無い。手元のファイルはこのスクリプトの今の版の出力そのものではない（前の版か、列を落とした写し）。そのためコードは「意味の根拠」、表 1 は「手元のファイルがその意味どおりか」の確認として読む。

## 2. 表 1: `src` の値 × 独立の所属（プールの行数）

出力 `src_audit.json`（3 回目の実行、b926ae4。§8 の実行の記録を参照）。分母は各 `src` の行数。

| 所属 | src 1（248,440 行） | src 2（436,515 行） | src 3（25,746 行） | 独立か |
|---|---|---|---|---|
| (a) train_raw: key ∈ train.parquet の inchikey14 | 246,335（99.15%） | 22（0.005%） | 25,106（97.51%） | 独立（競技データ） |
| (b) train_canon: key ∈ train_structs.key | 248,440（100%） | 0 | 25,746（100%） | 同じ作者の表 |
| (c) train_any: (a) か (b) か train_structs.rawkey | 248,440（100%） | 22（0.005%） | 25,746（100%） | (a) を含む |
| (d) coco_raw: key ∈ COCONUT 全件 CSV の standard_inchi_key[:14] | 462（0.19%） | 393,760（90.21%） | 23,849（92.63%） | 独立（COCONUT 公式の CSV） |
| (e) coco_np: pool_popularity の np_coconut == 1 | 9 | 436,515（100%） | 25,746（100%） | **独立でない**（§5） |
| (f) coco_any: (d) か (e) | 466（0.19%） | 436,515（100%） | 25,746（100%） | (e) に頼る |
| (g) pubchem: pubchem_cid > 0 | 232,938（93.76%） | 421,081（96.46%） | 25,452（98.86%） | 別のデータセット |
| 内部: train_sid ≥ 0 | 248,440（100%） | 0 | 25,746（100%） | 内部の列 |
| 内部: np_likeness がある | 0 | 436,515（100%） | 0 | 内部の列 |

- 入力の大きさ: プール 710,701 行（key は一意、pid = 行番号を assert）、train.parquet の inchikey14 275,810 種、train_structs 275,810 行、COCONUT CSV 738,827 行（raw キー 479,721 種、sha256 `1360a7f3…83d9da75`）、pool_popularity 710,701 行
- train_structs.key が自前の `casmi.chem.score_key(smiles)` と同じか: 無作為 2,000 行（seed 0）で 2,000/2,000 一致
- (a) が src 1・3 で 100% に届かない分（2,105 行・640 行）は、プールの key が正準キーで train.parquet の inchikey14 が raw キーなので、互変異性体の別名の分だけずれると考えられる（未検証。(b)・(c) は 100%）。src 2 の 22 行は、COCONUT の構造の正準キーが別の train 構造の raw キーと同じになったものと考えられる（未検証。(b) は 0）
- PubChem 層（`data/ext/pubchem_tier`、約 1.06 億行の SMILES、キーの列なし）との積は**数えていない**（未測定）

## 3. 判定

- 事前登録の判定（`judge`、コードどおり）: **「確かめた」**。(i) コードの定義あり（05_build_pool.py の sha256 と 3・29・30・37 行の文字列を照合）、(ii) src {1,3} の train_any 100%・train_sid ≥ 0 100%、src 2 の train_any 0.005%、(iii) src {2,3} の coco_any 100%、src 1 の coco_any 0.19%
- **ただし (iii) は独立でない列に頼っていた**: 1 回目の実行の後、`pool_popularity.csv` の作者の TECHNICAL_NOTES.md に「COCONUT membership follows the upstream pool」「NP bits 1/2/4 mean raw InChIKey14 membership in COCONUT-pool / LOTUS-knowledge / NPAtlas-knowledge keys」とあり、np_coconut は v2 プールの COCONUT の印の写しだと分かった。src 2・3 で 100%、src 1 で 9 行という一致は、独立の確認ではなく写しの確認にすぎない。事前登録で (e) を「別のデータセット」と書いたのは誤り
- 独立の列だけで読むと: (d) coco_raw だけでは src {2,3} が 417,609 / 462,261 = **90.34%** で、しきい値 99% に届かない（src 1 は 0.19% で ≤ 1%）
- 事前登録の外の追加の確認（§5、COCONUT の行を自前で標準化して正準キーを作る）を足すと src {2,3} が 462,025 / 462,261 = **99.95%**、src 1 が 481 / 248,440 = 0.19% で、同じしきい値を満たす
- **結論**: train 側（1・3 = train 構造、2 = train に無い）は**確かめた**（独立の (a)・(c) と、コードの 3・29・30・37 行）。COCONUT 側（2・3 = COCONUT にある、1 = 無い）は、事前登録の独立の列（raw キー）だけでは**未確認**（90.34%）で、事前登録の外の自前の正準キーを足すと 99.95% でコードどおり。しきい値は動かしていない。まとめると **src の意味はコードのとおり（1 = train だけ、2 = COCONUT だけ、3 = 両方）で、train 側は事前登録どおり確かめた、COCONUT 側は事前登録の外の確認で裏づけた**、と書く

## 4. 表 2: v3 の 4 集合の正解・おとりの `src`

出力 `src_audit.json`。おとり = run.py の `visible_window`（±10 ppm、pool: remove なら drop_keys を除く）の中の、key ≠ 正解の正準キーの**全行**。「窓の外」= 窓の中心（`pc_logits.npz` の `__targets__`）から ±10 ppm の外に正解がある分子。

| 集合（hide_rule・pool） | 正解の src（元のプール） | 正解の src（窓の中） | おとりの行数 | おとりの src 1 / 2 / 3 | 分子ごとの「正解と同じ src のおとり」の行数（p25 / p50 / p75） |
|---|---|---|---|---|---|
| v3_c1（enveda_libs・keep） | 1: 200 | 1: 200 | 27,304 | 74.85% / 23.59% / 1.56% | 34.75 / 67.5 / 144 |
| v3_c2（all・remove） | 1: 200 | 窓の外 200（プールから消してある） | 26,601 | 72.90% / 25.60% / 1.50% | —（正解が窓に無い） |
| v3_c2n（all・keep） | 3: 195 | 3: 177、窓の外 18 | 15,611 | 29.07% / 66.45% / 4.48% | 0 / 2 / 5（最小 0、最大 27） |
| enveda_v1（all・keep） | 1: 200 | 1: 200 | 26,115 | 72.40% / 26.29% / 1.31% | 28.75 / 66 / 127.75 |

- #155（PR の P5）の数「v3_c2n のおとりは 6.7% が src 3、v3_c1 は 72.3% が src 1」は、窓から seed 0 で 10 行ずつ抜いた値。全行では v3_c2n の src 3 は 4.48%、v3_c1 の src 1 は 74.85%。正解の内訳（v3_c1 は 200/200 が src 1、v3_c2n は 177/177 が src 3）は同じ
- v3_c2n の正解が全部 src 3 なのは**作り方による**: `build_groups.py` 163 行が `V3.select_c2p(meta, split_hold, coco, ...)` を呼び、`select_c2p`（`build_holdout_v3.py` 56〜63 行、61 行 `cand = sorted((ok & set(allowed) & set(coconut)) - set(exclude))`）は fold 0 の hold（train 構造）と `coconut_keys()`（`build_groups.py` 128〜131 行、`src ∈ {2,3}` の key）の積を取る。train 構造で COCONUT にあるもの = src 3 なので、作り方により src 3 になる（実測 195/195。raw キーと正準キーの差で src 2 の行が入る余地は理屈の上では残る）
- 他の 3 集合（600 構造）の正解は全部 src 1（COCONUT に無い train 構造）。v3 の 4 集合の正解の中で COCONUT の構造は v3_c2n だけ

## 5. 追加の確認（事前登録の外）

`--coco-canon`（ed50070、2 回目の実行）: COCONUT CSV の行のうち raw キーがプールの key に無い 93,640 行を、02_build_coconut.py と同じ手順（`casmi.chem.standardize_smiles`: 最大の断片 → 電荷を外す → 立体を落とす、電荷が残れば外す）→ `score_key` で自前に正準キーにした（53,700 種、2 ワーカーで 1,835 s）。

- src {2,3} で coco_raw に当たらない 44,652 行のうち 44,416 行（99.47%）が自前の正準キーで当たる。残りは 236 行
- raw か自前の正準キーで当たる行: src 1 は 481 / 248,440（0.19%）、src 2 は 436,298 / 436,515（99.95%）、src 3 は 25,727 / 25,746（99.93%）
- src 1 の 481 行（train 構造なのに COCONUT の全件 CSV にある）と src {2,3} の残り 236 行の理由は調べていない（候補: プールは CSV **lite** から、ここでは全件 CSV を使った版の差。02_build_coconut.py の質量 100〜1500 Da・重原子 5〜150 の絞り込み）
- 2 回目の出力の表 1・表 2・判定は 1 回目と完全に同じ（json の比較で一致）。ただし (g) pubchem の列は 3 回目で直す前の誤り（pubchem_cid = −1 を「ある」と数えていた）を含む

## 6. 読み方

- **run.py（v3 の対照の縮小版）は `src` を読まない**: `load_pool` が読む列は key・smiles・mass・pid と `pool_popularity.csv` の log1p_substances・log1p_pubmed だけ（`scripts/local_eval/run.py` 72〜77 行）、frag の段は mass・formula（317 行）。`src`・np_coconut・train_sid・np_likeness を読む段は無い（`scripts/local_eval/`、`data/ext/v4g/casmi26-glacier/*.py`、`data/ext/v4g/casmi26-v3-models/code/casmi/*.py`、`data/ext/frag/*.py`、#150 のブランチの `lfus.py`・`library.py` を grep して該当 0。`scripts/` の中で `pool_meta.src` を読むのは集合を作る側の `build_groups.py`・`build_holdout_v3.py`（の呼び出し元）と、監査の `lineage_leak_audit.py`・`msg_wild_repr_check.py`・この `src_audit.py` だけ）。したがって v3_c2n の判定が `src` そのものを近道にしたことは無い。影響がありうるのは `src` と相関する別の列を通る経路だけ
- **v3_c2n で `src` が見分けられる強さ**: 正解は全部 src 3、おとりの src 3 は 4.48%。分子ごとの「正解と同じ src のおとり」は中央値 2 行、少なくとも 4 分の 1 の分子で 0 行（窓の中で src 3 は正解だけ）。`src` と相関する特徴を持つ段は、v3_c2n では正解を当てやすく見える。他の 3 集合では正解は src 1 で、窓のおとりの約 73〜75% も src 1 なので、`src` はほぼ見分けに使えない
- **相関しうる経路（未測定、§7 で測り方を書く）**:
  - 0.479/0.487 系統の提出ノートブック（`kernels/lb479/notebook.ipynb` セル 17 の `_load_pop_prior`・`_score_cand`）は `NP_W * np_val`（np_val = np_coconut | np_lotus | src_natural_product）を足す。pool_popularity.csv の np_coconut は src ∈ {2,3} の写しなので（ノートブックが inchikey14 で引く `pool_popularity.csv` が手元の `data/ext/pop/pool_popularity.csv` と同じファイルかは確かめていない）、この項を縮小版に入れると v3_c2n では np_coconut が正解に必ず付き、src 1 のおとり（29%）にはほぼ付かない（src 1 で np_coconut = 1 は全体で 9 行）。v3_c1・v3_c2・enveda_v1 の正解は src 1 なので同じくほぼ付かない（正解の行ごとには数えていない。np_lotus・src_natural_product も数えていない）
  - 文献 prior（log1p_substances + log1p_pubmed）: pubchem_cid > 0 の率は src 1 93.76%・src 2 96.46%・src 3 98.86% で、src ごとに分布が違いうる（分布は数えていない）
  - lfus（#150）: 類縁体は train の参照の構造（sid）なので、参照の見える train 構造の候補（v3_c2n では src 1・3 のおとり、合わせて 33.55%）は自分自身が類縁体に出て T = 1 になりうる。v3_c2n の正解は参照を全部隠してあるので自分自身は出ない。lfus の −0.0387 がこの経路で出たかどうかは**未測定**（仮説）。この経路は隠れテストのクラス 2 型（正解が train に無い）でも同じ向きに働くので、v3_c2n だけの作りもの（近道）とは言えない
- **lfus の判定は変えない**（事前登録どおり。この監査は採点していない）
- 隠れテストの正解のうち COCONUT の構造がどれだけあるかは分からない。v3_c2n は「プールにある・参照なし・COCONUT と train の両方にある」構造だけの集合で、「プールにある・参照なし・COCONUT に無い」型は v3 の 4 集合のどれにも無い

## 7. 偏りの影響の測り方（案だけ。今回は採点しない）

次の計画で事前登録してから行う（ここでは採点していない）。どれも学習・定数の調整を含まない。

1. **src だけのランカー**（近道の強さの上限）: 各集合で、窓の候補を `src` の決まった順（3 → 2 → 1。今ここで決める。同じ src の中は対照 `score,prior,frag` の順）に並べた MRR@25 と、正解の src を知っているとした上限（正解と同じ src の候補を先頭に、中は対照の順）。対照の 0.6284（v3_c2n）より上限が大きく上なら、src と相関する特徴を足す変更は v3_c2n で過大に良く見えうる、と読む。v3_c1・enveda_v1 で同じ値を並べ、差が v3_c2n だけに出るかを見る
2. **段の特徴と src の相関**: v3_c2n の窓の候補（正解を除く）で、各段が読む特徴（文献 prior の pop、lfus の N(c)、lib の最大類似度、0.479 系統の np_val）について「src 3 か否か」を見分ける AUC。正解を使わないので結果で選び直す余地が無い
3. **lfus の −0.0387 の内訳**（記録だけ、判定は変えない）: #150 の分子ごとの出力から、lfus が対照の上位 1 を替えた分子について、新しい上位 1 の src と、その候補に参照が見えていたかを数える
4. **集合の作り方**: 「プールにある・参照なし・COCONUT に無い（src 1）」型の集合（v3_c2n と同じ規則で fold 0 の hold から、src 1 に限る）を作り、v3_c2n との差で「参照なし」と「COCONUT」の効果を分ける。作る前に事前登録し、作った後に今までの判定を選び直さない

## 8. 資源

- 時間はどれも `/usr/bin/time` の real
- 1 回目（ee1543a、事前登録の分）: 17.86 s、`/usr/bin/time -l` の最大 RSS 1.26 GB（1,264,664,576 B）
- 2 回目の 1 度目（ed50070、`--coco-canon --workers 6`）: 約 3.5 分で止めた。ps で見たワーカー 6 本の RSS が各 0.37〜0.51 GB、親 1.10 GB で、プロセス木の合計が約 3.3〜4.2 GB と**すでに 3 GB を超えていた**ため。出力は書いていない（`/usr/bin/time` の値は取れていない）
- 2 回目（ed50070、`--coco-canon --workers 2`）: 1,851.99 s、親の最大 RSS 1.33 GB、プロセス木（親＋ワーカー）の RSS の合計の最大 約 2.50 GB（2,442,992 KiB、5 秒ごとの ps の標本）。memory_pressure の空きは 68〜70%
- 3 回目（b926ae4、pubchem の列を直した後、`--coco-canon` なし）: 16.67 s、最大 RSS 1.23 GB。表 2 と判定は 1 回目と同じ、表 1 は pubchem の列だけが変わった（src 1: 248,440 → 232,938、src 2: 436,515 → 421,081、src 3: 25,746 → 25,452）
- GPU 0。PubChem 層は読んでいない
