# 調査: 公開ノート上位の差分と主催者スレッド（2026-11-12 昼）（#80）

計画 2026-11-12 昼 項目 4。読んだのはノートのコードと markdown だけで、どのノートも実行していない。**v3 の測定案（最後の節）は書いただけで、まだ実行していない。**

## 取得の条件
- 一覧: `kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --sort-by scoreDescending --page-size 30 --csv`（2026-11-12 03:33:51 UTC）。この一覧に点数の列は無い。以下の点数は、断りが無いかぎりノートの題や markdown にある**作者の申告で、こちらでは未確認**
- 取得: `kaggle kernels pull <ref> -p <scratch>/pull/<owner>__<slug> -m`（03:34 UTC。scratch は `~/.hermes/profiles/massspec/cache/scratch/t4_1112/`、合わせて約 4.4 MB）。版番号は Kaggle API の `get_kernel` の `current_version_number`（5 本は 03:34:49 UTC に取得して `meta_versions.txt` に保存、romanrozen は別に読み取りで取得）。同じ API の `last_run_time` はどのノートも「2026-11-09 03:34:xx」で、取得時刻からちょうど 3 日前になっていて信用できないので使わず、最終実行は一覧の `lastRunTime` を書く
- 比べる基準: 0.487 の写し `kernels/lb487/`（`taichiiiii/casmi26-lb487-copy`、ipynb sha256 380e2c82…、31 セル）。**このファイルは main には無く、`feat/77-lb487-owntables` のコミット c3d9992 にだけある**ので、`git show c3d9992:kernels/lb487/...` で取り出して比べた。セルの比較は、セルの本文が一致するかどうか（sha1）と difflib の差分による

## 出典（ノート 6 本。版は 11/12 03:34 UTC 時点）
1. ozertuu「👑 [0.500+] CASMI26 Dual-Neural & Physical Isomer」 https://www.kaggle.com/code/ozertuu/0-500-casmi26-dual-neural-physical-isomer — **版 21**、lastRunTime 2026-11-11 16:36:43、票 8、ipynb sha256 ab8126d1…
2. ndaedzomukhuba「enveda-0500plus」 https://www.kaggle.com/code/ndaedzomukhuba/enveda-0500plus — **版 1**、lastRunTime 2026-11-10 21:24:03、票 0、ipynb 486f79ca…
3. avazaku「CASMI26 Evidence Lab」 https://www.kaggle.com/code/avazaku/casmi26-evidence-lab — **版 9**、lastRunTime 2026-11-11 08:54:32、票 13、ipynb 16a7b8de…
4. heliosli「casmi26-0.488」 https://www.kaggle.com/code/heliosli/casmi26-0-488 — **版 1**、lastRunTime 2026-11-03 21:48:25、票 43、ipynb 0848adfa…
5. lucifer19「IONWRAITH v11 | Evidence-Guided Molecule ID」 https://www.kaggle.com/code/lucifer19/ionwraith-v11-evidence-guided-molecule-id — **版 18**、lastRunTime 2026-11-11 20:01:51、票 44、ipynb 8e575a75…
6. （生成コードを探すために読んだもの）romanrozen「[0.482] CASMI: From Peaks to Structures」 https://www.kaggle.com/code/romanrozen/0-482-casmi-from-peaks-to-structures — **版 17**、lastRunTime 2026-11-05 17:06:30、ipynb 5bdea66b…

一覧の上位 10 件（並び順は Kaggle の scoreDescending のまま。点数は表示されない）: heliosli/casmi26-0-488、lucifer19/ionwraith-v11…、prvsiyan/analog-propagation…、pablodiegoo/…starter、avazaku/casmi26-evidence-lab、alexchilton/casmi26-lb-0-487…、zz3t3rn41/…-v2、evgendvorkin/…EDA、ozertuu/0-500-…、guhongbin/casmi26-v589…。prvsiyan・zz3t3rn41 は前の調査（11/10・11/11 の日報、`docs/research/lineage-reproducibility.md`）で読んでいるので、今回は省いた。題に「0.500」を含むノートはほかに wenbodai/casmi26-apex-0500-probe（票 0）がある（読んでいない）。

## ノートごとの差分と分類
分類の言葉: **構造的** = 新しい候補源・新しい証拠チャネル・新しい並べ替えの規則・公開コードで学習し直したモデル。**定数** = 既存の重み・しきい値・切り替えを動かしただけ。定数の写しは案にしない（ハイパラ調整の禁止）。

### 1. ozertuu 版 21（`dataset_sources`・`kernel_sources` は 0.487 写しと**同じ**）
0.487 写しの 31 セルのうち 22 セルが同一（版 21 は 30 セル。0.487 写しのセル 2（markdown「Camouflage Snapshot」）が無い）。変わったのは版 21 の番号でセル 0・1（markdown）、4（設定）、14（スコアラーの差し替え）、20（最終層）、22（v8）、23（外部ライブラリの重ね）、25（最後の確認の print）。
- スコアラー（セル 14）: 0.487 写しの `0.5·z(sanpier) + 0.5·z(v80)` の平均をやめ、`popwenman/casmi26-scorer-train-v80` の出力（`final_model.pt`＋`config_used.json`）だけで点を付ける → **定数**（混ぜる重み 0.5→0 と同じ）。v80 は全 train で学習（`lineage-reproducibility.md` の部品表の 20 番）なので v3 では測れない
- `V8 = 'MASS'`→`'BOTH'`（セル 4）: 0.487 写しに入っていて切ってある romanrozen の学習済み同式異性体リランカー（LightGBM、ノートに zlib+base64 で埋め込み、モデル本文の sha256 3b80e1c9…）を有効にする → 0.487 写しから見ると**構造的**（証拠チャネルを 1 本足す）。あわせて `RR8_ALPHA` 1.0→0.20（**定数**）と、`rerank_list` に `frozen[smiles[0]] = True`（1 位を動かさない、**規則の変更**）
  - 生成コード: **見つからない**。探した場所: romanrozen の公開カーネル一覧（17 本、CASMI 関係は 0-482 だけ）、0-482 の版 17（学習の説明だけ: 「LightGBM lambdarank、102 特徴、enveda の天然物 5 個ずつの scaffold 群 OOF で MRR 0.7283」。`lgb.train` などの学習コードは無い）、romanrozen の公開データセット 4 件（CASMI 関係なし）
  - 学習データは説明によると train の同式異性体と enveda の天然物 → v3 の正解を含む可能性が高い（未確認）。v3 では測れない
  - 盤面の手がかり（作者の申告、未確認）: 0.487 写しのセル 1 の表に「romanrozen v9 code, `V8 = 'BOTH'`, bgunshui scorer v2 | 0.481 (page) | 0.461 on our board」。lucifer19 版 18 のセル 0 にも「IonWraith v16 0.461 ← Gold Apex V22 code, V8 = 'BOTH'」。ozertuu 版 21 の BOTH は土台が違うので、これで良し悪しは言えない
- 最終層（セル 20）に規則を 2 つ追加: (a) ライブラリで 1 位を差し替え（ライブラリ一致がある分子で、いまの 1 位に `spec_map` の印が無く 2 位にあり、2 位の候補の `cinfo` の 2 つの順位がどちらも 3 以内なら入れ替え）、(b) 天然物の昇格（(a) が起きず lib_max < 0.80 のとき、2 位が天然物で 1 位は天然物でなく、2 位の `cinfo` の 2 つ目の順位が 1 位より上で、点差が `NP_PROMO_MARGIN = 0.040` を超えれば入れ替え）。`cinfo` の 2 つの順位の意味はコードを追っていない（未確認）→ **構造的な規則**（既存の証拠で 1 位を動かす）。定数 0.80・0.040 は作者の値
- 外部ライブラリの重ね（セル 23）: (1) 付加イオンごとの質量差の表で中性質量をそろえて照合、付加イオンの組が合うものだけ比べる、候補の精密質量が 15 ppm を超えてずれたら捨てる（**構造的**: 既存チャネルの照合の直し）、(2) cos ≥ 0.80 で 1 位へ（従来どおり、旧 1 位は 2 位へ）、0.72 ≤ cos < 0.80 なら 2 位へ差し込み（1 位は動かさない）、0.65 ≤ cos < 0.72 で 3〜25 位にある候補なら 2 位へ（**構造的な規則**「2 位への差し込み」。heliosli の考えとセルに書いてある。しきい値は作者の値）
  - 使うデータは ExtLib（生成コードは見つからない、CC0）と MassBank harmonized（作り方の説明だけ、CC-BY-NC-SA-4.0）で、0.487 写しと同じ（`lineage-reproducibility.md` の部品表の 21・22 番）。timsTOF の条件では v3 で測れない（`docs/research/extlib-holdout.md`）
- markdown（セル 0）は「Clean LOTUS-free candidate pool」と書く。LOTUS のデータセットは添付されていない。11/11 昼の日報で読んだ ozertuu の旧ノート（`ozertuu/casmi26-dual-neural-physical-isomer`、11/10）は LOTUS を添付していたが、版 21 には無い
- 点数: 題の「0.500+」は作者の申告で**未確認**。セル 1 の表は 0.487 写しのセル 1 と同じ形の表（行の一部が違う）で、版 21 自体の盤面の値は書かれていない

### 2. ndaedzomukhuba 版 1（0.487 写しに `qwertykeypadapple/casmi26-lotus-candidates` を足しただけ）
ozertuu 版 21 とセル 0・4・14・23 だけ違う（題も同じ「Imperial Zenith」）。
- スコアラー: sanpier 0.50 + v80 0.25 + `ozertuu/casmi26-our-scorer-v80-model` 0.25 の z 平均（markdown の記述）。コードは `/kaggle/input` の `final_model.pt` を全部集める作りで、`ozertuu/casmi26-our-scorer-v80-model` は `dataset_sources` に入っていない → 実際にいくつのモデルで平均するかは未確認。**定数**（重み）と、全 train 学習のモデルの追加
- `POP_W, SPEC_W, NP_W` 0.12/0.12/0.12→0.10/0.05/0.02、`LIB_HIT_GATE` 0.78→0.87、`RR8_ALPHA` 0.25 → **定数**
- LOTUS 候補を添付（空いた枠を質量の合う天然物で埋める段が動く）→ **構造的**（候補源）だが、0.487 系統の既知の部品（`lineage-reproducibility.md` の部品表の 23 番。README に作り方、コードは見つからない、CC0）
- 外部ライブラリの重ねは ozertuu の付加イオン対応だけを入れ、2 位差し込みは無い
- 点数: 題の「0.500+」は**未確認**、票 0

### 3. avazaku 版 9（`dataset_sources` は 0.487 写しの 10 件＋名前の出ない 1 件（`""`、非公開とみられる））
98 セル。0.487 系統の中身を `readable_sources/` に書き出して読めるようにした作りで、0.487 写しと同じセルは 2 つだけ（セル単位では比べにくい）。markdown（セル 0）の主張（すべて作者の申告、未確認）: 公開の点 0.487、「final-slot reference experiment」も 0.487、今回の版は別の提出で改善は主張しない。
- `PUBLIC_JOINT_WEIGHT = 0.0`（版 8 は 0.5）: v80 の点だけで最終リストを並べる → **定数**（ozertuu と同じ向き）。作者の部品診断: 991 構造で v80 単独＋2 位規則 0.6637 対 平均＋2 位規則 0.6227、差 0.041 [0.027, 0.055]（エポックはこの holdout で選んだ、と作者自身が書く。盤面の予測ではないとも書く）
- **MassSpecGym 1.5 の実測参照スペクトル**（非公開の添付 `casmi26-msg15-measured-references`、`references.parquet` の sha256 e4a1996d… をセル 95 で assert）と、2 つの別々の問い合わせスペクトルが同じ構造を支持したときだけ（score ≥ 0.85、2 位との差 ≥ 0.10、2 本目 ≥ 0.75、一致ピーク ≥ 8）、**最終リストに既にある候補を 3〜25 位から 2 位へ**動かす規則（1 位・候補の集合・長さは変えない）→ **構造的**（新しい参照源＋新しい規則）
  - 生成コード: `spectral_match.py`・`consensus_rescue.py`・`apply_rank2.py`（照合と規則）はノートの中にある。`references.parquet` を MassSpecGym から作るコードはノートに無い（`.tsv` や MassSpecGym の読み込みは 0 件、`references.parquet` は読む側の 1 か所だけ）。添付は非公開で、作者の公開データセット 4 件にも無い（`kaggle datasets list --user avazaku`）。元の MassSpecGym 1.5 は公開（MIT、主催者 757307 で使用可）
- 外部ライブラリの重ね（セル 86）: 0.487 写しの重ねに「分子の中性質量と合わない一致は捨てる」質量の関門を足しただけ（**構造的**だが小さい直し。データは 0.487 写しと同じ）
- ライセンス: セル 97「Our audit code and this report are Apache-2.0」。上流の部品はそれぞれの条件

### 4. heliosli 版 1（`dataset_sources` から ExtLib・MassBank・RDKit wheel・v80 カーネルが無く、名前の出ないデータセットが 6 件）
0.487 写しより前の系統（Spectral Parallax＋busyaprime の m1 合流、0.477）に、heliosli の重ねを 6 つ足したもの。markdown（セル 0）は「public LB 0.477 (submission 57722897)」の説明のままで、重ねのことは書いていない。セル 11 のコメントに「as in the 0.489 version」とある。題の 0.488 も含め、点数は**未確認**。
- FRIGID＋E1（セル 11・27）: 2 枚目の T4 で、1 位の分子式に合う構造を **de novo 生成**（FRIGID、`DLM.ckpt`）し、予測指紋との類似 ≥ 0.7・重原子 ≤ 30 なら 2 位へ差し込む → **構造的**（新しい候補源）
- rr3 同式異性体リランカー（セル 24、グラフネット＋transformer、学習構造は動かさない FREEZE）→ **構造的**
- cvjoint（セル 25）: PubChem 層の同式異性体＋天然物表＋リストを、fold-safe FPNet model 0・チャネルの点・RT モデル（`heliosli/casmi26-rt-np`）・天然物の印で LightGBM（`heliosli/casmi26-joint-ranker`）が並べ、差が大きければ 1 位を差し替え、中くらいなら 2 位へ → **構造的**（学習済みの合同ランカー）
- push489（セル 26・28）: decoder v3（`heliosli/casmi26-denovo-v3`）が分子式どおりの構造を 256 個生成し、234 構造の天然物パネルで当てはめた条件付きロジット（重みはセルに直書き）で 1 つ選び、確率 ≥ 0.47 なら 2 位（または 3 位）へ → **構造的**（新しい候補源）。セルの記述では、decoder と FPNet model 0 は評価パネルを見ていない
- 生成コード: **見つからない**。`heliosli/casmi26-denovo-v3`・`casmi26-joint-ranker`・`casmi26-rt-np` は `kaggle datasets files` が 403（非公開）。heliosli の公開データセット 9 件に CASMI 関係は無い。セルが引く `reports/c3_design_v0.md` も公開されていない。FRIGID 本体の出典はノートに書かれていない（未確認）

### 5. lucifer19 版 18（`dataset_sources`・`kernel_sources` は 0.487 写しと**同じ**）
0.487 写しの 31 セルが**全部そのまま**入っていて（コードセル 16 本はすべて同一）、足したのは markdown 6 セル（0・1・2・30・36・37）と図のセル 1 本（セル 31、`submission` の文字は出るが書き込みは無い）。セル 0 の作者の申告: 版 17 が 0.487（同じ版を 2 回提出して同じ値）、版 18 は同じコードで読みやすくしただけ。→ **新しい部品なし**。

## 生成コードの判定のまとめ
- 構造的で、生成コードも公開の材料も揃うもの: **無い**
- 構造的で、規則のコードはノートにあるもの: 2 位への差し込み規則（ozertuu セル 23 の tier 2/3、avazaku セル 92〜95）、ozertuu の最終層の 2 規則、付加イオン対応の照合。どれも新しい学習済み部品を使わない規則で、こちらの縮小版でも同じ形を書ける
- 構造的だが、作り方が見つからない、または非公開のもの: romanrozen の RR8 リランカー、avazaku の MSG 参照表（元データは公開）、heliosli の de novo・joint・RT・rr3 の各モデル
- 定数の変更: v80 だけで並べる（ozertuu・avazaku の両方が同じ向き）、prior の重み、ゲート、`RR8_ALPHA`
- 0.487 写しから新しい部品が無いもの: lucifer19 版 18

## 主催者スレッド（`kaggle competitions topics show <slug>/<id> --format json`、取得 2026-11-12 03:41:53 UTC = 12:41 JST）
どれも参加者が立てたスレッドで、主催者（David Healey）の答えを待っている、または答えが付いたもの。
- 758233「Clarification request: Reproducibility of small fixed index lists shipped inside public datasets」（Bhavish Mansharamani、2026-11-12 00:03:48 UTC）: commentCount 0。**回答なし（変化なし）**
- 753662「Question on reproducibility: are public notebooks that rely on datasets without generation code eligible?」: commentCount 8。いちばん新しいのは 기욱 이 の質問（2026-11-12 00:56:33 UTC、公開の学習コードはあるが種や手順が一部書かれておらず、ビット一致しない重みをそのまま読んでよいか）で**未回答**。主催者の最新は David Healey 2026-11-11 00:15:52 UTC「Intermediate data and models need to be reproducible. If we can't reproduce it because parts of the generating code is missing, the solution is open to being disqualified. Use others' intermediate data artifacts at your own risk.」
- 756893「Are shipped ranker tables without a public generator reproducible?」: commentCount 1。David Healey 2026-11-11 00:16:58 UTC「Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible.」のまま。作り直した表が認められるかへの答えは無い
- 11/12 11:00 JST（02:00 UTC）以降の書き込みは 3 件とも**無い**（いちばん新しい書き込みが 00:56 UTC）
- 参考に読んだ 757307（MassSpecGym）: David Healey 2026-11-11 00:20:38 UTC「MassSpecGym and MIST models are fine」（変化なし）

## ライセンスと再現性の条件（この競技で使う場合）
- 主催者の基準（753662・756893・748882）: 中間データとモデルは再現できること。生成コードの欠けた他人の中間物は失格になりうる
- 2 位への差し込み規則: 学習した部品を使わない規則なので、自前のコードで書けば再現性の問題は無い。ozertuu・avazaku のノートの metadata にライセンスの項目は無い（avazaku は自分のコードを Apache-2.0 と書く）。コードは写さず、規則だけを自前で書く
- MassSpecGym 1.5: MIT（HF のカード）、757307 で使用可。参照表を自前のコードで作れば再現できる。ただし 4 集合では効く余地がほとんど測れない（`docs/research/mist-massspecgym.md` の重なり: v3_c1 1、v3_c2 0、enveda_v1 0、v3_c2n 117/200）。v3_c2n は参照を全部隠す集合なので、MSG の参照も隠す（drop_keys を当てる）必要があり、そうすると効く余地が残るのは v3_c1 の 1 構造だけになる
- RR8・heliosli の各モデル・ExtLib・MassBank harmonized: 最終提出には使えない、または使うなら作り直しが先

## v3 で測る案（1 つ。書くだけで実行しない）
**案: 「ライブラリ一致で、既存の候補を 2 位へ（1 位は動かさない）」の段 `lib2` を縮小版に足す**
- 出どころ: heliosli の「2 位への差し込み」の考え方。ozertuu 版 21 の tier 2/3 と avazaku 版 9 の existing-candidate rank2 が同じ形で使う。どちらも 1 位を守り、候補の集合を変えない。縮小版の既存の `lib` 段（一致の強い候補を先頭へ）は v3_c1 で score+prior 0.822 → 0.749 と下げた（`casmi26.md` の記録）。差し込みを 2 位に限れば、正しい 1 位は崩れない。崩れうるのは「正解が 2 位にいて、別の候補が 2 位に入る」ときだけ（0.5→1/3）。この損と得の差し引きを測る
- 規則（定数は既存の値をそのまま使い、動かさない）:
  - 対照: `scripts/local_eval/run.py --holdout <h> --stages score,prior,frag`（v3_c1 0.8535、v3_c2 0.0117、enveda_v1 0.8376、v3_c2n 0.6284）
  - 新: `--stages score,prior,frag,lib2`。frag の後、上位 25 に切る前のリストで、既存の `library.lib_scores`（エントロピー類似度、参照は hide_rule を当てた train のスペクトル）を使う。`lib_max ≥ LIB_TAU`（run.py の既存の 0.7、0.479 版の既存のしきい値）のとき、類似度が最大の候補（同点なら上の順位）が 3 位以下なら 2 位へ動かし、ほかは 1 つずつ下げる。1 位・2 位にいれば何もしない。1 位はどの場合も動かさない
  - `LIB_TAU` と差し込む順位（2 位）は事前に固定し、結果を見て変えない。一度だけ走らせ、結果が良くても悪くてもそのまま記録する
- 採否の規則（設計 §5 のまま）: 4 集合 v3_c1・v3_c2・enveda_v1・v3_c2n すべてで、`train_eval.bootstrap_diff`（分子ブートストラップ 2,000 回、seed 0、`scripts/cv/frag_v3_check.py` と同じ）の (新 − 対照) 95% 区間の下限 ≥ −0.02、かつ 1 つ以上で下限 > 0。1 集合でも欠けたら採用しない
- リークの確かめ方（この段は何も学習しないので、確かめるのは参照表が各集合の隠す規則どおりかどうか）:
  1. 学習する部品なし（FPNet は既存の fold0、prior・frag は既存）。新しい外部データも足さない
  2. 参照は `library.load_refs` の規則どおりであること: v3_c2・enveda_v1・v3_c2n（rule all。enveda_v1 は `hide_rule.json` が無く、`library.hide_rule` の既定値 rule all・pool keep で動く）は正解の構造の参照が 0 行、v3_c1（rule enveda_libs）は正解の参照に enveda-180・enveda-np-examples の行が 0 行。段を走らせる前にこれを数えて assert し、各集合の「正解に参照がある分子の数」と「lib_max ≥ 0.7 の分子の割合」を結果に書く
  3. 参照のキャッシュ `outputs/local_eval/<h>/refs_top60.parquet` は、作ったときのリストの候補キーしか持たない。frag の後のリストのキーが全部入っているかを確かめ、足りなければ別名のキャッシュを作り直す（足りないキーは「参照なし」として黙って扱われるため）
  4. 縮小版の対照が変わらないこと: `score,prior,frag` の csv が変更前の写しと `cmp` 同一、mrr25 が上の 4 値と一致
- この案で分からないこと: 縮小版の土台は弱い（LB 0.282）。ここで勝っても、0.487 の系統（すでに ExtLib の 1 位差し上げと `LIB_HIT_GATE` を持つ）の上で勝つとは言えない。盤面に出すなら、0.487 写しへの重ねとして別に事前登録する
- 費用の見込み: 1 集合およそ 40 秒（既存の縮小版と同じ程度）× 4、ピーク約 1.8 GB、GPU 0。別 Issue で、テストと Issue→PR の手順に沿って行う

## 案にしなかった理由（短く）
- v80 だけで並べる: 混ぜる重みの変更なので定数。v80 は全 train で学習していて v3 では測れない。fold-safe に作り直すのは学習の項目（#54 の 16 エポック版は #60 で不採用）
- RR8・heliosli の各モデル: 生成コードが見つからない、または非公開で、v3 の正解を学習に含む可能性がある
- 付加イオン対応の外部ライブラリ照合: データ（ExtLib・MassBank harmonized）が timsTOF の条件で v3 では測れない
- MSG 参照表: 上の重なりのとおり、4 集合では効く余地がほぼ測れない
- LOTUS 候補: 既知の部品（`lineage-reproducibility.md` の部品表の 23 番）。空いた枠を埋めるだけで、v3_c2 の正解（enveda-180 だけの構造）は COCONUT にほぼ無いので、LOTUS にも少ないと見込む（未確認）

## 再現のコマンド
```
S=~/.hermes/profiles/massspec/cache/scratch/t4_1112
kaggle kernels list --competition enveda-CASMI26-molecule-id-mass-spectra --sort-by scoreDescending --page-size 30 --csv
kaggle kernels pull ozertuu/0-500-casmi26-dual-neural-physical-isomer -p $S/pull/ozertuu__0-500-casmi26-dual-neural-physical-isomer -m   # ほかの 5 本も同様
git show c3d9992:kernels/lb487/casmi26-lb487-copy.ipynb > $S/lb487.ipynb
kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/758233 --format json   # 753662・756893・757307 も同様
kaggle datasets files heliosli/casmi26-denovo-v3   # 403（非公開）
```
