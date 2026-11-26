# 調査 H3: 同じ分子式の異性体を分ける特徴と、上位の公開ノート 2 本の追加部品（#276）

計画 2026-11-27 夜 項目 6（カード t_16b1f847）。親 #3、関連 #195（F2・F4）、誤り分析 `docs/experiments/error-analysis-2026-11-26.md` の H3。
**この文書では測っていない**: dev の当てはめ・MRR の計算・holdout の読み・提出はしていない。GPU 0。数えたのはラベルを使わないスペクトルの本数（§3）と、定義案の費用の試し（§4.4、候補 300 構造の列挙の速さ）だけ。

印: **確認** = 原文・コード・こちらの出力を今回読んだ。**主張** = 著者の申告で、こちらで再現していない。**未確認** = 調べたが分からない、または今回読んでいない。**推定** = こちらの見積もりで、測っていない。

## 0. なぜ調べるか

- dev（B8 の 20 集合・3,901 分子）で exp003 が対照より下げた 444 件のうち 429 件（96.6%）は、1 位と正解の分子式が同じだった（run_2026-11-26_noon 項目 2、誤り分析 H3）。分子式・質量で効く特徴（窓、同位体、付加イオン）では直せない
- いまの表の断片の列（`metfrag_lite`・`mf_*` 26 列・`cc`）は「候補の断片の**質量**が観測ピークに合うか」だけを見ている。同じ分子式の異性体は、断片の質量の集合がかなり重なるので、これらの差は小さい（誤り分析の型 b の抜き出し 4 件で、差の上位は 1〜4 sd の `mf_*` に散っていた）
- 既にある調査（`fragment-isomer-ranking.md` #124、`isomer-rerank.md` #118、`gems-isomer-search.md` #131）は MassSpecGym・MetFrag 2.2・MAGMa・CFM-ID 4・ICEBERG・FraGNNet・MIST の**同じ式の候補での数値**をまとめた。この文書は、そこで読んでいない部分（MetFrag の統計学習版の定義、MARASON のアブレーション、ICEBERG の断片の規則性）を開いて、「何の入力が異性体を分けたか」に絞り、次の特徴群の定義案を 1 つ出す

## 1. 出典（今回開いて読んだもの）

1. **MetFrag 2.4.5（統計学習の断片の注釈）** — Ruttkies C, Neumann S. "Improving MetFrag with statistical learning of fragment annotations". BMC Bioinformatics 20, 376 (2019). doi:10.1186/s12859-019-2954-7 、PMC6612146
   - 確認: Europe PMC の全文 XML（https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6612146/fullTextXML 、2026-11-27 05:5x JST に取得、92,914 字）。読んだ: 要旨、Methods の「断片の構造の指紋」「m/z の区切り」「平均事後（MP）推定」「断片の損失」「候補の点」、学習の 2 段、Table 1 の陰性モードの 10 行、Table 2、Availability。#124 では方法の前半までしか読めていなかった部分
   - 読んでいない: Fig. 3〜6 の図そのもの、Additional file、陽性モードの Table 1 の全行
   - ライセンス: 論文は CC BY 4.0（全文の permissions の文「provide a link to the Creative Commons license」と、#124 の記録）。コードは https://github.com/ipb-halle/MetFragRelaunched の branch `feature/statistical_scoring`（本文の Availability、確認）。リポジトリにライセンスのファイルが見当たらないこと（#124 の記録、今回は取り直していない）から、**コードは写さず、式だけから自前で書く**。学習に使ったスペクトルの一覧 https://github.com/c-ruttkies/metfrag_statistical_annotation は開いていない（未確認）
2. **MARASON** — Wang R, Wang R-X, Manjrekar M, Coley CW. "Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning". arXiv:2502.17874v2（2025-07-03 改訂）. https://arxiv.org/abs/2502.17874 、doi:10.48550/arXiv.2502.17874
   - 確認: HTML 版 https://arxiv.org/html/2502.17874v2 を `curl -sL -A Mozilla` で取り、タグを除いて読んだ（68,570 字）。読んだ: 貢献の節、Table 1 の説明、検索の設定（§4.2.2 の前段）、Table 3（アブレーション）、§4.2.2・§4.2.3・§4.2.4・§4.2.6
   - 読んでいない: §3 の式の細部、付録の Table 5〜7 の数値
   - ライセンス: arXiv の abs ページの license のリンクが `licenses/by-sa/4.0/`（CC BY-SA 4.0、確認）。コードは本文に「publicly available at https://github.com/coleygroup/ms-pred」（確認）。ms-pred は MIT（#118・#124 の記録、今回は取り直していない）。本文の主な重みは NIST20 で学習（**NIST 由来の重みは競技の規則 750330 で不可**）。MassSpecGym で学習し直したとある（§4.2.3、確認）が、その重みの配布は未確認
3. **ICEBERG** — Goldman S, Li J, Coley CW. "Generating Molecular Fragmentation Graphs with Autoregressive Neural Networks". arXiv:2304.13136v2. https://arxiv.org/abs/2304.13136
   - 確認: HTML 版 https://arxiv.org/html/2304.13136v2（93,937 字）。今回読んだ: §3.3 の断片の例（切れやすい結合）、§3.4（異性体の検索）。#118 で読んだ retrieval の節を、入力の観点で読み直した
   - ライセンス: 論文 CC BY-NC-SA 4.0（#118 の記録。今回の HTML からは license のリンクを取れなかった、未確認）。コード ms-pred（MIT、同上）。ICEBERG・GLACIER の MassSpecGym 版は主催者が名指しで承認（744338・745841、`glacier-iceberg-provenance.md` の記録）。GLACIER は項目 2（#268）で既に特徴にしている途中
4. 公開ノート 2 本（§2）と、その `dataset_sources` の dataset のメタデータ（`kaggle datasets metadata`、05:5x JST）

既に別の文書で読んだもの（今回は開いていない）: CFM-ID 4.0（`fragment-isomer-ranking.md` 出典 5）、MetFrag 2.2（同 出典 2）、MIST（`gems-isomer-search.md` 出典 4）、MS2Query（`ranker-input-features.md` 出典 1、群 N1 の元）。

## 1.1 各手法で異性体を分けた入力と、そのアブレーション

### MetFrag 2.4.5（出典 1）
- 入力（確認、Methods）: 学習スペクトルごとに**正解の構造**を MetFrag で断片にし、説明できたピークについて「ピークの m/z」と「そのピークを説明した**断片の構造**の指紋」の組 (m, f) を数える（N_mf）。m は断片の分子式の質量に直してから、`mzppm`・`mzabs` で決まる幅の区切り（bin）に入れる。f は断片の構造の指紋（Lingo・MACCS・Circular・GraphOnly を比べ、Circular と Lingo が同程度に良かった。Table 1 は Circular）
- 推定の式（確認）: θ_mf = (N_mf + π_mf) / Σ_f' (N_mf' + π_mf')。π_mf = α（f が断片）、β（f = ⊥、説明の無いピーク）。候補 c の点は S_RawPeak = 1 / (−Σ_k log θ(f_ck | m_k))、候補の中の最大で割って S_Peak。断片の損失（2 つのピークの差と、大きい断片から小さい断片を除いた部分構造）にも同じ形の S_Loss
- 最終の点（確認）: S_Fin = ω1 S_MetFrag + ω2 S_Peak + ω3 S_Loss。α・β・ω は学習の第 2 段（PubChem の同じ質量の候補、最大 500）で Top1 の数を目的に格子探索
- アブレーション・結果（主張、Table 2、CASMI 2016 の 87 スペクトル）: Top1 は MetFrag 2.3（断片の質量だけ）5 → MetFrag 2.4.5 21、Top10 39 → 55。同じ表で CFM-ID 15、CSI:IOKR 16、CSI:FID 13（陽性のみ）。重み ω2（S_Peak）の中央値は陽性 0.4855・陰性 0.4935 で、3 つの中で最大（ω1 は 0.2875・0.2840）
- 異性体との関係（推定）: 同じ式の異性体は、ピークの m/z（＝断片の式）は同じでも、そこを説明する**断片の部分構造**が違う。S_Peak は「この m/z のピークは、学習でいつもどの部分構造で説明されていたか」を見るので、質量だけの点（いまの `mf_*`）が同点になる異性体を分けられる。論文は同じ式の候補だけでの数を出していない（未確認）
- 注意（確認、本文）: 正解の候補が学習のスペクトルに含まれると、自分の (m, f) の組が数に入って有利になる、と著者も学習の分け方の段で書いている（InChIKey の第 1 ブロックで 70:30 に分けた）

### MARASON（出典 2）
- 入力（確認）: 予測したい候補の構造ごとに、学習のライブラリから似た構造（参照）を取り、ICEBERG の断片化で**参照のスペクトルのピークに断片を注釈**し、候補の断片と参照の断片を神経グラフ照合で対応づけ、参照の強度を候補の強度の予測に使う（retrieval-augmented）
- アブレーション（主張、Table 3、NIST20 random split、cosine）: RAG 無し 0.739、参照のスペクトルを入力に連結 0.737（−0.3%）、Hungarian 0.746、RRWM 0.742、NGM Softmax（GNN を分ける）0.757（+2.4%）
- 異性体の検索（主張、§4.2.2、Table 1、NIST20、正解 1 + PubChem の Tanimoto の高い異性体 49）: top-1 は ICEBERG 18.7% → 衝突エネルギーつき ICEBERG 20.2% → MARASON 27.8%。著者は「改善の大半は RAG による」と書く。MassSpecGym（§4.2.3）では FraGNNet 31.93% → 34.03%
- 参照の近さ（主張、§4.2.6、Table 4）: 参照の Tanimoto が 0.3 を超えると RAG の利得がはっきりし、0.1〜0.3 では有意な利得が無い
- この競技との関係（推定）: 「近い構造の参照スペクトルで、断片ごとの強度を借りる」は、群 N1（`neighbour-reference-features.md`、近傍の参照と問い合わせの一致）が分子単位でやっていることを、**断片単位**にしたもの。MARASON そのものは NIST の重みが不可、学習し直すと GPU の学習になる（G8 で新しいネットは入れない決まり）。考え方だけを下の H3 群の作り方（学習の参照の断片の統計）に使う

### ICEBERG（出典 3）
- 入力（確認、§3.3）: 原子を 1 つずつ外す断片の DAG と、断片ごとの水素の移動に応じた複数の強度。図の例の読みとして「C–O・C–N の結合は、C–C や環の開裂よりも切れやすい」と書く（定性的な観察で、数の表は無い）
- 異性体の検索（主張、§3.4、#118 と同じ）: PubChem の Tanimoto の高い異性体 49 を候補に、NPLIB1 で top-1 が次点より 9 ポイント上（相対 46%）。NIST20 の top-1 は MassFormer と同じ。CASMI22 陽性 303 で 12.9%（次点 NEIMS 8.6%）
- この競技との関係: GLACIER（ICEBERG と同じ ms-pred の、断片を列挙しない版）の予測との一致は項目 2（#268）で特徴にする途中なので、H3 群は**それと重ならない**入力にする

### まとめ（3 本から言えること、推定）
- 異性体を分けたのは、どれも「**どの部分構造がそのピークを生むか**」の情報だった: MetFrag 2.4.5 は学習の正解の (ピーク, 断片の構造) の頻度、MARASON は近い参照の断片の強度、ICEBERG は学習した断片化の確率
- 学習なしで自前のコードで作れ、GPU が要らないのは MetFrag 2.4.5 の形だけ（数え上げと pseudo count）。MARASON・ICEBERG の形は学習が要る（ICEBERG 側は GLACIER の項目 2 で別に入る）

## 2. 上位の公開ノート 2 本の追加部品

取得: `kaggle kernels pull <ref> -p <scratch>/i6/pull/<owner>_<slug> -m`（2026-11-26 20:48 UTC）。並びは `kaggle kernels list --competition … --sort-by scoreDescending --page-size 30`（同時刻）。比べる基準はこちらの床の写し `kernels/lb495/kernel-metadata.json`（taichiiiii/casmi26-lb495-floor、dataset 11 件、kernel_sources `popwenman/casmi26-scorer-train-v80`）。**どちらのノートも実行していない。点数は作者の申告で未確認**。

- A: alexchilton「CASMI26 0.509 | What Moved the Board: 71 Arms」 https://www.kaggle.com/code/alexchilton/casmi26-0-509-what-moved-the-board-71-arms — ipynb sha256 1ce135e7…、53 セル（コード 24）、lastRunTime 2026-11-25 04:03 UTC、票 73
  - 確認（セル 0）: 「ishankarnick の casmi26-lb-0-509-47-7m-virtual-natural-products を変えずに実行」「コードのセルは採点された写しとバイト一致」と書く。**中身は ishankarnick の 0.509 版の写し**で、alexchilton 自身の追加は最後のセルの提出の表（194 件）だけ。コードがバイト一致かは確かめていない（未確認、ishankarnick の原本は pull していない）
  - 確認（セル 1・最後のセル）: 「同じ土台で 1 つの値だけ変えた 71 本のうち +0.002 を超えたのは 5 本、最大 +0.006。39 本が下がり、最大 −0.047」。土台の上がりは主に他人の新しい公開ノートを走らせたことによる、と書く（主張）
- B: ozertuu「0.509+ CASMI26 Gold Baseline Analog Ranker」 https://www.kaggle.com/code/ozertuu/0-509-casmi26-gold-baseline-analog-ranker — ipynb sha256 ddda2aec…、51 セル（コード 30）、lastRunTime 2026-11-26 13:00 UTC、票 102
  - 確認（セル 0・コード）: A と同じ系統（0.505 の stack ＋ 仮想 NP 層）に、Dual-Mode Mass Sync（コードのセル 15）、ExtLib の Tier-2（`EXTLIB_TIER2_TAU = 0.72`）、LOTUS の候補を足したもの。題の「0.515+」は目標の記述で、盤面の値は書いていない

`dataset_sources` の差（lb495 から見て。確認）: A・B とも `bgunshui/casmi26-scorer-v2-model`、`ishankarnick/casmi-scorer-x1full`、`ishankarnick/casmi26-vt0`、`ishankarnick/casmi26-vtlp` を足す。B はさらに `qwertykeypadapple/casmi26-lotus-candidates` と `ozertuu/quantum-sim-linalg-cache` を足す。`kernel_sources` は 3 本とも `popwenman/casmi26-scorer-train-v80`。

部品の表（「分類」は構造 = 新しい候補源・証拠・規則・学習し直したモデル、定数 = 既存の値を動かしただけ。「場所」は確認したコードのセル。ライセンスは `kaggle datasets metadata` の値）:

| 部品 | A | B | 分類 | 生成・学習のコード | ライセンス | オフライン |
|---|---|---|---|---|---|---|
| 仮想 NP 層 vt0（候補 47.7M、プールの各構造に `derive.py` の 45 変換を 1 回） | セル 23 | セル 29 | 構造（候補源） | 作り方の説明だけ。生成のスクリプトは ishankarnick の公開カーネル 5 本・dataset の説明に見つからない | CC-BY-NC-SA-4.0 | 可（dataset 644 MB） |
| 層の学習した事前分布 vtlp（GBM と MLP 3 seed の平均、プールで leave-one-out） | 同上 | 同上 | 構造（学習した事前分布） | 学習のコードは見つからない | CC-BY-NC-SA-4.0 | 可 |
| スコアラー x1full（bgunshui の公開の学習コードで train だけ、210 epoch） | セル 7・23 | セル 29 | 構造（学習し直したモデル） | dataset の説明は bgunshui の casmi26-scorer-train-public を指すが、`kaggle kernels list -s` で見つからない（未確認） | CC-BY-NC-SA-4.0 | 可 |
| スコアラー bgunshui v2 | セル 7・23 | セル 8・29 | 構造（モデル） | 同上（未確認） | unknown | 可 |
| 層の合流の規則 `VT_RULE`（1 位は差 1.5、下位は 1.0、1 分子 10 本まで、3〜8 位に充填） | セル 23 | セル 29 | 規則は構造、1.5・1.0・10 は定数 | ノートの中にある | ノート（明記なし） | 可 |
| APRIOR（enveda-180 で学習した付加イオンの型の事前分布、`_AP_KEEP1` で 1 位を守る） | セル 19（`_AP_KEEP1 = True`） | セル 19（`APRIOR_KEEP1 = True`） | 構造（学習はノートの中）。KEEP1 の切り替えは定数 | ノートの中にある | ノート（明記なし） | 可 |
| UNIQ（分子をまたいで同じ構造が 1 位にならない制約） | セル 20 | セル 20 | 構造（規則） | ノートの中 | 同上 | 可 |
| NPX（天然物らしさの LightGBM、ノートの中で学習） | セル 21 | セル 21 | 構造（学習） | ノートの中 | 同上 | 可 |
| E180X（enveda-180 などの合成の構造を下げる） | セル 22 | セル 22 | 構造（規則） | ノートの中 | 同上 | 可 |
| P5 Tail の後に FR 3-25（順番の入れ替え） | セル 17・18 | セル 17・18（`RUN_FR325_AFTER_P5 = True`） | 規則の順番（構造）、`FR325_W = 3.0` は定数 | FR の学習コードは fr3to25-clean にある（`lineage-reproducibility.md` の記録） | CC BY-NC-SA 4.0（同） | 可 |
| romanrozen の同式異性体リランカー（`V8`） | セル 13、`V8 = 'MASS'`（切ってある） | `V8 = 'MASS'`（コメント「destructive RR8 isomer reranker OFF」） | — 使っていない | 見つからない（#80 の記録） | — | — |
| Dual-Mode Mass Sync（陽・陰の両方がある分子で中性の質量をそろえ、2〜25 位を並べ替え） | 無し | セル 15 | 構造（規則） | ノートの中 | 同上 | 可 |
| ExtLib Tier-2（0.72 ≤ cos < 0.80 を 2 位に差し込む） | 無し（`EXTLIB_TIER2_TAU` の語が無い） | セル 16、0.72 | 規則は既知（#80 の ozertuu 版 21 と同じ）、0.72 は定数 | ノートの中 | ExtLib は既知（部品表 21・22） | 可 |
| LOTUS の候補（149,983 構造） | 無し | セル 12・22 など | 構造（候補源）、既知（部品表 23） | README に作り方（#80 の記録） | CC0-1.0 | 可 |
| quantum-sim-linalg-cache | 無し | セル 1 で名前を print するだけ | 機能なし（別の競技の dataset。こちらの読みでは何の計算にも使っていない） | — | CC0-1.0 | — |

読み（推定）:
- 2 本とも、0.495 → 0.509 の差は「候補源（仮想 NP 層）」と「事前分布・規則の段（APRIOR・UNIQ・NPX・E180X）」から来ている。**同じ式の異性体を並べ替える新しい証拠（断片の証拠）は、FR 3-25（既に 0.495 にある）以外に足されていない**。romanrozen の異性体リランカーはどちらも切ってある
- 仮想 NP 層は計画（11/27 夜 計画の「採らなかった候補」）で analog-edit と同じ系統として却下済み。A の記録でも「弱い群（ライブラリの一致 0.825 未満など）では層の上位 25 が 0.000」と書かれている（主張）
- APRIOR・NPX・E180X は「ノートの中で学習・計算するので自前のコードで作り直せる」形。ただし分子の事前分布で、異性体の断片の証拠ではない（H3 の答えではない）。採るかは H3 と別の話で、この文書では決めない
- 盤面の差: A の作者自身が「一つの値の変更の利得は 1〜2 分子（0.004/分子）」と書く。公開の区分の SE（約 0.035、`kaggle-improvement-loop` の記録）より小さい差は根拠にしない

## 3. 1 分子あたりのスペクトル本数（ラベルを使わない数え）

`scripts/research/spectra_per_molecule.py`（このブランチで追加）。出力 `outputs/local_eval/spectra_per_molecule.json`、写し `docs/research/spectra-per-molecule.json`。1.2 s、最大 RSS 228 MB。dev の集合は特徴表 `dev_b8.parquet` の `set` 列の 20 集合（`b8_f{1..4}_{a..e}_t10`）、各集合の `data/holdouts/<set>/test.parquet` の行を `molecule_id` ごとに数えた。

- **見える test（400 分子、1,213 本）**: 平均 3.033、中央値 3、p10 1・p90 5、最大 9。1 本だけの分子 13.75%、3 本以下 62.75%。陽・陰の両方がある分子 97/400
- **dev 全体（3,901 分子、20,117 本）**: 平均 5.157、中央値 4、p10 2・p90 10、最大 16（16 本ちょうどが 293 分子。16 本が上限になっている。どこで切ったかは確かめていない、未確認）。1 本だけ 8.36%、3 本以下 37.43%
- dev の型ごと（平均 / 中央値 / 1 本だけの割合 / p90）: a 4.664 / 4 / 1.14% / 8、b 9.013 / 8 / 15.5% / 16、c 4.581 / 4 / 0.76% / 7、d 2.605 / 3 / 24.7% / 4、e 4.789 / 4 / 0.63% / 8
- 参考（train の inchikey14 ごと、275,810 構造・2,539,608 本）: 平均 9.208、中央値 6。ingest_lib ごとの平均 / 中央値: enveda-180 6.307 / 6、enveda-np-examples 4.736 / 4、gnps 4.827 / 2、msdial 4.466 / 2、pluskal_ms2 11.268 / 10、massbank 11.081 / 5、mona 7.912 / 3、riken 21.846 / 5
- 日報 Q6 の「test は約 3 本、train は約 4.6 本」（著者の見積もり）: 見える test の 3.03 は合う。train の 4.6 は、train 全体（9.21）とも enveda-180（6.31）とも合わない。何を数えた値かは未確認

読み（推定）:
- dev の分子は、見える test より 1.7 倍多くスペクトルを持つ（5.16 対 3.03）。型 d（2.61）だけが見える test に近く、型 b は 3 倍（9.01）。スペクトルを平均・最大でまとめる列（`metfrag_lite`、`mf_*`、`cc`、N1 の `nbr_ms_mean`、GLACIER の一致）は、dev では本数が多いぶん揺れが小さく、見える test の本数では弱くなる見込み。dev の値はこの点で楽観側
- 見える test は train の写しの形式例である可能性がある（`kaggle-improvement-loop` の記録）。隠れた test の本数は分からない。上の比は「見える test と比べると」の話にとどめる
- 次の群の評価でも、型ごとの値に加えて「分子のスペクトル本数 ≤ 3 の分子だけ」の値を補助に出す（§4.5、定義を先に固定する）

## 4. H3 群の定義案: 断片の構造の統計（MetFrag 2.4.5 の S_Peak の形、学習なし・数え上げ）

### 4.1 考え方
いまの断片の列は「候補の断片の質量が観測ピークに合うか」。H3 群は「**観測ピークのその m/z は、学習の参照では、どんな部分構造で説明されていたか**」を数え、候補の断片がそれと同じ部分構造かを点にする（出典 1 の N_mf と θ_mf）。同じ式の異性体でも、ピークを説明する部分構造は違うので、質量の点が同じ候補を分けられる見込み（推定、この文書では測らない）。

### 4.2 列（4 列、名前は案）
- `pfs_ll`: 問い合わせのスペクトルごとに Σ_k log θ(f_ck | m_k) を計算し、分子のスペクトルで平均。k はスペクトルのピーク（`frag_rescore._prep_spectrum` と同じ前処理のピーク）。f_ck は候補 c の断片のうちピーク k に合うもの（合うものが複数なら θ が最大のもの。この選び方はこちらの決め。MetFrag 2.4.5 の選び方は未確認）。合う断片が無ければ ⊥
- `pfs_ll_per_peak`: `pfs_ll` をピークの数で割ったもの（スペクトルごとに割ってから平均）
- `pfs_seen_frac`: 強度の重み（√I、`cc` と同じ）つきで、割り当てた (m, f) の組の数 N_mf（自分の構造の分を除いた後、§4.3）が 1 以上のピークの割合
- `pfs_ll_fdmax`: `pfs_ll` から、同じリストの中の**同じ分子式の候補**の `pfs_ll` の最大を引いたもの（同じ式の候補が自分だけなら 0）。いまの相対化（`relativize.py`）は分子のリスト全体が単位なので、式の違う候補が混ざる。異性体どうしの差だけを見る列

### 4.3 作り方（自前のコード、既定の定数）
- 断片: MetFrag-lite と同じ「0〜2 本の結合を切った連結成分」（2 本切りは結合 ≤ 100 の分子だけ、`fragment_levels` の `max_bonds_2=100` と同じ）を、**原子の集合**で列挙し、成分の正準 SMILES（RDKit `MolFragmentToSmiles`、水素は書かない）を断片の名前 f とする。MetFrag-lite の `frag_rescore.py` は断片の質量しか返さないので、列挙は新しく書く（`frag_rescore.py` は data/ext の外部のファイルなので変えない）
- m: ピークに合った断片の**分子式**と、電荷の担い手・水素の移動 dh の組（MetFrag 2.4.5 が m を断片の式の質量に直してから区切るのと同じ意味。式で持てば区切りの幅の定数が要らない）。照合の許容と dh は MetFrag-lite の `DEFAULT_CFG` のまま
- 数え上げ N_mf: train の参照スペクトル（構造ごとに先頭 16 本まで、`QUERY_CAP` と同じ）について、**その参照の正解の構造**の断片で説明できたピークの (m, f) を数える。fold ごとの数 N_mf^(fold) と、構造ごとの寄与を持っておく
- fold-safe と隠す規則: dev の問い合わせ（fold k）には Σ_{j≠k} N^(j) を使う。さらに集合の `hide_rule.json`・`drop_keys.txt` で隠す参照の分を引く（raw の inchikey14 と正準の鍵の両方、N1 と同じ厳しい側）。12/6 の holdout は fold 0 を除く数、提出は全 train
- **自分の構造の分を除く（own-out）**: 候補 c が train の構造なら、c 自身の参照の寄与を N から引いてから θ を計算する。正解・偽物のどちらにも同じ規則を当てる（参照のある構造だけが自分の (m, f) で有利になるのを防ぐ。出典 1 の著者も 70:30 の分け方で同じ点を避けている。`lib_max_sim` で起きた「学習では偽物だけが占める領域」の型の偏りを避ける狙い）
- θ の推定: θ_mf = (N_mf + α) / (Σ_f' N_mf' + α·|F_m| + β)、⊥ には β/(…)。|F_m| は m で観測された断片の種類の数（候補の断片が未観測なら |F_m| に 1 を足す）。**α = 0.0001、β = 0.0025**（出典 1 の Table 1 陰性モード 10 行の α・β の最頻値。dev で動かさない。論文の陽性モードの行は全部は読んでいない）
- 欠け: 問い合わせのスペクトルにピークが無い分子は全列 0、`pfs_seen_frac` は NaN にせず 0（`cc` と同じ扱い）。SMILES が読めない候補は全ピーク ⊥
- 学習・ハイパラ: 無い（数え上げと既定の pseudo count だけ）。LightGBM lambdarank は既定のまま

### 4.4 計算の費用
- 測った（候補 300 構造、dev の表の `cand_smiles` から seed 0 で無作為、scratch の試しのスクリプト、コミットしない）: 0〜2 本切りの連結成分の列挙と正準 SMILES の生成で **6.5 ms/構造**（1 コア、Python + RDKit）、最大 RSS 275 MB。結合の数の中央値 25、異なる断片の SMILES の数の中央値 99・p90 146
- 推定（上の速さからの外挿。ピークの照合と数え上げの時間は入っていない）:
  - dev の候補の異なる構造 371,391 → 約 40 分（1 コア）、6 プロセスで約 7 分
  - train の構造 275,810（参照の数え上げ）→ 列挙だけで約 30 分（1 コア）。参照のピークの照合は 2.54M 本のうち構造ごとに 16 本まで（平均で約 6.8 本／構造、§3 の `mean_capped16`）
  - 数の表の大きさ: (m, f) の組は、構造 × 説明できたピークの数。1 構造 20 組とすると約 550 万組（推定）。(64 bit の hash, 32 bit の数) で約 66 MB、fold ごとの数を持つと × 5
  - Kaggle（隠れた test 1,200 分子 × 約 110 候補、dev の表の 430,979 行 / 3,901 分子から）: 列挙 約 14 分（1 コア）。表は Kaggle の private dataset の残り（約 121 MB、11/19 の記録）には fold 別のまま載らない見込みで、提出用は全 train の 1 本（約 66 MB、推定）にするか、Kaggle Model（別の枠）に置く
- **時間の制約**: 項目 5（カード t_f27ab674）の測定では、見える test 400 分子で 5,736.6 s、1,200 分子への外挿が 4.78 h で、4.5 h の目安を既に超えている。H3 群を提出に入れるなら、PubChem 層の採点の短縮と合わせて時間の予算を先に決める（この文書では決めない）
- 最初の一歩（推定 1 h、CPU、peak < 3 GB の見込み）: dev 1 集合（例 `b8_f1_b_t10`）で、列の計算の時間と NaN・0 の数だけを測る（MRR は見ない）。N1 の #261 と同じ進め方

### 4.5 dev での測り方（台帳 1 行）
- 定義（列の名前・作り方・定数）を、値を見る前に Issue に固定し、実行前レビューを通す
- 表: 土台の表（その時点で済んだ最新の版。note に書く）に 4 列を足した表。`leak_check_ledger.py` all_ok（§4.2 (a)〜(e)。参照スペクトルを使う特徴なので hide_rule に従う、#270 の §4.2(c)）。EXTRA_ALLOWED に 4 列を登録しテストを足す
- 当てはめ: LightGBM lambdarank 既定（num_boost_round=100、早期停止なし、seed 0）、fold-safe（leave-one-fold-out）。台帳 1 行（exp0NN）に表の sha256・特徴一覧の sha256・commit
- 報告: 5 型 a〜e を別々に、土台との差の点推定と 95% 区間、exp003 との累計の差を note に。版 A に入れてよいのは、5 型すべてで土台との差の点推定 ≥ −0.02、かつ exp003 との累計も ≥ −0.02 のときだけ（共通の決まり、playbook §4.1）
- 補助の数（定義はいま固定する）: (i) 同じ式の誤りの直りと壊れ: 土台で 1 位 ≠ 正解かつ 1 位と正解が同じ式だった分子のうち、新しい版で正解が 1 位になった数と、土台で正解が 1 位だった分子のうち新しい版で 1 位が同じ式の別の候補になった数（両方を並べる。片方だけ数えない）、(ii) スペクトル本数 ≤ 3 の分子だけの型ごとの差（§3 の読み）、(iii) 正解の行の `pfs_seen_frac` と偽物の行の範囲を型ごとに並べる（own-out が効いているかの点検）。補助の数で採否を決めない
- 見込み（推定）: 出典 1 の利得（Top1 5 → 21/87）は「断片の質量だけ」の土台からのもので、こちらの土台には指紋（`fz`）・`mf_*`・`cc` などが既にある。dev での利得は小さい見込み（+0.005〜+0.02 程度）で、seed の揺れ（exp003・011・012 の幅: a 0.0060・b 0.0042・c 0.0120・d 0.0020・e 0.0149、#272）と同じ大きさになり得る。差が区間の中で 0 を跨いでも、それ以上の読みはしない

### 4.6 12/6 の判定用の 4 集合での読み方
- 判定は共通の決まりどおり 1 回: v3_c1u_t10（C1 型）・enveda_v1u_t10（プールにあり参照なし）・v3_c2_t10（C2 型）・v3_c2nu_t10 の 4 集合すべてで、score+prior の基準（0.8684 / 0.8402 / 0.0000 / 0.5987）との差の 95% 区間の下限 ≥ −0.02、かつ 1 つ以上で下限 > 0（#249・type-weights.md §3.2）。H3 群はこの判定を版の一部として受ける（群だけを別に判定しない）
- 期待する向き（推定、判定には使わない）: own-out により、参照の有無で点が変わらない設計なので、正解に参照の無い enveda_v1u_t10・v3_c2nu_t10 で同じ式の異性体の並べ替えが効く見込み。C2 型（v3_c2_t10）は正解が PubChem 層にだけある型。層の窓の中の同じ式の候補の数はこの文書では数えていない（未確認）。C1 型は正解が別の装置の参照を持つので、`lib_max_sim` 系との重なりが大きく、差は小さい見込み
- 漏れの点検（判定の前）: 4 集合の正解と、その正準の鍵の別名が、fold 0 を除いた数の表（own-out の後）に寄与していないこと（`leak_check_ledger.py` に足す）

### 4.7 採らなかった案と理由
- `mf_*`・`metfrag_lite` を同じ式の群の中で相対化した列だけ（費用は秒）: 異性体を分ける**新しい入力**ではなく、既にある値の並べ替え。H3 の仮説（断片の証拠が弱い）に答えない。§4.2 の `pfs_ll_fdmax` の形だけ採った
- MARASON・ICEBERG の再学習: GPU の学習で、新しいニューラルネットは入れない決まり（G8）。ICEBERG 系は GLACIER（#268）で別に入る
- romanrozen の同式異性体リランカー: 生成コードが見つからず、上位 2 本も切っている
- 断片の損失 S_Loss（出典 1）: 断片どうしの部分構造の関係が要り、費用が増える。S_Peak の形で効くか見てから

## 5. ライセンスと再現性の条件（まとめ）

- MetFrag 2.4.5（出典 1）: 論文 CC BY 4.0。コードはライセンスのファイルが見当たらないので写さない。H3 群は式（θ の MP 推定）だけから自前で書き、データは競技の train だけ。オフラインで動く。再現性: 自前のコードと train だけから作れる（学習なし、乱数なし）
- MARASON（出典 2）: 論文 CC BY-SA 4.0、コード ms-pred（MIT、過去の記録）。重みは NIST20 由来で不可。考え方だけを参照し、コード・重みは使わない
- ICEBERG（出典 3）: 論文 CC BY-NC-SA 4.0（過去の記録）。コード MIT。MassSpecGym 版の重みは主催者の承認あり（GLACIER の項目 2 で使う）。この文書の案では使わない
- 公開ノートの部品: §2 の表のとおり。仮想 NP 層・vtlp・x1full は CC-BY-NC-SA-4.0、bgunshui v2 は unknown、生成・学習のコードは見つからない（x1full は「bgunshui の公開の学習コード」と説明にあるが、そのノートは見つからない）。APRIOR・UNIQ・NPX・E180X・Dual-Mode Mass Sync はノートの中にコードがあり、オフラインで動く（ノート自体のライセンスの記載は見ていない、未確認）。この文書ではどれも写さない

## 6. 次の手順（この文書の外）
1. H3 群の定義（§4.2・§4.3 の列・定数・own-out）を新しい Issue に値を見る前に固定 → 実行前レビュー
2. dev 1 集合で列の計算の時間と欠けの数だけを測る（§4.4 の最初の一歩）
3. 20 集合 → 台帳 1 行（§4.5）
4. 提出に入れる場合の Kaggle の時間の予算（§4.4 の時間の制約）を、PubChem 層の採点の短縮と合わせて決める
