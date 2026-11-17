# 調査: 学習した並べ替え器の「train の構造への偏り」と、その扱い方

- Issue: #144（計画 2026-11-18 夜 項目 5）。親 #3、関連 #120（fragnn）・#122（e180_c2）・#138（公開系統の調査）
- 書いた日: 2026-11-18（JST）。この文書は調査と測定の**案**だけ。採点・学習・提出はしていない
- 印: 【未確認】= 出典か手元の数で確かめていない主張。【推測】= 自分の解釈

## 0. 問題（手元で確かめた事実）

- fragnn（#120、fold-safe、94k の best）の診断 (ii)（#120 2026-11-17T05:21:11Z、定義は 14:50:50Z の事前登録）:
  - sub_test・fragnn: d = **+0.2371 [+0.1991, +0.2772]**（n 397）、fragnnblind: +0.0834 [+0.0469, +0.1183]
  - v3_c1・fragnn: −0.0458 [−0.1048, +0.0175]、fragnnblind: +0.0750 [+0.0238, +0.1271]（v3_c1 は「読まない」と事前登録）
  - d = 分子ごとの窓の z（ddof 0）で mean(z | key ∈ T) − mean(z | key ∉ T)。T = fragnn の学習構造 214,772 キー（`outputs/local_eval/trainstruct_bias.json`）
  - **この d は正解の候補も含む**。v3_c1 では正解が T の外なので、正解が高く出るほど d は負に寄る。おとり（正解以外）が T だから押し上げられるのかは、この d では分けられない（#120・日報 Q2 で指摘済み）
- fragnn の学習の候補はすべて T の中（#120 の診断 (ii) の事前登録のレビュー: `flat_tr` のユニーク 207,433 はすべて学習の正解構造の中）。つまり fragnn は学習中に **T の外の候補を一度も見ていない**
- 悪くなった分子の内訳（#120 2026-11-17T16:03:26Z、PR #143）: fragnn・enveda_v1 で悪化した 19 分子の新しい 1 位はすべて同じ分子式の異性体で、うち 14 が T、残り 5 も train の構造。ただし基準の割合（T の候補がもともと窓の何割か）は数えていないので、「T だから上がった」とは言えない（同コメントの読み）
- heliosli の rr_v3（`heliosli/casmi26-rr-v3` の `rr_v3.py` docstring、手元の写し `cache/scratch/pub1117n/rrv3/`）: 「train の構造を同じリストの他より約 0.34 SD 低く出す。ベンチの正解は全部 train の外なので効くが、隠れテストの class 1 では損」→ FREEZE。**0.34 SD は作者の主張で、手元では測っていない【未確認】**（公開版の重みはデモで、`public_demo_manifest.json` は `competition_weights: false`）
- 向きが逆: fragnn は T を高く（sub_test、正解を含む d）、rr_v3 は train を低く出す（作者の主張）

## 1. 出典

### 1.1 MassSpecGym（Bushuiev et al., NeurIPS 2024 Datasets & Benchmarks）
- arXiv:2410.23326v3、DOI https://doi.org/10.48550/arXiv.2410.23326 、HTML https://arxiv.org/html/2410.23326v3
- 読んだ範囲: HTML 版の本文（序論・§2 関連研究の分割の段落・§3 の課題と指標・§4 のデータ・分割の段落・ベースラインの段落）。補足資料（分割の詳細・MCES の分布の図）は読んでいない
- 要点
  - 従来の分割は 2D InChIKey が違えば別の fold にするだけで、Tanimoto > 0.85 の近い分子が train と test にまたがる（図 3）。MassSpecGym は MCES（結合の編集距離）で単連結の凝集クラスタリングをし、閾値 10 で「fold の間に MCES < 10 の対が無い」分割にした
  - 検索の候補は、同じ質量の分子を (i) 生体・環境由来 1M → (ii) 4M → (iii) PubChem 118M の順に、上限まで無作為に取る（同じ式の版もある）。**候補に train の構造が混ざるか・その割合は本文に書いていない【未確認】**
  - 未ラベルの 4M 分子は、test の分子と MCES < 2 のものを除いてある（de novo での漏れを減らすため）
- この競技への示唆
  - MassSpecGym 型の評価（正解がすべて train から遠い）は、隠れテストの class 2・3 型だけを測る。class 1（正解が train の構造）はこの型の評価に入らない。だから「train を低く出す」偏りは MassSpecGym 型の評価では得にしか見えない（rr_v3 の docstring と同じ構図）【推測】
  - 手元の v3（fold-0 の構造を学習から外す）は InChIKey 型の分割で、MCES 型ではない。fragnn の T と v3 の正解の近さ（Tanimoto・MCES）は測っていない【未確認】
- ライセンス: arXiv の非独占配布ライセンス（abs ページで確認）。データセット（Hugging Face `roman-bushuiev/MassSpecGym`）は API で `license:mit`。この文書では考え方だけを使い、データは取り込まない

### 1.2 De Waele et al.「On objective mismatch in molecular retrieval from tandem mass spectrometry」（NeurIPS 2026 採択と記載）
- arXiv:2602.16507v2（2026-11-09）、DOI https://doi.org/10.48550/arXiv.2602.16507 、CC BY 4.0（abs ページで確認）
- 読んだ範囲: HTML 版の §1、§3 の regret bound の節（定理 3.2 の前後、不確かさと類似度の幅）、§4 の実験設定（分割・候補集合）、付録 G の表 6 への言及の段落、付録 H（対照学習の負例の候補集合の選び方）。**数式は HTML から文字として取り出せず、式そのものは読めていない**。定理 3.3 以降と付録 D の証明は読んでいない
- 要点
  - 候補集合を固定したとき、HR@1 の最適な予測と類似度（Tanimoto・cosine）の最適な予測のずれ（regret）は、条件つきの不確かさ（aleatoric）と、候補集合の類似度の幅（similarity band）が大きいほど大きい。本文は「類似度の幅が広いこと・候補の近い重複が regret を大きくする」と書いている（「確率の差（マージン）の効きが弱まる」という言い方は、式を読めていない自分の解釈【推測】）
  - 「正解は候補集合に入っている」（closed world）を仮定している
  - Enveda-180 でも MCES の単連結クラスタリングで分割（閾値 10 では全体が 1 つの塊になるので、重原子の数に応じた閾値）。候補は PubChem の同じ質量（10 ppm）から最大 256 を無作為に
  - 付録 H: 対照学習の負例を「同じ質量の候補」で学習したモデルは「同じ式の候補」でも、同じ式で学習したモデルより良かった。負例の候補集合の作り方が検索の成績を左右する（作者は「同じ質量の方が難しい負例を多く含むから」かもしれない、と書いている）
- この競技への示唆
  - fragnn は負例がすべて T（学習の正解構造）なので、「T の外の候補」は学習の分布の外にある。付録 H の「負例の集合の作り方が効く」は、fragnn の負例に T の外の候補（COCONUT・PubChem 層）を混ぜる学習の変更が構造的な候補になりうることを示す【推測】。偏りそのものを扱った節は無い（train の構造への偏りは論文の主題ではない）
  - 正解がすべて学習から遠い分割で測っているので、class 1 型の損は見えない（1.1 と同じ）
- ライセンス・再現性: 論文は CC BY 4.0。コードは §4 の直前に「https://github.com/gdewael/ms-mole 」とあり、GitHub API でライセンスは MIT（2026-11-18 に確認）。コードの中身は読んでいない。この文書では考え方だけ（レビューの指摘で、初稿の「公開先は未確認」を訂正）

### 1.3 heliosli `casmi26-rr-v3`（`rr_v3.py`、`rr3_config.json`、README）と公開ノートでの扱い
- 出典: Kaggle データセット `heliosli/casmi26-rr-v3`（https://www.kaggle.com/datasets/heliosli/casmi26-rr-v3 ）、ノート `heliosli/casmi26-0-486-public`（セル 10・11）、`liuzewen123/casmi26-h495-mirror`（セル 23・24）、`zz3t3rn41/casmi26-peaks-to-structures-v2` のセル 19（romanrozen の v8 isomer reranker）
- 読んだ範囲: `rr_v3.py` の全文、`rr3_config.json`、`README_PUBLIC_DEMO.md`、`public_demo_manifest.json`、`dataset-metadata.json`、heliosli 0.486 のセル 11 の冒頭のコメント、zz3t3rn41 のセル 19 の埋め込みコードの FREEZE の部分（正規表現で抜き出した範囲だけ）。heliosli が参照する `reports/reranker_criteria_v3.md` は公開されておらず読んでいない
- 要点
  - `rerank_v3` のモード: `none`（素のまま）・`freeze`（train の構造の候補は位置を固定し、他の候補だけを残りの位置で並べ替える）・`debias`（train の候補の z(点) に定数 delta を足す）・`skiptop`（1 位が train の構造ならリストを触らない）。点は z(−log 順位) + alpha·z(並べ替え器の点)
  - is_train = 候補の SMILES の InChIKey 先頭 14 文字（立体を外す）が train の inchikey14 にあるか
  - 公開版の設定は `mode: freeze`・`alpha 1.0`・`delta 0.0`・`lib_tau 0.9`（lib_max ≥ 0.9 のリストは触らない）
  - 作者の説明（docstring）: train の構造は学習で「正解でもおとりでもあった」、他の候補は分布の外。そのため train の構造を約 0.34 SD 低く出す【未確認、作者の主張】
  - romanrozen の v8（zz3t3rn41 のセル 19）も同じ FREEZE: 「ライブラリのスペクトルで学習したモデルは、ライブラリのスペクトルを持つ構造を下げてはいけない」。lib_max < lib_tau のリストだけ、先頭と同じ分子式の候補 G の中で、train の構造は位置を固定
  - つまり公開の 2 系統（heliosli・romanrozen）は、偏りの大きさを推定して直す（debias）のではなく、**train の構造の候補には触らない（FREEZE）**ことで class 1 を守っている。debias の delta は定数で、選ぶには class の比が要る
- この競技への示唆
  - FREEZE は定数を選ばない構造の規則なので、手元の禁止（ハイパラ調整）に触れずに使える形【推測】。ただし FREEZE は「train を低く出す」偏り（rr_v3 型）に対する守りで、fragnn の「T を高く出す」向きの危険（class 2 で T のおとりを押し上げる）には効かない。fragnn の向きで同じ考えを使うなら、「T の外の候補は位置を固定」（逆 FREEZE）になるが、それは class 1 を守らない【推測】
  - 手元の ref_last（`library_max < 0.7` で train の参照の無い候補を先に）は、train の構造を下げる向きの規則で、LB 0.443（−0.036）で不採用だった（`references/casmi26.md`）。train の構造を下げる規則が隠れテストで損をした手元の実例
- ライセンス: データセットは CC-BY-NC-SA-4.0（`dataset-metadata.json`）。公開版の重みはデモ（競技の重みではない）。ノートのコードのライセンスは確かめていない【未確認】。FREEZE の規則は 10 行程度の考え方で、使うなら自前で書く（コードは写さない）

### 1.4 Chao et al.「An Empirical Study and Analysis of Generalized Zero-Shot Learning for Object Recognition in the Wild」（ECCV 2016）
- arXiv:1605.04253、DOI https://doi.org/10.48550/arXiv.1605.04253 （ar5iv の HTML https://ar5iv.labs.arxiv.org/html/1605.04253 ）
- 読んだ範囲: 要旨・§1・§3（直接の重ね合わせの問題）・§4.1 calibrated stacking・§4.2 AUSUC・§4 の新規性検出の手法との関係・§5 の表 3 の段落。補足資料は読んでいない
- 要点
  - 学習で見たクラス（seen）と見ていないクラス（unseen）を同じ並びで比べると、unseen のデータもほとんど seen のクラスに分類される（学習で見たものを高く出す偏り）
  - calibrated stacking: 点 f_c(x) から seen のクラスだけ定数 γ を引く。γ は「unseen から来る事前の割合」とも読める。γ を動かすと seen の正解率と unseen の正解率が入れ替わる曲線（SUC）になり、その面積 AUSUC を γ に依らない評価として提案
  - 新規性検出（x が unseen かを先に判定して振り分け）は calibrated stacking に負けた
- この競技への示唆（【推測】）
  - fragnn の「T を高く出す」は、seen クラスへの偏りと同じ形（学習の分布の中を高く出す）。heliosli の `debias`（delta）は calibrated stacking と同じ形
  - γ（delta）の最適値は、テストの seen と unseen の比（隠れテストの class 1 と class 2・3 の比）で決まる。その比は手元に無い【未確認：主催者の公表の有無を確かめていない】。γ を v3 や LB で選ぶのは禁止のハイパラ調整にあたる。使えるのは「γ を選ばずに、偏りの有無と大きさを測る」部分（AUSUC の考え方）まで
- ライセンス: arXiv の非独占配布ライセンス。考え方だけを使う

### 1.5 Zhang et al.「Causal Intervention for Leveraging Popularity Bias in Recommendation」（SIGIR 2021、PDA）
- arXiv:2105.06067、DOI https://doi.org/10.1145/3404835.3462875 （ar5iv の HTML）
- 読んだ範囲: 要旨・§1・§3.1 因果の図・§3.2 deconfounded training・§3.3 推論での調整（PD と PDA）・アルゴリズム 1。実験の節は読んでいない
- 要点
  - 人気度 Z は、出された項目 I と観測された反応 C の両方に効く交絡因子。素直に学習すると、人気の項目を必要以上に上に出す（偏りの増幅）
  - PD: 学習では点を「照合の項 × 人気度の項」に分けて当てはめ、推論では照合の項だけで並べる。PDA: 推論で人気度の項を「望む人気度」（予測した将来の人気度）に差し替える。人気度の強さに定数（γ・α）がある
  - 逆確率重み（IPS）は理屈は良いが、重みの推定が難しく分散が大きいと書いている
- この競技への示唆（【推測】）
  - fragnn の T への偏りを「学習で見た回数」のような交絡と見るなら、PD 型は「学習時にだけ『学習で見た構造か』の手がかりを別の項として持たせ、推論では外す」構造の変更にあたる。ただし fragnn の学習の候補はすべて T なので、学習データの中では「T か」が常に真で、その項を学習で分けられない。分けるには学習の負例に T の外の候補を混ぜる必要がある（1.2 の付録 H と同じ結論）
  - PDA の人気度の差し替え（定数つき）は、手元の禁止に触れるので使わない
- ライセンス: arXiv の非独占配布ライセンス。考え方だけを使う

### 1.6 Kretschmer et al.「Coverage bias in small molecule machine learning」（Nature Communications 16, 554, 2025）
- DOI https://doi.org/10.1038/s41467-024-55462-w （オープンアクセス）
- 読んだ範囲: **要旨だけ**（検索結果の抜粋）。本文は読んでいない
- 要点（要旨）: 広く使われる分子のデータセットは既知の生体分子の構造の空間を一様に覆っておらず、MCES に基づく距離で覆い方を測る方法を提案
- この競技への示唆: 「T の構造」と「T の外の候補」は化学の分布が違いうる（例: train 由来とCOCONUT 由来）。偏りの測定では、学習で見たかどうかの効果と、化学の分布の違いの効果を分けて見る必要がある（下の案の 3 群に分けた理由）【推測】

## 2. まとめ（出典から言えること・言えないこと）

- 学習で見たものへの偏りは、向きも大きさもモデルと学習の候補の作り方しだい（1.3 の作者は「低く」、1.4 は「高く」、fragnn の正解込みの d は sub_test で「高く」）。理屈から向きは決まらないので、測るしかない【推測】
- 文献と公開ノートの扱い方は 3 種類
  1. 規則で守る: FREEZE（heliosli・romanrozen）。定数なし。class 1 を守る向き
  2. 定数で直す: calibrated stacking・debias（delta）・PDA。定数の最適値はテストの class の比で決まり、手元では選べない（禁止）
  3. 学習で分ける: PD 型の分解、負例の集合の作り直し（De Waele 付録 H）。構造的な変更だが学習のし直しが要る
- 評価の側: MassSpecGym・De Waele の分割（正解がすべて train から遠い）は class 1 を測らない。手元の v3 は fragnn から見て「正解は T の外」の集合で、さらに T の外の train の構造（fold 0 など）が窓に入る。この構造は「化学の分布は T と同じで、学習では見ていない」群として使える（下の案）

## 3. v3 と e180_c2（pool keep 版）で測る案（1 つ。この回は実行しない）

### 3.1 名前と問い
- 名前: **seen-decoy 偏り（d_seen）**
- 問い: 正解を除いたおとりだけで見て、fragnn は「学習で見た構造（T）」を「化学の分布は同じで学習で見ていない構造（train ∖ T）」より高く（低く）出すか。正解込みの d（#120 の診断 (ii)）では分けられなかった「おとりの押し上げ」を直接測る

### 3.2 集合
- 読む集合: **v3_c1**（正解は競技の train の構造で T の外）と **e180_c2k**（下の定義。正解は競技の train の外＝class 2 型）
- 記録だけ: enveda_v1・v3_c2n・v3_c2（どれも正解は T の外）。読む集合を class 1 型 1 つと class 2 型 1 つに絞るのは、集合を増やして都合のよい結果を拾うこと（多重比較）を避けるため
- e180_c2k の定義: e180_c2（#122、200 分子、timsTOF、正解は train・プールの外）の問い合わせのまま、`hide_rule` を `{"rule": "all", "pool": "keep"}` にした集合。#122 の数え上げで正解はプールに 0 件なので、窓は remove 版と同じになるはず。**最初に、全分子で窓の行（Pool の添字）が remove 版と一致することを確かめ、違えば止める**。この集合では正解が窓に無いので、指標はおとりだけで決まる（正解を足して順位を見る版は別の Issue、下の 3.6）
- 前提の準備（実行の回で）: e180_c2 の fold-0 ロジット（`local_fold0_logits.py`）と fragnn の npz（`fragnn_score.py`）が無い。run.py は `outputs/holdouts/<h>/truth.csv` を読むので、e180_c2 の読み先を決める必要がある（#122 の限界の節）

### 3.3 指標
- 候補: run.py の score 段の窓（±10 ppm）を pool の key でまとめた行。**正解の key（生と正準化後の両方、別名も）を先に除く**。**fragnn と MetFrag-lite の両方（fragnnblind と比べるときは両方）で点が有限の候補だけ**にして、2 つの d_seen を同じ候補集合の上で計算する
- 3 群（key は生と正準化後の両方で判定。どちらかで入れば入る）
  - S = T（fragnn の学習構造、214,772 キー、`EXPECTED_T`）
  - U = 競技の train の構造（train_structs の key・rawkey・inchikey14、和 278,555）のうち T に入らないもの
  - O = それ以外（プールの train の外の候補）
- 分子ごと: 点を窓の中で z（ddof 0、正解を除いた後）にし
  - **d_seen = mean(z | S) − mean(z | U)**（主）
  - d_dom = mean(z | U) − mean(z | O)（記録だけ。化学の分布の違いの効果）
  - 両方の群に 1 件以上あり、比べる 2 つの点のどちらも窓の中で sd > 0 の分子だけ（Δ_seen の n はこの分子の数）。外した数を理由別（群が片方だけ・sd 0・点の無い候補だけ）に記録
- 点の種類: fragnn（主）、fragnnblind（スペクトルと無関係な成分）、MetFrag-lite（`frag_rescore`、学習していない点＝学習で見たかに依らない基準。S と U の化学の違いの効果が 2 つの点の z の上で同じ大きさで出る、と仮定して差し引く【推測：この仮定は確かめていない】）、FPNet の `fp[bits] @ z`（記録だけ。fold-0 の FPNet の学習構造は T と同じではないので解釈しない）
- 主の対比: **Δ_seen = d_seen(fragnn) − d_seen(MetFrag-lite)**（分子ごとの差の平均、対の bootstrap 2,000 回・seed 0、95% 区間）
- 最小検出効果の目安: v3_c1 の正解込みの d の区間幅（0.122、n 200）から sd ≈ 0.44 と見て、MDE ≈ 2.8 × 0.44 / √200 ≈ 0.09【推測：Δ_seen の sd は測っていない】。これは**計画の目安**（200 分子で足りるか）にだけ使い、読みには使わない。読みは下の帯（区間の端）だけで決め、平均が 0.09 より小さいのに区間が 0 を外れた場合も帯どおり読み、「効果は MDE の目安より小さい」と併記する（レビューの指摘で、目安と帯の食い違いを解消）

### 3.4 読み方（実行前に事前登録で固定する案）
- v3_c1 と e180_c2k のそれぞれで
  - Δ_seen の区間の下限 > 0 → `higher_seen`（学習で見た構造のおとりを押し上げる。class 2 型の危険）
  - 区間の上限 < 0 → `lower_seen`（rr_v3 型。class 1 型の危険）
  - それ以外 → `none`
- 結果の使い道（結果を見る前に固定する）: 次の自前の並べ替え器の設計に、
  - 2 集合とも `higher_seen` → 学習の負例に T の外の候補を混ぜる（1.2 付録 H・1.5 の考え）を設計の前提に入れる
  - 2 集合とも `lower_seen`、または片方が `lower_seen` で他方が `none` → FREEZE（1.3）を設計の前提に入れる。入れる FREEZE は lib_tau の門を付けない形（すべてのリストで train の構造の候補の位置を固定。定数なし）。門などの定数を足すなら、その値の出どころを別に事前登録する
  - 片方が `higher_seen`、他方が `lower_seen` → 判定なし（手当てを入れず、記録して次の計画で扱う）
  - それ以外 → 偏りへの手当ては入れない
- 決めないこと: delta・γ・lib_tau などの定数は、この測定の数で選ばない（禁止のハイパラ調整）。v3 の採否の規則（4 集合の下限 ≥ −0.02）は変えない。fragnn の採否（不採用）も変えない
- **事前登録が要る**: 定義（群・正解の除き方・点の種類・bootstrap）、読む集合と記録だけの集合、帯、使い道を #120 か新しい Issue に書き、独立レビュー（Opus）PASS の後にコードを書き、CI pass の後に 1 回だけ数える

### 3.5 費用の見込み【推測、未測定】
- v3 の 4 集合: fragnn・fragnnblind の npz は既にある（#120）。追加は MetFrag-lite の点を窓の全候補に付けること（v3_c1 の窓は 27,504 行）。時間は未測定。#120 の `trainstruct_bias.py` は `chem.score_key` に 666 s かかったので、T の key の計算は同じ程度
- e180_c2k: fold-0 ロジット（CPU 数秒の見込み）と fragnn の npz（各集合 7〜9 s・ピーク 2.1 GB の実績）が要る
- メモリ 3 GB 以下、GPU 0、Kaggle 0 の見込み

### 3.6 この案に入れないもの（別の Issue）
- e180_c2 に正解を足した閉じた候補集合（De Waele の closed world）で、正解の順位が T のおとりより下がるかを見る版。プールに無い SMILES に fragnn の点と MetFrag-lite の点を付けるコードが要り、窓の形（npz の idx はプールの添字）も変わるので、別に事前登録する
- 0.487 系統での測定（Kaggle が要る）

## 4. 未確認のまま残っていること
- rr_v3 の「約 0.34 SD」（作者の主張、公開の重みはデモ）
- MassSpecGym の検索の候補に train の構造が入るか・その割合
- De Waele の定理の式（HTML から読めなかった）、ms-mole のコードの中身
- 隠れテストの class 1・2・3 の比（主催者の公表があるか確かめていない）
- fragnn の T と v3 の正解の近さ（Tanimoto・MCES）、T と train ∖ T の化学の分布の違い（fold の分け方が無作為か）
- Kretschmer et al. の本文（要旨だけ読んだ）
- 公開ノートのコードのライセンス
