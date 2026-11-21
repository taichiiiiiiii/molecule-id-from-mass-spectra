# 調査: train の構造に依らない「候補 × スペクトル」の証拠（次の自前の段の条件）

- Issue: #178（計画 2026-11-21 昼 項目 5、カード t_72f99867）。親 #3、関連 #171（rep）・#150（lfus）・#146（d_seen）・#144（偏りの調査）・#177（11/22〜の方針）
- 書いた日: 2026-11-21（JST）。採点・学習・提出はしていない。v3 の採否はこの文書では決めない
- 印: 【未確認】= 出典か手元の数で確かめていない主張。【推測】= 自分の解釈

## 0. なぜ調べるか

- rep（#171）は v3 の 4 集合で不採用（v3_c1 −0.0318 [−0.0566, −0.0069]・enveda_v1 −0.0540・v3_c2n −0.0723、#171 の記録）で、偏り Δ_dom = d_dom(rep) − d_dom(MetFrag-lite) も v3_c1・e180_c2k で higher_train だった（`outputs/local_eval/rep_dseen.json`）。lfus（#150）も不採用
- rep・lfus はどちらも「train のライブラリにある構造（とその参照スペクトル）」から点を作る。次の自前の段は、**候補の構造と問い合わせのスペクトルの対だけから出る量**（train に同じ・近い構造があるかで値が変わらないはずのもの）から選びたい
- ただし「対だけから出る量」でも、train の構造（U）と train に無い構造（O）とで化学の分布が違えば、点の分布も違いうる。そこで、既にある対だけの量 = MetFrag-lite（`frag` 段）の d_dom を基準として測る（§2）

## 1. 出典（この回に読んだもの）

この回に本文を取り直して読んだのは 1.1〜1.6。取り方: arXiv の PDF は curl で落として pypdf でテキストにした（scratch、リポジトリには入れない）。PMC・BMC・Nature は web の抽出（BMC と Nature は途中で切れた）。

### 1.1 MassSpecGym（Bushuiev et al., NeurIPS 2024 D&B、arXiv:2410.23326v3）
- URL: https://arxiv.org/abs/2410.23326 （PDF v3 全 49 ページをテキスト化。読んだのは §3.2 の課題の定義、§3.3 のデータ集め、表 3・表 4 の見出しと数値、付録の候補の作り方〈Algorithm 1〉と分割の段落）
- 要点（確認）
  - molecule retrieval の候補 C は「正解と同じ質量（実験誤差の範囲）」の分子で、|C| ≤ 256、多ければ無作為に取る。bonus は同じ分子式に絞る（§3.2）
  - 候補は (i) 生体・環境由来の 1M → (ii) 4M → (iii) PubChem 118M の順に、256 に達するまで同じ質量（Algorithm 1 では ±10 ppm）か同じ式で取る（§3.3、付録）。**縮小版 v3 の窓（±10 ppm）と同じ幅**
  - 表 3 の見出し「Baseline results for the molecule retrieval challenge … 99.9% confidence intervals upon bootstrapping (20,000 resamples)」。MIST の Hit@1 は質量の候補で 14.64%、bonus（同じ式）で 9.57%
  - データ集めで「部分式への分解で全強度の 50% 超を説明できないスペクトル」を外している（§3.3）。部分式による説明はスペクトルと分子式だけで出る量で、学習を要らない
  - 分割は MCES < 10 の対が fold をまたがないようにしたもの（§3.4 Dataset splitting、図 2）。de novo 用の未ラベル分子からは test の分子と MCES < 2 のものを除いてある
- この文書への意味: MassSpecGym 型の評価は正解がすべて train から遠いので、「train の構造の有無で点が偏るか」はこの型の評価では見えない（#144 の読みと同じ）【推測】

### 1.2 ICEBERG（Goldman, Li, Coley、arXiv:2304.13136v2）
- URL: https://arxiv.org/abs/2304.13136 （PDF 49 ページをテキスト化。読んだのは要旨、データの段、retrieval の段〈図 5 の見出し〉、表 1、付録 S1.1 の CASMI22）。論文のライセンス CC BY-NC-SA 4.0（arXiv の表示）
- 要点（確認）
  - 学習データは NIST20（35,129 スペクトル、24,403 構造）と NPLIB1（GNPS）。NPLIB1 は天然物が多く、SA スコアの平均 3.75（NIST20 3.01）
  - retrieval の評価: テストのスペクトルごとに PubChem から正解と Tanimoto の高い順に最大 49 個の「decoy」の**異性体**を取り、予測スペクトルとの cosine で並べる（＝同じ分子式の候補の中の順位）。NPLIB1 の top-1 で 29% 対 次点 20%
  - CASMI22（正イオン 303）では PubChem の全異性体（平均 1,784 候補/式）で top-1 12.9%
  - 本文の主張「decoys not appearing in the training set に汎化する」は著者の解釈で、train に有る decoy と無い decoy の点の差は測っていない（本文に数値なし）【未確認】
- この文書への意味: 学習した順方向の予測器は、学習した構造の集合（NIST20 か MassSpecGym）に対して偏りうる。train に依らない証拠には当たらない（学習元が競技の train でなくても、その集合と train の重なり次第）【推測】

### 1.3 MetFrag 2.2（Ruttkies et al., J. Cheminform. 8:3, 2016、doi:10.1186/s13321-016-0115-9、PMC4732001、CC BY 4.0）
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4732001/ （全文。読んだのは方法の候補の取り方と断片の点、評価の段、表 1・表 2 の見出し）
- 要点（確認）
  - 候補は中性の単同位体質量 ± 相対誤差（例 ±5 ppm）か分子式か ID で取る。表 1 の見出し「Comparison of in silico fragmentation results for 473 Eawag Orbitrap spectra (formula search)」＝**同じ分子式の候補**
  - 断片の点（式 2）: SC_Frag = Σ_p RelMass_p^α · RelInt_p^β / (Σ_b BDE_b)^γ。合う断片の切った結合の**結合解離エネルギー（BDE）の和**で割る。α = 1.84、β = 0.59、γ = 0.47 は「この論文の評価に使わない別の部分集合（MassBank の IPB・RIKEN）」で著者が決めた値
  - 断片の点は学習しない（結合の切断の組み合わせ）。点を足すと強くなるのは文献の数・保持時間など、スペクトルの外の情報（1 位 PubChem 30 → 336）
  - InChIKey の第 1 ブロックで立体異性体をまとめ、同点は悲観的な順位で数える
- この文書への意味: MetFrag 2.2 の断片の点は「候補 × スペクトル」だけから出る。縮小版の MetFrag-lite（`data/ext/frag/frag_rescore.py`）は同じ考えの別の実装で、BDE を使わず切った結合の数（0/1 回 = 1、2 回 = w2 = 0.6）で重みをつける（同ファイルの docstring）
- ライセンス: 論文の「Availability and requirements」に MetFrag 2.2 のソフトウェアは「GNU LGPL version 2.1 or later」とある（確認）。今のリポジトリ MetFragRelaunched にはライセンスのファイルが見当たらない（`fragment-isomer-ranking.md` の出典 3 の欄）ので、LGPL がそこにも及ぶかは【未確認】

### 1.4 CSI:FingerID（Dührkop et al., PNAS 112:12580, 2015、doi:10.1073/pnas.1509788112、PMC4611636）
- URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC4611636/ （全文。読んだのは要旨、方法の 3 段〈学習・予測・採点〉、評価と議論の段）。PNAS のオープンアクセス
- 要点（確認）
  - スペクトルから断片木を作り、断片木とスペクトルのカーネルで SVM が分子指紋を予測し、候補の指紋と比べて並べる。**断片木の計算は構造のデータベースを要らない**（"without the need of any (structural or spectral) database"）が、指紋の予測は参照の MS/MS で学習する
  - 候補は正しい分子式の構造（分子式は先に分かると仮定）
  - 独立データで「学習データに対応する構造が無い 358 の novel な化合物」では、**すべての方法で同定率が大きく落ちた**。著者は原因を説明できていない（議論の段）
- この文書への意味: 学習した指紋の方法は「学習に構造があるか」で成績が変わる、という報告。今の FPNet・rep の偏りと同じ向きの現象だが、この論文は decoy の側の偏りは測っていない【未確認】

### 1.5 Fragmentation trees reloaded（Böcker & Dührkop, J. Cheminform. 8 (2016)、号・論文番号は未確認、doi:10.1186/s13321-016-0116-8）
- URL: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-016-0116-8 （web の抽出は 50,000 字で切れた。読んだのは背景、断片木の定義、評価の段の前半、事前分布の段の冒頭。点の式の細部〈各事前分布の式〉は読んでいない）。論文のライセンスは今回確かめていない【未確認】
- 要点（確認）
  - 断片木は各ピークに部分式を当て、辺を中性の損失とする木。最大事後確率で選ぶ（SIRIUS 3）
  - 「事前分布のハイパーパラメータは構造のデータベースから決めるが、手法をそのデータベースで学習するわけではない」（"we do not train our method using these databases"）。独立の 2 データではハイパーパラメータを決め直さずに評価した
  - 断片木の点は**候補の分子式**ごとに決まる（構造ごとではない）
- この文書への意味: 断片木の点は train の構造に依らないが、同じ分子式の異性体を分けない。届くのは「正解と分子式が違う候補」が上にいる分子だけ（§4.1 で上限を測った）

### 1.6 SIRIUS 4（Dührkop et al., Nat. Methods 16:299–302, 2019、doi:10.1038/s41592-019-0344-8）
- URL: https://www.nature.com/articles/s41592-019-0344-8 （web の抽出は本文の短報と補足図の見出しまで。方法の細部と補足の本文は読んでいない）
- 要点（確認）: SIRIUS 4 は Java、**GNU GPL v3** のオープンソース。CSI:FingerID を組み込み、構造データベースの検索をする
- 未確認: CSI:FingerID の学習データの範囲（NIST を含むか）、指紋の予測が手元で動くか（web サービス経由か）。Kaggle の採点はネットにつながらない
- この競技で使えるか: 断片木の部分（学習なし）は考え方として使えるが、GPL v3 のコードを提出のノートブックに入れるなら配布の条件が付く【未確認】。CSI:FingerID の重みは学習データが確かめられないので使わない側に倒す

### 1.7 CFM-ID 4.0（Wang et al., Anal. Chem. 2021、doi:10.1021/acs.analchem.1c01465）
- 読めたのは要旨だけ（ACS は本文が有料）。要旨（確認）: 分子の位相に基づく特徴からパラメータを学習し、環の開裂を単純な結合の切断の列として扱い、手書きの**規則に基づく予測器**の対象をアシルカルニチン・アシルコリン・フラボノール・フラボン・フラバノン・フラボノイド配糖体に広げた
- 学習データの出所（NIST を含むか）はこの回も確かめられていない【未確認】（#124 の `fragment-isomer-ranking.md` の欄と同じ）。規則に基づく予測器だけを取り出して使えるか、そのライセンスも【未確認】
- この文書への意味: 手書きの規則は train に依らない証拠の候補だが、対象のクラスが限られ、v3 の正解のうちそのクラスがどれだけあるかは数えていない【未確認】。今回の候補には入れない（§4.3）

### 1.8 競技の規則（引用のみ、今回は取り直していない）
- 756893（David Healey、2026-11-11、`docs/research/organizer-question-756893.md` の記録）: "Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible."
- 750330（NIST は不可、重みなどの派生物も含む）、748882（承認済みのチェックポイントは学習コードと許可データへの参照と自前の推論コードで足りる）は `fragment-isomer-ranking.md`・`glacier-iceberg-provenance.md` の記録から

## 2. 偏りの測定の事前登録（このコミットで固定し、区間はこの後に出す）

### 2.1 既に分かっている値（引用、区間なし）
`outputs/local_eval/rep_dseen.json`（sha256 `aad24b808deccc3167da40365bce36543fe4bac9de0a01095f4150b514bcff12`、#171 の rep_dseen.py〈origin/feat/171-rep-stage の commit c5bf237、script sha256 `099f346a…9c244`〉が書いたもの）の `d_dom.metfrag.mean`（小数 4 桁に丸めた値）:
- v3_c1 0.1871（n 180）、v3_c2 0.1666（n 175）、enveda_v1 0.2581（n 179）、v3_c2n −0.0125（n 117）
- 参考: e180_c2k 0.2735（n 187）

### 2.2 定義（替えない）
- `frag` 段は MetFrag-lite そのものなので、rep_dseen.py の Δ_dom（d_dom(対象) − d_dom(MetFrag-lite)）を frag に当てると定義から 0 になり意味が無い。代わりに **d_dom(MetFrag-lite) の絶対値と 95% 区間** を集合ごとに測る
- d_dom は rep_dseen.json と同じもの（rep_dseen.py c5bf237 の `molecule_d`）: 候補 = run.py の score 段の窓（±10 ppm）を pool の key でまとめたもの、正解の除き方は `dseen.truth_exclusion`、群 S・U・O は `dseen.group_of`（S = fragnn の学習構造 T、U = S でなく train の構造、O = それ以外）。**rep の点と MetFrag-lite の点の両方が有限の候補に絞り、sd0 の判定も rep と MetFrag-lite の組**。分子ごとに z（ddof 0）にして d_dom = mean(z|U) − mean(z|O)
- 今日は「frag だけの定義」（rep の点が無い候補も入れるもの）には替えない。替えると rep_dseen.json の値と比べられなくなるため
- 計算: 既存の分子ごとの表 `outputs/local_eval/rep_dseen_per_molecule.csv`（sha256 `1e170a337563f20212b0df56f2012c42a40c6202fed5f3fc51f8928ca859b569`、838 行）の `d_dom_metfrag` 列に、main の `scripts/local_eval/dseen.py` の `bootstrap_ci`（2,000 回・seed 0・molecule_id の昇順、丸める前の lo・hi）をかけるだけ（`scripts/research/ddom_metfrag_ci.py`）。再計算はしない（重いジョブは無い）
- 先に CSV の集合ごとの平均（4 桁に丸め）と n が 2.1 と一致することを確かめる。一致しなければ区間を出さない（スクリプトは終了コード 2）。このコミットの前に一度だけ平均と n の一致を確かめた（5 集合とも一致。区間はまだ計算していない）
- 帯（rep_dseen.band と同じしきい値、丸める前の lo・hi）: lo > 0 → higher_train、hi < 0 → lower_train、それ以外 → none
- 1 回だけ測る。測った後に定義・集合・帯を替えない

### 2.3 読み方（測る前に固定）
- 4 集合（v3_c1・v3_c2・enveda_v1・v3_c2n）それぞれで読む。e180_c2k は参考として記録だけ
- **d_dom(MetFrag-lite) の区間が 0 を含まなければ（higher_train か lower_train）、候補とスペクトルの対だけから出る量でも train の構造の有無で偏る。** そのときは「d_dom > 0 なら train を覚えている」とは読めない（化学の分布の違いでも出る）。新しい段は、rep と同じ Δ_dom = d_dom(新しい段) − d_dom(MetFrag-lite)（rep_dseen.py の定義、対の bootstrap）で測り、**この値と比べて偏りが大きくならないこと**（rep_dseen.py で読む集合 v3_c1・e180_c2k のどちらでも Δ_dom の帯が higher_train でない）を、実装して採点する前の条件にする
- 区間が 0 を含む集合では、その集合では対だけの量の偏りは見えなかった、と読む（偏りが無いとは言わない）。条件は同じ（Δ_dom が higher_train でない）
- この測定は採否に使わない。v3 の採否は §5 の採用規則だけで決める

### 2.4 結果（1 回だけ、事前登録のコミット e673e15〈committer 2026-11-21T12:46:14+09:00〉の後、12:46:29 JST に計算）
`/usr/local/bin/python3 scripts/research/ddom_metfrag_ci.py` → `outputs/local_eval/ddom_metfrag_ci.json`（sha256 `f8d3c59823414e5cbb350e6d4e4ddcf3eca9611f2d5f0417b0cf95eae9c43397`、git_head e673e15、dseen.py sha256 `9b46a112…e363`、ピーク RSS 59 MB）。平均と n は 5 集合とも rep_dseen.json と一致（means_match_ref_json = true）。

- **v3_c1**: n 180、d_dom(MetFrag-lite) **+0.1871 [+0.0628, +0.3016]**、higher_train（正 119・負 61）
- **v3_c2**: n 175、**+0.1666 [+0.0424, +0.2745]**、higher_train（正 109・負 66）
- **enveda_v1**: n 179、**+0.2581 [+0.1152, +0.3932]**、higher_train（正 121・負 57）
- **v3_c2n**: n 117、**−0.0125 [−0.1866, +0.1633]**、none（正 55・負 62）
- 参考 e180_c2k: n 187、+0.2735 [+0.1589, +0.3854]、higher_train

読み（§2.3 のとおり）:
- 4 集合のうち 3 集合（と参考の e180_c2k）で区間が 0 を含まない。**候補とスペクトルの対だけから出る学習なしの点（MetFrag-lite）でも、正解を除いたおとりのうち train の構造（U）は train に無い構造（O）より、同じ窓の中で約 0.17〜0.26 SD 高く出る。** よって「d_dom > 0 なら train を覚えている」とは読めない
- rep の Δ_dom（v3_c1 +0.2592、e180_c2k +0.4695）はこの基準の上に乗った分で、rep の d_dom そのもの（0.4464・0.743）のうち約 4 割（0.1871/0.4464 = 0.42、0.2735/0.743 = 0.37）は MetFrag-lite と同じ大きさの分だった（rep_dseen.json の値からの割り算）
- なぜ U が高く出るかは測っていない【未確認】。U（train の構造）と O（COCONUT・PubChem などにしか無い構造）の大きさ・元素・切れやすさの違いが考えられる【推測】
- 新しい段の条件（§2.3 で固定）: Δ_dom = d_dom(新) − d_dom(MetFrag-lite) が、v3_c1 と e180_c2k のどちらでも higher_train でないこと。実装して採点する前に確かめる

## 3. 既存の段との重なり

- **frag（MetFrag-lite、`data/ext/frag/frag_rescore.py`）**: 学習なし、候補 × スペクトルの対だけの点。上位 60 の同じ分子式の群の中で FPNet の点と z で合わせる（`run.py` の `frag_rerank` → `gl_fuse.rerank_multi`、λ = FRAG_LAM）。ただし既定の設定（w2 = 0.6、強度は線形、許容 0.005 Da、H のずれ −2〜+3、切断 2 回まで）は docstring によれば「掲示板のアブレーション（241 構造の同じ式の異性体のパネルで +0.065 MRR）」で選ばれたもので、そのパネルの構造が train から来たかは【未確認】。§2.4 のとおり d_dom は 3 集合で higher_train
- **nbz（#103）**: 評価集合の中の問い合わせ同士の相互最近傍のロジットを混ぜる。点の源は他の問い合わせの FPNet のロジット（train で学習）で、「候補 × スペクトル」の新しい証拠ではない。v3 で −0.0175（v3_c1）・−0.0075・−0.0057（`net_mde.json` の diff）
- **Engine 2 の ICEBERG / GLACIER**: 学習した順方向の予測器。MassSpecGym 版は MassSpecGym の train で学習（`glacier-iceberg-provenance.md`）。縮小版 v3 には入っておらず、v3_c2n の正解の 58.5% が MassSpecGym にある（#66）ので v3 では測れない。MassSpecGym の構造と競技の train の構造の重なりを数えていないので、train の構造への偏りがあるかは【未確認】
- **FPNet（score 段）・fragnn・rep・lfus**: どれも train（か train のライブラリ）で学習するか、それを引く点。train に依らない証拠には当たらない

## 4. 候補

### 4.1 候補 A（分子式だけの証拠）の届く上限の測り方（測る前に固定）
- 分子式だけで決まる証拠（SIRIUS 型の断片木の点、同位体の形、元素の規則など）は、同じ分子式の異性体に同じ点を出すので、正解より上にある「分子式が違う候補」しか動かせない
- 上限の定義（`scripts/research/formula_oracle_bound.py`）: 対照 `submission_score+prior+frag.csv`（4 集合）で、正解より上の「正解と式が違う候補」をすべて正解の下へ動かす神託を考え、正解の新しい順位 r' = 1 + (正解より上の同じ式〈か式の分からない〉候補の数)。上限 = Σ (1/r' − 1/r) / n。正解が上位 25 に無い分子は 0 として数え、数だけ記録する。照合は `mrr25.to_key`、式は RDKit の CalcMolFormula
- 先にスクリプトで数えた MRR が run.py の値（0.8535・0.0000・0.8376・0.6284）と 4 桁で一致することを確かめる
- 読み方: 上限が集合の MDE の最小（`net_mde.json` の mde_min: v3_c1 0.0176、enveda_v1 0.0175、v3_c2n 0.0171。v3_c2 は対照が 0 で MDE なし）より小さい集合では、候補 A は採用規則の「下限 > 0」を取れない見込み。3 集合すべてで小さければ、候補 A は作らない
- 1 回だけ測る。採否には使わない

### 4.1 の結果（1 回だけ、事前登録のコミット f949a19〈committer 2026-11-21T12:50:58+09:00〉の後に計算）
`.venv/bin/python scripts/research/formula_oracle_bound.py` → `outputs/local_eval/formula_oracle_bound.json`（sha256 `d4b8219eb8aa100e23f3626ef663c534b8e75c29704119909d3308d199c8a5a5`、RDKit 2026.09.1、58 s、ピーク RSS 143 MB）。スクリプトで数えた MRR は 4 集合とも run.py の値と 4 桁で一致。この json には git_head・スクリプトの sha256 を入れていないので、事前登録の後に計算したことの証拠はファイルの更新時刻（12:51:57 JST ＞ f949a19 の committer 12:50:58）だけ。

- v3_c1: 対照 0.8535 → 神託 0.8588、**上限 +0.0053**（動く分子 3、神託で 1 位になる分子 1、上位 25 に無い 3）。mde_min 0.0176 より小さい
- v3_c2: 0.0000 → 0.0000、上限 0（正解が 200 分子すべてで上位 25 に無い。±10 ppm の窓にも正解が無い〈`iso_rank_diag.json` の excluded.truth_absent 200〉ので、並べ替えの段では届かない）
- enveda_v1: 0.8376 → 0.8412、**上限 +0.0036**（3・1・7）。mde_min 0.0175 より小さい
- v3_c2n: 0.6284 → 0.6467、**上限 +0.0183**（10・6・25）。mde_min 0.0171 をわずかに超える

読み（§4.1 のとおり）: 3 集合すべてで小さい、には当たらない（v3_c2n だけ上限が mde_min を 0.0012 超える）ので、規則の上では候補 A を「作らない」とは決まらない。

上の値の範囲（独立レビュー 1 回目の指摘で直した）: これは**対照の上位 25 の中だけ**の神託の値で、厳密な天井ではない。正解が上位 25 の外にいても ±10 ppm の窓の中にいる分子は、分子式の違う候補が下がれば上位 25 に入りうるが、この定義では 0 と数えている（§4.1 の事前登録のとおり。定義は替えない）。その分子の数は `iso_rank_diag.json` から、v3_c1 3（上位 25 に無い 3、窓に無い 0）、enveda_v1 7（7、0）、v3_c2n 7（25、窓に無い 18）。それぞれが 1 位まで上がるとした粗い上乗せは ≤ 3/200 = 0.015、7/200 = 0.035、7/195 = 0.036 で、これを足すと MDE を超えうる。分子式だけの証拠はその正解を同じ式の異性体より上に上げられないので、実際の上乗せはこれよりずっと小さいはずだが、窓の全体の並びが要るので今日は測っていない【未確認】

判断: v3_c1 で神託が動かせるのは上位 25 の中で 3 分子で、外れの大半（対照で 1 位が同じ式の異性体: 44 / 49）には分子式だけの証拠は届かない。今日は候補 A を作らない（調査の回で、v3 の採点はしていない）。進めるなら、先に窓の全体の並びで神託の値を測り直す（上位 25 の外の正解を、同じ式の候補の数から順位を決めて数える）。採否はその後でも §5 の採用規則だけで決める

### 4.2 候補 B（BDE で重みをつけた断片の点。frag 段の点を差し替える）
- 中身: MetFrag 2.2 の式 2（§1.3）の形で、合った断片の切った結合の BDE の和で割る点。α・β・γ は論文の値（1.84・0.59・0.47、著者が別の部分集合で決めたもの）をそのまま使い、こちらでは選ばない。MetFrag-lite の点 f を置き換え、frag 段の他（上位 60、同じ式の群、λ = 0.5、z の合わせ方）は変えない（新しい重みを足さない）
- train に依らないか: 点は候補の構造と問い合わせのスペクトルと BDE の表だけから出る。学習しない。ただし §2.4 のとおり、学習なしの点でも U と O で点が違いうるので、Δ_dom で確かめる
- 届く先: 同じ分子式の異性体の並び（v3_c1 の外れ 49 のうち対照で 1 位が同じ式の異性体なのは 44、enveda_v1 39、v3_c2n 64。`iso_rank_diag.json` の flip.control_c）。上位の候補の BDE が違えば並びが変わる
- 既存との重なり: MetFrag-lite と同じ断片の列挙（切断 2 回まで）を使えば、違いは「切断の回数の重み（1・0.6）」を「BDE の和」に替えることと、質量・強度の指数だけ。違いが小さければ効きも小さい【推測】。同じ式の中の順位で MetFrag-lite（B_metfrag、v3_c1 0.7346）と比べる記録を先に取る
- 足りないもの（作る前に片付ける）: (1) BDE の表の出どころとライセンス。論文には MetFrag 2.2 は LGPL-2.1 以上とあるが、MetFragRelaunched のリポジトリにはライセンスのファイルが見つからない（§1.3）ので、そのコードや表は写さない。文献の表を自分で入力するなら、その出典と転載の条件を確かめる【未確認】。(2) 結合の種類ごとの BDE を決める規則（原子の組と結合次数）。表に無い結合の扱いを先に決める
- 756893 に照らす: 学習した中間物を読まないので、コードと BDE の表（出典つき）を一緒に出せば再現できる。表を「自分で作った中間物」とみなされても、生成の手順（文献の値の転記）を一緒に出せる【推測】

### 4.3 候補にしなかったもの
- CFM-ID 4.0 の規則に基づく予測器（§1.7）: 規則は train に依らないが、本文を読めず、学習データ（NIST の有無）・規則部分だけを使えるか・ライセンスが確かめられない。対象のクラスに v3 の正解が何分子入るかも数えていない
- SIRIUS の断片木（§1.5・§1.6）: 候補 A に含めた（分子式ごとの点）。上限は §4.1
- CSI:FingerID・ICEBERG・GLACIER（§1.2・§1.4・§1.6）: 学習した重みで、train か別の参照集合の構造に依る。ICEBERG・GLACIER は既に Engine 2 にある
- 部分式で説明できる強度の割合（MassSpecGym の品質のふるい、§1.1）: 分子式ごとに決まるので候補 A と同じ上限に入る

## 5. v3 の 4 集合での測り方（候補 B を作る場合。この文書では採点しない）

- 事前登録: 新しい Issue に、変更（frag 段の点 f を BDE の点に替える）、1 回だけ測る、採用規則、「上がっても下がっても規則どおりに決める」を書き、独立レビューを通してから実装・採点する
- 作る前の条件（§2.3・§2.4）: Δ_dom = d_dom(BDE の点) − d_dom(MetFrag-lite) を rep_dseen.py と同じ定義（対の bootstrap 2,000 回・seed 0、sd0 の判定は 2 つの点の組）で測り、**v3_c1 と e180_c2k のどちらでも higher_train でない**こと。higher_train なら採点せずに止める。rep_dseen.py は main に無い（PR #172 は不採用）ので、同じ関数を写す場合は AST で同一と確かめる
- 採点: `run.py --holdout v3_c1|v3_c2|enveda_v1|v3_c2n` で、新 = `score,prior,frag`（f を差し替えた版）、対照 = 今の `score,prior,frag`（v3_c1 0.8535、v3_c2 0.0000、enveda_v1 0.8376、v3_c2n 0.6284）
- **採用規則（共通の決まり）: 4 集合（C1 = v3_c1、C2 = v3_c2、enveda_v1、v3_c2n）すべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、かつ 1 つ以上で下限 > 0**（分子の bootstrap 2,000 回・seed 0、下限は 4 桁に丸めた値）。集合が 1 つでも欠ければ不採用。C1・C2 の両方を必ず報告する
- C2（v3_c2）は窓に正解が無い（§4.1、200/200）ので、frag 段の差し替えでは定義から差 0 になる見込み。それでも 4 集合の 1 つとして測り、0 であることを確かめる
- 集合ごとの MDE（`outputs/local_eval/net_mde.json`、過去の変種の Δrr の sd から 2.8·sd/√n。最小・中央・最大）:
  - v3_c1: 0.0176 / 0.0324 / 0.0434（対照 0.8535、n 200）
  - v3_c2: なし（対照 0、比べる変種が無い）
  - enveda_v1: 0.0175 / 0.0321 / 0.0401（0.8376、n 200）
  - v3_c2n: 0.0171 / 0.0428 / 0.0538（0.6284、n 195）
  - 読み: 変える分子が少ない変種ほど MDE は小さい（nbz は 10 分子で 0.0176）。frag 段の点の差し替えは上位 60 の同じ式の群の中だけを動かすので、動く分子の数は nbz と fragnn（24 分子、0.034）の間に入る見込み【推測】。効きが 0.02〜0.03 より小さければ判定できない
- 記録だけ（採否に使わない）: 同じ式の中の順位（`iso_rank_diag.py` と同じ定義）で BDE の点と MetFrag-lite を比べる、対照の (c) のうち新で 1 位になった数

## 6. 未確認のまま残っていること
- U が O より MetFrag-lite で高く出る理由（§2.4）
- MetFrag-lite の既定の設定を選んだ 241 構造のパネルが train から来たか（§3）
- BDE の表の出典とライセンス（§4.2）。CFM-ID 4.0 の学習データと規則部分のライセンス（§1.7）
- MassSpecGym の構造と競技の train の構造の重なり（§3）
- 「frag だけの定義」（rep の点が無い候補も入れる d_dom）での値。今日は替えていない（§2.2）

## 7. 再現のコマンド
```
/usr/local/bin/python3 scripts/research/ddom_metfrag_ci.py        # §2.4（numpy だけ、≤ 1 GB）
.venv/bin/python scripts/research/formula_oracle_bound.py          # §4.1（RDKit、約 1 分）
.venv/bin/python -m pytest -q tests/test_ddom_metfrag_ci.py tests/test_formula_oracle_bound.py
```
