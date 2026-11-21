# 調査: train の構造に依らない「候補 × スペクトル」の証拠（次の自前の段の条件）

- Issue: #178（計画 2026-11-21 昼 項目 5、カード t_72f99867）。親 #3、関連 #171（rep）・#150（lfus）・#146（d_seen）・#144（偏りの調査）・#177（11/22〜の方針）
- 書いた日: 2026-11-21（JST）。採点・学習・提出はしていない。v3 の採否はこの文書では決めない
- 印: 【未確認】= 出典か手元の数で確かめていない主張。【推測】= 自分の解釈

## 0. なぜ調べるか

- rep（#171）は v3 の 4 集合で不採用（v3_c1 −0.0318 [−0.0566, −0.0069]・enveda_v1 −0.0540・v3_c2n −0.0723、#171 の記録）で、偏り Δ_dom = d_dom(rep) − d_dom(MetFrag-lite) も v3_c1・e180_c2k で higher_train だった（`outputs/local_eval/rep_dseen.json`）。lfus（#150）も不採用
- rep・lfus はどちらも「train のライブラリにある構造（とその参照スペクトル）」から点を作る。次の自前の段は、**候補の構造と問い合わせのスペクトルの対だけから出る量**（train に同じ・近い構造があるかで値が変わらないはずのもの）から選びたい
- ただし「対だけから出る量」でも、train の構造（U）と train に無い構造（O）とで化学の分布が違えば、点の分布も違いうる。そこで、既にある対だけの量 = MetFrag-lite（`frag` 段）の d_dom を基準として測る（§2）

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

（§1 出典・§3 既存の段との重なり・§4 候補・§5 v3 の測り方は、このコミットの後に書く）
