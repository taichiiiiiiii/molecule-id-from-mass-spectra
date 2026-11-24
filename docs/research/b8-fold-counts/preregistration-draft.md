## B8 振り分け器 事前登録の下書き（2026-11-24 昼 項目 4。**当てはめはまだしない**、次の計画で実行）

数え: issuecomment-7121356702（規則）→ issuecomment-7121428480（表）。数字は `outputs/local_eval/b8_fold_counts.json`。

### 0. #216 §3 から外れる点と理由
- §3 は「学習も評価も fold-0 の保留側だけ、構造の鍵の hash で 2 つに分ける」だった。**fold 1〜4 の OOF（各 fold の構造を、その fold を学習していない fpnet_fold{k}.pt で採点）に変える**
- 理由: fold-0 の (a) 型（v3_c1u 型）は eligible 414 のうち v3_c1・v3_c1b とその別名で 404 を使い切り、残り 10（`docs/research/a3a-holdout-expand/a3a-counts.json` の eligible_all 414、今回の数え fold 0 の (a) = 10）。(a) が欠けると §3 の採否の規則でコードが採用を拒む。fold 1〜4 では (a) が 449〜492 ある
- §3 の他の点（特徴・向き・C=1・4 型 1:1:1:1・採否の規則・「v3_*_t10 の今の分子は評価に入れない」）は変えない

### 1. 集合（新しい分子、1 回だけ作る）
- fold k = 1〜4 のそれぞれで 4 型を **各 200 分子**（v3 の集合と同じ大きさ。数えた N_bal 449〜492 の範囲内。ここで固定し、結果を見て増減しない）。合計 16 集合・3,200 分子
- 引き方（seed 20261124、fold k・型 t ごとに `seed + 100·k + t`、t = 0..3）: (a) A から `select()` と同じ `rng.choice` → (b) B∖G(選んだ a) から → (c) C∖G(a∪b) から → (d) D∖G(a∪b∪c) から（1 つの identity_group は 1 つの型にだけ）
- 作り方は元の型のスクリプトと同じ: (a) `write_holdout(rule=enveda_libs, pool=keep, query=enveda)`、(b) `rule=all, pool=keep, query=prefer_enveda, test_adducts`、(c) `rule=all, pool=remove, query=enveda`（v3_c2 と同じ）、(d) `holdout_expand.write_enveda`（072ec66 と同じ、rule=all・pool=keep）。作った後に #116 の `rebuild_set`、10 種の付加イオンに絞る `_t10`（残り 0 本の分子は除く）
- 除外: v3 の 4 集合・A3a の 3 集合の正解（数えでは fold 1〜4 で 0）。以前のランカー（#29・#32）の学習の集合 `rk_f{k}_*` の構造は除かない（そのランカーは使わない。重なりの数を記録する）
- 名前 `b8_f{k}_{a,b,c,d}`、置き場所 `data/holdouts/`・`outputs/local_eval/`（クローンの実のディレクトリ）
- ロジット: `scripts/cv/local_fold0_logits.py --model data/ext/foldsafe/fpnet_fold{k}.pt --out-dir outputs/local_eval/b8_f{k}_{t}/`。fold k の集合は fold k のモデルだけで採点する（他の fold のモデルは使わない）。`pubchem.py` のキャッシュ名は `pubchem_fold0.json` のままだが、集合ごとのディレクトリなので中身は fold k のロジットのもの（実装で集合の `pc_logits.npz` の sha256 をキャッシュの隣に記録する）

### 2. ラベルと特徴（§3 の 8 個、向きは §3 のまま）
- ラベル y = 正解（`mrr25.to_key`）がプールの ±10 ppm 窓（隠したキーを除く）にある = 1、無い = 0。(c) は pool remove なので 0、(a)(b)(d) は 1 の見込み（実際の値を記録、型で決め打ちしない）
- 特徴（対照 score,prior,frag の列 L と、`pubchem.py` の PubChem 層の列 Q（プールのキーを除く、f·z 順、最大 60））:
  1. κ_conf(L)（+）、κ_gap(L)（+）: selpred.py の定義（softmax T=1、f(L0) − f(L1)）
  2. κ_conf(Q)（−）、κ_gap(Q)（−）: 同じ定義を Q の f·z に当てる
  3. d = プールの窓の f·z の最大 − Q の 1 位の f·z（+）（run.py の pubchem 段の rel の符号を反転し、1.5 倍はしない）
  4. κ_lib（+）: selpred.py の lib_max（run.py の lib 段の diag と同じ）
  5. log(1 + |プールの窓|)、log(1 + |PubChem 層の窓|)（向きは制約しない）
- 欠け・無限: Q が空なら 2 は 0、3 は学習側の d の最大値で埋める。|L| = 1 の +inf・|L| = 0 の −inf は学習側の有限の最大・最小で置く（数を記録）

### 3. モデル（既定のまま、選ばない）
- 向きを制約したロジスティック回帰: 特徴は学習側の平均・標準偏差で標準化、目的 = Σ logloss + ‖w‖²/(2C)、C = 1、切片は罰なし・向きなし、`scipy.optimize.minimize(method="L-BFGS-B")` の既定の許容誤差で、+ の係数 ≥ 0・− の係数 ≤ 0・5 は制約なし
- 重みなし（型ごとに同数なので 1:1:1:1 のまま）。温度・しきい値は選ばない
- 並べる相手（記録だけ、採否に使わない）: 「κ_conf(L) だけ」の 1 変数モデル（向き +、同じ手順）

### 4. 合成の形（§3 では「別の Issue」だったが、採否の測定に要るのでこの下書きで 1 つに固定する）
- 分子ごとに p を出し、L の r 番目の候補に p/r、Q の s 番目の候補に (1 − p)/s の重みを付け、重みの大きい順に並べる（同点は L 側を先、InChIKey14 で重複除去、上位 25）。定数なし（1/順位は MRR の重みそのもの）
- 対照 = 今の対照 score,prior,frag（L だけ）。参考（採否に使わない）として 0.479 版の差し込みの規則 score,prior,frag,pubchem も同じ集合で記録する

### 5. 評価（fold 1〜4 の交差、1 回だけ）
- 4 回: fold j（1〜4）を評価、残り 3 fold の 12 集合（各型 600 分子）で当てはめ、fold j の 4 集合に p を出す。評価の分子は、その fold の FPNet にもその回の振り分け器にも見られていない
- 型ごとに 4 回分を合わせ（_t10 で 1 型あたり約 800 分子）、分子ごとの rr の差（新 − 対照）を分子の bootstrap 2000 回（seed 0、パーセンタイル 95%）で
- MDE の目安（net_mde.json の sd_delta の中央値、n = 800、2.8·sd/√n）: (a) 0.0162・(b) 0.0212・(d) 0.0161、(c) は対照が 0 で過去の変種の sd が無い（算出できない）。振り分け器の Δ は過去の変種より広く動くので sd はもっと大きい見込み
- fold 0 の残り（(a) 10）では 4 型の評価ができない → fold 0 は評価に使わない

### 6. 採否の規則（#216 §3 のものに一本化、当てはめの前に確定）
- 4 型すべてで「新 − 対照」の 95% CI の下限 ≥ −0.02、かつ 1 つ以上で下限 > 0。1 型でも欠けたらコードが採用を拒む
- 上がっても下がってもこの 1 回で決める。特徴・C・合成の形・集合を結果を見て変えない。不採用なら理由を記録し、次の版は別の Issue で事前登録

### 7. 分かっている危うさ（記録）
- 提出の経路では FPNet が公開のアンサンブルなので、特徴（κ の尖り方）の分布が fold k の単独モデルと違う。採用になっても、提出に載せる前に別の事前登録（分布の比べ）が要る（skill の「下流は上流の出力の分布に合わせて学習される」）
- §2.4 のとおり 1 位の確信度は (b) と (c) を弱くしか分けない。分けられない分は特徴 2・3 に頼る（効き目は未測定）
- 0.479 版の定数・fp_bits は fold k を見ていないとは確かめられない（数えの §4）

### 8. 費用の見込み（未測定）
- 集合作り: train.parquet を集合ごとに行群で 1 回流す × 16（1 回 20〜60 s の見込み）
- ロジット: CPU、fold ごとにモデル 1 本
- 採点: score,prior,frag ＋ lib ＋ PubChem 層（PubChem 層は 1 分子あたりの時間を最初の集合で測ってから全体を見積もる）。peak は selpred.py の 4.66 GB 程度の見込み、memguard MIN_FREE_PCT=15、GPU 0
- 当てはめ自体は CPU 数秒
