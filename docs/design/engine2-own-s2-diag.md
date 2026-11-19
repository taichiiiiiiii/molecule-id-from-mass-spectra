# Engine 2 を「自前の 2 表＋自前で作り直した S2 の FP 列」にする診断の設計（#158）

この文書は設計と事前登録だけです。S2 の追加学習・表の生成・Kaggle 実行・提出はしていません（別 Issue、オーナー許可の範囲で）。親 #3、関連 #77・#152・#156。

## 0. 位置づけ
- 棚卸し（#156、`docs/research/final-repro-inventory.md` 91–92 行・127 行）: (A) 0.487 写しは Engine 2 の 2 表（行 8 `rank_train.npz`、行 9 `sim_rank_rows_nofp.npz`）が「不可」で、最終提出の要件を満たさない。行 10（prvsiyan の FP モデル `fp_single_s2.pt`・`fp_merged_m1.pt`）は「未決」
- 自前表を入れた唯一の LB は #77（提出 58411499、Public 0.435、0.487 写しに対し Δ −0.052、事前登録の帯では「判定なし」）
- #77 は表と FP 列の出どころを同時に変えたので、−0.052 がどちらのせいか分からない（日報 2026-11-19 昼 Q2）。この文書は、その 2 つのうち 1 つだけを変える診断を決める
- これは自前のコードの部品（自前の表生成 `scripts/engine2/build_tables.py` と、公開の学習コードで自前に学習した S2〈#152〉）を入れる構造の変更で、公開ノートの写しや定数の変更ではない

## 1. #74/#77 の自前表で、学習行の FP 列はどのモデルのどの分割の出力か（受け入れ条件 (i)）

参照したコード（行番号はこのコミットで確かめた）:
- `scripts/engine2/build_tables.py`（`origin/feat/74-engine2-generate` = PR #78、head f45833b）
- `scripts/engine2/base_fs.py`（main と PR #78 で同じバイト、最後の変更 2d46dd8）
- `scripts/submit/make_lb487_owntables.py`（`origin/feat/77-lb487-owntables` = PR #79、head ad87e80）
- 元の Engine 2: `workspace/casmi_daily/ref/frozen_487/eng/casmi_engine.py`・`pv_fp.py`（0.487 写しの凍結コードを復号したもの、`docs/design/engine2-rebuild.md` §1c）

学習行（`own_rank_train.npz` の X の列 15–20・25–30 の 12 列）:
- **モデル: fold-safe FPNet（`dmitriigluzdov/casmi26-fold-safe-fpnet`）の単独モデル `fpnet_fold{k}.pt`、k = クエリの `fold_identity`。つまり OOF（そのクエリの構造を見ていないモデル）で、全 train のモデルではない**
  - クエリは `fold_identity` 1–4 だけ: `build_tables.py` 121–136 行 `select_queries`（129 行 `f.fold_identity.isin([1, 2, 3, 4])`）。fold 0・hold・np・v3 の 4 集合は 91–106 行 `load_exclusions` で除き、139–144 行で積が空を assert
  - モデルの読み込み: 613 行 `banks = {k: base_fs.load_bank(.../fpnet_fold{k}.pt) for k in (1, 2, 3, 4)}`
  - クエリごとの選び方: 418–423 行 `k = int(fold.loc[ik, "fold_identity"])`、`assert k in (1, 2, 3, 4)`、`z = mol_logits(..., k, banks, raw)`
  - ロジットの計算: 366–376 行 `mol_logits` → `base_fs.fs_logits(banks[k], ...)`。`base_fs.py` 149–180 行で、単スペクトルの平均と極性ごとに束ねたスペクトルの平均を 0.5 ずつ（180 行）。先頭 16 本だけ使う（154 行）
  - fold k のモデルは fold k の分子と NP パネルの identity group（split の np 265 構造）を除いて学習（`data/ext/foldsafe/README.md` 29–30 行、`make_split.py` 21–23 行）
- **空間: 10,226 ビット**（v2-pool の `fp_bits.npy`）。Engine 2 の候補指紋（6,930 ビット）とは別なので、FP の 12 列だけを `base_fs.py` 56–82 行 `fp_columns` で 10,226 ビットの候補指紋 `cfp_fs` から計算し、85–91 行 `blk_base_fs` で base 31 列に差し替える（`build_tables.py` 330–335 行 `rows_for_molecule` の 334 行）
- 保存: `own_fp_logits.npz`（`build_tables.py` 496–503 行）

推論（#77 の診断ノート）:
- `make_lb487_owntables.py` 108–109 行: fold-safe FPNet **5 本**（181 行 `fpnet_fold{0..4}.pt`）を読み、`base_fs.zlog_fs_for_test`（`base_fs.py` 183–192 行、191 行で 5 本の平均）。120 行で prvsiyan ランカーの採点を `base_fs.score_base_fs` に替える
- つまり学習行は「単独モデルの OOF」、推論は「5 本の平均」で、ロジットの鋭さが揃っていない（`engine2-rebuild.md` §4.4 でリスクとして記録済み）

元の Engine 2（0.487 写し）:
- 推論の FP: `casmi26-fp-models-v2` の `fp_*.pt` 2 本（`eng_runner.py` の glob、`fp_single_s2.pt` と `fp_merged_m1.pt`）。`casmi_engine.py` 368–370 行 `pv_fp.molecule_logits` → `pv_fp.py` 167–182 行で「single の各スペクトルの平均」と「merged を束ねたスペクトルに」の平均。6,930 ビット
- 元の `rank_train.npz` の学習行の FP 列: 作者 README によると `val_model_logits_combo.npy` / `hard_model_logits.npy` で、どのモデルのどの分割かは作者自身が「unresolved」と書く（`engine2-rebuild.md` 74 行・80 行）

まとめると #77 が 0.487 写しから変えたのは次の 3 つ:
1. (T) 2 表を自前に（クエリ 6 集合 1,872 分子、`own_sim_rank_rows.npz` 402,240 × 51、`own_rank_train.npz` 402,240 × 31）
2. (F) prvsiyan ランカーの FP 12 列の出どころ（学習行も推論も）を、fold-safe FPNet・10,226 ビット・`base_fs` に
3. 再実行で Engine 2 が時間切れ・失敗したら v4g の元のリストに戻る（#77 の 14:43Z の事前登録）。戻ったかは LB からは分からない

## 2. FP 列の鋭さと、全 train の S2 で学習行を作るとどうなるか（OOF が要る理由）

数えたもの（読むだけ、§9 のスクリプト）:
- 元の `rank_train.npz`: 142,762 行、819 グループ、M=0 側 87.2 行/グループ。M=0 の正解行で、FP 列 1 位（列 16 `fp_raw_rank` = 0）**0.402**、同値も 1 位（列 20 = 1）0.420、列 15 `fp_raw_z` の中央値 **1.499**（平均 1.639）
- 自前 `own_rank_train.npz`（#74 run2、sha256 66a8cef9…）: 402,240 行、1,872 グループ、M=0 側 107.4 行/グループ。同じ割合 **0.480**（同値込み 0.480）、`fp_raw_z` の中央値 **1.959**（平均 2.040）
- → 自前表の学習行の FP 列は、OOF でも元の表より強い。元の表の FP 列は、作者の分からない作り方で、自前の OOF より弱い

全 train の S2（#152 の本番 run1）を今の自前表のクエリに使うと:
- S2 の学習は `train_fp3.py` 29 行 `self.tr = np.where(~self.is_val)[0]`。`is_val` は `build_dataset.py` 40–54 行で構造ごとに決まる（VAL = `enveda-np-examples` の 250 構造＋NP ライブラリ ∩ COCONUT から無作為の 2,500 構造 = **2,750 構造**、スペクトルは `is_val` = False が 2,224,546・内部検証 110,018。30–31 行で候補プールに無い構造〈s2p < 0〉を落とした後の実際の学習は 2,222,324〈#152 のログ〉）
- #74 run2 のクエリ 1,872 分子のうち、S2 の VAL に入るのは **99**、残り **1,773（94.7%）は S2 が学習した構造**。class-1 模擬ではクエリのスペクトルそのものを S2 が学習している
- 見た分子と見ていない分子の差の大きさ: fold-safe FPNet の作者の記録で、正解 1 位は見た分子 97%、見ていない分子 68%（`data/ext/foldsafe/README.md` 1 行）。S2 でも同じ向きの差が出ると見込む（S2 では未測定）
- 隠れテストは S2 が見ていない。学習行の FP 列が推論より強いと、ランカーは FP 列を実際より信じる（`engine2-rebuild.md` §4.4 案 D と同じ理由）→ **学習行には S2 が見ていない構造のロジットが要る**（OOF か、S2 の内部検証の構造に限るか）

## 3. 選択肢（受け入れ条件 (ii)）

1 本の時間は #152 speed2 の実測（0.89061 step/s、24,000 step、評価込みで **7.69 h**、footprint 12.67 GB）。重いジョブは同時に 1 本（共用の Mac）。他のプロジェクトの GPU のジョブと重なると約 1/3 の速さになった記録がある（speed2 の 2 回目）ので、上限も書く。

- **(A) S2 を k 分割の OOF で作る（k = identity fold 1–4 の 4 本）**
  - 中身: fold k の構造（inchikey14 と scorer_key の両方）を学習から除いた S2 を 4 本。クエリは #74 run2 と同じ 1,872 分子（fold 1–4）を使え、fold k のクエリには model k のロジット。推論は全 train の S2（#152 run1）
  - 学習: **4 本 × 7.69 h ≈ 30.8 h**（ローカル、1 本ずつ。重なれば最大で約 92 h）。ラッパーに「fold k を学習から除く」差分とテストが要る（`is_val` は内部検証にも使われるので、別のマスクにする）
  - ローカルで回せるか: 1 本の関門は #152 で通っている（12.67 GB ≤ 14 GB）。ただし Mac をほぼ 1.3 日占有し、その間ほかの重いジョブ（merged_m1 の作り直しなど）を入れられない
  - 違い: OOF の各本は 80% のデータで学習、推論は 100% の 1 本。鋭さの差は残る（小さいと見込むが未測定）
- **(A2) 2 分割（fold {1,2} と {3,4}）**: **2 本 × 7.69 h ≈ 15.4 h**。各本が見るのは fold 0・残りの 2 fold（約 60%）で、推論の 1 本との鋭さの差が A より大きい
- **(B) 表のクエリを S2 の内部検証の構造（VAL）に限る — 追加の学習 0 本**
  - 中身: クエリを選ぶ条件に「inchikey14 が S2 の VAL に入る」「同じ scorer_key の別の inchikey14 が S2 の学習側に無い」を足す。FP 12 列は #152 run1 の S2 そのもの（6,930 ビット、`pv.rank_features` の元の式。`base_fs` は使わない）。S2 は VAL 構造のスペクトルを 1 本も学習していない（`is_val` は構造単位）
  - 使える量（§9 で数えた）: 今の除外（v3 の 4 集合・hold・np・fold 0、inchikey14 と scorer_key）と fold 1–4 を満たし、VAL に入り、別名の漏れが無いもの: gnps 1,308・riken 948・mona 720・msdial 525・massbank 421・pluskal_ms2 519（重複を除いて 1,729 構造）。どの集合も今の割り当て（400/300/300/300/300/400、計 2,000）以上ある。ただしユニークは 1,729 で 2,000 より少なく、`select_queries` は集合ごとに独立に選ぶので、**少なくとも 271 クエリは別の集合の同じ構造の重複**になる（#74 run2 では重複 45、ユニーク 1,827）。抜き取りの率も高い（pluskal_ms2 400/519、massbank 300/421）。どちらも事前に記録する違いで、条件は変えない。別名の漏れ: VAL 2,750 のうち 143 構造は、同じ scorer_key の別の inchikey14 が VAL の外（＝S2 の学習側として扱う。S2 のフィルタで落ちる構造も含むので、除く数が多めになる安全側の近似）にある → 除く。実装の assert も同じ定義にする
  - 学習: **0 本**。学習行のロジットは、S2 が見ていない構造に対する S2 そのものの出力（漏れなし）
  - 鋭さについて（1 回目のレビューの指摘で書き直した）: 学習行と推論の FP が同じモデルで揃うのは、将来の最終形（(T) と (F) の両方を自前にし、推論も S2）のときだけ。§5 の診断（(T) だけを変え、推論は prvsiyan の single＋merged のまま）では、A でも B でも「学習行は S2 single、推論は別の 2 本の平均」のずれが残る。これは #77 の「学習行は単独モデル、推論は 5 本平均」と同じ種類のずれ（§1）で、診断の読みの交絡になる（§6）
  - 費用: 表の生成 1 回（#74 run2 は 2,000 分子で 22,636 s ≈ 6.3 h、Kaggle CPU。GPU 枠は使わない）。S2 のロジットはクエリごとに CPU で 1 回（未測定。実装時に 50 分子で測って外挿）。S2 の重みを private の model か dataset に上げる（36.0M パラメータで重みだけなら約 144 MB の見込み、opt を含めない。残り約 0.9 GB に収まる）
  - 違い（事前に記録する）: クエリが NP ライブラリ ∩ COCONUT の構造に寄る（VAL の作り方のため）。#74 run2 の無作為のクエリとも元の表（作者 README では NP 型のレコードと hard クエリ、内訳は未確認）とも分布が違う
- **(C) FP 12 列を 0（prvsiyan ランカーも FP なし）**: 学習 0 本。ただし Engine 2 から FP の証拠を消す別の変更で、表の効果の切り分けにならない → この診断には採らない
- **(D) 全 train の S2 のロジットを今のクエリ（#74 run2）に使う**: 学習 0 本。クエリの 94.7% を S2 が学習している（§2）→ 漏れるので **採らない**

## 4. 推奨: (B)（受け入れ条件 (iii)）
- 理由: 追加の学習が 0 本で、学習行の FP 列に漏れが無く（S2 の VAL 構造）、#152 run1（20:00 JST 頃に終わる見込み）が終わればすぐ表を作れる。最終形（推論も S2）では学習行と推論が同じモデルになる。A は Mac を約 1.3 日（重なれば約 4 日）占有する
- A の利点（比べたうえで採らない理由）: A は #74 run2 と同じ無作為のクエリ 1,872 を使えるので、#77 の表との違いが「FP 列の出どころ」だけになり、#77 と比べやすい。一方、§5 の診断の比較相手は 0.487 写しで、#77 の表との比較ではない。診断の構成での学習行と推論のずれは A でも B でも残る（§3）。30.8 h の学習に見合う差は無いと判断した
- 実装の受け入れ条件（別 Issue。今は書くだけ）:
  1. クエリの条件: 上の 2 つを `select_queries` に足し、クエリ ∩ 除外集合 = 空、クエリ ⊂ S2 の VAL、別名の漏れ 0 を assert。件数を report に書く（`ds/split.pkl` の sha256 も）
  2. FP 12 列: `E.blk_base(dict(rec, zlog=z_s2), cfp)` で 31 列を作る（`pv.rank_features` そのまま）。z_s2 は最終形の推論と同じ関数 `pv_fp.molecule_logits(([S2], [], dev), ...)`（merged なし。`pv_fp.py` 172–182 行は merged が空なら `if merged:` の枝を飛ばし、single の各スペクトルの平均だけを返す。これを単体テストに入れる）。FP 以外の 19 列が #74 の `blk_base_fs` の出力と同じ入力で一致することを単体テストで確かめる
  3. `own_sim_rank_rows.npz` は FP を使わないので生成のコードは今のまま（クエリの集合だけが変わる）
  4. 恒等式チェック（`scripts/research/engine2_table_schema.py`）が自前表でも全式 1.0
  5. 生成は 1 回だけ。表の統計（正例率・候補数・M=0 の正解の FP 1 位の割合）を元の表と並べて記録するが、それを見て生成条件を変えない
- 止める条件（結果を見て案を替えないための事前の規則）: 生成した表の分子数（グループ数）が 819（元の表のグループ数）を下回ったら、その表を使わず Issue に記録して止める（A に自動では替えない。替えるかは次の計画で決める）。#74 run2 は 2,000 から窓の外 125・質量なし 3 の計 128（6.4%）を捨てて 1,872。同じ率なら約 1,870 の見込み

## 5. 診断提出の事前登録（受け入れ条件 (iv)）
- 比較相手（土台）: 0.487 写し（提出 58208946、Public 0.487、`kernels/lb487/casmi26-lb487-copy.ipynb`。main には無く `origin/feat/77-lb487-owntables` にある、入れたコミット 605e29c）
- 変えるのは **(T) Engine 2 の 2 表だけ**: `rank_train.npz` → (B) の `own_rank_train.npz`、`sim_rank_rows_nofp.npz` → (B) の `own_sim_rank_rows.npz`
  - 推論の FP は元のまま（`casmi26-fp-models-v2` の 2 本、6,930 ビット、`casmi_engine.score_molecules(..., ["base"], use_fp=True)`）。`base_fs` と fold-safe FPNet は使わない
  - 再実行で時間切れ・失敗したら元のリストに戻る枝は #77 と同じ形で入れる（オーナー判断 #108 09:19Z の 2）。戻りは診断として読まない
  - 残る違い（事前に書く）: 学習行の FP 列は自前の S2（single 1 本）、推論は prvsiyan の single＋merged の平均。元の表も「学習行は作者の分からないロジット、推論は fp-models-v2」で、学習と推論の FP の出どころはもとから違う
- (F) だけを変える診断（元の 2 表のまま、推論の FP を自前の S2 に）はこの次に別に事前登録する。merged_m1 の作り直し（`train_fp4.py`）が済むまでは single だけになり、FP の構造も変わるため、今は登録しない
- 提出前の関門（すべて満たしたときだけ提出 1 回）:
  1. #152 run1 が `ALL_DONE` で終わり、best の checkpoint が事前登録の規則（2000 step ごとの内部検証）で決まっている
  2. §4 の実装の受け入れ条件 1–5 が通り、表の sha256 を Issue に書いた
  3. 時間: 0.487 写しより表が大きい（rank_train 142,762 → 約 40 万行の見込み）ので HGB の学習が延びる。12 分子の commit に加え、Engine 2 の段の時間を測る版（#77 の timing400 と同じ形、提出しない）で、隠れテスト 1,200 分子（#77 の見積もりと同じ仮定）に外挿した Engine 2 の段 ≤ 3.0 h（上限 `timeout=4 * 3600` の 75%）。超えたら提出しない
  4. Kaggle GPU の今週の残りを確かめ、実行（約 2.7 h）＋再実行に足りる
  5. 提出前の独立レビュー PASS
- 帯の幅: 0.035（計画 2026-11-19 昼 §2 項目 3 の指定。#37 の 160 分子の仮定で、逆数順位の標準偏差 約 0.45 から出した **1 本のスコアの SE** 相当。同じテストに対する 2 本の差 Δ の SE そのものではない〈対応があれば小さく、独立なら約 √2 倍〉）。#77 は 0.04 を使った。結果はまだ無い
- この帯の検出力（Δ の SE を 0.035 と仮定した場合。2 本が独立なら Δ の SE は約 0.049 で、「決まらない」に入る確率はもっと上がる）: 1 SE の帯なので、本当の差が 0 でも約 32% の確率で上下どちらかの帯に入る（片側ずつ約 16%）。本当の差が −0.05（#77 と同じ大きさ）でも「決まらない」に入る確率は約 33%。1 回の提出で強い結論は出ないことを先に書く

## 6. 読みの帯と各帯の次の手（受け入れ条件 (iv)）
Δ = Public(診断) − 0.487。

- **Δ ≤ −0.035「自前表で失点（または再実行で戻った、区別できない）」**
  - 次の手: (B) の表をそのまま最終に入れない。追加の提出はしない。表の中身（クエリの分布、候補数、FP 列の鋭さ、正例率）を元の表と並べる比較を LB を使わずに行い、Issue に書く。最終 2 本の組み方はオーナー判断に回す
  - #77 の読み: −0.052 は「表のせい」と矛盾しない（ただし #77 の表とは別の表なので確定ではない）
- **−0.035 < Δ < +0.035「決まらない」**
  - 次の手: **追加の提出はしない**。表・条件は変えない。(B) の表は「最終の要件を満たす Engine 2 の表」の候補として残し、使うかはオーナー判断。(F) の診断を別に事前登録する
  - #77 の読み: −0.052 のうち表の分は 1 SE より小さい可能性がある、までしか言えない
- **Δ ≥ +0.035「自前表で下がっていない（上がった）」**
  - 次の手: (B) の表を最終の Engine 2 の表の候補にする（オーナー判断）。(F) の診断を別に事前登録する。上がった分を「表の改善」とは呼ばない（1 回、1 SE）
  - #77 の読み: −0.052 は表よりも (F) の差し替えか再実行の戻りから来た可能性がある、までしか言えない（#77 とはクエリも学習行の FP の出どころも違う別の表なので、#77 の内訳は確定しない）
- 共通の交絡（どの帯でも読みに入れる）: 学習行の FP 列は自前 S2 single、推論は prvsiyan の single＋merged の平均（§3・§5）。帯が示すのは「(B) の表一式（クエリの分布・FP 列の出どころを含む）を入れた効果」で、表の非 FP の列だけの効果ではない。再実行の戻りも区別できない
- 共通: 提出は 1 回だけ。**LB を見て、表の生成条件（クエリの条件・割り当て・アナログ数）、S2 の学習条件・checkpoint、ランカーの重み（`W_A`・priors・seed）を選び直さない**。帯も後から動かさない

## 7. v3 で測れない理由と、v3 で確かめられること（受け入れ条件 (v)）
- 測れない理由:
  1. Engine 2（`casmi_engine` の 2 つのランカーと ICEBERG/GLACIER・consensus の後段）は、v3 の縮小版（`scripts/local_eval`）に入っていない。表を替えても v3_c1 / v3_c2 の採点の経路は変わらない（`engine2-rebuild.md` §5）
  2. S2（#152 run1）は全 train（VAL 以外）で学習しており、v3 の正解構造（fold 0 の hold 由来）の大半を見ている（drop_keys のキー〈別名を含む〉v3_c1 202・v3_c2 203・enveda_v1 202・v3_c2n 208 のうち、S2 の VAL に入るのは 0・0・0・21）。S2 のロジットを v3 で採点すると漏れる
- v3 で確かめられること: 縮小版の対照が変わらないことだけ（`run.py --holdout v3_c1|v3_c2|enveda_v1|v3_c2n` の結果ファイルが変更前と `cmp` で同一、`engine2-rebuild.md` §4.5 の 4）。採否の根拠にはならない

## 8. リスク
- S2 の best の checkpoint は VAL の hardneg-top1 で選ぶ（最大 12 回の評価から 1 つ）。(B) のクエリは VAL の一部なので、その分だけ学習行の FP 列が「完全に見ていない構造」より少し良く見える可能性がある（小さいと見込む、未測定）
- (B) のクエリは NP 寄りで、元の表・#74 run2 と分布が違う（§3）。違いは事前に記録したもので、合わせるために条件を選ばない
- 表が大きくなる分、Engine 2 の段の時間が延びる（§5 の関門 3）。再実行で戻ると LB からは区別できない
- 学習行の FP 列（自前 S2 single）と推論（prvsiyan single＋merged）の出どころの違い（§5）。A を選んでも同じく残る
- 1 SE の帯なので 1 回の読みは弱い（§5）

## 9. 数えた方法
- スクリプト（scratch、git の対象外）: `~/.hermes/profiles/massspec/cache/scratch/e2diag/count.py`（sha256 eb1b9f71…）、`alias.py`（dcc2a788…）、出力 `counts.json`（3c08d465…）。クローンの `.venv` ではなく元のフォルダの `.venv/bin/python`。ピーク RSS 690 MB（`/usr/bin/time -l`）
- S2 の分割: `outputs/engine2_fp_retrain/work/ds/split.pkl`（`VAL`・`forced`）と `ds/spec.npz` の `is_val`（#152 の入力、作者の `build_dataset.py` のまま）
- クエリの条件: `build_tables.select_queries` と同じ（6 集合の inchikey14 を train.parquet の `ingest_lib`・`inchikey14` から、`fold_identity` 1–4、除外集合は inchikey14 と scorer_key の両方）。train.parquet は 2 列だけを 65,536 行ずつ読んだ
- 別名: `casmi26-identity-disjoint-validation-folds/structures.parquet` の inchikey14 → scorer_key で、同じ scorer_key の別の inchikey14（train に出るもの）が VAL に入らないものを「漏れ」とした
- FP 列の鋭さ: 各表の M=0 かつ Y=1 の行で、列 16 = 0、列 20 = 1、列 15 の中央値・平均

## 10. 出典
- 棚卸し: `docs/research/final-repro-inventory.md`（#156）
- 自前表の設計: `docs/design/engine2-rebuild.md`（#64）
- #77 の事前登録と結果: Issue #77 のコメント（2026-11-15 14:43Z、2026-11-16 の提出 58411499 と結果）
- S2 の作り直し: Issue #152（§1 出どころ・§4 違い・§6 関門）、PR #153（head c44c989）。上流 `cubres/enveda-casmi-2026` commit 4778979（Apache-2.0）
- fold-safe FPNet: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet（README、ローカルの写し `data/ext/foldsafe/README.md`）
