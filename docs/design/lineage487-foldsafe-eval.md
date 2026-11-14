# 0.487 系統を fold-safe に組み直して v3 でリーク無しに測る評価の設計（#112）

関連: #108（懸念 1、コメント 6875852190）、#3、部品表 `docs/research/lineage-reproducibility.md`、Engine 2 の設計 `docs/design/engine2-rebuild.md`（#64）、MSG との重なり `docs/research/mist-massspecgym.md`（#66）

この文書は設計だけです。学習・Kaggle の push・提出はしていません。数値は既存の出力から読んだもの、または下に書いた式・コマンドでこちらが計算したものだけです。

印の意味:
- **[確認]**: ファイル・コード・出力・Issue の記録をこちらで読んで確かめた
- **[未確認]**: 推測、作者の申告、レビューの指摘で、こちらでは確かめていない
- **[計算]**: 既存の出力から、この文書に書いた式・コマンドでこちらが計算した

改訂 r2（独立レビュー 1 回目 FAIL の 12 件への対応）。主な変更: 分母を truth.csv に統一し MSG と重なる分子を保守側で全部外す（§2-10・§5.2）、Tier 0 は「部分的に漏れている」と明記して使ってよいかを決める規則を事前登録（§2.1）、Tier 1 のランカーは OOF の特徴で学習（§2-4）、較正に「ホールドアウトで勝ち LB で負けた」変更の再現 K6 と大きさの範囲を追加（§5.3）、監査に train 由来の固定表を追加（§4）、検出力を等重みの E で計算し直し（§6）、評価を大きくする案（§6.2）、CPU の試走を push の前に入れる（§7）。

## 0. 要点
- いまの v3 の評価は縮小版（LB 0.282）しか回せない。その上での勝ちは LB に移らなかった（ランカー 0.246）[確認: #37、#3]。0.487 系統そのものを v3 で回すには、全 train で学習した部品を、v3（fold 0）を見ていない版に替える必要がある
- 段階を 3 つに分ける
  - **Tier 0**: 学習しない。既にある fold-safe のモデルと自前の表に差し替えるだけ。**Tier 0 は部分的に漏れている**: fe_v4 の fragnet は作者の train fold（fold 0 の大半を含む）で学習していて、外せない（ランカーがその特徴を必須にしている）。Tier 0 の値を較正・採否に使ってよいかは、監査の数で決める（§2.1、事前登録）。見込みは「使えない」で、その場合 Tier 0 は配線と実行時間の確認だけになる
  - **Tier 1**: 公開の学習コードで、fold 0 と v3 の正解を外して学習し直す（Engine 1 ランカー、fragnet、v80）。ランカーの学習の特徴は fold ごとの OOF で作る
  - **Tier 2**: 重いもの（fe_D DreaMS の微調整、FPNet A/B、ICEBERG/GLACIER）
- 回す場所は Kaggle。0.487 写しを生成スクリプトで書き換え、ホールドアウトを test として渡す（`references/holdout-pipeline-eval.md` の手順）。GPU を使う前に、書き換えた凍結コードを手元の CPU で数分子だけ試走する（§7）
- GPU の重い段を出し終えたところで、候補ごとの表を書き出す。後段は手元の CPU でやり直す（再生）
- **信じる前に較正を通す**（§5）。大きい既知の差（Engine 2 off −0.101、縮小版 −0.197）が向きと大きさ（LB の差の 0.5〜2 倍）で出ること、さらに「ホールドアウトで勝ち LB で負けた」ref_last（#22、LB −0.036）が v3 の上でも負けとして出ることを条件にする
- 検出力（§6）[計算]
  - 採点に使う分子は 199 + 200 + 200 + 79 = 678（MSG と重なる分子を保守側で外したあと）
  - 等しい重みの E で検出できる差（対応のある差、80%・両側 5%、sd_d = 0.2 の仮定）は 0.023。Public の 2 本の差（約 125 分子と仮定）の約 2.2 倍鋭い
  - プールの中だけで効く変更は v3_c2 で差が 0 になるので、集合ごとの効きに直すと検出できる差は 0.028
  - +0.01 の部品の検出力は約 22%。+0.01 を検出するには 1 集合あたり約 780 分子（計 約 3,100）が要る → §6.2 で fold 0 の分子を増やす案（リーク無し、最も安い）を出す
- 最初の到達点（§7、約 3 日に分割）: 1 日目に監査、2 日目に生成スクリプトと CPU の試走、3 日目にデータセットと GPU 1 本

## 1. 0.487 系統の骨格（評価に関わる所だけ）
対象は 0.487 写し `taichiiiii/casmi26-lb487-copy`。
- ノートは `kernels/lb487/casmi26-lb487-copy.ipynb`。main には無く、`origin/feat/77-lb487-owntables` のコミット c3d9992 にある [確認: `git ls-tree c3d9992 kernels/lb487/`]。ipynb の sha256 は 380e2c82…、31 セル
- 凍結コードはセル 12 の `FROZEN_B64` で、復号すると sha256 d1e757c2…、59 セル。0.479 版（`kernels/lb479` セル 10）と 59 セルすべて一致する [確認: engine2-rebuild.md §1c]
- 復号したものは `~/.hermes/profiles/massspec/workspace/casmi_daily/ref/frozen_487/c00–c58.py` と `eng/` に置いてある。以下「F cNN」は、このファイルの行番号で引く

| 段 | 場所 | 学習済みの部品 |
|---|---|---|
| Engine 1（v4n） | F c36・c37 | FP バンク `fpnet_full1.pt`（c37 15–16 行。無ければ v4b の `fpnet_0.pt` に黙って落ちる）。`V1FE` の model_views は `fe_A`/`fe_B`/`fe_D`（19 行、MANIFEST の source は fe_D = `work/analysis/dreams_ft/s2/pilot_d1_step20k_ema.pt`）。LightGBM ランカー `ranker_v4b` は 4 本（25 行）。`rank_score` はランカーの特徴が 1 つでも欠けると assert で止まる（28–29 行）。固定の表 `train_structs.parquet`・`train_fp_sel.npy` を `POOL_DIR` から symlink し（c36 4–8 行）、`train_fp_sel.npy` を Engine の `train_fp` に渡す（c37 14・18 行）。Engine は `EngineCfg(generate=True)` [確認] |
| prior の並べ替え | F c39 | 無し（`POOLPOP_MU = 0.15`、6 行）[確認] |
| PubChem チャネル | F c34 | `ModelBank(models + [fold0])`（4 行）。models は v3-models の `fpnet_*.pt` = A+B。lib_max < `LIB_TAU`=0.9 のときだけ差し込む（F c26 12 行、F c43 68 行）[確認] |
| Engine 2 | F c32 → `eng/casmi_engine.py` | `main`（483–506 行）は次のことをする [確認]<br>・`load_library(train)`<br>・prvsiyan の FP モデル 2 本（493 行）<br>・`fit_rankers(sim_rank_rows_nofp.npz)`（502 行、FP なし）<br>・`fit_pv_rankers(rank_train.npz)`（505 行、FP あり）<br>FP のロジット `zlog` が出てくるのは 320・368・374・434 行だけで、prvsiyan ランカーの base の FP 列に入る [確認: grep。ほかの経路は未確認] |
| 融合 | F c45 | 無し（`ALPHA, KRR = 0.6, 3.0`）[確認] |
| ICEBERG/GLACIER | F c41・c45 | MassSpecGym で学習した重み。`BASE_ICE = False`（c41 9 行）なので、入力は Engine 2 の候補と PubChem リストだけ [確認: #62 の結果コメント]。予算は `ICE_BUDGET = 5400` 秒（F c26 13 行）、`GL_BUDGET = 4000` 秒（c41 114 行）。GLACIER は [M+H]+ のスペクトルがある分子だけ（c41 120–121 行）。**ICEBERG の点は最終の並べ替えに入らない**: c45 2 行 `CELL19_ICE = False` で、40・43 行の並べ替えには `{}` が渡り、GLACIER の点だけが入る [確認]。ICEBERG の点がほかの段で使われるかは未確認 |
| 外部参照スペクトル | F c36 | 学習なし。85 行で `in_train_structure` の印を使う。拡張ライブラリの作成・読み込みが失敗すると、`_EXTERNAL_REFERENCE_STATE['error']` に理由を入れて `Library('work')` に黙って戻る（158–169・186–190 行）。作成時に `train_structs.parquet` の sha256 を pin と照合する（192 行の `base_files`）[確認] |
| 空き順位の DreaMS 構造埋め | F c48 | 学習なし（hengck23 の構造表）[確認] |
| 神経スコアラー | 外側セル 13・15 | sanpier の公開重み v1 と v80（popwenman）を `0.5·z + 0.5·z` で混ぜる [確認: 部品表 20 行] |
| v8 | 外側セル 5・23 | `V8 = 'MASS'` → 質量の規則（学習なし）だけが入る。romanrozen の LightGBM リランカー（RR8）は `RR8 = V8 in ('RR','BOTH')` で**切れている** [確認: セル 5] |
| 外部ライブラリの重ね | 外側セル 24 | ExtLib・MassBank harmonized（学習なし、cos ≥ 0.80 で 1 位へ）[確認] |

## 2. 部品ごとの (a) 漏れ・(b) 置き換え・(c) 費用

「v3」は v3_c1 / v3_c2 / enveda_v1 / v3_c2n の正解 4 集合を指す。どれも fold-safe 分割の fold 0（hold）から作っている [確認: `scripts/cv/build_holdout_v3.py`、部品表の「測れるか」の列]。分母はすべて `outputs/holdouts/<set>/truth.csv` の分子数（v3_c1 200、v3_c2 200、enveda_v1 200、v3_c2n 195）に揃える [確認: 行数]。#66 の数（v3_c1 200、enveda_v1 202、v3_c2n 200）は構造の数え方が違うので、この文書では使わない（違いの原因は未確認）。

費用の前提:
- 0.487 写しの 1 回の実行は T4 で約 2.7 時間 [確認: #46・#62 の「GPU 約 2.7 時間」の記録]
- Kaggle GPU は週 30 時間を共有し、1 日 1 本まで（オーナーの許可の範囲）
- 手元の Mac は大きな学習のとき 10 GB の空きが要る。今は約 6 GB しか空いていない [確認: #108 懸念 5]

| # | 部品 | (a) v3／fold 0 を学習で見ているか | (b) fold-safe の置き換え | (c) 費用・置き場 | Tier |
|---|---|---|---|---|---|
| 1 | `fpnet_full1.pt`（Engine 1 の FP バンク） | **見ている**: 全 train で学習（2,539,608 スペクトル）[確認: 部品表 2 行、TC README] | 推論は `fpnet_fold0.pt`（fold 0 と NP パネル 265 を除いて学習、sha256 ddf6eaaa… を F c26 が assert）[確認: 部品表 17 行]。fold 1〜4 のモデルは fold 0 を見ているので、**v3 の推論には使わない**（#46 の 5 本平均は v3 では使えない）。Tier 1 のランカー学習の特徴は 4 の OOF の規則で作る | 0（既存） | 0 |
| 2 | FPNet A+B（PubChem バンク、model_views の fe_A/fe_B） | **見ている**: fold 0 の 89% を 1 位にする [確認: fold-safe README、部品表 3 行] | PubChem バンクを `ModelBank([fold0])` にする。model_views の A・B の枠にも fold0 を入れる（3 枠が同じモデルになる） | 0（Tier 0）／Tier 2 で A・B を別々に fold 0 を外して学習する: T4 で 1 本 14.4 時間 [確認: #82 の試走 1.608 it/s × 82,855 ステップ]。`train_fpnet.py` には再開の仕組みが無く 1 セッションに収まらない [確認: #82] | 0 / 2 |
| 3 | `fe_D.pt`（DreaMS の微調整） | **見ている**: 作者の train fold で学習していて、fold 0 を含む [確認: 部品表 4 行] | Tier 0: D の枠にも fold0 を入れる（アーキテクチャが違う代わりの物として記録する）。`fpnet6.load_model` は checkpoint の `arch` で分岐し、`fpnet_fold0.pt` には `arch` が無い（キーは model・step・nbits=10226・d・layers・heads・args）ので FPNet として読まれる [確認: TC の `fpnet6.py` 185–208 行と checkpoint のキーを読んだ]。V1FE は枠ごとに別の `ModelBank` を作る（`v1engine.py` 78 行）ので、枠の間で nbits は照合されない。fe_D の nbits、model_views の特徴が DreaMS 特有の出力を前提にしていないかは**未確認** → §7 の CPU 試走で確かめる。Tier 2: TC の `dreams_ft/` を fold 0 と v3 を外して回す | Tier 2 は GPU で重い（未測定）。backbone は 1.39 GB [確認: #108 懸念 4] | 0 / 2 |
| 4 | Engine 1 ランカー `ranker_v4b.pkl`（LightGBM 4 本・160 特徴） | **間接に見ている**: 特徴を作る A・B・fe_D・fragnet が fold 0 を見ている [確認: 部品表 1 行]。学習の模擬クエリ（作者の holdout fold の 5,350 構造・15,701 件）と v3 の重なりは**未確認** | Tier 0: そのまま残す（漏れの扱いは §2.1）。Tier 1: TC の 3 段（`simulate_v1ctx` → `build_feats` → `train_v4_ranker`）を回し直す。**クエリ**: fold-safe 分割の fold 1〜4 から選び、fold 0・np・v3 の正解を外す（assert）。**特徴は OOF**: fold k（k = 1〜4）のクエリの FP 依存の特徴（Engine のバンク、model_views の 3 枠）は `fpnet_fold{k}.pt` で作る（#74 案 A と同じ考え方）。理由: fpnet_fold0 は学習した構造を 97% 1 位にするが、見ていない構造では 68% [確認: `data/ext/foldsafe/README.md` 1 行]。学習した構造の上の特徴でランカーを学習すると、v3 の推論（fold0 は v3 を見ていない）で FP を信じすぎる。fragnet・train 由来の表に依存する特徴も同じ規則で OOF にする（5 を参照）。推論は fold0 と、fold 0 を外した fragnet | 段 1 の推定は 3 カーネル × 2 シャードで合計 12.1 時間、補正すると 8.7 時間（T4）[確認: #88 の試走]。OOF にすると FP の推論は fold ごとに 1 回で、総量は変わらない見込み [未確認]。段 2・3 は未測定。LightGBM の学習自体は手元の CPU で回せる見込み [未確認] | 0 / 1 |
| 5 | fe_v4 の表（fragnet `stageA.npz`/`stageB.txt`、`analog_struct/*`、`frag2/*`、`prior_tables.pkl`、derivation の prior） | fragnet は作者の train fold の構造で学習していて、**fold 0 の大半を含む** [確認: 部品表 5 行。割合は未確認]。fragnet の族は外せない: ランカーの 160 特徴に入っていて、欠けると `rank_score` が assert で止まる（F c37 28–29 行）[確認]。`analog_struct`・derivation に v3 の正解が載っているかは**未確認**（§4 の 9〜11） | Tier 0: 外せないので残し、§2.1 の規則に従う。Tier 1: `fe_v4/res_fragnet.py` などを fold 0 と v3 を外して回し直す。4 の OOF のために、fold k を外した版（k = 1〜4、いずれも fold 0 も外す）を 4 本と、推論用に fold 0 だけを外した 1 本を作る | 未測定（CPU の見込み、未確認）。OOF のため 5 回分 | 0 / 1 |
| 6 | v2 プール・`fp_bits.npy`（10,226） | 学習なし。v3 の正解はプールに残っている [確認: 部品表 6・7 行] | そのまま。v3_c2 だけは正解をプールから見えなくする（窓の関数にキー集合のパッチを当てる。`make_holdout_kernel.py --hide-pool` と同じやり方） | 0 | 0 |
| 7 | Engine 2 の 2 表と prvsiyan ランカー | 元の表はクエリと v3 の重なりが**未確認**。FP 列は全 train に近い範囲で学習した FP モデルから作られている [確認: 部品表 8–10 行] | 自前の表（#74）。クエリから v3 の 4 集合の正解・hold・np・`fold_identity == 0` を除くことを assert する [確認: engine2-rebuild.md §4.3]。学習行の FP 列は fold k のモデルによる OOF [確認: §4.4 案 A]。**v3 で推論するときの `zlog_fs` は fpnet_fold0 だけ**にする（#77 の提出版は 5 本平均。v3 では fold 1〜4 が漏れるので使えない） | Kaggle CPU（#74 本番、GPU 0）[確認] | 0（#74 の本番待ち） |
| 8 | prvsiyan の FP モデル（`fp_single_s2`・`fp_merged_m1`） | 学習範囲は**未確認**（fold 0 を含む可能性が高い）[部品表 10 行] | 7 の `blk_base_fs` で FP 列を fold0 から作る。この 2 本の読み込みは禁止にする（§3.1 の読み込みの監視） | 0 | 0 |
| 9 | Engine 2 の候補バンク（COCONUT・ChEBI/LMSD・PubChemLite） | 学習なし。v3 の正解が入っている可能性がある [未確認] | v3_c2 だけは、窓の関数（`pool_window`）に同じキー集合のパッチを当てる | 0 | 0 |
| 10 | ICEBERG/GLACIER | MSG との重なり（truth.csv で数え直し、MSG 1.0 ∪ 1.5、inchikey14）: v3_c1 1（train 1）、v3_c2 0、enveda_v1 0、**v3_c2n 116 / 195**（train 95・val 10・test 11）[計算: 下のコマンド]。ICEBERG の重みは MSG 全体（train ∪ val ∪ test、`msg_all`）で学習 [未確認: レビューの指摘。こちらでは重みの出所を読んでいない]。GLACIER の学習分割は**未確認** | **保守側に倒す**: MSG のどの fold にでも載っている分子は全部、採点から外す。v3_c1 は 1 を外して 199、v3_c2n は 116 を外して **79**（MSG の train ∪ val だけなら 90 になるが使わない）[計算]。外すのは全部の版で同じ分子（版によって ICE/GL を使わない版も含む）。外した分子の値は参考として別に出すが、較正にも採否にも使わない。Tier 0 から外す（Tier 2 まで待たない）。最終の並べ替えに入るのは GLACIER の点だけ（ICEBERG は `CELL19_ICE = False`、§1）[確認]。Tier 2: MSG と重ならない NP 集合を作り直すか（#66 の案）、v3 を外して学習し直す | 0（Tier 0）。Tier 2 の学習し直しは GPU で重い（未測定） | 0 / 2 |
| 11 | sanpier v1（公開重み） | **見ている**: 全 train で学習 [確認: 部品表 18 行] | #54 の重み。キットの既定のまま、fold-safe の hold/np と v3 の正解 55,369 構造を外して学習（リーク 0）[確認: #54] | 0（既存） | 0 |
| 12 | v80（popwenman） | **見ている**: 全 train で 80 epoch [確認: 部品表 20 行] | Tier 0: #54 の重みを v80 の枠にも入れる（2 つのスコアラーが同じになり、混ぜても 1 本と同じ順位）。**このため Tier 0 の L487fs と L479fs の差は、外部ライブラリの重ね（15）の効きだけになる**（§5.3 K3）。Tier 1: キットを epochs=80 にし、#54 と同じ除外で学習 | Tier 1: T4 で前処理 約 1.7 時間＋学習 約 1 時間 [未確認: #54 の 6,175 秒・16 epoch 11 分からの外挿] | 0 / 1 |
| 13 | PubChem 層・文献 prior（`pool_popularity.csv` ほか） | 学習なし。競技データは含まない [確認: 部品表 15・16 行] | そのまま。v3_c2 は PubChem 層からしか届かないので、プール外の扱いにする（`--unhide-pool` と同じ）| 0 | 0 |
| 14 | MetFrag-lite・質量の規則（v8 MASS） | 学習なし [確認] | そのまま | 0 | 0 |
| 15 | 外部参照スペクトル（知識表、782 キー）・ExtLib・MassBank harmonized | 学習なし。v3 の正解のスペクトルが入っている可能性がある。**ExtLib の構造表に train の構造は 0**（#52、部品表 21 行）[確認]。v3 の正解はすべて train の構造なので、ExtLib が v3 の正解を 1 位に上げることはありえない。cos ≥ 0.80 の一致で別の構造を 1 位に置くことしかできず、**v3 の上では効きが 0 か負にしかならない**（推論、構造上の偏り）。隠れテストでは正解が ExtLib に載っていることがありうるので、v3 は ExtLib の採否を判定できない | 知識表の外部参照は hide_rule に合わせる: `all` の集合（v3_c2・enveda_v1・v3_c2n）は v3 のキーの行を落とす。`enveda_libs`（v3_c1）は公開ライブラリの参照なので残す。F c36 の `in_train_structure` の印が、train から消した構造でも立つかを監査で確かめる（§4 の 8）。ExtLib・MassBank harmonized の重ねは、版の比較では「重ねの前」と「後」の両方を出し、採否には「前」を使う | 0 | 0 |
| 16 | hengck23 の DreaMS 構造（空き順位の埋め） | 学習なし。v3 の正解が入っているかは**未確認** | v3_c2 では窓のパッチを当てる | 0 | 0 |
| 17 | romanrozen RR8 | 0.487 では**切れている**（`V8 = 'MASS'`）[確認] | 対象外 | 0 | — |
| 18 | `train.parquet`（Engine 1/2 のスペクトルのライブラリ） | v3 の行が入っている | holdout の手順のとおり、`hold_comp/train.parquet` から集合ごとの hide_rule で行を落とす [確認: `scripts/cv/make_holdout_kernel.py`]。F c36 の `build_spec_cache` は COMP の train.parquet から作る（10 行）ので、ここは hide_rule が効く [確認] | 0 | 0 |
| 19 | `train_structs.parquet`・`train_fp_sel.npy`（`POOL_DIR` の固定表） | 学習ではないが **train 由来の固定表**。hold_comp の train.parquet から行を落としても、この 2 表は全 train のまま symlink される（F c36 4–8 行）[確認]。そのため v3 の正解が「train の構造」として残り、`EngineCfg(generate=True)` の候補生成・`analog_struct`（`ctxlib.py` 187 行で `train_fp` を引く）・derivation が「正解は train の構造」という近道を持つおそれがある [未確認] | §4 の 9〜11 で、表と v3 の正解の積、それを使う特徴の一覧を数える。積が 0 でなければ、hide_rule に合わせて 2 表を作り直すパッチを足す。ただし `train_structs.parquet` の sha256 は外部参照の pin（F c36 192 行）に入っていて、作り直すと拡張ライブラリが失敗し黙って `Library('work')` に戻る → pin も一緒に書き換え、§7 の受け入れ条件で `error is None` を確かめる | 0 | 0 |

順番は Tier 0（配線の確認）→ §2.1 の判定 → Tier 1 の 4・5（ランカーと fragnet）→ Tier 1 の 12 → §5 の較正 K1〜K6 → Tier 2。

### 2.1 Tier 0 の値を使ってよいかの規則（事前登録）
Tier 0 は部分的に漏れている。fragnet（5）は fold 0 の大半を学習で見ていて [確認: 部品表 5 行]、ランカーがその特徴を必須にしている（`rank_score` の assert）[確認] ので、族ごと外すことも、重なった分子を採点から外すこともできない（外すと v3 がほぼ残らない見込み [未確認: 割合は監査で数える]）。

監査（§4）の数で、次のとおり決める。結果を見てから閾値を変えない。
- 集合ごとに、正解が次のどれかに入っている分子の割合 c を出す: fragnet の学習構造、`ranker_v4b` の模擬クエリの構造、`analog_struct`・`frag2`・derivation の表のキー、§4 の 9〜10 で積が 0 でない表
- **どの集合でも c ≤ 5%** → その分子を全部の版の採点から外し、Tier 0 の値を「Tier 0（部分的に漏れあり）」の名で較正 K1〜K6 に使ってよい。採否には使わない
- **どれか 1 集合でも c > 5%、または fragnet の学習構造の一覧を作れない** → Tier 0 の値は較正にも採否にも使わない。Tier 0 の実行は、配線・読み込みの監視・実行時間・段の飛ばしの確認だけに使う。較正は Tier 1（4・5・12 の作り直し）のあとに回す
- 見込みは後者 [未確認]。§7 の到達点は、後者でも無駄にならないように組む

## 3. どこで回すか（d）

### 3.1 Kaggle で回す（主案）
- 生成スクリプト `scripts/cv/make_lineage_fs_kernel.py`（未作成）が、`kernels/lb487/casmi26-lb487-copy.ipynb`（c3d9992）からノートを作る。手では書き換えない。中身は 4 つ
  1. 先頭にセルを足す。ホールドアウトのファイルを `/kaggle/working/hold_comp/` に写し、集合ごとの hide_rule で `train.parquet` を流しながら行を落とす。落とした行数を assert する。外側の COMP（セル 6）は train・test・sample の 3 つがそろったディレクトリを探すので、hold_comp を先に探させる
  2. `FROZEN_B64` を復号し、§2 の差し替えを入れる。入れる先は F c37（FP バンクと model_views）、F c34（PubChem バンク）、F c32（Engine 2 の表のパス・`blk_base_fs`・`zlog_fs`）、窓のパッチ、（§4 の結果しだいで）F c36 の固定表と pin。どの置き換えも「ちょうど 1 回当たる」ことを assert する。入れ子の文字列リテラルの中は、リテラルと同じ書き方でエスケープする。最後に再エンコードし、`FROZEN_SHA256` を書き換える
  3. 外側のセル 15 の `_swap_to_ours` を #54 の重みに向ける
  4. 全セルと入れ子のモジュールを `compile()` してから出力する。compile は構文しか見ないので、動くかどうかは §7 の CPU 試走で見る
- 読み込みの監視: `torch.load`・`lgb.Booster`・`np.load`・`pickle.load` を包む。禁止の名前を読み込んだら例外にし、読んだファイルを全部 `opened_files.json` に書く。受け入れ条件は `opened_forbidden == []`
  - 禁止の名前: `fpnet_full1.pt`、v3-models の `fpnet_0.pt`/`fpnet_1.pt`、v4b の `fe_A.pt`/`fe_B.pt`/`fe_D.pt`/`fpnet_0.pt`、sanpier v1 の公開 `final_model.pt`、v80 の `final_model.pt`、`fp_single_s2.pt`/`fp_merged_m1.pt`、`rank_train.npz`、`sim_rank_rows*.npz`、`fpnet_fold1〜4.pt`（v3 の推論では禁止）
  - Tier 0 で残る `ranker_v4b` と fragnet は監視の対象外。§2.1 の判定と一緒に記録する
  - v4b の MANIFEST の照合（F c26）は sha256 を計算するだけで、モデルとしては読まない。なので `open` は包まず、上の読み込み関数だけを包む [確認: #88 の fe_D コメントの 72〜78 行]
- ホールドアウトのデータセット: `taichiiiii/casmi26-holdout-v3mix`（未作成）に 2 つのサブディレクトリを置く
  - A = v3_c1 + enveda_v1（どちらもプールに残す、400 分子）
  - B = v3_c2 + v3_c2n（395 分子。プールから消すのは v3_c2 のキーだけ）
  - MSG と重なる分子（§2-10）も実行には入れる（ICE/GL の負荷を本番と揃えるため）。採点だけで外す
  - 構造は 4 集合で重ならないので、構造ごとに hide_rule を当てられる [未確認: 重なりは監査で assert する]
  - molecule_id には集合の接頭辞を付ける
  - `truth.csv` は手元にだけ置く
- 1 回の実行は約 400 分子で、隠れテスト（約 400 分子 [未確認: Public 31% ≈ 125 からの逆算。§6 で感度を出す]）とほぼ同じ大きさ → T4 で約 2.7 時間の見込み。1 つの版を A と B の 2 本で回すと約 5.5 GPU 時間
  - `TIME_GUARD_HOURS = 7.5` を超えると、任意の段が飛ばされる [確認: セル 5]
  - ICEBERG は `ICE_BUDGET = 5400` 秒、GLACIER は `GL_BUDGET = 4000` 秒で打ち切られ、残りの分子は黙って点が付かない（F c26 13 行、c41 105・114・123 行）[確認]
  - 外部参照の拡張ライブラリが失敗すると、黙って `Library('work')` に戻る（F c36）[確認]
  - どれかが起きたら、その回は無効として記録する（定数は変えない）。判定は §7 の受け入れ条件
- テストの署名が見本と違うので `IS_RERUN` が真になり、コミットの実行で全件が回る [確認: セル 6]
- 出力（`variants/*.csv`、`submission.csv`）は `scripts/cv/fetch_kernel_output.sh` で取る。採点は `scripts/cv/score_variants.py`（MRR@25・top-1・hit@25）と、`scripts/ranker/train_eval.py` の `bootstrap_diff`（分子ブートストラップ 2,000 回、seed 0）

### 3.2 手元での再生（GPU を何度も使わないため）
- GPU の段（Engine 1/2、PubChem、ICEBERG/GLACIER、神経スコアラー）が終わった所で、分子 × 候補ごとの表を parquet に書き出す
  - 中身は、各エンジンのリスト・生の点・ICE/GL の生の点・sanpier の点・lib_max・prior
  - 書き出しのセルは 1 つ足すだけ。下流のコードは変えない
- 後段（融合・V0〜CUSTOM・frag・prior・重複除去・ExtLib の重ね）を手元の CPU で再実装する。同じ回の `variants/*.csv` と分子ごとに一致することを確かめてから使う [確認: スキルの規則]
- 一致すれば、後段に入れる自前の部品はその表の上で手元で測れる（例: #99 の整列チャネル、class 2 向けの段、K6 の ref_last）。GPU は使わない
- Engine 2 off の版も再生できる見込み: ENG を空にし、PubChem の候補の ICE/GL の生の点を再利用して z を取り直す [未確認]。1 回だけ Kaggle の本物の noeng2 の出力と照らし合わせてから使う

### 3.3 手元で全部回さない理由
- ICEBERG/GLACIER と FPNet の推論は CUDA を前提に書かれている。MPS で動くかは未確認 [未確認]
- Engine 2 の `load_library` は全 train のピークを持ち、約 3.5 GB [確認: engine2-rebuild.md §4.6]。プールの指紋は 909 MB [確認: #82]、PubChem 層もある。今の空き約 6 GB では足りない
- 学習（Tier 1・2）は、オーナーの規則どおり手元の MPS を先に試す。メモリの見張り（mem_guard）を付けて起動し、小さく切った分で速さを測ってから決める

## 4. 監査（漏れが無いことを数で示す）
CPU で読むだけの監査 `scripts/cv/lineage_leak_audit.py`（未作成）を作り、結果を `outputs/research/lineage_leak_audit.json` に書く。数えるのは、v3 の各集合の正解キー（truth.csv の inchikey14、および正準化した scorer のキー）と次のものとの積。分母は truth.csv の分子数。
1. fold-safe の split の hold/np（v3 は hold の部分集合のはず: assert）
2. #54 の sanpier の除外集合（v3 の正解はすべて除外側のはず）
3. #74 の自前表のクエリ（0 のはず: 表の json の件数と照合）
4. `ranker_v4b` の模擬クエリの構造（TC の段 1 の選び方で再現する。#88 で動いたコード）
5. fe_v4 の各表（fragnet の学習構造、`analog_struct`、`frag2`、derivation の prior）
6. MSG の 1.0 ∪ 1.5 の各 fold（§2-10 の値の再現。今の値は次のコマンドで出した [計算、2026-11-14]: `pandas.read_csv(MassSpecGym*.tsv, sep='\t', usecols=['inchikey','fold'])` の inchikey 先頭 14 文字を fold ごとの集合にし、`truth.csv` の inchikey14 との積を数える。結果は §2-10）
7. ExtLib・MassBank harmonized・知識表の外部参照・hengck23 の DreaMS 構造・Engine 2 の候補バンク（漏れではないが、hide_rule を当てる対象を決めるため）
8. F c36 の `in_train_structure` の印が v3 の正解で立つか（v3_c2・v3_c2n で立つなら、hide_rule と食い違う証拠。パッチの対象にする）
9. **`POOL_DIR` の `train_structs.parquet`**（F c36 が symlink し、外部参照の拡張でも symlink する、115 行）と v3 の正解の積 [未確認: まだ数えていない]
10. **`train_fp_sel.npy`**（F c37 14 行で Engine の `train_fp` へ）の行が、9 の表のどの構造に対応するかと、v3 の正解に当たる行の数
11. 9・10 を使う特徴とコードの一覧: `EngineCfg(generate=True)` の候補生成、`analog_struct`（`ctxlib.py` 187 行）、derivation、`in_train_structure`。grep で使う所を全部挙げ、各々が「正解は train の構造」を手がかりにできるかを書く。ランカーの 160 特徴のうち、これらに依存する特徴の数も出す
12. 各正解に最も近い、残った train 構造との Tanimoto（§8 のリスク 12 のため。漏れではなく、記録だけ）

1〜11 の数は §2.1 の判定に使う。9〜10 が 0 でなければ、2 表を hide_rule に合わせて作り直すパッチを §3.1 に足す（pin も書き換える）。

## 5. 較正: 既知の LB の版を正しい順と大きさで並べるか（e）

### 5.1 使う版
| LB の版（Public） | 出典 | v3 での対応する版 |
|---|---|---|
| 0.487（0.487 写し、提出 58208946） | #108 [確認] | **L487fs**: §2 の Tier の差し替えを全部入れた版 |
| 0.479（lb479） | #10 [確認] | **L479fs**: `kernels/lb479` に同じ差し替え（凍結コードは同一）。v80・ExtLib の重ねが無い [確認: 部品表 20・21 行] |
| 0.378（0.479 から Engine 2 とその後段を外した版、提出 58215746） | #62 [確認] | **L378fs**: L479fs に #62 のパッチ（F c32・c41・c50）。生成スクリプト `kernels/casmi26_lb479_noeng2` の手順 |
| 0.416（0.479 の Engine 1 と PubChem の FP バンクを fold-safe 5 本の平均に替えた版、提出 58145202） | #46 [確認] | **L416fs**: Tier 1 以降でしか作れない。**前提は未確認**: v4n は「v4i + engine FP bank fpnet_full1」で、full1 のバンクは後から足された（F c26 の docstring）[確認: 部品表 29 行]。`ranker_v4b` の model_views の特徴は A/B/fe_D から作られていて [確認: MANIFEST]、ランカーが full1 のバンクの特徴で学習されたかは分からない [未確認]。なので #46 の −0.063 を「ランカーが学習したときと違う FP を受けた」せいとは言えない。Tier 1 のランカーを作るときに、TC の段 1 が Engine のバンクに何を使ったかを読み、仕組みが分かってから L416fs の作り方を決める。仕組みが「学習時は学習済みの構造の上の特徴、推論は見ていない構造」なら、L416fs = Tier 1 のランカーを OOF ではなく in-sample の特徴（fold 1〜4 で学習した 1 本 `full1_fs` を全クエリに使う）で学習し、推論は fold0、とする |
| 0.443（0.479 + ref_last、提出 58073753） | #22 [確認] | **L479fs+RL**: L479fs の出力に `scripts/submit/ref_last.py` を掛ける（§3.2 の再生で、GPU 0） |
| 0.282（縮小版 score+prior、提出 58119572） | #37 [確認] | **R282**: `run.py --stages score,prior`（既存の出力） |
| 0.246（ランカー model_c2p） | #37 [確認] | **R246**: 既存の `submission_ranker_c2p.csv` |

### 5.2 指標
- E = 4 集合の MRR@25 の等しい重みの平均。採点に使う分子は §2-10 と同じ: v3_c1 199、v3_c2 200、enveda_v1 200、v3_c2n 79（MSG のどこかに載る分子を全部外したあと）[計算]。§2.1 で外す分子があれば、さらに外して数を記録する。重みと除外は事前に決め、結果を見て変えない
- v3_c2n の 79 は #66 の「100 未満は測定不能」の規則に当たるが、#66 の規則は MIST の採否のためのもの。ここでは等しい重みのまま入れ、SE が大きくなることを §6 で数える
- 集合ごとの値も必ず並べる。外した分子の値も参考として別に出す
- 差の区間は、分子の対応のあるブートストラップ（2,000 回、seed 0。集合ごとに再抽出してから E を作る）

### 5.3 通す条件（事前登録。1 つでも欠けたら、この評価で採否を決めない）
大きさの範囲は「v3 の点推定が LB の差の 0.5〜2 倍」とする。漏れは差を広げる向きに働きやすい（漏れた部品ほど強く見える）ので、上限も置く。fold-safe の部品は全 train の部品より弱く、絶対値はずれる（§8 リスク 4）。0.5〜2 倍はそのずれを許す幅として先に決める（根拠は無く、仮の幅）。
- **K1**（Engine 2）: E(L479fs) − E(L378fs) の 95% 区間の下限 > 0、かつ点推定が **+0.05〜+0.20**。LB の差は +0.101（約 2.5〜3 SE）
- **K2**（系統 vs 縮小版）: E(L479fs) − E(R282) の下限 > 0、かつ点推定が **+0.10〜+0.39**。LB の差は +0.197
- **K3**（0.487 vs 0.479）: 条件にしない（記録だけ）。理由: Tier 0 では v80 = #54 で 2 つのスコアラーが同じになり（§2-12）、差は ExtLib の重ねだけになる。ExtLib は v3 の上では 0 か負にしかならない（§2-15）。なので「下限 ≥ −0.03」はほぼ自動で通り、何も確かめない。Tier 1 で v80 を作り直したあと、ExtLib の重ねの**前**どうしで E(L487fs) − E(L479fs) を出して記録する（LB の差 +0.008 は 1 SE 未満で、順を求められない）
- **K4**（Tier 1 以降、記録だけに下げる）: E(対照) − E(L416fs)。§5.1 のとおり #46 の仕組みが未確認なので、L416fs の作り方が決まるまで条件にしない。決まったら、点推定 > 0 かつ下限 > −0.01（LB の差 +0.063）を条件に上げるかを、v3 の値を見る前に Issue に書く
- **K5**（記録だけ、条件にしない）: E(R282) − E(R246)。LB は +0.036。縮小版の上では v3 の 4 集合すべてで逆向き（ランカーが +0.12〜+0.36）[確認: #37、スキルの記録]。これは既存の出力どうしの比較で結果は分かっているので、系統の評価を確かめる材料にならない
- **K6**（「ホールドアウトで勝ち LB で負けた」の再現）: ref_last（#22）は削除のホールドアウトで +0.024 / +0.060（悪化 0）だったのに、LB で −0.036 だった [確認: #22]。条件は E(L479fs+RL) − E(L479fs) の**点推定 < 0、かつ −0.072〜−0.018**（LB の差の 0.5〜2 倍）
  - 見込み [未確認]: hide_rule が `all` の集合（v3_c2・enveda_v1・v3_c2n）では正解の train の参照が全部消えるので、ref_last は構造上必ず勝つ。負けが出うるのは v3_c1（`enveda_libs`、他のライブラリの参照が残る）だけ。縮小版の上では v3_c1 で −0.036 が再現した記録がある（score+prior 0.822 → +lib-first 0.749 → +ref_last 0.714）[確認: スキルの `references/casmi26.md` 15 行]
  - E で K6 が通らなければ、この評価は「train の参照・lib_sim・in_train_structure を手がかりにする変更」を判定できないと記録し、その種の変更には使わない（v3_c1 だけで判定する、とは今は決めない。必要なら別の Issue で、値を見る前に決める）。K6 が通らないことだけで K1・K2 の結果を捨てはしない
- lib_sim を下げる規則（#36 の PubChem の入口のゲート）は LB に出していない（手元の反実仮想で不採用）[確認: `references/casmi26.md` 25 行] ので、較正には使えない

### 5.4 通らなかったとき
- 漏れ・バグが原因と特定できたときだけ直す。原因は監査（§4）か、出力の食い違いで示す。直したら、全部の版を回し直す
- ホールドアウトの作り方・重み・定数・大きさの範囲を、較正が通るように変えることはしない（結果を見て選ぶことになる）
- K1 か K2 が通らなければ、この評価は「信じない」と Issue に記録する。そのあとは、LB の診断提出（帯つき）だけで判断する

### 5.5 費用
- 較正には L487fs・L479fs・L378fs の 3 版が要り、1 版あたり 2 本 → 6 本・約 16 GPU 時間。§2.1 で Tier 0 が使えなければ、これは Tier 1 のあと
- §3.2 の再生が使えれば、L487fs と L479fs の差（外側のスコアラーと重ね）と L479fs+RL は手元で作れる。L378fs も再生で作れる見込み [未確認]。その場合 Kaggle は L487fs の 2 本（約 5.5 時間）＋ 照合用の L378fs の A 1 本（約 2.5 時間）で足りる
- 1 日 1 本の枠で 3〜6 日かかる

## 6. 検出力（v3 と Public LB の比較）

既存の出力から計算した [計算]。分子ごとの値は `outputs/local_eval/<set>/submission_*.csv` と `outputs/holdouts/<set>/truth.csv` に `scripts/validate/mrr25.py` を当てたもの。下の表の n は truth.csv の分子数（MSG で外す前）。

| 集合 | n | MRR（score+prior+frag） | 分子ごとの RR の標準偏差 | 対応のある差の標準偏差（frag あり／なし） | 同（pmi 段あり／なし） |
|---|---:|---:|---:|---:|---:|
| v3_c1 | 200 | 0.8535 | 0.268 | 0.125（25 分子が動く） | 0.219（40） |
| v3_c2 | 200 | 0.0117 | 0.102 | 0.000（0） | 0.047（1） |
| enveda_v1 | 200 | 0.8376 | 0.300 | 0.140（20） | 0.203（37） |
| v3_c2n | 195 | 0.6284 | 0.408 | 0.203（51） | 0.268（78） |

**sd_d は仮定**: 上の sd_d は縮小版の 1 段の変更（frag・pmi）から測ったもの。系統の上の変更で同じ大きさになるかは確かめていない [未確認]。以下は sd_d = 0.2 を主の仮定とし、表の値も並べる。

### 6.1 計算（等しい重みの E）
E は 4 集合の平均なので、差の SE は SE(E) = ¼ √(Σ_i sd_i² / n_i)。採点に使う n = (199, 200, 200, 79)。MDE（対応のある差、両側 5%・検出力 80%）は 2.80 × SE。[計算: 2026-11-14 に python で計算]
- sd_d = 0.2（4 集合とも）: SE(E) = 0.0083 → **MDE 0.023**。795 分子をまとめた計算（0.2 / √795 = 0.0071 → 0.020）より SE が 17% 大きい。前の版の「795 分子で 0.012〜0.027」は E の定義と合っていなかった
- 表の frag の sd_d（0.125, 0, 0.140, 0.203）: MDE 0.019。pmi の sd_d（0.219, 0.047, 0.203, 0.268）: MDE 0.026
- **プールの中だけで効く変更**（v3_c2 は正解がプールに無いので差 0、表でも frag は 0 分子が動く）: E の効きは集合ごとの効き δ の ¾ に薄まる。sd_d = (0.2, 0, 0.2, 0.2) で SE(E) = 0.0075、集合ごとの効きに直した MDE は 2.8 × 0.0075 / 0.75 = **0.028**
- +0.01 の部品の検出力: 4 集合とも効く場合 22%、プールの中だけの場合 17%
- +0.01 を 80% で検出するには、4 集合とも m 分子（sd_d = 0.2）で 2.8 × ¼ × 0.2 × √(4/m) = 0.01 → **m ≈ 780、計 約 3,100 分子**（MSG で外したあとの数）
- 1 集合（n = 200）だけ: sd_d = 0.125 → 0.025、0.20 → 0.040、0.27 → 0.053（前の版と同じ）
- Public LB
  - Public の分子数は約 125 と仮定する [未確認: 隠れテスト約 400 × Public 31% の逆算で、どちらも確かめていない]
  - 2 本の提出の差は同じ分子の上での対応のある差なので、SE は sd_d / √n_pub。sd_d は隠れテストでは分からない。0.2 を当てはめると、n_pub = 100 → MDE 0.056、125 → **0.050**、200 → 0.040、400 → 0.028
  - E の鋭さは Public の約 2.2 倍（n_pub = 125）。n_pub が 100〜400 なら 2.4〜1.2 倍
  - 1 本の値の SE は 0.45 / √125 ≈ 0.040（逆数順位の標準偏差約 0.45 は #3 の記録 [確認: engine2-rebuild.md §6]）。いまの帯（SE 0.035〜0.04）は、差の読み方としては約 2 倍控えめ。それでも +0.01〜0.02 は見えない
- まとめ: 利点は鋭さだけではない（分子ごとの差と区間が見える、提出の枠を使わない、Public に寄せる過学習が無い）。ただし +0.01 級の部品は今の大きさでは判定できない → 6.2

### 6.2 評価を大きくする案
| 案 | やること | リーク無しか | 費用 | 判断 |
|---|---|---|---|---|
| O1: fold 0 から分子を増やす（v4） | `build_holdout_v3.py` と同じ作り方で、fold 0（hold 55,102 構造 [確認: `mist-massspecgym.md` 33 行]）から 4 集合を各 780〜800 分子に増やす | **リーク無し**（v3 と同じ扱い）: fold-safe の部品は fold 0 全体を外している（fpnet_fold0、#54 は hold を外す、#74 は `fold_identity == 0` を外す、Tier 1 も fold 0 全体を外す）。MSG と重なる分子は同じ規則で外す（hold の 10.3% が MSG にある [確認: 同 33 行]。NP 寄りの集合はもっと多く外れる） | Kaggle の時間は分子数にほぼ比例するとして、3,200 分子 × 1 版 ≈ 22 GPU 時間（約 2.7 h / 400 分子）[計算、線形は仮定]。週の枠の 7 割。§3.2 の再生と組み合わせ、GPU は「GPU の段の状態が変わるとき」だけにする | **主案**。Tier 1 と同時期に別 Issue。上限は class ごとの元データ（timsTOF の構造が fold 0 にいくつあるか、NP 寄りで MSG と重ならない構造がいくつあるか）。数は未確認 |
| O2: ほかの fold（k = 1〜4）で集合を作る | fold k の集合を、fold k を外した部品の組で採点する。ホールドアウトは `outputs/holdouts/rk_f{1..4}_c{1,2,2p}`（各 121〜125 分子）が既にある [確認: truth.csv の行数]。作り方が v3 と同じかは未確認 | 部品が全部 fold k 版のときだけリーク無し。FPNet は `fpnet_fold{1..4}.pt` がある [確認: `data/ext/foldsafe/`]。sanpier（#54）・#74 の表・Tier 1 のランカーと fragnet・v80 は fold 0 しか外していないので、fold k ごとに作り直す必要がある。1 つでも fold 0 版を使うと漏れる | 学習が fold の数だけ増える: sanpier 4 本（1 本 約 2.7 h [未確認]）、#74 の表 4 組（Kaggle CPU）、ランカー・fragnet 各 4 組、v80 4 本。GPU の推論も fold ごとに別の回 | Tier 1 が fold 0 で動いてから考える。費用が O1 より大きい |
| O3: fold をまたぐ OOF | O2 を 5 fold 全部でやり、版の差を fold ごとに出してまとめる | O2 と同じ条件 | O2 の 5 倍の学習 | 測るのは「作り方」の差で、1 組の部品の差ではない。O2 が回ってから |
| O4: NP パネルを足す | fold-safe FPNet と #54 が外している np（265）を集合に足す | FPNet・sanpier はリーク無し [確認: 部品表 17 行、#54]。fragnet・ランカーの学習範囲に np が入るかは未確認。MSG と 239 / 265 が重なる [確認: `mist-massspecgym.md` 34 行] ので、保守側で外すと 26 しか残らない | 小さい | 増える数が少なすぎるので採らない |

## 7. 最初の到達点（f、約 3 日に分割）
1 日目（GPU 0）: 監査だけ
1. 漏れの監査 `scripts/cv/lineage_leak_audit.py` と単体テスト（合成のキー集合で積の数え方を確かめる）。§4 の 1〜12 の数を json に書く
2. §2.1 の判定を出して Issue に書く（Tier 0 を較正に使えるか）。§4 の 9〜11 が 0 でなければ、固定表の作り直しを 2 日目のパッチに足す

2 日目（GPU 0）: 生成スクリプトと CPU の試走
3. 生成スクリプト `scripts/cv/make_lineage_fs_kernel.py` と単体テスト
   - どの置き換えもちょうど 1 回当たる
   - 全セルと入れ子のモジュールが `compile()` を通る（構文だけの確認）
   - 禁止の名前の一覧がコードに入っている
   - 差し替え後の凍結コードの sha256 を記録する
   - 変えたセルの一覧が §2 の Tier 0 の行と一致する
4. **CPU の試走（push の前に必ず）**: 書き換えた凍結コードの Engine 1 の部分（F c36・c37 と V1FE の model_views）を、手元の CPU で 3〜5 分子だけ回す
   - 材料: 手元の v4b の写し（`~/.hermes/profiles/massspec/cache/scratch/tc1114/v4b/`。中身は `MANIFEST.json`・`ranker_v4b.pkl`・`code/fe_v4/` だけで、`casmi` パッケージと fe_*.pt は無い [確認: ls]）。`casmi` のコード（`v1fe_code_casmi` データセット。手元には `fpnet6.py` しか無い [確認]）と fe_v4 の表は、小さい物だけ取ってくる
   - `train.parquet` は試走する分子の候補が入る範囲に切ったものを使う（意味は本番と違い、配線の確認だけ）。メモリの見張りを付け、3 GB を超えたら止める
   - 確かめること: V1FE が `fpnet_fold0.pt` を 3 枠とも読める（D 枠で `arch` が `fpnet` になる）、model_views の特徴が全部出て `rank_score` の assert を通る、点が有限、読み込みの監視が禁止の名前を 0 件にする、`_EXTERNAL_REFERENCE_STATE['error']` が None（または試走で外部参照を切る理由を記録する）
   - 試走が通らなければ push しない。D 枠に fold0 が入らないと分かったら、Tier 0 の D 枠の扱いを Issue で決め直す
5. データセット `taichiiiii/casmi26-holdout-v3mix`（A・B の 2 つ。test.parquet・sample・drop_keys・hide_rule・HOLDOUT.json）を作る。upload は 3 日目の push の直前（private の容量は数 MB）

3 日目（GPU 1 本）:
6. 1 本 push する。どれを回すかは §2.1 の判定と #74 の状態で決める
   - Tier 0 を較正に使える、かつ #74 本番 run1 の表が一致の判定 ①〜③ を通っている → L487fs を A（v3_c1 + enveda_v1）で
   - それ以外 → **L378fs を A で回す**（Engine 2 を使わないので #74 を待たない。較正 K1 に必ず要る版で、Tier 1 のあとも配線の確認として使える）
7. 採点: 版ごと（V0・V1・V2・CUSTOM・最終・submission）に、集合ごとの MRR@25・top-1・hit@25 と区間を出す。MSG で外した分子の値は別に出す。同じ集合の縮小版（score+prior+frag: v3_c1 0.8535、enveda_v1 0.8376）と並べる

受け入れ条件:
- カーネルが COMPLETE
- `opened_forbidden == []`
- 段の飛ばしが 0（`TIME_GUARD` のログ）
- ICEBERG がすべての入力分子に点を付けた（`ICE input molecules` の数 = 点の付いた分子の数）。GLACIER が [M+H]+ のあるすべての入力分子に点を付けた（`v29 GLACIER [M+H]+ only: molecules` の数 = 点の付いた分子の数）。どちらも、比べる版どうしで分子の集合が同じ（Engine 2 off の版は入力が違うので、その差は記録する）
- `_EXTERNAL_REFERENCE_STATE['error'] is None`（ログの `external reference spectra: … added`）。`engine FP bank:` のログが FALLBACK でない
- 監査の json がある
- 数値はすべて Issue に貼った出力と一致する

この時点では較正（K1〜K6）は済んでいない。数値は「Tier 0 の値」とだけ呼び、採否には使わない。

## 8. リスク（g）
1. **Tier 0 に残る漏れ**: fragnet（fold 0 の大半を学習で見ている、外せない）、`ranker_v4b` の模擬クエリ、fe_v4 の表、train 由来の固定表（§2-19）。重なった分子を外すだけでは足りない見込みなので、§2.1 の規則で Tier 0 の値を使うかを決める。ICEBERG/GLACIER と v3_c2n の重なり（116 / 195）は採点から外し、v3_c2n は 79 分子になる → 検出力が落ちる（§6）
2. **Tier 0 の対照は低く出うる**: ランカーが学習したときと違う FP を受け取る。ただし #46 の仕組みは未確認（§5.1）
3. **ホールドアウトの作り方が生む近道**: 縮小版のランカーは v3 の 4 集合で勝ったが LB では負けた（K5）。hide_rule が `all` の集合では正解の train の参照が全部消えるので、「参照を持たない候補を上げる」種類の変更は構造上勝つ（K6 で確かめる）。固定表（§2-19）が全 train のままなら、逆に「正解は train の構造」という近道も生まれうる。隠れテストの class の混ざり方（重み）も分からないので、E の等しい重みは仮のもの
4. **fold-safe の部品は全 train の部品より少し弱い**: 学習データが約 1/5 少ない。絶対値は本物の 0.487 より低く出る。差の大きさも変わりうる（K1・K2 の 0.5〜2 倍はこの幅を許すための仮の幅）[未確認]
5. **GPU の枠と待ち**: 1 版 5.5 時間 × 数版。キューで 75 分以上待つことがある [確認: スキルの記録]。§3.2 の再生が成り立たなければ、較正だけで 1 週間かかる
6. **パッチの結びつき**: Engine 2 を外すと、ICEBERG/GLACIER の入力や最終段の並べ替えも一緒に消える [確認: #62]。版の違いを 1 つの部品のせいにしない
7. **実行ごとのぶれ**: 同じコードでも約 0.002 ぶれる [確認: 0.48x 系統のノートの表、スキルの記録]。比べる 2 つの版は同じ回の中で作る（§3.2）か、ぶれより大きい差だけを読む
8. **結果を見て選ぶ危険**: この評価で定数や重みを選ばない（ハイパラ調整の禁止）。採否の規則は #48 と同じで、4 集合すべてで (新 − 対照) の 95% 区間の下限 ≥ −0.02、1 つ以上で下限 > 0。それに K1・K2・K6 を通っていることを足す（K6 が通らなければ、K6 の種類の変更にだけ使わない）
9. **手元の資源**: 空き約 6 GB、ほかの仕事の swap がある。Tier 1・2 の手元での学習は空きが戻るまで始められない（#108 懸念 5）。§7 の CPU 試走は 3 GB 以内に収める
10. **規則の上で 0.487 系統そのものが除外されうる**（756893）: この評価が測るのは fold-safe の系統。提出に使う系統と同じ部品の組かは、版ごとに書く
11. **ExtLib は v3 で判定できない**: ExtLib の構造表に train の構造は 0 [確認: #52] で、v3 の正解はすべて train の構造なので、ExtLib の重ねは v3 の上で 0 か負にしかならない。Tier 0 では v80 = #54 なので、L487fs − L479fs ≈ 「ExtLib の害だけ」になる。ExtLib・外部ライブラリを足す変更の採否には、この評価を使わない（重ねの前の値で比べる）
12. **fold は identity で分けている**: fold-safe 分割は identity-disjoint [確認: `data/ext/foldsafe/README.md` 1 行]。identity のキーが inchikey14 かは未確認 [未確認]。骨格がほぼ同じ構造は fold をまたぎうる（分け方は骨格の近さを見ていない [未確認]）。v3 の正解に近い構造が train に残ると、`analog_struct` や構造の近さを使う特徴が有利になり、隠れテスト（新しい骨格が多い可能性がある）より値が高く出うる。§4 の 12 で最近傍の Tanimoto を出し、E を近さの帯ごとにも記録する（条件にはしない）
13. **ICE/GL の予算と外部参照の黙った失敗**: 予算を超えた分子や外部参照の失敗は、エラーにならず値だけが変わる（§3.1）。§7 の受け入れ条件で毎回確かめる

## 9. 出典
- 0.487 写しのノート: scratch `nb_1114noon/taichiiiii_casmi26-lb487-copy/`（セル 5・6・15・23・24 を読んだ）。リポジトリでは c3d9992 の `kernels/lb487/casmi26-lb487-copy.ipynb`
- 凍結コード: `workspace/casmi_daily/ref/frozen_487/`（c26・c32・c34・c36・c37・c39・c41・c43・c45・c48、`eng/casmi_engine.py`）
- TC のコード: scratch `tc1114/v4b/`（MANIFEST・`code/fe_v4/v1engine.py`・`ctxlib.py`）、scratch `tc1114/dl/g/v1fe_code_casmi/`（`fpnet6.py`）
- 部品表 `docs/research/lineage-reproducibility.md`、Engine 2 の設計 `docs/design/engine2-rebuild.md`、MSG の重なり `docs/research/mist-massspecgym.md`
- Issue: #10、#15、#22、#36、#37、#46、#52、#54、#62、#66、#74、#82、#88、#108
- fold-safe FPNet: メインのチェックアウトの `data/ext/foldsafe/README.md`（リポジトリには入っていない）、`train_fold0.log`（fold0 の hold の top-1 0.676、np の top-1 0.329）、`fpnet_fold0.pt` の checkpoint のキー
- MSG: メインのチェックアウトの `data/ext/msg/MassSpecGym.tsv`・`MassSpecGym1.5.tsv`
- 手順: スキル `kaggle-improvement-loop` の `references/holdout-pipeline-eval.md`、`kaggle-competition-workflow` の `references/casmi26.md`
