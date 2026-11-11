# Engine 2 の 2 表を自前の生成コードで作り直す設計（#64）

この文書は設計と事前登録だけです。表の生成・Kaggle 実行・提出はしていません（別 Issue、設計レビューとオーナー許可の後）。

## 0. なぜ作り直すか
- 主催者回答 756893（2026-11-11 00:16 UTC）: 再現できない中間物を読み込む解は再現可能ではない
- 0.487 写し（提出 58208946、LB 0.487）と 0.479 版はどちらも Engine 2（`casmi_engine.py`）で 2 表を読み込み、HistGradientBoosting をその場で学習する
  - `rank_train.npz`（prvsiyan/casmi26-ranker-features V6）→ `fit_pv_rankers`（prvsiyan のランカー、`base` 31 列、FP あり）
  - `sim_rank_rows_nofp.npz`（megayak/casmi26-simulated-ranker-rows）→ `fit_rankers`（「ours」、`base+libx+anapol+fragx` 51 列、FP なし）
- Engine 2 を外すと 0.479 → 0.378（#62、提出 58215746）。外す道は失点が大きい

## 1. 前提の確認（手順 1b・1c）

### 1b-(a) v80 の出力は Engine 2 に入るか → 入らない
- Engine 2 は凍結コードのセル 32 で `eng_runner.py` をサブプロセスとして起動する。引数は `test.parquet`・`train.parquet`・`sample_submission.csv`・出力先の 4 つだけ
- `eng_runner.py` が読む資産: `sim_rank_rows_nofp.npz`、`rank_train.npz`、`casmi26-fp-models-v2` の `fp_*.pt`、COCONUT（`coco_fp.npy` ほか）、ChEBI/LIPID MAPS、PubChemLite CC0、`fp_bits.npy`（6,930 ビット）
- v80（popwenman/casmi26-scorer-train-v80 の `final_model.pt`）は凍結の外のセル（cell 12 の `MODEL_PINS` / `find_bundle`）で読まれる。凍結コードのセルのどれにも `final_model.pt` は出てこない（grep で 0 件）
- 流れは逆向き: Engine 2 の出力 `ENG` が後段（凍結セル 41・45・48–56）の候補に入り、そこを v80 を含むスコアラーが採点する
- 確かめていないこと: `casmi_engine.find`（41 行）は `/kaggle/input` 以下を再帰で探して最短のパスを取る。v80 のカーネル出力に同じ名前のファイル（`rank_train.npz`、`coco_fp.npy` など）があれば Engine 2 が拾う可能性は残る。言えるのは「v80 のモデル（`final_model.pt`）は Engine 2 に入らない」まで

### 1b-(b) prvsiyan 本人のノートブック
- `kaggle kernels pull prvsiyan/analog-propagation-casmi-2026-baseline -m`（11/11 12:09 に取得）。2026-11-10 版は 1 つ目の markdown に「alex chilton の 0.487 のコードそのもの」と書かれていて、凍結コード（セル 10 の `FROZEN_B64`）の sha256 は d1e757c2…（0.487 写しと同じ）。dataset_sources・kernel_sources も 0.487 写しと同じ 10 件 + v80
- このため、本人のノートでも `rank_train.npz` の読み込みと特徴関数の場所は 0.487 写しと同じ（`casmi_engine.fit_pv_rankers`、特徴は `pv.rank_features`）。2026-09-16 版（`pv.py` の見出しにある元の版）は `kernels pull` で取れない（最新版だけ）ので、元の版での使い方は**未確認**
- 表の作り方の一次資料は prvsiyan/casmi26-generation-code の README（`make_ranker3.py --model`、欠けは §2 の表）

### 1c 復号コードの出どころ
- 0.487 写し（taichiiiii/casmi26-lb487-copy、scratch の `k487/…ipynb`、ノート全体の sha256 380e2c82…）のセル 12 の `FROZEN_B64` を復号し直した: sha256 d1e757c2…、宣言値と一致、59 セル
- 0.479 版（`kernels/lb479/notebook.ipynb` セル 10）の凍結コードと 59 セルすべて一致（セルごとの sha256 で照合）
- scratch の `frozen/c*.py` は 0.487 写しの復号ではなく、#62 の診断ノート（`kernels/casmi26_lb479_noeng2`）の凍結コードを書き出したもの（`scratch/dec.py` が `fb` = noeng2 側を書き出している）。0.487 写しの復号と比べると 59 セル中 57 セルが一致、c32（Engine 2）と c50 が違う（#62 の変更そのもの）
- 復号し直した 0.487 写しの凍結コードを正とし、`workspace/casmi_daily/ref/frozen_487/`（c00–c58.py と `eng/` に `ENG_FILES` の 3 ファイル）に置いた。`eng/casmi_engine.py` の sha256 は e461349b…で、ローカルの写し `outputs/holdout_eval/np_v1/eng/casmi_engine.py` と一致
- 「`rank_train.npz` が 6 回、`sim_rank_rows_nofp` が 5 回出る」（日報 調査 1）は数え方が違う。凍結コードのセルの文字列としては `rank_train.npz` 2 回・`sim_rank_rows_nofp` 1 回（c32 の中、`ENG_FILES` の文字列を含む）。依存していることは変わらない

## 2. 2 表の中身（`outputs/engine2_ref/schema.json`、`scripts/research/engine2_table_schema.py`）

ファイルは `kaggle datasets download` で `data/ext/engine2_ref/` に取得（読むだけ）。

- `rank_train.npz`（9,553,020 バイト、sha256 41ccec87…＝作者 README の V6 の値と一致）
  - X (142,762, 31) float32、Y/M (142,762,) float64、G int64。部品表の既載値 142,762×31 と**一致**
  - 正例率 0.011474、グループ 819。どのグループも M=0 と M=1 の両方に出る（同じ候補行、Y・G は同一）。正例はグループあたり 2（M ごとに 1）
  - M=0 は class-1 模擬（正解の参照が残る: 正解の lib > 0 が 68.1%）、M=1 は class-2 模擬（正解の lib > 0 が 0%）
  - FP の 6 列（15–20）は M=0 と M=1 で同じ値。正解が FP 列で 1 位になる割合（M=0 の正解行で、列 16 fp_raw_rank が 0。同値は argsort の順で切る）は 40.2%。同値も 1 位に数える（列 20 fp_raw_is_max = 1）と 42.0%（FP ロジットはクエリのスペクトルだけから計算されるので、M=0 と M=1 で同じ）
- `sim_rank_rows_nofp.npz`（26,872,207 バイト、sha256 c19ae76f…。megayak/casmi26-simulated-ranker-rows から取得。作者の公開ハッシュは見つけていないので照合なし。0.487 写しが実際に読むのは v4g の写しで（部品表 9 行）、その写しと同じバイトかは**未確認**）
  - X (399,790, 51) float32、Y/S/Q int8、G int32、`blocks` = base, libx, anapol, fragx、`sets` = np, gnps, riken, mona, msdial, massbank, pluskal_ms2。部品表の既載値 399,790×51 と**一致**
  - 正例率 0.011256、グループ 4,500 = 2,250 分子 × S（偶数 G が S=0、奇数 G が S=1）。各グループに正例 1
  - 集合ごとの分子数（グループ数 / 2）: np 250、gnps 400、riken 300、mona 300、msdial 300、massbank 300、pluskal_ms2 400（megayak README の値と一致）
  - S=0 は正解の lib > 0 が 86.0%、S=1 は 0%。全 0 の列は 15–20 と 25–30 の 12 列（FP 列、`nofp`）
  - 候補数（`log_nc` 列から、S=0 のグループごと）: 平均 88.8、中央値 53（行で平均すると 212）。`rank_train` は平均 87.2、中央値 52

## 3. 列の対応（推論側のコードとの照合）

- 推論側の列: `casmi_engine.BLOCKS`（341 行）の各関数を合成の 1 分子で呼んで数えた幅は base 31・mass 3・libx 8・anapol 5・fragx 7。`sim_rank_rows` の `blocks` の順の合計は 31+8+5+7 = **51 で X の列数と一致**。`rank_train` の 31 は base と一致
- `zlog=None`（Engine 2 の「ours」は `use_fp=False`）で base を呼ぶと全 0 になる列は 15–20, 25–30 で、`sim_rank_rows` の全 0 の列と一致
- 列名と区分は `engine2_table_schema.py` の `BLOCK_COLS`（`pv.rank_features` 210–255 行、`casmi_engine.py` 299–341 行の順）。区分は推論コードを読んで付けたもので、データから導いたものではない

区分の集計（own = 自前で作れる、fp = FP ロジットが要る、unknown = 定義不明）:
- `rank_train`（31 列）: own 19、fp 12、unknown 0
- `sim_rank_rows`（51 列）: own 39、fp 12、unknown 0。ただし fp の 12 列は表でも推論でも常に 0 なので、作り直しに FP ロジットは要らない

恒等式による定義の確認（グループ × M/S ごとに表の値から再計算。成り立つグループの割合。`schema.json` の `identities`）:
- `rank_train`: **22 列が全グループ（1,638 = 819 分子 × M）で成立（1.0）**
  - 1 列から: lib_rank / lib_max / lib_minus_max / lib_pos / ap_rank / ap_max / ap_minus_max / log_nc / fp_raw_rank / fp_raw_minus_max / fp_nrm_rank / frag_rank / frag_minus_max / frag_z
  - グループ内で一定であることだけ: top_sim（値そのものはアナログの最上位の類似度で、表からは出せない）
  - 複数の列から（`pv.py` 234–243 行、mr = fp_raw_rank）: fp_raw_is_max = (fp_raw_minus_max == 0)、x_lib_mr = lib·(1 − mr)、x_ap_mr = ap·(1 − mr)、x_agree_lib = 1 − mr[argmax lib]、x_agree_ap = 1 − mr[argmax ap]、x_agree_lib_lvmax = x_agree_lib·lib_max、x_corr_lib_mr = corr(lib, −mr)
- `sim_rank_rows`: **24 列が全グループ（4,500）で成立（1.0）**: 上の FP 以外の 1 列の式 10（lib 系 4、ap 系 3、log_nc、frag_rank、frag_minus_max）と frag_z、一定の 2（top_sim、anapol_top_w）、libx の 5（lib_mean_rank / lib_mean_minus_max / libsh_rank / libsh_minus_max / log_cnt_minus_max）、anapol の 2（anapol_ap_rank / anapol_ap_minus_max）、fragx の 4（frag_ad_rank / frag_ad_minus_max / frag_mean_rank / frag_mean_minus_max）。FP の 10 個の式は列が全 0 なので対象外
- `log_nc` が全グループで log(グループの行数) と一致 → 表の行は `rank_features` に渡された候補そのもの（特徴を計算する前に間引いたかどうかは、この式では分からない）
- 式を壊すと検出できること: 単体テスト（`test_identity_detects_break`）。1 回目のレビューで列を行の間で入れ替えると成立の割合が 0.0002–0.0122 に落ちることも確認済み

式で確かめていない「元の値」の列（定義は推論コードで決まるが、表を作ったときの条件は表から分からない。`schema.json` の `unchecked_cols`）:
- `rank_train`（9 列）: 0 lib、5 ap、9 a1、10 best_tan、11 top_tan、12 mean_tan、15 fp_raw_z、18 fp_nrm_z、21 frag
- `sim_rank_rows`（15 列）: 0 lib、5 ap、9 a1、10 best_tan、11 top_tan、12 mean_tan、21 frag、31 lib_mean、34 libsh、37 log_cnt、39 anapol_ap、42 anapol_ap6、44 frag_ad、47 frag_mean、50 frag_pol
- これらの条件で分かっている違い:
  - アナログ数: 表を作った側は 80（作者 README「reference cache has 80」、`pv.N_ANALOG = 80`）。推論は `eng_runner.py` で `E.CFG.N_ANALOG = 200`。ただし `pv.rank_features` は渡されたアナログを全部使い、`top_analogs` が `CFG.N_ANALOG` で切る
  - 候補プール: megayak 版は COCONUT + 学習構造（README）。推論は ChEBI/LIPID MAPS と PubChemLite CC0 も足す
  - FP ロジット: `rank_train` は `val_model_logits_combo.npy` / `hard_model_logits.npy`（作者 README、どのモデルのどの分割かは未解決と作者自身が書く）。推論は `casmi26-fp-models-v2` の 2 本

作者版生成コードの欠け（prvsiyan/casmi26-generation-code、`source-files/assets/casmi26-ranker-features/README.md`）:
1. `c2_np_recs2.pkl` の正確な生成コードが見つからない（近い `run_c2_base.py` は 250 件で 10 項目中 9 一致、衝突エネルギーが 0、`idxs` が無い）
2. `feat_c2.pkl` の正確な生成コードが見つからない（`gen_feats.py` はアナログ 120、元のキャッシュは 80）
3. `hard_feats.pkl` は public rebuild に含まれない
4. FP ロジットのキャッシュ 2 つ（`val_model_logits_combo.npy`、`hard_model_logits.npy`）は作り直せない（A/B のロジットの系統が未解決）
5. chain-3 の public rebuild は再構成で、少数の照合だけ。V6 の完全な再現ではない

## 4. 生成手順の設計

### 4.1 方針: 1 つの模擬で 2 表を同時に作る
- 2 表とも「1 分子 × {class-1 模擬, class-2 模擬} × 候補」の行で、行の形は同じ（§2）。ただし元の 2 表が同じ模擬から作られたわけではない: `rank_train` は作者 README によると `c2_np_recs2.pkl`・`feat2_c1.pkl`・`feat_c2.pkl`・`hard_feats.pkl`（NP 型のレコードと「hard」クエリ、計 819 分子、集合の内訳は**未確認**）、`sim_rank_rows` は 7 集合 2,250 分子
- **元からの違い（事前に記録する）**: 自前の `own_rank_train.npz` は §4.2 の 2,000 分子から作る。prvsiyan ランカーの学習データのクエリの分布は元と変わる（元の内訳が分からないので合わせられない）
- 生成コードは 1 本（`scripts/engine2/build_tables.py`、未実装）。各分子・各模擬で `casmi_engine` の推論コード（`molecule_channels`・`compute_frag`・`BLOCKS`）を呼ぶ。列の定義が推論とずれる余地を無くすため。例外は prvsiyan ランカーの base で、学習と推論の両方を §4.4 の `blk_base_fs` に替える
- 出力: `own_sim_rank_rows.npz`（キー・dtype・`blocks`・`sets` は megayak 版と同じ形）と `own_rank_train.npz`（X/Y/M/G、M は 0 = class-1、1 = class-2）

### 4.2 模擬クエリ
- 集合: megayak 版の 7 集合から `np`（enveda-np-examples、split の `np` と同じ NP パネル）を除いた 6 集合 gnps 400・riken 300・mona 300・msdial 300・massbank 300・pluskal_ms2 400 = **2,000 分子**（`ingest_lib` 列で選ぶ）。enveda-180 は元の設計に無いので入れない（入れるのは別の構造変更として別 Issue）
- 抽出: 集合ごとに分子（inchikey14）を seed 0 で無作為に選ぶ。クエリのスペクトルはその集合のその分子のスペクトル全部
- class-1 模擬（S=0 / M=0）: ライブラリからクエリのスペクトル（その集合のその分子の行）だけを除く。他の集合の同じ分子の参照は残す
- class-2 模擬（S=1 / M=1）: ライブラリから正解構造のスペクトルを全部除く。構造はプールに残す
- 候補: 推論と同じ `build_pool`（COCONUT + ChEBI/LIPID MAPS + PubChemLite CC0 + 学習構造）、±10 ppm（0 件なら 30 ppm）。正解が窓に無い分子は捨て、捨てた数を記録する
- アナログ数は推論の 200 に合わせる（`eng_runner.py` と同じ `E.CFG.N_ANALOG = 200`）。元の表（80）との違いは推論と揃えるためで、値を比べて選んだものではない。これ以外の定数は推論コードのまま変えない

### 4.3 リーク除外（生成コードで assert）
- クエリから除く構造: v3 の 4 集合（`data/holdouts/{v3_c1,v3_c2,enveda_v1,v3_c2n}/drop_keys.txt`）の正解、`data/ext/foldsafe/split.parquet` の hold（55,102 行）と np（265 行）、identity-disjoint folds の `fold_identity == 0` の構造。hold の inchikey14 は 55,102 件すべて `fold_identity == 0` に入り、`fold_identity == 0` で hold に無いのは 59 件だけ（inchikey14 どうしで比較。2 回目のレビューによると 59 件はすべて split の np 行で、`make_split.py` が NP パネルの identity group を hold より先に np に入れるため）。fold-safe の 5 分割と揃えるため、inchikey14 と split の `key` の両方で除く
- 生成コードは「クエリの inchikey14 と上の除外集合の積が空」を assert し、件数を出力の json に書く
- v3 の正解が負例の候補として窓に入ることは許す（負例のラベルしか持たない）。その行数を記録する
- この表で学習したランカーを v3 で測ることはしない（§5）。除外は将来の測定と、hold を使う他の部品との混線を避けるため

### 4.4 FP ロジット（`own_rank_train.npz` の 12 列だけに要る）

推論の FP 列は `cfp @ zlog`（候補の 6,930 ビット指紋 × FP モデルのロジット）。学習行のクエリには、その構造を見ていないモデルのロジットが要る。

- 案 A（推奨）fold-safe FPNet の 5 分割 OOF: `dmitriigluzdov/casmi26-fold-safe-fpnet` の `fpnet_fold{k}.pt` は fold k（`fold_identity == k`）と NP パネルを除いて学習。クエリは fold 1–4 だけから取り、fold k のクエリには model k のロジットを使う（`make_split.py` の規則を k で再現し、クエリが model k の除外集合に入ることを assert）
  - リーク: 無し（どのクエリもそのロジットのモデルに見られていない。v3 は fold 0 由来でクエリから除外）
  - 違い: 出力は 10,226 ビット（v2-pool の `fp_bits.npy`）で、Engine 2 の 6,930 ビットと別の空間。`pv.rank_features` は 1 つの候補指紋 `cfp` を FP 列（15–20、25–30）とアナログの Tanimoto 列（5–13）の両方に使うので、そのままでは差し込めない
  - 推論コードの変更（事前に決める）: 自前のモジュールに `blk_base_fs(r, cfp, cfp_fs, zlog_fs)` を作る。FP 以外の 19 列は `pv.rank_features(cfp, …, model_logits=None)` の出力をそのまま使い（6,930 ビットのまま）、FP の 12 列だけを同じ式（`pv.py` 227–243 行の写し）で `cfp_fs`（候補の 10,226 ビット指紋）と `zlog_fs` から計算して差し替える。Engine 2 の prvsiyan ランカーの学習（自前表）と推論（`score_molecules` の `["base"]`）の両方でこの関数を使う。単体テスト: `cfp_fs = cfp`、`zlog_fs = zlog` のとき `pv.rank_features` の出力と完全一致
  - 推論の `zlog_fs` は fold-safe 5 本のロジットの平均（隠れテストはどの 5 本にも見られていない）。学習行は単独モデル、推論は 5 本平均なので、ロジットの鋭さが違う（リスクとして記録）。`casmi26-fp-models-v2` への依存は外れる
  - 費用: ロジットは CPU で計算（`scripts/cv/local_fold0_logits.py` と同じ読み込み、2,000 分子で数分の見込み、未測定）。候補の 10,226 ビット指紋は v2-pool の作り方（TC の `casmi/structs.py`）で計算
  - 再現性: 学習コード・分割・SHA256 が公開（部品表 17 行）
- 案 B prvsiyan の FPNet（6,930 出力）を 5 分割で学習し直す（作者版の `train_fp3.py` / `train_fp4.py`）
  - リーク: 無し。違い: 推論の空間は今のまま
  - 費用: GPU が要る（1 本あたり T4 で数時間、5 本＋merged 版）。今週の GPU 枠では重い
- 案 C prvsiyan ランカーも FP なしにする（`own_rank_train.npz` の FP 12 列を 0、推論も `use_fp=False`）
  - リーク: 無し。費用: 0。違い: Engine 2 の中から FP の証拠が消える（パイプラインの他の段の FP は残る）
- 案 D `casmi26-fp-models-v2`（推論で使っている 2 本）のロジットをそのまま学習行に使う
  - リーク: 有り（学習範囲が全 train に近いと見られ、学習行のクエリ構造を見ている可能性が高い。部品表 10 行、未確認）。FP 列が学習行で実際より強く見え、ランカーが FP を過信する。モデル自体も部分的にしか再現できない → **採らない**

事前登録: **案 A を使う**。案 A の単体テスト（§4.5 の 1）が通らない場合は、案 C に切り替えるのではなく、生成と提出を止めて Issue に記録する（結果を見て案を選び直さない）。

### 4.5 実装時の受け入れ条件（別 Issue で使う）
1. 列の一致: 生成コードで作った 1 分子分の行が、同じ入力で `casmi_engine.BLOCKS`（base は §4.4 の `blk_base_fs`）を直接呼んだ結果と完全一致（単体テスト）。恒等式チェック（`engine2_table_schema.py`）が自前表でも全式 1.0
2. 形: `own_sim_rank_rows.npz` は 51 列・`blocks` が同じ順、`own_rank_train.npz` は 31 列。グループ数・正例数（グループあたり 1）を記録。分子数は 2,000 から窓外で捨てた数を引いたもの
3. リーク: §4.3 と §4.4 の assert が通り、除外の件数が json にある
4. 縮小版の対照は変わらない: `scripts/local_eval/run.py --holdout v3_c1|v3_c2|enveda_v1|v3_c2n` の結果ファイルが変更前と `cmp` で同一
5. 生成は 1 回だけ。表の統計（正例率・候補数）を元の表と並べて記録するが、それを見て生成条件を変えない

### 4.6 費用の見積もり（未測定の部分は実装時に 50 分子で時間を測ってから外挿する）
- ローカル CPU: `load_library` は全 train のピークを float32 で持つ（コードのコメントで約 3.5 GB）。分子ごとのライブラリ検索・アナログ検索・MetFrag-lite（候補はグループあたり平均約 89 × 2,000 分子 × 2 模擬 ≈ 36 万行。MetFrag-lite は候補の重複を除いて 1 回）。時間は未測定。メモリのピークは 10 GB 以下を見込む（train の列を row group ごとに読む実装のまま）
- 候補プールの資産（COCONUT・ChEBI/LMSD・PubChemLite の `*_fp.npy`）はローカルに無いものがある（`data/ext/v4g/casmi-chebi-lmsd-structures` はある、`coco_fp.npy` は無い）。ダウンロードの大きさは実装時に `kaggle datasets files` で確かめ、1 GB を超えるなら Kaggle の CPU カーネルで生成する（GPU 枠は使わない）
- Kaggle GPU: 生成には不要。診断提出の実行に約 2.7 時間（0.487 写しの実行時間の目安）、提出 1 回
- private dataset: 2 表で約 40 MB（残り約 0.9 GB に収まる）

## 5. v3 で何が測れるか
- Engine 2 の後段（ICEBERG/GLACIER、consensus）は縮小版（`scripts/local_eval`）に無いので、自前表に替えても v3_c1 / v3_c2 の数値は直接は動かない。v3 で確かめるのは §4.5 の 4（縮小版の対照が変わらない）だけ
- 将来、自前表で学習した Engine 2 型ランカーを縮小版の新しい段として測るなら、(答えの出所 × 参照の有無) の 4 集合（v3_c1、v3_c2、enveda_v1、v3_c2n）で #48 と同じ採用規則にする: 4 集合すべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、v3_c1 と v3_c2 の両方を含む、1 つ以上で下限 > 0。1 集合でも欠けたら採用しない

## 6. LB の読み方（事前登録。提出はオーナー許可の後）
- 比較相手: 0.487 写し（提出 58208946、Public 0.487）。変更は「Engine 2 の 2 表と prvsiyan ランカーの FP ロジット源を自前の生成に替える」1 つ
- SE は 0.04 とする（#3 の記録: Public は約 125〜130 分子〈lb479 の cell 20 が引く推定〉、逆数順位の標準偏差約 0.45 で約 0.04。#37 の 160 分子の仮定では約 0.035。大きい方を使う）。Δ = LB(新) − 0.487
  - Δ ≥ −0.04（1 SE 以内）: 同等とみなす。自前表の版を「再現できる 0.487 系統」として最終候補に置ける
  - Δ ≤ −0.08（2 SE 以上の低下）: 自前表では Engine 2 の効きを取り戻せていない。最終 2 本の組み方はオーナー判断に回す
  - その間（−0.08 < Δ < −0.04）: 判定なし。追加の提出はせず、表の条件も変えない
- 結果を見て生成条件・定数・案を変えない（ハイパラ調整の禁止）

## 7. 出典
- 主催者回答 756893・757307・757309: `kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/<id>`
- prvsiyan/casmi26-generation-code（Apache-2.0、コードのみ）: https://www.kaggle.com/datasets/prvsiyan/casmi26-generation-code
- prvsiyan/casmi26-ranker-features: https://www.kaggle.com/datasets/prvsiyan/casmi26-ranker-features
- megayak/casmi26-simulated-ranker-rows: https://www.kaggle.com/datasets/megayak/casmi26-simulated-ranker-rows
- dmitriigluzdov/casmi26-fold-safe-fpnet: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-fold-safe-fpnet
- prvsiyan/analog-propagation-casmi-2026-baseline（2026-11-10 版）: https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline
- 部品表: `docs/research/lineage-reproducibility.md`（8–10 行、17 行）
