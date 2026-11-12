# Engine 1 ランカー `ranker_v4b.pkl` を TC の学習コードで作り直す準備（#88）

対象: 0.487 系統の Engine 1 ランカー `ranker_v4b.pkl`（LightGBM lambdarank 4 本・160 特徴、sha256 6172e926…）。公開の学習コード `ahmedberatozer/casmi26-training-code`（以下 TC。2026-11-13 00:07 JST に全 363 ファイル・2.4 MB を取得）の 3 段で作り直す本番の学習を、次の計画で事前登録して回せるようにするための記録。

- 範囲は**提出用**（全 train 側のデータで作る学習）。v3 の評価には使わない。特徴を作る FPNet A/B と fe_D が v3 の正解構造を学習で見ているため（`lineage-reproducibility.md` 部品表 1・3・4 行目）
- 今回は本番の学習をしていない。Kaggle GPU 0、提出 0
- 印: **確認**＝ファイルを読んだ・sha256 を計算した・実行した。**主張**＝TC README などが書いているだけ。**推定**＝コードと他のログから計算した数字で、実測ではない

## 1. 3 段と起動のしかた（TC README §2.3 と同梱の .sh。確認）

段 0 として、README §4 のとおりに配置する（ROOT = リポジトリの根）。

- `fe_v4/` → `work/analysis/fe_v4/`
- `v1fe_code_casmi/` → `work/analysis/v1fe/code/casmi/`（段 1 が使う v1 エンジン）
- `engine_v6_casmi/` → `work/analysis/engine_v6/code/casmi/`（段 2・3 が `ctxlib` 経由で使う v6 エンジン。`RankerBag` はここの `ranker.py`）
- 入力のデータは `work/` に置く（§2）

段 1 から 3 は次のとおり。

1. **段 1 シミュレーション** `simulate_v1ctx.py`。`launch_sim1x.sh` が 6 シャードを並行に起動する（`--shard s --nshard 6`、`NUMBA/OMP/MKL/OPENBLAS_NUM_THREADS=2`）
   - 既定の引数: `--folds np,nplib,syn,plusk --regimes c1,c2,c3 --seed 21 --part_every 40 --engine_models work/fpnet_a.pt`
   - 出力: `--out work/analysis/v1fe/sim1x/rows` と `--ctx work/analysis/v1fe/sim1x/ctx`
   - 途中から再開できる。40 構造ごとに part を書き、`_done_sids.txt` も残す
2. **段 2 特徴の族** `build_feats.py --fams ids,derivation,frag2,analog_struct,model_views,fragnet --procs 2|3`
   - 環境変数: `FE_SIM=…/rows FE_CTX=…/ctx FE_FEATS=…/feats`
   - `build_sim1x_loop.sh` は、シミュレーションが走っている間は 300 秒おきに追加分を作り、終わったら 70 秒待って `--procs 3` の最終パスを 1 回回す
   - 最終更新から 60 秒未満の ctx ファイルは飛ばす。`--procs` の上限はコードで 3。`FE_FEATS/.slot{0,1}` の 2 枠ロックがある
3. **段 3 学習**（README で「再構成」とされるコマンド）
   - `train_v4_ranker.py --no_lock --fams derivation,frag2,analog_struct,model_views,fragnet --only "$(cat fe_v4/v4b_only.txt)" --rounds 500 --lr 0.03 --seeds 0,1,2,3 --threads 12 --out work/analysis/v1fe/ranker_v4b.pkl`
   - 環境変数は段 2 と同じ `FE_*`
   - 同じ行が `after_sim1x_d2.sh` に v4d 用として残っている
   - `eval_sim1x.sh` は go/no-go の判定（OOF）だけで、ブースターの作成には要らない

### seed・乱数・決定性（確認）

- 段 1: クエリの引き方は構造ごとに `np.random.default_rng([21, sid])` で決まる（シャードの分け方に依らない）。README は、後から編集されたスクリプトでも 15,701 件の引き方が同じになることを `after_sim1x_d2.log` で確かめたと書く（主張。ログには sim1x と sim1x_d2 の 15,701 件の sid・regime・n_cand・has_truth が一致する行がある。確認）。FPNet の logits は CUDA の fp16 autocast（`assert bank.dtype == torch.float16`）なので、GPU と版によっては小さくずれる
- 段 2: 族は決定的で順序に依らない、と ctxlib の docstring にある（主張）。RDKit の版に依る部分がある。derivation の site-prior 表は「unfolded Morgan ids と RDKit reaction の列挙順」に依り、`parity_fixture.pkl` は rdkit 2026.03.3 で作られている（`fams/derivation.py` の docstring。主張）
- 段 3: `RankerBag.fit` は群の中で行を並べ替え（`shuffle_seed=0`）、ブースターごとに `seed=bagging_seed=feature_fraction_seed=s`（s = 0..3）。`deterministic` は指定がなく、`num_threads=12` なので、バイトは一致しない（README §5 も同じことを書く）
- TC の環境（主張、README §4）: python 3.14.5、torch 2.14.0+cu130、lightgbm 4.7.0、rdkit 2026.03.3

### CPU と GPU のどちらが要るか（確認）

- 段 1 は **CUDA の GPU が要る**。`fpnet.ModelBank(em, device='cuda')` と `V1FE(…, device='cuda')` が直書きで、その後に `assert bank.dtype == torch.float16` がある。CPU に切り替える引数は無い。この Mac で TC のモジュールそのままに 139 行目を通すと `AssertionError: Torch not compiled with CUDA enabled`（§3）。docstring が「fp16 autocast forced (Kaggle T4)」と書くので、作者は Kaggle T4 の精度に合わせている
- 段 2・3 は CPU だけで回る（torch を使わない。fragnet は LightGBM の stageB を読み、frag2・fragnet は numba を使う）

## 2. 入力の表（3 段が開くファイルのすべて）

sha256 の出どころ:

- v4b = `casmi26-v4b-models/MANIFEST.json`（64 ファイル。2026-11-13 に取得）
- spec = `casmi26-spec-cache/docs/README.md`
- v4g = `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets` の `MANIFEST.json`（v2-pool の 7 ファイル）

「ローカル一致」は、この Mac にある写しの sha256 を 11/13 に計算し、記録の値と同じだったもの。

### 段 1 `simulate_v1ctx.py`（`Library('work')`・`Pool('work')`・`Ctx`・モデル）

- `work/spec_off.npy`: 配布元 `ahmedberatozer/casmi26-spec-cache` `cache/`。20,317,000 B。sha256 af2be108…（spec）。生成コード有り（TC `scripts/03_build_spec_cache.py` → `casmi/build.py::build_spec_cache`）
- `work/spec_mz.npy`: 同上。532,912,560 B。482332fe…（spec）。生成コード有り（同上）
- `work/spec_it.npy`: 同上。532,912,560 B。9e127f00…（spec）。生成コード有り（同上）
- `work/spec_meta.parquet`: 同上。54,036,239 B。4f968592…（spec）。生成コード有り（同上）
- `work/split.parquet`（TC の分割。fold-safe 版の split とは別物）: 同上。9,641,941 B。d05a2f9f…（spec）。生成コード有り（`scripts/04_make_split.py`、rng 2026）
- `work/train_structs.parquet`: 配布元 `ahmedberatozer/casmi26-v2-pool`。23,306,834 B。f20aa1ca…（v4g、ローカル一致）。生成コード有り（`scripts/01_build_train_structs.py`）
- `work/pool_meta.parquet`: v2-pool。36,692,358 B。82b8c403…（v4g、ローカル一致）。生成コード有り（`02_build_coconut.py`・`05_build_pool.py`）
- `work/pool_fp.npy`: v2-pool。908,986,707 B。f02d91ee…（v4g）。生成コード有り（`05_build_pool.py`）
- `work/fp_bits.npy`: v2-pool。41,032 B。c3869e86…（v4g、ローカル一致）。生成コード有り（`04_make_split.py` の `select_bits`）
- `work/pool_frag_off.npy`: v2-pool。5,685,744 B。354015a6…（v4g）。生成コード有り（`05b_build_frags.py`）。ファイルがあれば読む（`load_frag`）
- `work/pool_frag_mass.npy`: v2-pool。357,116,708 B。9afbdc02…（v4g）。生成コード有り（同上）
- `work/train_fp_sel.npy`: v2-pool。352,761,118 B。0e6ddfbf…（v4g、ローカル一致）。生成コード有り（`scripts/09_predict.py::build_train_fp_packed`。`train_fp_raw.npy` と `fp_bits.npy` から作る）
- `work/fpnet_a.pt`（エンジンのバンク＋見方 A）= `casmi26-v4b-models/fe_A.pt` = v3-models `fpnet_0.pt`: 198,790,165 B。edccf479…（v4b。11/13 に落として一致を確認）。生成コード有り（`scripts/06_train_fpnet.py`、seed 7、ログ `logs/log_train_a.txt`）
- `work/fpnet_b.pt`（見方 B）= v4b `fe_B.pt` = v3-models `fpnet_1.pt`: 198,790,165 B。6f60a22b…（v4b）。生成コード有り（`06_train_fpnet.py`、seed は記録なし）
- `work/analysis/dreams_ft/s2/pilot_d1_step20k_ema.pt`（見方 D）= v4b `fe_D.pt`（v4b MANIFEST の `fe_models[].source`）: 486,278,727 B。8ae5bee2…（v4b）。生成コード有り（`dreams_ft/review_fixes/train_dreams_fp.py` → `extract_step20k_ema.py`。後者は後から書かれ、再実行されていない＝主張）。土台は DreaMS SSL `ssl_model.ckpt`（sha256 4b73da58…、TC README の主張）
- コード: `work/analysis/v1fe/code/casmi/*.py`（17 本）と `fe_v4/{ctxlib,v1engine,hooks_v1,derive6,fragcaps,chemfeat}.py`。TC 同梱。v4b MANIFEST の sha256 と全 23 本が一致（確認。生成物ではない）

### 段 2 `build_feats.py`（`ctxlib.Resources`＋族の `prepare`）

- 段 1 の出力: `FE_SIM` の `rows_<s>.npz`・`rows_<s>_meta.parquet`（無ければ part ファイル）と、`FE_CTX` の `ctx_s<s>_c<NNN>.pkl`。sha256 は記録なし（作る物）。生成コード＝段 1
- `work/spec_{off,mz,it}.npy`・`spec_meta.parquet`（lean 列）・`train_structs.parquet`・`pool_meta.parquet`・`pool_fp.npy`・`fp_bits.npy`・`pool_frag_{off,mass}.npy`・`train_fp_sel.npy`: 段 1 と同じファイル（v6 の `Library`/`Pool` で mmap）
- derivation: `fe_v4/res/derivation/prior_tables.pkl`。v4b `code/fe_v4/res/`。20,738,118 B。21790581…（v4b）。生成コード有り（`class3_v2/priors_build.py build 30000` → `tables` → `fe_v4/res_derivation.py`。`priors.py` は build の後に編集された＝主張）
- frag2: `res/frag2/{pool_cap.npy, caps.txt, pool_f1_off.npy, pool_f1_mass.npy}`。15,635,550 / 149 / 5,685,744 / 62,531,984 B。6be43a73… / 0ebe04b7… / de25d731… / 0fab0a20…（v4b）。生成コード有り（`res_frag2.py --step caps`、`--step f1`）
- analog_struct: `res/analog_struct/{pool,train}_{hash,num,fvec}.npy` と `{pool,train}_{el,core}smi_{buf,off}.npy`。計 14 本、186,994,707 B。sha256 は v4b に全 14 本（例: pool_hash 9835d50a…）。生成コード有り（`res_analog_struct.py --step rec / str`。rec の出力の後でスクリプトが編集された＝主張）
- fragnet: `res/fragnet/{stageA.npz, stageB.txt, rec_ok.npy, rec_aoff.npy, rec_atom.npy, rec_boff.npy, rec_bond.npy}`。268,385 / 544,829 / 710,829 / 5,685,744 / 66,704,036 / 5,685,744 / 71,602,625 B。3a501ecb… / 6916c25c… / efa48d27… / 9aadeb10… / 4b33b9c3… / 0d15c8c4… / ca1caa5d…（v4b）。生成コード有り（`res_fragnet.py --step rec / sets --nspec 32000 / countA / featB / fitA / fitB`。sets と fitA にはログが無く、`rec_bond.npy` はコマンドの記録なしに uint8 で保存し直された＝主張）
- model_views・ids: ファイルは読まない（ctx の z と `P.bits` だけ）
- 読まないもの（確認）: `parity_fixture.pkl`（`self_test()` 専用）、`fragnet/{sets.pkl, countsA.npz, featB.npz}`（評価の関数だけが使う）
- コード: `fe_v4/fams/{ids,derivation,frag2,analog_struct,model_views,fragnet}.py`（`ids.py` 以外は v4b MANIFEST と一致を確認。`ids.py` は v4b に無い診断用の族）、`work/analysis/engine_v6/code/casmi/*.py`（TC にだけあり、sha256 の記録なし）

### 段 3 `train_v4_ranker.py`

- `FE_SIM` の rows・meta（段 1）、`FE_FEATS/<fam>/*.npz`（段 2、ids と 5 族）、`fe_v4/v4b_only.txt`（TC 同梱、66 特徴の名前）。sha256 は記録なし（作る物、または TC 同梱）
- 出力の比べ先: `fe_v4/ranker_v4b.json`（TC 同梱。n_groups 11,086、n_rows 1,204,705、160 特徴の名前と params）、`ranker_v4b.pkl` 6172e926…（v4b）

段 1 から 3 の入力（段の出力を除く）の合計は 4,161,058,302 B（約 4.2 GB）。内訳は spec 1.15 GB、v2-pool 1.68 GB、モデル 3 本 0.88 GB、res 26 本 0.44 GB。

## 3. ローカルの小規模試走（2026-11-13 00:12–00:15 JST。確認）

計画の上限は、ディスク 3 GB 以下と「1 GB を超える入力は落とさない」。spec cache（1.15 GB）・`pool_fp.npy`（0.91 GB）・`fe_D.pt` などは落とさなかった。さらに段 1 は CUDA が要る。このため「入力がそろう場合」に当たらず、3 段とも**通していない**。

確かめたのは、TC のスクリプトをそのまま本物の入力だけで起動したときに、どこで何が足りないかである。ごまかしの入力は作っていない。

- 試走の配置: README §4 のとおりに置いた。`work/` には、ローカルにあって sha256 が記録と一致した 4 ファイル（train_structs・pool_meta・fp_bits・train_fp_sel）だけを symlink した。CUDA の確認用に `fe_A.pt`（199 MB）を落とした（sha256 一致）
- 環境: `.venv` の python 3.12、torch 2.15.0（CUDA なし）、lightgbm 4.7.0、RDKit 2026.03.3（`PYTHONPATH` で先に読ませた）
- 起動のしかた: 包み `scripts/train/mem_guard_run.py` で起動した。`/usr/bin/time -l` で包み、5 秒おきに `memory_pressure` の空き率（10% 未満で止める）・プロセス木の RSS（8 GB で止める）・経過時間（30 分で止める）を見る
- 起動前の確認: 空きは free＋inactive＋speculative で約 17.3 GB。`memory_pressure` の空き率 59%。swap は 2,048 MB 中 1,040 MB 使用。このプロジェクトの重いジョブと、他のプロジェクトの `train*.py` は無かった。他のプロジェクトの動画生成 `sd-cli`（RSS 約 10.5 GB、学習ではない）は動いていたが、触っていない

結果:

- 段 1 `simulate_v1ctx.py --out … --ctx … --shard 0 --nshard 6 --max_structs 20`: 終了コード 1、0.87 s、RSS ピーク 314 MB（time -l）。エラーは `FileNotFoundError: [Errno 2] No such file or directory: 'work/spec_off.npy'`（`Library('work')`、v1 `library.py` 32 行目）。足りない入力は spec cache 5 本（1.15 GB）、`pool_fp.npy`、`pool_frag_{off,mass}.npy`、`fpnet_b.pt`、`fe_D.pt`
- 段 1 の CUDA の確認（`fe_A.pt` だけを置き、TC の `simulate_v1ctx` モジュールを import して 139・140 行目と同じ `fpnet.ModelBank(['work/fpnet_a.pt'], device='cuda')` を呼んだ）: 終了コード 1、0.93 s、RSS ピーク 711 MB。`casmi/fpnet.py` 184 行目 `m.to(device)` で `AssertionError: Torch not compiled with CUDA enabled`。差し替えた `autocast_dtype('cuda')` は torch.float32 を返したので、ここを越えても次の `assert bank.dtype == torch.float16` で止まる
- 段 2 `build_feats.py --fams ids,derivation,frag2,analog_struct,model_views,fragnet --procs 2`（FE_* は sim1x のパス）: 終了コード 1、0.50 s、RSS ピーク 137 MB（親だけ）。「0 ctx files」の後、ワーカーの `ctxlib.Resources()` で `FileNotFoundError: … 'R/work/spec_off.npy'`。足りない入力は段 1 の出力（rows・ctx）、spec cache ほか段 1 と同じもの、`fe_v4/res/` の 26 本（v4b `code/fe_v4/res/`。TC には生成スクリプトだけがあり、ファイルは無い）
- 段 3 `train_v4_ranker.py --no_lock …`（README の再構成コマンドそのまま）: 終了コード 1、0.27 s、RSS ピーク 111 MB。`ctxlib.load_meta_all()` の `pd.concat` で `ValueError: No objects to concatenate`。足りない入力は段 1 の rows・meta と段 2 の feats

ログは scratch の `tc1113/trial/stage{1,2,3}.log`・`probe_cuda.log` と、同じ名前の `.json`（包みの報告）にある。24 時間で消える場所なので、数値はこの文書と #88 に写した。

## 4. 本番（15,701 件）の見積もり（推定。作者のログからの換算で、Kaggle では測っていない）

作者の実測（TC の `v1fe_sim1x/log_*.txt`。確認）:

- 6 シャードを並べて回した。各シャード 891〜892 構造
- 各シャードの経過時間（途中で止まって再開したシャード 0・1・2・5 は、最初の起動分の最後の進み具合の行の経過秒と、再開分の「done」行の秒の和）: 6,317 / 5,983 / 6,156 / 5,226 / 5,552 / 6,865 s。合計 36,099 シャード秒（約 10.0 シャード時間）
- 進み具合の行の 1 クエリあたりの時間は、最初の起動分の最後の行で 2.00〜2.52 s、再開分で 1.69〜3.43 s（スレッド 2）。GPU は RTX 5090 Laptop 1 枚を 6 本で共有（README）
- 再開したシャードでは、最後の進み具合の行から止まるまでの時間が入っていない（少なめになる）。一方、最後の part より後の構造はやり直している（多めになる）。どちらの大きさも記録が無い
- ライブラリの読み込みは 1 s 未満、プールは 0.7 s、エンジンの初期化は 4 s（固定費）

見積もり:

- **段 1（GPU 必須）**
  - Kaggle T4・vCPU 4 なら、スレッド 2 のシャードを 2 本並べる
  - CPU が作者と同じ速さなら 10.0 / 2 = 5.0 h。Kaggle の CPU の遅さを 1.5〜2 倍と見ると 7.5〜10 h
  - 1 セッション 12 h に収まる見込みだが、余裕は小さい。part ごとに再開できるので、12 h で切れたら、出力を次のカーネルの `kernel_sources` に入れて続きを回せる
  - GPU の消費は 7.5〜10 h（週 30 h の 1/4〜1/3）
  - GPU のメモリ: 1 プロセスがモデル 4 本（A がバンクと見方の 2 回、B、D）を持つので、重みだけで約 1.1 GB。2 本で 3 GB 弱になり、T4 の 15 GB に収まる
  - RAM: v1 の `Library` は spec を mmap せずに全部読む（1.07 GB）。ほかに pool_fp 0.91 GB、train_fp_sel 0.35 GB、frag 0.36 GB、meta 類があり、1 プロセス 3〜4 GB。2 本で 7〜8 GB、Kaggle のセッションの RAM（約 29〜30 GB）に収まる
- **段 2（CPU）**
  - TC の別の ctx 集合での実測（`build_*.log` の ms/q。確認）: fragnet 223、derivation 101、analog_struct 66、frag2 29〜31 ms/q。model_views と ids はログなし（軽い）。合わせて約 0.42 s/q/プロセス
  - 15,701 件で 1.8 プロセス時間。`--procs 3` なら作者の機械で約 40 分、Kaggle で 1.0〜1.5 h
  - RAM: 1 ワーカーあたり BaseRows の X（全 15,701 群で約 1.7 M 行 × 94 × 4 B ≈ 0.64 GB）と mmap。3 ワーカーで 6〜9 GB
- **段 3（CPU）**
  - 作者の実測は、同じコマンドの v4d で 193 s（スレッド 12、`after_sim1x_d2.log`。確認）。vCPU 4 では 10〜15 分
  - RAM: X 1,204,705 × 160 × 4 B = 0.77 GB、BaseRows 0.64 GB、LightGBM の Dataset で、ピーク 3〜4 GB
  - 段 1・2 の出力をこの Mac に持ってくれば、ローカルで回せる（ピーク 8 GB の上限内の見込み）
- **ディスク**
  - 入力 4.2 GB
  - 段 1 の rows: 約 1.7 M 行 × 94 × 4 B ≈ 0.64 GB（npz 圧縮前）
  - 段 1 の ctx: 1 クエリに z_s・z_m（モデル 3 本 × (スペクトル数の平均 3.6 + 極性ごとの合成 1) × 10,226 ビット × float32）≈ 0.57 MB。15,701 件で**約 9 GB**（pickle、圧縮なし）
  - 段 2 の feats: 族の全列（約 100〜150 列）× 1.7 M 行 × 4 B ≈ 0.7〜1 GB
  - 合計 約 15 GB。Kaggle の `/kaggle/working`（20 GB）に収まる見込み。ローカルでは計画の上限（3 GB）を越えるので、段 1・2 は Kaggle で回す
- **全体**: GPU カーネル 1 本（段 1、7.5〜10 h）の後に CPU カーネル 1 本（段 2＋3、1.5〜2 h）で回す。段 1 と同じ GPU カーネルの中で段 2・3 も続けると GPU 時間がさらに 2 h かかるので、CPU カーネルに分ける

## 5. 事前登録の下書き（実行の約束ではない。次の計画でレビューを通してから採る）

- レシピ: TC の既定のまま。段 1 は `launch_sim1x.sh` の引数（seed 21、folds np,nplib,syn,plusk、regimes c1,c2,c3、nshard 6、エンジンのバンク `work/fpnet_a.pt`）。段 2 は `build_sim1x_loop.sh` の族 6 つ（最終パス `--procs 3`）。段 3 は README §2.3 の再構成コマンド（`--rounds 500 --lr 0.03 --seeds 0,1,2,3 --threads 12`、`--only v4b_only.txt`）
- 変えるのは次の 3 つだけ
  - パス: `/c/Users/omen/Desktop/envedaCasmi` を Kaggle の作業場所に置き換え、README §4 の配置にする
  - 起動の包み: `nohup … &` と `powershell` の待ちを、Python の `subprocess` の待ちに置き換える
  - 6 シャードを vCPU 4 に合わせて 2 本ずつ順に回す（シャードの分け方は引き方に影響しない）
- 環境: rdkit==2026.03.3 と lightgbm==4.7.0 を入れる（学習用のカーネルなのでインターネット可）。最初に `fams/derivation.self_test()`（parity_fixture、0 mismatches が条件）を通し、`torch`・`lscpu`・GPU 名を記録する
- 入力の固定: §2 の sha256 を、起動前にすべて照合する。1 つでも外れたら止める（v4b・spec・v4g の記録の値）
- 受け入れの記録（判定の数値は先に固定し、学習の設定は動かさない）
  - (a) 段 1 のクエリ 15,701 件・truth あり 11,086 群・行数 1,204,705 が `ranker_v4b.json` と同じか（同じと予想。違えば差を記録する。止めはしない）
  - (b) 160 特徴の名前と並びが `ranker_v4b.json` と同じ（**必須**）
  - (c) 作り直したブースターと配布の `ranker_v4b.pkl` の予測を、作り直した特徴の表の上で比べる（群ごとの Spearman の中央値、1 位の一致率）。ここは記録するだけで、合否の線は引かない
  - 呼び方は「自前のコードで作り直した」とし、「再現した」とは書かない
- v3 には使わない（特徴の FPNet A/B と fe_D が v3 の正解構造を見ているため）
- 提出への差し替え: 別の Issue で、診断提出として計画する（0.487 写しの `ranker_v4b.pkl` だけを差し替える。読み方の帯は公開分の SE ≈ 0.035 から先に固定する）。この下書きでは決めない
- 資源: GPU は段 1 の 1 本だけ（7.5〜10 h。push の前に週の残りを確かめる）。段 2・3 は CPU カーネル。push の前に、生成したノートの埋め込みコードを `compile()` で確かめる

## 6. 未確認・判断待ち

- 段 1 の Kaggle での実際の速さ（§4 は作者のログからの換算）。最初に `--max_structs` を小さくした GPU の試走を 1 本回して s/q を測るかどうかは、GPU の使い方の判断になる（今日は GPU 0）
- ctx の大きさ（約 9 GB はコードの配列の形からの計算）
- 11-05 の TC への質問（学習の後で編集されたスクリプト、seed の欠落）への主催者の返答（`lineage-reproducibility.md` §3）
- fe_D の土台の DreaMS SSL と NIST20+MoNA の扱い（748882 で質問中、部品表 4 行目）

## 7. 縮小版が変わらないことの確認（2026-11-13 00:19–00:21 JST。確認）

この変更は包み（`scripts/train/mem_guard_run.py`）とテストと文書を足すだけで、pipeline のコードには触れていない。念のため、ブランチの上で `scripts/local_eval/run.py --holdout <h> --stages score,prior,frag` を 4 集合で回した。実行前に写しておいた `submission_score+prior+frag.csv` と比べた結果は次のとおり。

- v3_c1（C1）: `cmp` 同一、mrr25 0.8535
- v3_c2（C2）: `cmp` 同一、mrr25 0.0117
- enveda_v1: `cmp` 同一、mrr25 0.8376
- v3_c2n: `cmp` 同一、mrr25 0.6284
