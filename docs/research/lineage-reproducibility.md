# 0.479 系統ノートブックの部品ごとの再現性（生成コード・ライセンス・置き換え・リーク無し検証）: 調査メモ（2026-11-09）

関連: #44、親 #3、#24

この日は**測っていない**。読んだのはノートブック・データセットのカード／ファイル一覧／小さいファイル・ディスカッションだけ。
「確認」はこちらで実物（コード・メタデータ・ハッシュ）を読んだもの、「主張」はカードや投稿に書かれているだけでこちらで再現していないもの。

## 出典（すべて 2026-11-09 に読んだ）
1. 0.479 版ノートブック `kernels/lb479/notebook.ipynb`（24 セル）と `kernels/lb479/kernel-metadata.json`（dataset_sources 8 件）。
2. 0.464/0.465 版 `~/.hermes/profiles/massspec/workspace/casmi_daily/ref/frozen465.ipynb`（lb479 cell 10 の `FROZEN_B64` を復号したもの、59 セル。以下「F セル n」）と Engine 2 のコード `ref/eng_casmi_engine.py`（F セル 32 の `ENG_FILES['casmi_engine.py']` と同一であることを文字列比較で確認）。
3. データセットのメタデータ（licenses）: `kaggle datasets metadata <owner/slug> -p ~/.hermes/profiles/massspec/cache/scratch/lineage_meta/<owner_slug>`（24 件。下の表の配布元すべて＋ `ahmedberatozer/casmi26-training-code`）。
4. ファイル一覧: `kaggle datasets files <owner/slug> --page-size 100`（必要なものは `--page-token` で続きも）。
5. 小さいファイルの中身: `kaggle datasets download <owner/slug> -f <file>`（scratch の `lineage_meta/dl/` に保存）。読んだもの: v4b の `MANIFEST.json`・`LICENSE-NOTICE.txt`・`code/casmi/{ranker,traindata,build,structs,chem}.py`、ICEBERG/GLACIER の `README.md`・`LICENSE-NOTICE.txt`、megayak の `README.md`、sanpier の `README.txt`・`config.json`・`data_audit.json`・`training_result.json`、v4g の `ATTRIBUTION.md`・`SOURCE-LICENSES.md`・`MANIFEST.json`、training-code の `README.md`・`scripts/{01_build_train_structs,02c_build_pubchem,04_make_split,05_build_pool}.py`。
6. ローカルの公開データ: `data/ext/foldsafe/{README.md,MODEL_INPUTS.json,make_split.py,train_fivefold.py,SHA256SUMS_FIVEFOLD.txt}`、`data/ext/v4g/casmi26-glacier/README.md`、`data/ext/frag/frag_rescore.py`、`data/ext/v4g/casmi26-v2-pool/fp_bits.npy`（形・dtype・SHA256 だけ）。
7. Kaggle ディスカッション（`kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/<id> --format json`）:
   - 756893「Are shipped ranker tables without a public generator reproducible?」(Shishir Adhikari, 2026-11-08)。**本文もコメントも取得結果に無く、タイトルだけ**（commentCount 0）。
   - 754610「Clarification on permitted external resources for final submissions」: 主催者 David Healey（11-05）「最終提出は競技データと公開文書化された資源から端から端まで再現可能でなければならない」「規則違反の解を失格・除去する権利を留保」。wata（11-08）が `casmi26-v2-pool/fp_bits.npy` の選択コードが無いと質問（返答はまだ無い）。
   - 754292（prvsiyan/ranker-features, megayak/simulated-ranker-rows について）: David Healey（11-05）「再現できない中間物やモデルを使う解は違反」。
   - 755373「low GM participation and enforcement of the NIST restriction」: コメント 5 件はどれも ICEBERG/GLACIER の重みの由来に触れていない。**重みの由来はこのトピックからは分からない**（下の 11・12 の出典で確認した）。
   - 748882（再現性の質問まとめ）: Marie Killian（10-13）「承認済みの MassSpecGym・GLACIER/ICEBERG・DreaMS のチェックポイントは、公開チェックポイント＋著者の学習コードと許可された学習データへの参照＋自前の微調整・推論コードで足りる」「参加者共有のアーキテクチャを許可データで学習し直して学習コードを付けるのは可。既存の第三者重みは、元の学習コードと学習データが揃うなら可」。同（10-14）「元の学習コードと許可データで再現できない第三者重みは不可」。yesterdays_rebel（11-05）と Ashot Mirzoyan（11-05）が `ahmedberatozer/casmi26-training-code` の扱い（学習後に編集されたスクリプト、seed 欠落）を質問中（読んだ時点で主催者の返答は見当たらない）。
8. GitHub:
   - https://github.com/coleygroup/ms-pred （README: 公開の重みは「open-source MassSpecGym-trained weights」、NIST 版は NIST ライセンス保持者に申請で配布。`msg_all` は MassSpecGym 全件で学習する設定）
   - https://github.com/cubres/enveda-casmi-2026/blob/main/reproduction/legacy-v2/README.md （prvsiyan の FP モデル・ランカー特徴量の回収ソース。README だけ読んだ）
   - https://github.com/Majyles/kaggle-casmi-2026 （README だけ。MIT、train.parquet と COCONUT から再生成できる LightGBM ランカー、公開 LB 0.313）
   - https://github.com/Gluzdov-Dmitrii/Enveda-CASMI-2026 （カード画像だけ。コードは無い）
9. 自前コード: `scripts/local_eval/run.py --help`、`scripts/ranker/{train_eval,apply_eval,build_groups,ranker_features}.py --help`、`scripts/cv/{local_fold0_logits,build_holdout_v3,score_variants}.py --help`、`scripts/ranker/train_eval.py` の `bootstrap_diff`、`outputs/local_eval/v3_c{1,2}/result_score+prior.json`。

## 1. 系統の骨格（確認したこと）
- **Engine 1（v4n）**: F セル 26 の docstring「v4n = v4i + ICE_LAM = GL_LAM = 1.0 + engine FP bank fpnet_full1」。F セル 37 で `fpnet_full1.pt` を glob で探して FP バンクにし（**無ければ黙って v4b の `fpnet_0.pt` に落ちる**）、`MANIFEST['ranker']` の `ranker_v4b.pkl` から LightGBM ブースターを読む（MANIFEST に `n_boosters: 4`, `n_features: 160`, `rounds: 500`）。F セル 26 で `TOPN = 60`、F セル 39 で `POOLPOP_MU = 0.15` のプール文献 prior 並べ替え。
- **Engine 2**: F セル 32 の `eng_runner.py` が `E.main(test, train, sample, find('sim_rank_rows_nofp.npz'), find('rank_train.npz'), fp_models, w_pv=0.88)`。`main` は `fit_rankers(sim_rank_rows_nofp.npz)`（`RANK.W_A=(0.35,0.55)`×`SEEDS=(0,1)` で HistGradientBoosting 4 本）と `fit_pv_rankers(rank_train.npz)`（重み 0.30/0.60 × seed 0–3 で 8 本）を**ノートブック内で学習**する（`eng_casmi_engine.py` の `_hgb`, `fit_rankers`, `fit_pv_rankers`, `main`）。FP モデルは `casmi26-fp-models-v2` の `fp_*.pt`、指紋ビットは COCONUT バンク側の 6,930 ビット（`_find` の上書きで v4g の 10,226 ビットを避ける）。
- **PubChem チャネル**: F セル 34 で FPNet A+B（v3-models の `fpnet_*.pt` が 2 本であることを assert）＋ fold-safe `fpnet_fold0.pt` のバンク。F セル 26 `LIB_TAU = 0.9`、F セル 43 `elif lib_max >= LIB_TAU: untouched` → **lib_max < 0.9 のときだけ差し込む**（確認）。
- **融合**: F セル 45 `ALPHA, KRR = 0.6, 3.0`、`1/(3+r) + 0.6/(3+r2)` で上位 40（確認）。
- **ICEBERG+GLACIER**: F セル 40 の説明「同じ組成式の候補を z(score)+1.0 z(ICEBERG)+1.0 z(GLACIER) で並べ替え」「GLACIER は [M+H]+ だけ」。GLACIER README の `rerank_multi` は「slot-preserving same-formula groups」（確認）。なお F セル 45 は `CELL19_ICE = False`（最終並べ替えで ICE の寄与を切る旨のコメント）で、最後の段で ICEBERG が効いているかは未確認。
- **0.479 版で足された層**（lb479）: cell 10 に `MODULE_SHA256`・`MODEL_PINS`（sanpier の `features.py`/`model.py`/`config.json`/`final_model.pt`）・`FROZEN_SHA256`、cell 12 に未学習の断片チェック（`_physical` の SHA256 を assert）、cell 15 に provenance reranker（`frag_rescore` を import）、cell 17 に `pool_popularity.csv` の文献 prior と InChIKey14 重複除去。`sanpier/catalogue-experiment-saner` は「this run does not read it」と cell 23 に書かれている。

## 2. 部品の表（23 行。20〜23 は 11/11 に追加）
略号: F = frozen465.ipynb のセル、L = lb479 のセル、TC = `ahmedberatozer/casmi26-training-code`、v4g = `dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets`（上流の写し。ライセンス CC-BY-NC-SA-4.0。中の各フォルダは上流のライセンスのまま）。
「測れるか」は v3_c1 / v3_c2（`scripts/cv/build_holdout_v3.py`、fold-safe 分割の fold 0 から作る）で、置き換えた後に**リーク無しで**比べられるか。

| # | 部品 | 使われる場所 | 配布元 | 生成コード | ライセンス（metadata の licenses） | SHA256 の固定 | 学習データの範囲 | 置き換え案 | v3_c1 / v3_c2 で測れるか | 出典 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Engine 1 ランカー `ranker_v4b.pkl`（LightGBM 4 本、160 特徴） | F セル 37 `RK_B`、`rank_score`；F セル 39 の base リスト | `ahmedberatozer/casmi26-v4b-models`（v4g に写し） | **有り（TC）**: `fe_v4/simulate_v1ctx.py` → `build_feats.py` → `train_v4_ranker.py`。ただし学習ログ無し、起動コマンドは「再構成」、学習後に編集されたスクリプトあり、ビット一致はしないと TC README 自身が書く | other（カード: コード MIT、重み・表は train.parquet と COCONUT 由来） | **固定**: F セル 26 が v4b `MANIFEST.json` の全 64 ファイルの SHA256 を assert（ranker は 6172e926…） | 作者の holdout fold（np / nplib / syn / plusk）の 5,350 構造から作った模擬クエリ 15,701 件（主張: TC README）。特徴量は全 train 近くで学習した FPNet A/B・fe_D に依存 | 自前 `scripts/ranker/`（fold-safe な学習グループ rk_f{1..4}）、または TC の手順を fold 0 を外して回し直す、Majyles（MIT） | **測れない**: 特徴を作る FPNet A/B/fe_D が fold 0 の構造を学習で見ている（19 行目の fold-safe README「公開 v4g FPNet は fold-0 分子の 89% を 1 位にする＝見ている」）。模擬クエリの構造と v3 の重なりも未確認 | 出典 3–5、TC README §2.3 |
| 2 | `fpnet_full1.pt`（Engine 1 の FP バンク） | F セル 37 `_FULL` → `fpnet.ModelBank` | `ahmedberatozer/casmi26-fpnet-full1`（v4g に写し） | **有り（TC）**: `scripts/06_train_fpnet.py --holdout "" …`（コマンドは再構成、seed 未記録） | other | **固定されていない**: glob で読むだけ。v4g MANIFEST にも載っていない。TC README に d13a3611… の記載のみ | **全 train**（enveda-np-examples 含む、2,539,608 スペクトル。主張: カードと TC README） | fold-safe `fpnet_fold0.pt`（既にローカル評価の基準）、または 06_train_fpnet.py を fold 0 を外して学習 | **測れない**: 全 train で学習＝v3 の正解構造を学習で見ている | 出典 3–5、TC README §2.1 |
| 3 | FPNet A+B（v3-models `fpnet_0.pt`/`fpnet_1.pt` = v4b `fe_A`/`fe_B`/`fpnet_0`） | F セル 34（PubChem チャネルのバンク）、F セル 37（`V1FE` の model_views）、`fpnet_0` は full1 が無いときの代替 | `ahmedberatozer/casmi26-v3-models`、`casmi26-v4b-models` | **有り（TC）** `06_train_fpnet.py`（A は seed 7、B は seed 未記録） | other（コード MIT） | v4b 側は MANIFEST で固定。v3-models 側はノートブックで固定されていない（本数の assert だけ）。v4g MANIFEST と TC README に edccf479… / 6f60a22b… | 作者の holdout（np, nplib, syn, plusk, twin）以外の train 全部 | fold-safe fold0 | **測れない**: fold 0 の構造の大半を学習で見ている（fold-safe README の 89%） | 出典 3–6、TC README §1 |
| 4 | `fe_D.pt`（DreaMS の微調整、v4b の model_views） | F セル 37 `V1FE(… MAN['fe_models'] …)` | `casmi26-v4b-models` | **有り（TC）** `dreams_ft/review_fixes/train_dreams_fp.py` → `extract_step20k_ema.py`（後から書いたもので未再実行と README） | other（DreaMS は MIT / CC BY 4.0、LICENSE-NOTICE） | MANIFEST で固定 | 作者の train fold（holdout は使わないと主張）。土台は公式 DreaMS SSL | 外す（fold-safe 化するなら fold 0 を外して再微調整） | **測れない**（作者の分割で学習、fold 0 を見ている）。DreaMS SSL の NIST20+MoNA プロービング集合の扱いは 748882 で質問中＝**要確認** | 出典 5、7（748882）、TC README §2.2 |
| 5 | fe_v4 の表（`prior_tables.pkl`、`fragnet/stageA.npz`・`stageB.txt`、`analog_struct/*`、`frag2/*`） | F セル 37 `V1FE` の families（derivation, frag2, analog_struct, model_views, fragnet） | `casmi26-v4b-models/code/fe_v4/res/` | **有り（TC）** `class3_v2/priors_build.py`、`fe_v4/res_fragnet.py` など。学習後に編集されたものあり | other | MANIFEST で固定 | prior_tables: COCONUT 親 30,000（作者 holdout を除外）。fragnet: 作者の train fold の構造（主張） | ranker を自前にするなら不要 | **測れない／要確認**: fragnet は作者の train fold で学習（fold 0 を含む）。prior_tables は COCONUT 由来で学習ラベルを使うかは未確認 | TC README §2.4–2.6 |
| 6 | v2 プール（`pool_meta.parquet`、`pool_fp.npy`、`pool_frag_*`、`train_structs.parquet`、`train_fp_sel.npy`） | F セル 36（`work/` へ symlink、`Pool('work')`）、F セル 34 | `ahmedberatozer/casmi26-v2-pool`（v4g に写し） | **有り（TC）** `scripts/01_build_train_structs.py`、`02_build_coconut.py`、`05_build_pool.py`。再生成してバイト一致するかは**未確認** | other（コード無し、train.parquet と COCONUT CC BY 4.0 由来とカード） | 部分的: F セル 36 の `_EXTERNAL_REFERENCE_PINS['base_files']` が `pool_meta`（82b8c403…）と `train_structs`（f20aa1ca…）を照合（**外れると外部参照を無効にするだけで止まらない**）。v4g MANIFEST に全 7 ファイル | 学習なし（構造の表） | 現状のまま（ローカル評価もこれを使い、v3 は hide_rule / drop_keys で答えを消す） | **測れる**（既に `scripts/local_eval/run.py` の候補窓）。学習した部品ではないのでリークの経路は無い | 出典 3–5、TC README §3 |
| 7 | `fp_bits.npy`（10,226 ビット、Engine 1・PubChem チャネル・fold-safe FPNet 共通） | F セル 34（`bits = np.load(POOL_DIR/'fp_bits.npy')`、fold-safe 側と一致を assert）、`casmi/pool.py`、`traindata.py` | `casmi26-v2-pool`（fold-safe にも同じもの） | **有り（TC、11/9 に確認）**: `scripts/04_make_split.py` の末尾で `structs.select_bits(train_fp_raw, lo=0.005, hi=0.995, sample=200000)`（`casmi/structs.py` の既定 seed 0）→ `work/fp_bits.npy`。754610 で wata が「選択コードが無い」と書いたのは v2-pool データセットの中に無いという意味では正しいが、TC にはある。再生成して一致するかは**未確認** | other | ノートブックでは fold-safe 側との一致 assert のみ。SHA256 c3869e86…（v4g MANIFEST、fold-safe `MODEL_INPUTS.json`。ローカル 2 つのファイルで同じ値を確認） | 全 train 構造の指紋のビット頻度（ラベル・スペクトルは使わない） | 現状のまま。必要なら 04_make_split.py で再生成して一致確認 | **測れる**（ローカル評価で既に使用）。fold 0 の構造の指紋が頻度集計に入るが、正解ラベルは使わない | 出典 5–7（754610）、TC `04_make_split.py` |
| 8 | `rank_train.npz`（142,762 行 × 31 特徴、prvsiyan） | F セル 32 → `fit_pv_rankers`（ノートブック内で HGB 8 本を学習） | `prvsiyan/casmi26-ranker-features`（v4g に写し） | **部分的**: カードが cubres/enveda-casmi-2026 に回収した `make_ranker3.py` を挙げるが、「FP ロジットの 12 列を作ったキャッシュの生成元が未特定、完全な再生成は主張できない」と自ら書く。HGB の学習コードはノートブック内にある | CC-BY-NC-SA-4.0 | **記録はあるが照合は無い**: frozen465 のノートブック metadata（`casmi.historical_configurations.v39〜v42_config.engine_inputs`）に 41ccec87… が記録されている。コードセルには読み込みも assert も無い。カードにも同じ値（9,553,020 バイト） | 未確認（カード: 「候補単位の学習特徴で、held-out のクエリ集合ではない」。FP 列は全 train 近くで学習した FP モデル由来） | 自前ランカー（`scripts/ranker/`）または Majyles 手順で Engine 2 ごと置き換え | **測れない**: 生成元が揃わず fold 0 を外した版を作れない。FP 列は fold 0 を見たモデル由来 | 出典 3、7（754292, 756893 はタイトルのみ）、8（cubres） |
| 9 | `sim_rank_rows_nofp.npz`（399,790 行 × 51 特徴、megayak） | F セル 32 → `fit_rankers`（HGB 4 本） | `megayak/casmi26-simulated-ranker-rows`（v4g に写し） | **無し**（データセットは README・npz・pt だけ。README にも生成スクリプトの所在は無い） | CC0-1.0 | **記録はあるが照合は無い**: 上と同じ metadata に c19ae76f… が記録されている。コードセルでの照合は無い | 7 つのクエリ集合（enveda-np-examples 250, gnps 400, riken 300, mona 300, msdial 300, massbank 300, pluskal_ms2 400）の 2,250 分子の模擬、候補は COCONUT＋学習構造（README の主張）。v3 との重なり未確認 | 自前ランカー／Majyles で置き換え | **測れない**: 生成コードが無く fold 0 を外した行を作れない | 出典 3–5（megayak README）、7（754292） |
| 10 | prvsiyan FP モデル（`fp_single_s2.pt`、`fp_merged_m1.pt`、6,930 出力） | F セル 32 `fp_models` → `pv_fp.load_fp_models`（Engine 2 の特徴） | `prvsiyan/casmi26-fp-models-v2`（v4g に写し） | **部分的**: cubres に元の学習スクリプト `train_fp3.py`/`train_fp4.py` と起動引数を回収。カードは「学習入力の版が未固定、seed が初期化後に設定、公開重みへのバイト一致は未確立」 | CC-BY-NC-SA-4.0 | **記録はあるが照合は無い**: 上と同じ metadata に fp_single_s2.pt c19aa290…、fp_merged_m1.pt 261529c8… が記録されている。カードにも同じ値 | カードが調べた legacy cache は 2,334,531 スペクトルで、学習と検証の両方にまたがる構造は 0（主張）。学習がそのどこまでを使ったか、fold 0 を含むかは未確認（含む可能性が高いが推測） | 自前ランカーで Engine 2 を置き換えるなら不要。fold-safe 化するなら fold 0 を外して再学習（megayak の leak-free FPNet はシミュ構造を外しただけで fold 0 は外していない） | **測れない**: fold 0 を外した版が無く、学習範囲も未確認（fold 0 を外して学習し直すまでは使えない） | 出典 3、8（cubres） |
| 11 | Engine 2 の候補バンクと 6,930 ビット（COCONUT: `prvsiyan/coconut-casmi26-candidates`、ChEBI/LMSD: `sergiomalv/casmi-chebi-lmsd-structures`、PubChemLite: `franciscoangulo/casmi26-pubchemlite-cc0-candidate-fingerprints`） | `eng_casmi_engine.py` の `build_pool`、`_fp_init`（`fp_bits.npy`） | 各データセット（v4g に写し） | COCONUT: カード「ダウンロード日・生成手順はマニフェストに無い」＝**未確認**。sergiomalv: 公開源だけから作ったとカード、コードの有無は未確認。franciscoangulo: 「hashes and build manifest」を含むとカード（中身は未読）。6,930 ビットの選択規則: 未確認 | CC-BY-NC-SA-4.0 / CC BY 4.0 / CC0-1.0 | **記録はあるが照合は無い**: 上と同じ metadata に coco_*/bio_*（fp・mass・meta）と Engine 2 側の fp_bits.npy（a61e372f…、6,930 ビット版）が記録されている。コードセルでの照合は無い | 学習なし（構造の表） | Engine 2 を自前ランカーで置き換えるなら、候補は v2 プール＋PubChem 層にまとめる | **要確認**: 学習はしないが、v3_c2 は答えをプールから消す設計なので、これらのバンクにも drop_keys を当てないと答えが漏れて届く | 出典 3（カード） |
| 12 | MetFrag-lite（`frag_rescore.py`、未学習の断片説明率） | F セル 51 `FRAG_GAP`（v7c）、L セル 12（`parallax_physical`）・L セル 15（provenance reranker が `import frag_rescore`）。Engine 2 では `pv.py` の `explain_score*` を `compute_frag` で使用 | ノートブックに埋め込み（F セル 51 のコメント: ahmedberatozer/casmi26-v7c-e006-fraggap v1 から写し、Huseyin Emre Aksoy に帰属） | **有り**（ソースそのもの、学習なし）。ローカル写し `data/ext/frag/frag_rescore.py` | Apache-2.0（F セル 51 のコメントの記載。データセットの metadata ではない） | L セル 12 で `_physical` ソースの SHA256 を assert | 学習なし（既定値は「241 構造の同一組成式異性体パネルの forum アブレーション」に従うと docstring） | 現状のまま（`run.py --stages …,frag` が同じ λ 0.5 で実装済み） | **測れる**（学習しないのでリーク経路なし） | 出典 2、6 |
| 13 | ICEBERG の重み（`ckpt/gen.pt`、`ckpt/inten.pt`） | F セル 41（ICE ランナー）、F セル 45 | `ahmedberatozer/casmi26-iceberg`（v4g に写し） | 学習コードは ms-pred（MIT）に公開。重みは ms-pred 公開の **MassSpecGym `msg_all` 版**（LICENSE-NOTICE: 「ICEBERG msg_all (MassSpecGym-trained) weights … re-saved」）。**NIST 版ではない**（ms-pred README: NIST 版はライセンス保持者に申請配布） | other（ms-pred は MIT） | v4g MANIFEST に deef0451… / c49e330e…。ノートブックでの assert は未確認（F セル 41 はランナー類の SHA256 を固定） | MassSpecGym（公開ライブラリ由来）。主催者は 748882（10-13）で承認済み | 現状のまま（承認済み）。推論コード（`ice_runner.py`, `fuse.py`）の著作権・ライセンスは**未確認** | **要確認**: v3_c1 の構造は公開スペクトルを持つので MassSpecGym の学習構造と重なり得る（重なりは未確認）。v3_c2（enveda-180 だけ）は重ならない見込みだが未確認。ローカル評価環境に無く GPU が要る | 出典 5、7（748882）、8（ms-pred） |
| 14 | GLACIER の重み（`ckpt/glacier.pt`） | F セル 41（GL ランナー、[M+H]+ だけ）、F セル 45 | `ahmedberatozer/casmi26-glacier`（v4g に写し） | 学習コードは ms-pred（MIT）。重みは ms-pred README の「open-source MassSpecGym-trained weights」（LICENSE-NOTICE、sha256 5a47cecc… の best.ckpt を再保存） | other（ms-pred MIT 扱いと LICENSE-NOTICE） | F セル 41 で `N1_GL_RUNNER_SHA256` などランナーを固定。重み自体の assert は未確認 | MassSpecGym | 13 と同じ | **要確認**（13 と同じ理由） | 出典 5、8 |
| 15 | PubChem 層（`pc_mass.npy`、`pc_off.npy`、`pc_smiles.npy`、約 1.06 億構造） | F セル 34（`init_worker`）、F セル 43 の差し込み | `ahmedberatozer/casmi26-pubchem-tier`（ローカル `data/ext/pubchem_tier/`） | **有り（TC）** `scripts/02c_build_pubchem.py`（CID-SMILES + CID-Mass、有機元素のみ、150–1250 Da、立体を落とす。docstring を確認）。入力の PubChem 版は未確認 | other | v4g MANIFEST に 3 ファイル（pc_smiles 7a888630…）。ノートブックでの assert は無い | 学習なし（PubChem 公開データ） | 現状のまま | **測れる**（`scripts/local_eval/pubchem.py`。v3_c2 の答えはここからしか届かない） | 出典 3–5、TC `02c_build_pubchem.py` |
| 16 | 文献 prior（`pool_lsid/lpmid.npy`、`pc_lsid/lpmid/lsynonyms/np/src_mask/lotus_refs.npy`、delta/growth 補遺、`pool_popularity.csv`） | F セル 26（環境変数と存在 assert）、F セル 34（PubChem チャネルの S）、F セル 39（`POOL_POP`）、L セル 17（`pool_popularity.csv`） | `dmitriigluzdov/casmi26-pubchem-popularity-prior` | **有り**: データセットの `scripts/`（`build_pop.py`、`build_annot.py`、`build_delta.py`、`build_sources.py`、`download_sources.py` ほか）。PubChem Extras 2026-09-27 を MD5 照合と主張 | other（PubChem は NCBI 方針、LOTUS CC BY 4.0、NPAtlas CC BY-NC 4.0） | `aligned_public_priors.py` だけ SHA256 を assert（89873c73…）。配列は存在の assert のみ | 学習なし（PubChem の件数。「競技データは含まない」とカード） | 現状のまま | **測れる**（`run.py --stages score,prior` が v19 と同じ μ 0.15 で使用） | 出典 3–5（カード、ファイル一覧 2 ページ目） |
| 17 | fold-safe FPNet（`fpnet_fold0.pt` ほか 5 本） | F セル 26（SHA256 を assert して `CASMI_FPNET_FOLD0`）、F セル 34（PubChem バンクの 1/3） | `dmitriigluzdov/casmi26-fold-safe-fpnet`（ローカル `data/ext/foldsafe/`） | **有り**: `train_fivefold.py`、`train_fpnet.py`、`make_split.py`、学習ログ。アーキテクチャとデータ読みは v3-models の `casmi` | other（チェックポイントは競技データ条件、スクリプトは Apache-2.0） | **固定**: ddf6eaaa…（F セル 26 の assert、`MODEL_INPUTS.json`、ローカルで一致を確認） | fold 0（55,102 構造）と NP パネル（265）を除いた train | — （これが置き換え先） | **測れる**（ローカル評価の基準: v3_c1 0.8219 / v3_c2 0.0117、score+prior） | 出典 3、6 |
| 18 | sanpier のスペクトル–候補ニューラル（`casmi_spectrum_model_v1/final_model.pt`） | L セル 10 `MODEL_PINS`、L セル 13 `score_final_lists` / `score_outside_candidates`、CUSTOM の w_mid 0.75 | `sanpier/casmi-spectrum-model-v1` | **有り**: `train.py`、`prepare.py`、`config.json`、`spectrum_split_manifest.parquet`、学習履歴。README「再現には config・学習ソース・入力版・environment.json」 | apache-2.0 | **固定**: L セル 10 `MODEL_PINS`（features.py, model.py, config.json, final_model.pt 44436b2d…） | **全 train**: `training_result.json`「final_model は former validation を含む全データ」。分割は無作為の分子 holdout（`data_audit.json`） | `train.py` を fold 0 の構造を外して学習し直す（引数は未確認） | **測れない**（全 train で学習）。fold 0 除外で学習し直せば測れる。→ 11/11: #54 でキットの既定のまま fold-safe の hold/np と v3 の正解 55,369 構造を外して学習（リーク 0）。#60 で縮小版 score+prior+frag に等重みで足して測り、**不採用**（v3_c1 +0.0194 [−0.0008, +0.0404]、v3_c2 0、enveda_v1 −0.0209 [−0.0489, +0.0048]、v3_c2n −0.0255 [−0.0565, +0.0051]。enveda_v1 と v3_c2n が下限 −0.02 を割る） | 出典 3–5、#54、#60 |
| 19 | 外部参照スペクトル（`external_delta.parquet` ほか、782 キー・1,384 行） | F セル 36 `_build_external_reference_cache`（Engine 1 のライブラリに追加） | `dmitriigluzdov/casmi26-natural-product-knowledge-table` の `external_reference_spectra/`（MassBank / MoNA / GNPS） | 未確認（このメモでは知識表のスクリプトを読んでいない） | other（各レコード CC BY 4.0 / CC0 と F セル 58） | **固定**: `_EXTERNAL_REFERENCE_PINS['files']` 3 ファイル（外れたら無効化） | 学習なし（参照スペクトル） | 外す、または生成コードを確認してから使う | **要確認**: v3 の hide_rule（c1 は enveda 系だけ隠す、c2 は全部隠す）をこの追加参照にも当てる必要がある | 出典 2、F セル 58 |
| 20 | 2 本目のスペクトル–候補スコアラー v80（`casmi_spectrum_model_ours_v80/final_model.pt`、sanpier キットを epochs 80 で学習） | 0.487 版（guhongbin v589 をそのまま実行）の `kernel_sources`。公開重み（18）と `0.5*z(public)+0.5*z(ours)` で平均（11/10 の日報で確認） | カーネル https://www.kaggle.com/code/popwenman/casmi26-scorer-train-v80 （出力が重み） | **有り**: そのカーネル自体（52 行、11/11 に `kaggle kernels pull -m` で読んだ）。キット（18）の config から epochs 16→80 と時間予算だけ変え、prepare.py に normalized_smiles の別名を足す | キットは apache-2.0。カーネルのライセンスは metadata に無い（未確認） | 0.487 版では固定されていない（kernel_sources で最新版の出力を読む） | **全 train**（`external_paths` は空、holdout はキットの無作為 10% で、refit で全部を学習） | #54（キットの既定 16 epoch で、fold-safe の hold/np と v3 の正解 55,369 構造を外して学習し直す） | **そのままでは測れない**（全 train で学習）。#54 の重み（エポック数だけ違う同じキット）で測った結果は 18 行のとおり不採用（#60） | v589 の 1 つ目の markdown セルの表（作者の申告、こちらで未確認）: 「bgunshui 480 neural ensemble, verbatim 0.480」「the same with our scorer v80 as the second model 0.480」 |
| 21 | ExtLib（`takumuhata/casmi26-extlib`、GNPS+MassBank+MoNA を合わせた外部 MS2 ライブラリ、`ext_mz.npy`・`ext_in.npy`・`ext_meta.parquet` ほか 6 ファイル、77.6 MB） | 0.484／0.487 系統の「外部ライブラリで 1 位に上げる」重ね（10 ppm＋0.002 Da、cosine ≥ TAU、一致ピーク ≥ 4、自前の lib_max ≥ 0.90 なら上げない） | https://www.kaggle.com/datasets/takumuhata/casmi26-extlib | **見つからない**: データセットのファイル 6 件、作者の他の公開データセット 24 件（`kaggle datasets list --user takumuhata` の 2 ページ。fp-models、pc-index、v2ckpt、con-v2 ほか。名前で関係しそうな `casmi26-colab-scripts`（中身は con_v2 の学習セル 2 本）と `casmi26-con-train-ext`（学習用の npz・parquet）/ `casmi26-con-ext-out`（npz 1 件、学習の出力とみられる）はファイル一覧まで確かめた）、作者の公開カーネル一覧（CASMI 関係はすべて casmi26-codex-sp… の提出版、中身は読んでいない）のどれにも ExtLib を作るコードが無い（11/11）。#52 の調べでも見当たらない | CC0-1.0 | 0.487 版では固定されていない | 学習なし（参照スペクトル）。構造表に train の構造は 1 件も入っていない（#52） | 外す、または公開ライブラリ（GNPS / MassBank / MoNA）から自前で組み直す | **timsTOF の条件では測れない**: timsTOF の問い合わせで、hold の構造と重なりスペクトルもあるのは 2 構造だけ（#52、`docs/research/extlib-holdout.md`）。装置の違う公開ライブラリのスペクトルを問い合わせにする代わりの集合（#52 の `v3e` 案、1,143 構造）なら、近い性質だけは測れる（設計だけで未作成） | v589 の表（作者の申告）: ozertuu の版をそのまま 0.478、TAU 0.85→0.80 で 0.484。ExtLib だけを外した盤面の値は見つからない（guhongbin v590「extlib only」・v591「massbank only」はコードが v589 と同じで添付だけが違い、markdown は v589 の写しで値が書かれていない＝未確認） |
| 22 | MassBank harmonized（`samartalwar/casmi-2026-spectral-library-massbankharmonized`、13.9 万スペクトル＋指紋、`spectra.parquet` ほか 3 ファイル） | 21 と同じ重ね（`is_casmi_subset` の行を落としてから使う） | https://www.kaggle.com/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized | **手順の説明だけ有り、コードは見つからない**: metadata の説明文に「Build pipeline (reproducible)」として 6 手順（MassBank/MassBank-data の clone、レコードのパース、RDKit で SMILES 正規化と InChIKey、ND ライセンスのレコードを除く、InChIKey で重複除去と Morgan(2,1024)・MACCS、parquet 出力）。データセットは 3 ファイル（spectra.parquet、spectra_preview.csv、structures_with_fingerprints.parquet）だけで、コードは無い。作者の公開カーネルのうち名前に CASMI・MassBank・spectr を含むものは無い（11/11。レビュー役が `samartalwar/notebook28db8539b3` の中身を見て、データを読むだけで生成コードではないことを確かめた） | **CC-BY-NC-SA-4.0** | 0.487 版では固定されていない | 学習なし（参照スペクトル） | 外す、または MassBank-data からレコードごとのライセンス（CC0・CC BY・CC BY-SA・CC BY-NC・CC BY-NC-SA が混ざる、と説明文）で絞って自前で組み直す | **timsTOF の条件では測れない**: MassBank 側で hold の構造と重なり timsTOF のスペクトルもあるのは 0 構造（#52）。装置の違う問い合わせの代わりの集合なら 1,916 構造から作れる（#52、設計だけで未作成） | 21 と同じ（単独の値は未確認） |
| 23 | LOTUS 候補（`qwertykeypadapple/casmi26-lotus-candidates`、149,983 構造の `candidates_lotus.parquet`） | 0.484 系統の「空いた枠を質量の合う天然物で埋める」（候補が 25 未満のときだけ）。**0.487 版（v589）には添付されておらず、この段は飛ばされる**（0.487 のノートの説明文「LOTUS candidate dataset is not attached, so the LOTUS vacant-slot backfill skips」） | https://www.kaggle.com/datasets/qwertykeypadapple/casmi26-lotus-candidates | **手順の説明だけ有り、コードは見つからない**: README に手順（Zenodo 19360665 の LOTUS frozen metadata、RDKit 2026.03.3 で立体除去・最大断片・中和、100–1500 Da、inchikey14 で重複除去、709,740→149,983）。データセットは README と parquet の 2 ファイル。作者の他の公開データセット 2 件・公開カーネル 2 件は無関係（11/11） | CC0-1.0（上流の LOTUS も CC0 と README） | 固定されていない | 学習なし（構造の表） | README の手順は短いので、自前で組み直せる（未実施） | **測れる見込み**（学習しない構造の表）。ただし v3_c2 の答えがこの表に入っていれば届いてしまうので、drop_keys を当てる必要がある。重なりは未確認 | 0.487 のノートの説明（作者の申告）: 「同じコードの前の版 0.484 に対して 0.487（submission 58185447）」。前の版との違いは添付で、LOTUS を外し v80 を足した（11/10 の日報の調べ）。どちらが効いたかは測られていない |

補足（表に入れなかった小さい部品）:
- `hengck23/hengck23-dreams-enveda-casmi26`（MIT）: F セル 48 で空いた順位を質量の合う DreaMS 構造で埋める。生成コード未確認。
- RDKit wheel（`metric/rdkit-2026-3-3-wheel`、licenses unknown、RDKit 自体は BSD-3）と simulator 用 `rdkit-2025.3.6` wheel（ICEBERG データセット内）。部品ではなく環境。
- `ahmedberatozer/casmi26-training-code` 自体の metadata の licenses は `other`、README は MIT と書く。
- lb479 の 6 つの補助モジュール（runner.py ほか）と consensus.py / outsiders.py は L セル 10 にソースごと埋め込みで SHA256 固定。生成物ではなくコードなので表から外した。

### 規則スレッドの状況（11/11 00:16 JST = 11/10 15:16 UTC に `kaggle competitions topics show … --format json` で取得）
- 754610「Clarification on permitted external resources for final submissions」（blaise yonga、10-31、コメント 3）: 主催者の回答は David Healey の 11-05 の 1 件だけ（前回までと同じ）。その後は参加者の質問 2 件（wata 11-08: v2-pool の候補の選び方、Aravindhan Chandran 11-10: DreaMS の ssl_model.ckpt も「公式チェックポイント」に入るか）で、どちらにも主催者の返答は無い
- 756893「Are shipped ranker tables without a public generator reproducible?」（Shishir Adhikari、11-08）: コメント 0。**主催者の回答なし**
- 757307「Is MassSpecGym (and MIST weights trained only on it) allowed? NIST was used only for data cleaning」（ujihash、11-09）: コメント 1（参加者 Nidhal 11-09 の同意）。**主催者の回答なし**
- 757309「Clarification on PubChem/COCONUT candidate-index redistribution」（Phil Mohrman、11-09）: コメント 0。**主催者の回答なし**

## 3. まとめ
### 分類
- **置き換え必須（生成コードが無い、または揃わない）**
  - 9 `sim_rank_rows_nofp.npz`（megayak）: 生成コードがどこにも見当たらない。
  - 8 `rank_train.npz`（prvsiyan）: 特徴の組み立てコードは回収されたが、FP ロジット列のキャッシュの生成元が未特定とカード自身が書く。754292 の主催者回答（再現できない中間物は違反）にそのまま当たる。
  - → Engine 2 の 2 つのランカーは、学習コードはノートブック内にあっても学習データが再現できない。Engine 2 ごと自前ランカーに置き換えるのが筋。
- **要確認**
  - 1–5 Engine 1 一式（ranker_v4b、full1、A+B、fe_D、fe_v4 の表）: TC で学習コードは公開されたが、seed 欠落・コマンド再構成・学習後の編集があり、ビット一致はしない（TC README §5）。主催者（Marie Killian、10-27）は LeeDongHyuk（10-24）の質問 2 に「Yes, that is sufficient」と答えている。質問 2 は「元の作者が学習・生成コードを公開していれば、チームが再実行しておらず、学習し直してもビット一致しなくても十分か」。同じ回答の 3 は「候補表を元のダウンロードから少しでも変えたなら、その手順のコードを付けること」。この回答に従えば、TC があることで Engine 1 は通る方向に読める。ただし、学習後に編集したスクリプトや seed の欠落の扱いを問う 11-05 の質問（yesterdays_rebel、Ashot Mirzoyan）には返答が見当たらないので、要確認のままにする。fe_D は DreaMS SSL の NIST 関連の扱いも未決。
  - 10 prvsiyan FP モデル: 元の学習スクリプトは回収済みだが、入力の版とバイト一致が未確立。
  - 11 Engine 2 の候補バンク、19 外部参照スペクトル、DreaMS 埋め: 生成手順を読んでいない／記載が無い。
  - 13・14 ICEBERG/GLACIER: 主催者が承認済み（748882）。再現性ではなく、リーク無しで測れるか（MassSpecGym と v3_c1 の構造の重なり）が未確認。
  - 18 sanpier ニューラル: コードは揃っているが公開重みは全 train で学習。
- **再現可能（コードあり。バイト一致の確認はまだ）**
  - 6 v2 プール、7 `fp_bits.npy`（TC の 01/02/04/05）、15 PubChem 層（TC の 02c）、16 文献 prior（データセットの scripts/）、17 fold-safe FPNet（学習コード＋SHA256 固定）、12 MetFrag-lite（未学習のソース）。

### 「再現できる部品だけの版」を作るなら
2 つの要件を分けて考える。
- **提出の再現性**（規則）: 生成コードがあればよく、全 train で学習してよい。
- **検証のリーク無し**（v3 で測る）: fold 0 を学習で見ていないことが要る。

| | 残すもの | 作るもの・外すもの |
|---|---|---|
| 候補 | 6 v2 プール＋7 fp_bits、15 PubChem 層 | Engine 2 の候補バンク（11）は外す。必要なら PubChem 層に寄せる |
| 指紋スコア | 17 fold-safe FPNet（検証用）。提出用には fold-safe 5 本、または 06_train_fpnet.py / train_fpnet.py で全 train を学習し直したもの | 2 full1・3 A+B は外す（規則上は要確認なので、置き換えれば判断待ちが要らない） |
| 並べ替え | 16 文献 prior、12 MetFrag-lite、13・14 ICEBERG/GLACIER（承認済み） | Engine 1 ranker（1）と Engine 2（8–10）の代わりに自前ランカー（`scripts/ranker/`、fold-safe な学習グループ）。#29 の版は enveda_v1 で 0.472 に落ちて不採用。#32 の model_c2p は v3 の 4 集合で縮小版の規則に勝ったが、公開 LB は 0.246 で、同じ土台の score+prior 0.282 より低い（#34 / #37） |
| 追加 | — | 18 sanpier は fold 0 を外して学習し直すまで外す。19 外部参照は生成コードを確かめるまで外す。11/11 追加分: 20 v80 は全 train 学習なので #54 の fold-safe 版に置き換える。21 ExtLib と 22 MassBank harmonized は生成コードが見つからず（22 は CC-BY-NC-SA-4.0）、v3 でも測れないので外す。23 LOTUS は手順が README にあり CC0 なので、組み直せば使えるが、0.487 版はそもそも使っていない |

現状のローカル縮小版（fold-safe fold0 + prior、プールだけ）は公開 LB 0.282（提出 58119572。11/9 に `kaggle competitions submissions` で確認）。0.479 との差の大部分は、置き換えが必要な部品と「要確認」の部品にある。

## 4. 測る案（この日は測らない）
### 案 A（主案）: MetFrag-lite（12）を score+prior に 1 つ足す
- 理由: 学習しない部品なのでリークの経路が無く、v3_c1 / v3_c2 の両方でそのまま測れる。λ = 0.5 は 0.479 版 v7c の値で固定（掃かない）。以前の記録（np_v2 / enveda_v1）では prior の上ではほぼ差が無かった。v3 ではまだ測っていない（`outputs/local_eval/v3_c{1,2}/` に `+frag` の結果が無いことを確認）。
- 対照: `score,prior`（v3_c1 0.8219 / v3_c2 0.0117。`outputs/local_eval/v3_c{1,2}/result_score+prior.json`）。
- コマンド（引数は `--help` で確認。ロジット `outputs/holdout_eval/v3_c{1,2}_fold0/pc_logits.npz` は既にある。無ければ `.venv/bin/python scripts/cv/local_fold0_logits.py --holdout v3_c1`）:

```bash
cd "/Users/taichi/コンペティション/Kaggle/Molecule ID From Mass Spectra"
for h in v3_c1 v3_c2; do
  .venv/bin/python scripts/local_eval/run.py --holdout $h --stages score,prior        # 対照（既存と同じ値になることを先に確かめる）
  .venv/bin/python scripts/local_eval/run.py --holdout $h --stages score,prior,frag   # 新
done
.venv/bin/python - <<'EOF'
import sys; sys.path.insert(0, 'scripts/validate'); sys.path.insert(0, 'scripts/ranker')
import mrr25
from train_eval import bootstrap_diff          # 分子を 2,000 回再抽出、seed 0（N_BOOT=2000 を確認済み）
def rr(h, tag):
    t = mrr25.read_truth(f'outputs/holdouts/{h}/truth.csv')
    p = mrr25.read_submission(f'outputs/local_eval/{h}/submission_{tag}.csv')
    return {m: mrr25.reciprocal_rank(mrr25.to_key(s), p.get(m, [])) for m, s in t.items()}
for h in ('v3_c1', 'v3_c2'):
    print(h, bootstrap_diff(rr(h, 'score+prior+frag'), rr(h, 'score+prior')))
EOF
```

  この採点スニペットは、既存の `submission_score+prior.csv` 同士で動かして 0.8219 / 0.0117 と差 0 を返すことだけ確かめた（新しい数値は出していない）。
- 受け入れ条件: v3_c1 と v3_c2 の両方で「新 − 対照」の 95% 区間（分子ブートストラップ 2,000 回、seed 0）の下限 ≥ −0.02、かつ少なくとも片方で下限 > 0。v3_c2 は対照が 0.0117 で、プールに答えが無いので、frag（プール内の並べ替え）では動かない見込み。その場合は「v3_c1 で下限 > 0、v3_c2 で差 0（区間 [0, 0]）」が実際の判定になる。
- 注意: この案は「足す」比較で、置き換えの比較ではない。0.479 版の部品を置き換えた効果は、0.479 版の側が v3 で測れない（表の 1–5, 8–10, 18）ので、この形では直接比べられない。

### 案 B（概要だけ、引数は未確認）: sanpier ニューラル（18）を fold 0 を外して学習し直し、score+prior に足す
- `casmi_spectrum_model_v1/train.py` の分割（無作為の分子 holdout、`val_fraction` 0.1）を、fold-safe の `split.parquet` の fold 0 構造を必ず除く形に替えて学習する。`train.py` の引数・分割の差し替え方は読んでいないので未確認。GPU が要る。
- 受け入れ条件は案 A と同じ。

## 5. この日に測らない理由と次の判断
- 系統の部品を置き換えて効果を見るには、置き換え先（fold 0 を外して学習した版）と、それを差し込める縮小パイプラインの段がまず要る。表の「測れる」は 6, 7, 12, 15, 16, 17 だけで、残りはどれも、学習し直すかリーク無しの集合に合わせる作業（drop_keys / hide_rule を候補バンクや外部参照に当てるなど）が先に要る。
- 0.479 版の側は v3 で測れない部品を含むので、「0.479 版 − 置き換え版」の差はリーク無しでは測れない。測れるのは「縮小版 + 部品 1 つ − 縮小版」の形だけ。
- どれをいつ測るか（案 A を先にやるか、Engine 2 の置き換えを優先するか）は 11/10 以降の計画で判断する。

## 6. 不明・未確認の一覧
- 756893 の本文（取得結果はタイトルだけ）。
- 755373 は ICEBERG/GLACIER の重みに触れていない。重みの由来は ms-pred README と LICENSE-NOTICE から確認した。
- 11-05 の TC についての 2 件の質問への主催者の返答（10-27 の「Yes, that is sufficient」は LeeDongHyuk の質問 2 への答えと特定済み。§3 を参照）。
- TC の v2 プール・fp_bits・PubChem 層の再生成がバイト一致するか（コードの存在だけ確認）。
- `rank_train.npz` / `sim_rank_rows_nofp.npz` の分子と v3（fold 0）の重なり、`ranker_v4b` の模擬クエリ（作者の np/nplib/syn/plusk）と v3 の重なり。
- MassSpecGym の学習構造と v3_c1 / v3_c2 の正解構造の重なり。
- ICEBERG/GLACIER の推論ランナー（ahmedberatozer 作）のライセンス表記、F セル 45 の最終並べ替えで ICEBERG が効いているか。
- Engine 2 の 6,930 ビットの選択規則、COCONUT バンク・ChEBI/LMSD・PubChemLite の生成コード、外部参照スペクトルと DreaMS 埋めの生成コード。
- sanpier `train.py` の引数と分割の差し替え方。
- fe_v4 `prior_tables.pkl` が学習ラベルを使うか。

## 7. 以前の記録の訂正
11/8 の日報（Issue #3 のコメント）とスキルの参照メモには、以下の点が事実と違う、または裏づけの無い形で書かれていた。
- 「`fp_bits.npy` には選び方のコードが無い」
  - 正しくは、`ahmedberatozer/casmi26-training-code` の `scripts/04_make_split.py` に、`structs.select_bits(fp, lo=0.005, hi=0.995, sample=200000)` → `work/fp_bits.npy` がある（11/9 にファイルを読んで確認）
  - v2-pool データセットの中に無い、という点だけは正しい
- 「Engine 2 の 2 表（`rank_train.npz`、`sim_rank_rows_nofp.npz`）には生成コードが無い」
  - `sim_rank_rows_nofp.npz` は生成コードが見当たらない（表の 9）
  - `rank_train.npz` には、カードが特徴の組み立てコード（cubres/enveda-casmi-2026 の `make_ranker3.py` ほか）を挙げている。ただし、FP ロジット列のキャッシュの生成元は未特定と、カード自身が書いている（表の 8）
  - 「生成コード無し」ではなく「部分的」が正しい
- 11/8 の記録の「両ファイルの SHA256 はノートブックの metadata に固定」について
  - metadata に記録されているのは正しい（`engine_inputs`）
  - ただし、コードセルでの照合（assert）は無い。実行時に別のファイルが入っても止まらない
- 11/8 の記録は 756893 の本文の内容に触れていたが、今回取得できたのはタイトルだけ（本文・コメントとも 0）。本文の内容は、今回の取得では裏づけられない
- どの訂正でも、Engine 2 を置き換えが必要な部品とする結論は変わらない

