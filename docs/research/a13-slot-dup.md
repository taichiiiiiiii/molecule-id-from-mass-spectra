# A13 25 枠の重複と A5 の外部ライブラリとの一致（#230、#195 A13・A5）

定義は #230 の本文（2026-11-24 15:40:24Z、数える前に書いた）。ここでは結果と読みを書く。結果を見て定義は変えていない。照合の単位は A1（#197）で metric と合わせた `scripts/validate/mrr25.py` の `to_key`（立体を外す → 互変異性体の正準化 → InChIKey14）。RDKit は手元の .venv の 2026.09.1（metric の 2026.03.3 とは違う。#197 の既知の差）。

## 1. A13: 25 枠の中の「同じ答え」の候補

### 1.1 やったこと
- `scripts/local_eval/run.py` に、深い順位を書き出すだけの引数 `--dump-ranks <path>` を足した（a0311d0）。中身は「最終の並び（dedup 後の上位 TOPN=60、prior・frag の後）」に続けて「窓の残りを score の順（pool の key で dedup、すでに出た key は飛ばす）」。`pubchem`・`joint` とは併用できない
- 既定の出力は変わらない: 引数ありで `score,prior,frag` を 4 集合で実行した `submission_score+prior+frag.csv` の sha256 は、前からある CSV と同じ（v3_c1u_t10 `ebc59d2c…`・v3_c2_t10 `7716fc27…`・enveda_v1u_t10 `cc200529…`・v3_c2nu_t10 `acfaade8…`）。MRR も前と同じ（0.8878 / 0.0000 / 0.8513 / 0.6357）。深い順位の先頭 25 が submission の並びと同じことは、run.py の中の assert と `a13_slot_dup.py --sub` の両方で確かめた
- 数えた: `scripts/research/a13_slot_dup.py`（70d13c2 で数え、be410dd で補助の数だけを足して数え直し。主の値は同じ）。peak RSS は run.py 0.88〜1.14 GB、a13_slot_dup.py 0.07 GB。ログ `logs/a13_dump_1124.log`・`logs/a13_count3_1124.log`（git の対象外）
- 深い順位の CSV（git の対象外、`outputs/local_eval/a13_1124/ranks_<h>.csv`）の sha256: v3_c1u_t10 `4ff334d3…`・v3_c2_t10 `d251cd5d…`・enveda_v1u_t10 `3689a330…`・v3_c2nu_t10 `97f41e83…`
- 分子ごとの表と集合ごとの json の写し: `docs/research/a13-1124/`

### 1.2 結果（score,prior,frag、上位 25）
- v3_c1u_t10（394 分子）: 重複 0 件、重複のある分子 0（Wilson 95% [0, 0.0097]）、正解より上に重複がある分子 0、重複を除くと 25 位以内に入る分子 0（正解が 26 位以下の 3 分子は、重複を除いても入らない）、`to_key` が None の候補 0
- v3_c2_t10（197 分子）: 重複 0 件、重複のある分子 0（[0, 0.0191]）、正解より上の重複 0、入る分子 0。正解は 197 分子とも深い順位に無い（C2 型は正解をプールから隠している）
- enveda_v1u_t10（373 分子）: 重複 0 件、重複のある分子 0（[0, 0.0102]）、正解より上の重複 0、入る分子 0（26 位以下の 1 分子は入らない）
- v3_c2nu_t10（394 分子）: 重複 0 件、重複のある分子 0（[0, 0.0097]）、正解より上の重複 0、入る分子 0（26 位以下の 15 分子は入らない）。正解が深い順位に無い 25 分子（うち 3 分子は窓が空）は headroom.md の (i-a) 25 分子と同じ数
- 26 位以下・深い順位に無いの分子の数は、headroom.md §2.1 の (ii)・(i-a) の数（3・0・1・15、25）と同じ
- 補助（定義の外、結果の後に足した説明）: 上位 25 の候補で、run.py の dedup が使う pool の `key` と `to_key` が同じものは v3_c1u_t10 9,525/9,525、v3_c2_t10 4,781/4,781、enveda_v1u_t10 8,925/8,925、v3_c2nu_t10 7,949/7,950。run.py は pool の key で重複を除いており、その key がほぼ全部 metric の単位と同じなので、metric の単位の重複が残らない。違う 1 件（v3_c2nu_t10 の mb_00164 の 14 位、pool の key `HYLADRACZRPRLK`・`to_key` `JLLAPBWQMOZKNV`）も、上位 25 の中で別の候補と同じ答えにはなっていない

### 1.3 公開の系統と最良の提出
- 公開の系統のフルの出力（見える test、400 行が本物）は手元に**無い**。400 行の CSV のうち公開の系統（`outputs/kaggle/lb479_repro_v1`・`foldsafe5_v1`・`noeng2_v2`・`outputs/reflast`）は 388 行が同じ置き換えの行、`outputs/lb495/v1`（0.495 の fast commit）は 400 行すべて置き換えの行。400 行が本物なのは自前の縮小版・自前の ranker の出力だけ
- 補助: lb479_repro_v1 の本物の 12 行では、上位 25 の重複は 0 件（12 分子、Wilson [0, 0.2425]。見える test に正解は無いので重複の数だけ）
- いまの最良の提出（0.495 の写し、提出 58594647）は数えられない: 点が付いたのは Kaggle の隠れテストでの再実行の出力で、手元には取れない。commit の出力は 400 行すべてが同じ置き換えの行（`CO;CCO;…`）なので、候補の重複を表さない

### 1.4 読み（#230 §2 の規則どおり）
4 集合のどこでも「正解より上に重複がある分子」と「除くと 25 位以内に入る分子」がともに 0。**重複の除去は自前の構成の MRR を変えない（順位が上がる分子が無い）ので、A13 は「変更なし」で終える。** 重複を除いて次の候補を詰める変更の事前登録はしない。

限界: 数えたのは自前の縮小版（score,prior,frag）だけ。0.495 / 0.487 系統（PubChem の差し込み・Engine 1 など、pool の key 以外の経路で候補が入る）の 25 枠に重複があるかは、手元に出力が無いので分からない。

## 2. A5 の残り: 見える test と外部ライブラリの一致

### 2.1 やったこと
- headroom.md §1.2 と同じ方法（`library_coverage.py` の `cosine`、同じ付加イオン・前駆体 ±10 ppm、分子ごとの最大値）を、MassSpecGym（`data/ext/msg/MassSpecGym.tsv`・`MassSpecGym1.5.tsv` の 462,208 行）の参照に当てた: `scripts/research/a5_extlib_match.py`（主の値は a445486 で数え、c609d10 で補助の構造の重なりを足して数え直した。主の値は同じ）。数え直しは peak RSS 2.0 GB（補助なしの 1 回目は 3.6 s・1.22 GB）。ログ `logs/a5_extlib_1124.log`・`logs/a5_extlib2_1124.log`（git の対象外）
- MassSpecGym は [M+H]+ と [M+Na]+ だけ。見える test 1,213 スペクトルのうち 388 は比べる相手が無い（負イオン・ほかの付加イオン・±10 ppm に参照が無い）。比べる相手が 1 件以上ある分子は 315/400
- ExtLib（takumuhata/casmi26-extlib）は手元にピークが無い（`ext_meta.parquet` だけ）ので、スペクトルの一致は数えていない。MassBank harmonized のスペクトルも手元に無い
- 分子ごとの表: `docs/research/headroom-1123/a5_extlib_per_molecule.csv`、集計: `docs/research/headroom-1123/a5_extlib.json`

### 2.2 結果（見える test 400 分子）
- (a) MassSpecGym の参照と best_cos ≥ 0.99: **0/400 = 0、Wilson 95% [0, 0.0095]**
- (b) そのうち最良一致の参照の構造が train に無い分子: **0/400、[0, 0.0095]**
- 補助（定義どおり）: best_cos ≥ 0.7 は 3/400（0.0075、[0.0026, 0.0218]）。best_cos の中央値 0.1675
- 補助（定義の外、結果の後に足した説明）: 見える test の各分子の train の最良一致の構造（`a5_visible_test_per_molecule.csv` の `best_key`）は、MassSpecGym の構造（28,929 件）に 0/400、ExtLib の `ext_meta.parquet` の構造（151,338 件）に 0/400、MassBank harmonized の `structures_with_fingerprints.parquet`（20,204 行、構造 18,316 件）に 0/400（`a5_extlib.json` の `aux_visible_best_key_in_ext_struct`）。見える test の分子は、手元の 3 つの外部ライブラリのどれにも構造として無い。ExtLib のピークを取って数えても、同じ構造の参照との一致は起こりえない（違う構造の参照と 0.99 で一致することだけが残る）

### 2.3 読み
- 見える test の 400 分子は train（enveda 系の timsTOF）の写しらしく（headroom.md §5: 400/400）、手元の外部ライブラリには構造として無い。train 以外の参照との一致は 0
- headroom.md §1.2 の注意のとおり、見える test は training dataset の例なので、**この値も隠れテストの型の比率の推定には使えない**。隠れテストの比率は headroom.md §4 のとおり分からないまま
