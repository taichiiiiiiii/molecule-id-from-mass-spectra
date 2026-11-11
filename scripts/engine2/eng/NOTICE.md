# scripts/engine2/eng/ の出典

この 3 本は第三者のコードを**無改変で**置いたものです（#72）。自前の表の生成（`scripts/engine2/build_tables.py`）が、推論と同じ列の定義・候補・特徴の関数を使うためです。

| ファイル | sha256 | 中身 |
|---|---|---|
| `casmi_engine.py` | e461349b37f5097fa2c2e52cedc0716e6fa26567513bab0bb3841cc37e1d5b4f | Engine 2 本体（ライブラリ・候補プール・チャネル・MetFrag-lite・特徴・ランカー） |
| `pv.py` | ac2e7c527f327cfe2acdef6c1a6128a0a360f1f91d5e3e0c8ade338a4a9078f6 | prvsiyan の analog-propagation の部品（見出しに「public notebook, version of 2026-09-16」） |
| `pv_fp.py` | 86a760b10a75f3b527a75fca8533e030783c3fd542daaca4711e051734914bcd | prvsiyan の FPNet（推論用、torch） |

## どこから取ったか
- 0.487 写しのカーネル `taichiiiii/casmi26-lb487-copy`（提出 58208946、Public 0.487）のセル 12 の `FROZEN_B64` を復号した凍結コードの、セル 32 の `ENG_FILES` 辞書の値（2026-11-11 に復号、`workspace/casmi_daily/ref/eng_files/`）。sha256 は上の値で、`casmi_engine.py` はローカルの写し `outputs/holdout_eval/np_v1/eng/casmi_engine.py` とも一致
- 0.487 写しの原本: alexchilton の公開ノート https://www.kaggle.com/code/alexchilton/casmi26-lb-0-487-peaks-to-structures-scorer
- `pv.py` の部品の元: prvsiyan の公開ノート https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline（2026-09-16 版。この版は `kernels pull` で取れず、照合していない）
- `casmi_engine.py` の作者: 0.479/0.487 系統のノートに組み込まれた「Our engine (two-ranker + BIO + AFIX)」。コード中のローカルパス（`C:\Users\HW-LEE\Desktop\CASMI`）以外に作者名の記載は見つけていない

## ライセンス
- **未確認**。Kaggle の公開ノートのライセンス表示を、上の 2 つのノートについて確かめていない
- 生成コード（`build_tables.py`・`base_fs.py`）からは import して呼ぶだけで、この 3 本は書き換えていない（sha256 で確認できる）
