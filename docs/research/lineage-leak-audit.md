# 0.487 系統の fold-safe 評価: 漏れの監査（1 日目、#114）

設計 `docs/design/lineage487-foldsafe-eval.md`（#112）§4 の 1〜12 と §2.1 の判定。生の出力は `outputs/research/lineage_leak_audit.json`（追跡しない）と、分子ごとの最近傍 Tanimoto `outputs/research/lineage_leak_audit_tanimoto.csv`。

    .venv/bin/python scripts/cv/lineage_leak_audit.py      # 約 40 秒、RSS 最大 1.47 GB（2026-11-14 に手元で測った。レビュー 1 回目の修正の後）

学習・推論・Kaggle の push はしていない。Kaggle からは小さいファイル 7 本を読むだけで取ってきた（`data/ext/lineage_audit/`、置き場はスクリプトの docstring）。

## 数（分母は truth.csv: v3_c1 200、v3_c2 200、enveda_v1 200、v3_c2n 195）

| # | 対象 | v3_c1 | v3_c2 | enveda_v1 | v3_c2n | 備考 |
|---|---|---:|---:|---:|---:|---|
| 1 | fold-safe split の hold | 200 | 200 | 200 | 195 | 全部 hold（assert 通過）。np とは 0 |
| 2 | #54 の除外（55,369） | 200 | 200 | 200 | 195 | 除外の数は 55,369 で期待どおり |
| 3 | #74 の試走のクエリ（48） | 0 | 0 | 0 | 0 | 本番の表は手元に無い（not_computable） |
| 4 | ranker_v4b の模擬クエリの構造（5,350） | 7 | 3 | 4 | 13 | 作者の split（sha256 一致）の np・nplib・syn・plusk |
| 5 | fragnet の学習構造（近似: 作者の train fold 270,304） | 193 | 197 | 196 | 182 | 近似。analog_struct・frag2・derivation・stageA/B は手元に無い |
| 6 | MSG 1.0 ∪ 1.5（どれかの fold） | 1 | 0 | 0 | 116 | §2-10 の値を再現。採点に残るのは 199・200・200・79 |
| 7 | ExtLib のスペクトル行（ext_meta） | 0 | 0 | 0 | 18（MSG で外したあと 7） | 構造表（ext_fp_keys）とは 0 |
| 7 | MassBank harmonized の構造 | 0 | 0 | 0 | 44（同 12） | 上限（is_casmi_subset で落とす前） |
| 7 | 知識表の外部参照 | 0 | 0 | 0 | 5（同 2） | |
| 7 | ChEBI/LMSD・PubChemLite | 0 / 0 | 0 / 0 | 0 / 1 | 111 / 96 | COCONUT・hengck23 は手元に無い |
| 8 | in_train_structure の印で F c36 が参照に足す行 | 0 | 0 | 0 | 5 行・4 分子（同 2） | v3_c2n は hide_rule = all なので食い違い（パッチの対象） |
| 9 | train_structs.parquet（パッチ前） | 200 | 200 | 200 | 195 | sha256 は pin と一致 |
| 9 | 同（hide_rule を今の drop_keys.txt のまま当てた train から作り直した後。正解の ik14 と scorer のキーで数える） | 200 | 0 | 0 | **8** | v3_c1 は公開ライブラリの参照が残るのが本来の設計。v3_c2n の 8 は tautomer の別名の行（食い違い (d)） |
| 9 | 同（drop を scorer のキーでも当てた場合） | 200 | 0 | 0 | 0 | drop_keys を作り直した後の値（#116） |
| 9b | 縮小版（scripts/local_eval）で別名の行に届く分子 / プールに残る正解 | 0 / — | 0 / **3** | 0 / — | 0 / — | 別名の ik14 は pool key に無いので load_refs は読まない。v3_c2 の 3 は #92（pool=remove の集合だけ数える） |
| 10 | train_fp_sel.npy | — | — | — | — | 手元に無い。行 = sid と推定（train_structs の sid は行番号と一致） |
| 11 | ランカーの 160 特徴のうち train 由来の表に依存 | | | | | 56（名前からの推定）、学習済みの族（fragnet・model_views）21 |
| 12 | 最近傍 Tanimoto の中央値（残った train 220,443 構造） | 0.634 | 0.650 | 0.647 | 0.727 | ≥ 0.9 は 1・2・2・32 分子 |

## §2.1 の判定（事前登録どおり）
- 判定は設計 §2.1 の文のとおり: 「どれか 1 集合でも c > 5%、または fragnet の学習構造の一覧を作れない → 使わない」。前の版は「どれかの成分の一覧が無ければ使わない」で、規則より厳しかった（レビュー 1 回目の指摘 2）。直した版では analog_struct・frag2・derivation の一覧が無いことは判定に入れず、c が下限であること（`c_is_lower_bound`）だけ書く
- パッチの後の c: 4 集合とも 1.0（fragnet の近似と模擬クエリの和） → 1 つ目の条件で **Tier 0 は較正にも採否にも使わない**。fragnet の一覧は作者の train fold での近似だけなので、2 つ目の条件でも「使わない」になる
- v3_c1（hide_rule = enveda_libs）は公開ライブラリの参照を残す設計なので、9〜10（train_structs に正解が載る）は c の成分から外した（隠れテストの「train に参照あり」と同じで漏れではない）。外さないと v3_c1 は 9〜10 だけで自動的に c = 1.0 になる（指摘 3）。外しても fragnet の近似で 1.0
- drop を scorer のキーでも当てた場合（`post_patch_scorer_key_drop`）も判定は同じ
- パッチの前の c（記録だけ）: 4 集合とも 1.0
- 感度（判定には使わない）: fragnet の近似を使わず模擬クエリだけでも、v3_c2n は 13 / 195 = 6.7%（MSG で外したあと 4 / 79 = 5.1%）で 5% を超える

## 設計との食い違い
- (a) §3.1 の「4 集合は構造で重ならない」は成り立たない: v3_c1 と enveda_v1 で 1 構造（VHNSOMLGZMGWNH）、v3_c2 と enveda_v1 で 2 構造（LBLAVYDUPXGKQX、MUAYFYDPOHQAJL）が重なる。データセット A（v3_c1 + enveda_v1）では、この構造に 2 つの hide_rule が当たる。enveda_v1 の rule = all が、v3_c1 の 1 分子（m_00159）の公開ライブラリの参照 8 行まで落とす（v3_c1 の設計では残すはずの行）。v3_c2 と enveda_v1 の 2 構造は別のデータセット（B と A）なので同じ train には当たらない
- (b) §2-15 の「ExtLib は v3 の正解を 1 位に上げられない」は構造表についてだけ正しい。スペクトル行（ext_meta）には v3_c2n の正解が 18 構造（MSG で外したあと 7）ある
- (c) §4-8: 知識表の外部参照は、v3_c2n の正解 4 分子について hide_rule = all に反して参照を戻す（F c36 85 行の条件を当てた数。付加イオン・モードの条件は数えていない）
- (d) §4-9・hide_rule: `drop_keys.txt` は truth の ik14（正準化なし）だけなので、同じ構造が train に別の ik14（tautomer の別名。train_structs.key = `mrr25.to_key(smiles)` = 正解のキー）で載っている行は隠れない（#92 と同じ種類の穴）。v3_c2n の **8 分子**（m_00017・m_00040・m_00059・m_00060・m_00061・m_00067・m_00071・m_00083。8 つとも MSG と重ならないので、採点に残る 79 分子の 10.1%）で、train.parquet に **97 行**が残る（gnps 31・pluskal_ms2 22・riken 21・mona 18・msdial 5）。例: CUAPOUHDJKWLOB → LJDAYEAXTWNKEL（49 行）、HKDLNTKNLJPAIY → IGMLRLGLIXZKQZ（23 行）。他の 3 集合は 0。前の版の §4-9 のパッチ後の数は ik14 ∈ drop_keys だけを見ていたので 0 と出ていた（レビュー 1 回目の指摘 1）
  - 0.487 系統の評価: 別名の行は hide_rule の後のライブラリに残り、train_structs の key で正解の構造に写る。2 日目のパッチ（§3.1 の 1、9〜10 の作り直し）は scorer のキーでも落とす
  - 既にある縮小版（`scripts/local_eval`）の v3_c2n の評価にも同じ穴がある: hide_rule = all を当てた後のライブラリ（train.parquet）に別名の参照 97 行が残る。ただし `library.load_refs` は候補の pool key と inchikey14 が等しい行しか読まず、8 つの別名の ik14 はどれも pool key ではないので、今の lib・reflast の段には届いていない（届く分子 0、`refs_top60.parquet` の中も 0）。プール側では pool = remove の v3_c2 で正解 3 分子が残る（#92 の m_00010・m_00020・m_00112）。v3_c1 は scorer のキーが drop_keys に無い正解が 2 あるが、train に別名の行は無く pool = keep
  - drop_keys を正準キーで作り直すのは別の Issue（#116）。§2.1 の判定は変わらない
