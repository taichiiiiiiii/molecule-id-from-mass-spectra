# #278 Kaggle の時間測定 版 2（PubChem 層の採点を並列にした版）の結果と関門の判定

- カーネル: `taichiiiii/casmi26-e2e-timing-274` 版 2（CPU、提出しない）。状態 **COMPLETE**（poller の `DONE COMPLETE`、13:48:05 JST）
- 埋め込みのコードの commit: `608c9b75ee1b536dc74aadf8d677da3a4e8e7630`（timing.json の `code_commit` と一致）
- 数値の元: `e2e_timing_278_v2.json`（同じフォルダ）。カーネルの出力は `outputs/local_eval/e2e_timing/v2/out/`（git の外、sha256 は json に）

## 段ごとの秒（見える test 400 分子）
- prepare 1.17 / logits 137.31 / pubchem 3,258.8 / qrows 32.33 / features 656.97 / cc 62.52 / rank 0.4、合計 4,149.5 s
- 一致の確かめの段（pubchem_check）1,109.8 s は合計に入れない（事前の決めどおり）。それを含む wall は 5,259.6 s。ノートの setup 35.72 s はどちらにも入っていない（#274 と同じ数え方）

## 関門（1,200 分子への外挿、段の秒 × 3）
- prepare 3.5 / logits 411.9 / pubchem 9,776.4 / qrows 97.0 / features 1,970.9 / cc 187.6 / rank 1.2、**合計 12,448.5 s（3.46 h）**
- **関門 16,200 s を満たす**（超えた段なし、余裕 3,751.5 s = 関門の 23.2%）
- 参考（関門の判定には使わない）: 確かめの段が先頭 40 分子の窓のページを温めた分の楽観を打ち消す上側として、並列の確かめの 322.74 s × 3 を足すと 13,416.7 s、setup を足すと 12,484.2 s。どちらも 16,200 s 未満

## worker の数（事前に固定した式）
- `MemAvailable` 31,302,795,264 B（29.15 GiB）、`cpu_count` 4、共有 18,454,937 B、専用 840,501,240 B
- `floor(0.8 × (MemAvailable − 共有) / 専用)` = 29 → `min(cpu_count, 29)` = **workers 4**（mode auto。mem_available は null でない）

## 同じセッションの 40 分子の直列と並列の一致
- 直列（workers=1）787.06 s、並列（workers=4）322.74 s（2.44 倍）
- 候補の鍵・SMILES の順・f·z の値（float の `==`）が 40/40 分子で**完全一致**（`identical: true`、`first_diff: null`）
- 本番の段（400 分子・並列）の先頭 40 分子も直列と完全一致（`main_vs_serial.identical: true`）
- macOS では fork の先で Accelerate の BLAS が落ちた（#278）が、Kaggle（Linux）では `VECLIB_MAXIMUM_THREADS` 無しで通った

## PubChem の段の速さ
- 8.147 s/分子（版 1 の直列は 12.798 s/分子）→ 版 1 の 0.637 倍
- features の段は版 1 の 408.0 s に対して 656.97 s（並列にしていない段なので、セッションの機械の違いと見る。原因は確かめていない）

## submission.csv の点検
- 400 行、id の集合と順序が sample_submission と同じ、重複なし、各行 25 件（1〜25 の範囲）、空の SMILES 0、行の中の重複 0、CCO の予備 0（LambdaRank の並びが 400）
- `scripts/validate/mrr25.py` の `validate_submission(rows, expected_ids)` が通る
- sha256 `6fcc3353…`。このモデルは dev の全行で学習した時間測定用の 1 本なので、中身の点は評価しない

## 版 1 との比べ（事前の基準ではない、参考）
- 版 1 の submission（sha256 `7cb2df24…`）と 3 行が違う（m_8071e1 の 4・11 位、m_9e99f7 の 4 位、m_b047da の 11 位）。どれも `mrr25.to_key` が同じで、同じ構造の SMILES の書き方（`[CH]` を明示するか）だけが違う。残りの文字列はバイト一致なので、400 分子すべてで鍵の列は同じ、1 位も 400/400 同じ
- 別のセッション（別の機械）の比べなので、この差の原因（同じ鍵の行のどれを選ぶか）はここでは確かめていない

## 提出
- 0 回（`kaggle competitions submissions` の最新は 2026-11-22 の 58699228 のまま）
