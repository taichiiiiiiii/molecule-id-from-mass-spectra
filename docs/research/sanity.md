# 健全性の点検 A2(a): 0.495 の床の再実行と commit の出力の一致（Issue #212）

#195 A2(a)。オーナーの承認 2026-11-23T14:01Z (2)。比べ方は Issue #212 の本文と、version 2 の出力を見る前に書いた追記コメント（2026-11-23T15:15:12Z。version 2 の COMPLETE は 15:21:04Z）で固定した。

## 何をしたか
- 対象: `taichiiiii/casmi26-lb495-floor`（#164 の床。Public 0.495 の提出 58594647 は version 1 から）
- version 2 は version 1 と **同じノート** を push した（`git show origin/feat/164-lb495-floor:kernels/lb495/…` で取り出したもの。notebook.ipynb の sha256 `82a14431a2b4c59d9fe0f3d1423519747f9a77d8f8e2d5eb010aab7a6610152a`、kernel-metadata.json `000a1a61…`）。push の後に pull したノートも `82a14431…` で同じ。**ノートの変更 0 バイト、提出 0**（`kaggle competitions submissions` の最新は 11/22 の 58699228 のまま）
- push: 2026-11-23T15:13:49Z「Kernel version 2 successfully pushed」→ 見張り（PPID 1）が 15:21:04Z に COMPLETE を見た
- version 1 の出力は push の前に取った（18 ファイル）。version 2 の出力も同じ正規表現 `\.(csv|json|log)$` で 18 ファイル
- 比べたスクリプト: `scripts/research/commit_output_compare.py --dirs`（単体テスト `tests/test_commit_output_compare.py`）。結果の JSON: `docs/research/sanity_a2a_compare_v1_v2.json`

## GPU と実行時間（受け入れ条件 (a)）
- push の前の `kaggle quota`: GPU used 18.64 h、**remaining 11.36 h**（refreshAt 2026-11-28T00:00Z）
- version 1 の実行時間: ログの最後の行が開始から 288.5 s。version 2: 363.0 s
- 実行後の `kaggle quota`: used 18.75 h、remaining 11.25 h（この 1 回で約 0.11 h）

## 実行環境
- 両方とも metadata は `docker_image` = `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…d461`、`machine_shape` = NvidiaTeslaT4（version 2 は push 後の pull で確認。version 1 は #164 の記録で同じ metadata）
- ログに GPU の名前は出ていない。分かるのは P5 の Engine 1 の `dev` = cuda（両方）と、公式の指標用 RDKit 2026.03.3（両方）まで

## 結果
**まず大事な点**: commit の実行は「fast commit」で、`submission.csv` の 400 行は **すべて同じ置き換えの行**（`CO;CCO;…`、25 個のアルコール）。ログも「Fast commit: unique dummy SMILES written cleanly; scoring rerun runs full pipeline.」。このため submission.csv の一致は自明で、提出に使われる本当のパイプライン（再実行で隠れテストを処理する部分）の決定性はこの比較では分からない。commit の中で本当に計算しているのは、P5 の Engine 1 の回し直し（12 分子）とその選び、fr3to25 の dry run（20 分子）、apex-extlib の読み込みだけ。

1. ファイルの sha256（18 ファイル）: 同じ 11、違う 7
   - 同じ: `submission.csv` と他の submission 系 csv 5 つ（すべて `e00a3a7c…`）、`bootstrap.log`、`merge_report.json`、`v8_report.json`、`ours_bundle/config.json`、`ours_bundle/config_used.json`
   - 違う 7 つの中身: `apex_extlib_audit.json`・`n035net_audit.json` は `elapsed_s` だけ、`fr3to25_audit.json` は `load_s`・`total_s` と一時ディレクトリの名前（`model.code_dir`）だけ、`p5tail_audit.json` と `p5tail_e1.json` は時間の項目（`wall_s`・`timings`・`total_s`）だけ、`p5tail_runner.log` は各セルの秒数とオブジェクトのアドレスだけ、カーネルのログは時刻を含むので違う
2. `submission.csv`: 行数 400 / 400、molecule_id の集合と行の順も同じ、候補の列が完全に同じ分子 **400 / 400**（置き換えの行が 400 / 400 で、本当に処理した分子は 0）
3. `p5tail_e1.json`（Engine 1 の回し直し、12 分子、cuda）: 分子の集合が同じ、分子ごとの記録（base・p6・p5・n_tr・gen・called）が完全に同じ分子 **12 / 12**
4. `p5tail_audit.json`: 時間の項目を除くと完全に同じ（選びの 49 件の挿入なども同じ）。`fr3to25_audit.json`: 時間の項目を除いて違うのは一時ディレクトリの名前だけ
5. 違う分子が 0 なので、1 位の一致率・順位の差は計算する対象がない
6. 実行時間は揺れた: P5 の回し直し 212.6 s → 289.8 s、全体 288.5 s → 363.0 s

## 読み
- commit の範囲では、時間と一時パスを除いて **出力は決定的**（同じ T4 の構成、同じ docker_image）
- ただし commit の submission.csv は置き換えの行だけなので、**提出した csv（隠れテストでの再実行）の再現性は確かめられていない**。確かめられたのは、commit で走る部分（Engine 1 の 12 分子の回し直し・選び・fr3to25 の dry run）が 2 回で同じ答えを出すことまで
- 実行時間は 1 回で 3 割ほど揺れた（P5: +36%）。再実行の時間の余裕を見積もるときは、この揺れを入れる
- この結果で定数は動かしていない

## 残ること（この文書の範囲の外）
- 提出した csv は 403 で取り出せない（日報 Q2）。隠れテストの再実行の決定性を見るには、fast commit を切った版（見える test 400 分子を全部処理する）を別の slug で回すしかない。GPU の時間が要るので、やるかどうかは計画で決める（このカードではしない）
- #195 A2(b)(c) は B11 と一緒に行う
