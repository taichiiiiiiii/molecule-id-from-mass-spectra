# 調査: MIST と MassSpecGym（使用条件、学習データ、v3 の正解との重なり）（#66）

## 出典
1. 主催者回答 757307「Is MassSpecGym (and MIST weights trained only on it) allowed? NIST was used only for data cleaning」: `kaggle competitions topics show enveda-CASMI26-molecule-id-mass-spectra/757307`。David Healey のコメント（2026-11-11 00:20:38、CLI の表示のまま）: "MassSpecGym and MIST models are fine"
2. Zenodo 23303289「MIST: Processed Data, Synthetic Training Features, and Pretrained Models」（2026-10-20 公開、v1）: https://zenodo.org/records/23303289 、API https://zenodo.org/api/records/23303289
3. coleygroup/MIST（GitHub、README）: https://github.com/coleygroup/MIST
4. MassSpecGym の HF データセットカード: https://huggingface.co/datasets/roman-bushuiev/MassSpecGym （API https://huggingface.co/api/datasets/roman-bushuiev/MassSpecGym）
5. MIST の論文: Goldman et al., Nature Machine Intelligence 5, 965–979 (2023), https://doi.org/10.1038/s42256-023-00708-3
6. MassSpecGym の論文: Bushuiev et al., NeurIPS 2024 Datasets and Benchmarks（HF カードの引用）

## 要点
- 757307 の質問の内容（質問者 ujihash の記述で、こちらで確かめていない部分を含む）: MIST の `mist_msg_noaug.ckpt` は MassSpecGym の実スペクトルだけで学習。MSG は MoNA・MassBank・GNPS・MSnLib 由来で、NIST はデータの掃除（NIST20 と合わない約 24,000 スペクトルの除去）にだけ使った。主催者の答えは「MassSpecGym と MIST のモデルは問題ない」（質問の 2 つ目「この競技の gnps に GNPS-NIST14-MATCHES は入っているか」への答えは無い）
- Zenodo 23303289 の内容: MSG の compact 特徴は train 194,119 / val 19,429 / test 17,556 行。チェックポイントは 4 本（`mist_msg_noaug.ckpt` = MSG の実スペクトルだけ、289.2 MB、md5 80594e4d…）。「noaug」は合成スペクトルを使わないという意味で、ピークと強度の確率的な揺らしは使う
- 学習分割（GitHub README の記述）: 「Validation always uses real held-out spectra. EMA checkpoints are selected by validation fingerprint Tanimoto; the test set is not read during training.」→ 学習側は MSG の train（勾配）と val（checkpoint の選択）。test は読まない（作者の主張。こちらで学習ログは確かめていない）。Zenodo の記録そのものには分割の記述は無い
- 再学習: `scripts/train_recipe.sh msg-noaug`（GitHub README）と Zenodo の `msg_compact.tar.gz`（15.7 MB）で重みを作り直せる、と作者は書く（こちらで再学習はしていない）

## ライセンス
- MIST の重みとデータ（Zenodo 23303289）: **MIT**（Zenodo API の `metadata.license.id = mit-license`）
- MIST のコード（coleygroup/MIST）: **MIT**（README「Code is distributed under the MIT license」、LICENSE.md）
- MassSpecGym（HF）: **MIT**（HF API の `cardData.license = mit`、タグ `license:mit`）
- MSG の元のライブラリ（MoNA、MassBank、GNPS、MSnLib）それぞれのライセンスがどう引き継がれるかは**未確認**
- この競技での使用: 757307 で可（上の要点）

## 重なりの測定（`scripts/research/msg_overlap.py` → `outputs/research/msg_overlap.json`）
- 入力: `data/ext/msg/MassSpecGym.tsv`（262,334,768 バイト、231,104 行、sha256 0c9cc504…）と `MassSpecGym1.5.tsv`（261,197,365 バイト、231,104 行、sha256 50cfdd1d…）。HF の datasets-server の parquet は `val` の 1 ファイル（133 MB）しか無く、構造の列だけを取る軽い経路にならなかったので tsv を取った。キーは `inchikey` 列（14 文字）。RDKit は使っていない
- 構造の数（2 版とも同じ、和集合も同じ）: train 22,746、val 3,185、test 2,998
- v3 の正解との重なり（和集合、2 版とも同じ値）:
  - v3_c1（200）: train 1、val 0、test 0（0.5%）
  - v3_c2（200）: 0
  - enveda_v1（202）: 0
  - v3_c2n（200）: train 96、val 10、test 11、計 117（58.5%）
  - split hold（55,102）: train 4,435、val 616、test 614、計 5,665（10.3%）
  - split np（265）: train 217、val 12、test 10、計 239（90.2%）
- 事前に決めた規則（重なりを見る前に Issue #66 と計画に固定）: 学習側 = MSG の train ∪ val（1.0 と 1.5 の和集合）。学習側と重なる構造を除いた残りが 100 分子未満の集合は「測定不能」で、4 集合が揃わなければ採用しない
  - 残り: v3_c1 199、v3_c2 200、enveda_v1 202、**v3_c2n 94（測定不能）**
  - 保守的な側（MSG 全行を学習側とみなす）: v3_c1 199、v3_c2 200、enveda_v1 202、v3_c2n 83
- 結論: **今の v3 では MIST msg-noaug をリーク無しに 4 集合すべてで測れない**（v3_c2n が測定不能）。規則により、この形では採用の判定ができない。v3_c2n は NP パネル寄りの集合で、MSG に NP が多く入っていることと合う（split np の 90.2% も MSG にある）

## この競技で使えるか
- 規則上: 可（757307）。ライセンス: 重み・コード・MSG とも MIT
- 再現性: 重みは公開、学習コードと学習用の特徴も公開（作者の記述。こちらで再学習は未確認）。最終提出に使うなら、Zenodo の md5 を固定して読む
- 測定: 上の結論のとおり、今の v3 の 4 集合では採否を決められない

## 試すなら（案。今は実行しない）
- 変更: MIST msg-noaug の指紋予測を、縮小版の新しい証拠チャネルとして等重みの z 合成で足す（#60 の sanpier と同じ形。重みは調整しない）
- リークの扱い（固定済み）: 各集合から学習側（MSG の train ∪ val、1.0 と 1.5 の和集合）と重なる構造を除いた部分集合で測る。残り 100 未満の集合は測定不能
- 採用規則: #48 と同じ。v3_c1・v3_c2 を含む 4 集合すべてで (新 − 対照) の 95% CI 下限 ≥ −0.02、1 つ以上で下限 > 0。**v3_c2n が測定不能なので、今の v3 のままでは採用できない**。進めるには、MSG と重ならない NP の集合を v3_c2n と同じ作り方で作り直すこと（別 Issue、作り方を事前に決めてから）が先に要る
- 費用の見込み（未測定）: チェックポイント 289.2 MB のダウンロード。MIST は入力に分子式の割り当て（部分式のラベル）を使う設計で（論文の chemical formula transformer）、v3 のスペクトルに対してその前処理をどう作るかは**未確認**。CPU 推論で 800 分子程度なら数十分以内と見込むが、測っていない。Kaggle GPU は不要の見込み
