# 指紋予測と検索のずれ（arXiv 2602.16507v2）と MSAlign（arXiv 2605.19752v3）

Issue #56。計画 2026-11-11 項目 3。本文は HTML 版を読んだ（2026-11-11、独立の読み手が全文を通読し、表・節番号つきで抜き出したものをここにまとめた）。数値はすべて論文の値で、こちらでは再現していない。

## 出典
- A: De Waele et al., "On objective mismatch in molecular retrieval from tandem mass spectrometry", NeurIPS 2026。https://arxiv.org/abs/2602.16507v2 （DOI 10.48550/arXiv.2602.16507、CC BY 4.0）。コード https://github.com/gdewael/ms-mole （MIT）
- B: Krzakala et al., "MSAlign: Aligning Molecule and Mass Spectra Representations for Metabolite Identification"。https://arxiv.org/abs/2605.19752v3 （DOI 10.48550/arXiv.2605.19752、CC BY 4.0）。コード https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026 （MIT）
- Enveda-180: Zenodo https://zenodo.org/records/20436851 （v1.0.0、CC BY 4.0）、https://github.com/enveda/enveda-180 （MIT）。A の本文の文献 [21] には URL が無く、これは読み手が別に見つけた配布元

## A の要点
- 主張: 指紋の当たり（Tanimoto）を良くする損失と、検索（HR@k）を良くする損失は一致しないことがあり、両者はパレート前線を作る（図 1、§4）。理論では、候補どうしの類似度の幅（σ_min〜σ_max）が広いほどずれが大きくなる上界を示す（定理 3.2・3.3）
- Enveda-180（§4）: 184,330 分子。分割は MCES 距離の単連結クラスタリング（適応しきい値 max(3, 0.1·min 重原子数)）で、テスト分子は学習分子とほぼ同一のものを含まない（図 3）。候補は PubChem の同質量（±10 ppm）から最大 256。装置は本文に書かれていない（第三者のミラーには timsTOF Pro 2 とあるが、未確認）
- 表 1（Enveda-180、同質量候補、Transformer、HR@1 / HR@5 / HR@20 / Tanimoto、%）
  - BCE: 53.01 / 68.20 / 79.28 / 32.49
  - IoU: 43.14 / 57.02 / 69.88 / **34.38**（指紋の当たりは最良、検索は最低）
  - 対照損失 Fp-Cos: **63.11** / 81.04 / 89.97 /（τ を変えたモデルで 4.64）
  - 対照損失 Emb-Cos: 63.05 / **82.54** / **92.98** / N/A
  - MRR は本文に無い
- 提言（§5、App. C）: 検索が目的なら、候補のリストに対する対照（listwise、InfoNCE 型）損失を使う。対照損失は HR@1 の滑らかな代理になる
- 学習の要らない変更として本文にあるもの
  - 採点の類似度を Tanimoto から cosine に替えると、どの損失のモデルでも良くなる（表 6、例: IoU モデルの HR@1 は cosine 4.29 / Tanimoto 3.86、MassSpecGym）
  - 指紋の種類（MAP4 が有望、表 2・表 8）
  - 予測確率から候補ごとの期待類似度を計算して並べ替える採点法は、本文に**無い**（式 5 は理論の枠組みで、実験は無い）
- 候補集合（App. H、表 9）: 同質量候補で学習したモデルは、同組成式候補での評価でも良い

## B の要点
- 凍結した DreaMS（スペクトル側）と MolDeBERTa（分子側）の埋め込みを、候補ベースの InfoNCE で整列させる軽い MLP（§3、表 8）
- 表 3（R@1 / R@5 / R@20、%）: MassSpecGym 組成式分割で MSAlign 47.5 / 71.3 / 87.0、スコア融合版 58.2 / 80.7 / 92.8。MCES 分割では MSAlign 19.0 / 41.2 / 64.9、融合版 30.8 / 59.3 / 82.0
- スコアの融合（§3 式 7・8、App. B.2）: 4 つの整列モデルの cosine と 3 つの質量スコアの凸結合。重みは検証の R@1 で選ぶ（式 23・24）
- 損失の比較（表 5、R@1）: 候補ベースの InfoNCE 47.5 ／ バッチ内 InfoNCE 19.5 ／ 回帰 16.7
- 学習データは MassSpecGym と Spectraverse（表 7）。NIST は本文に書かれていない
- 配布: MSAlign の重みは Zenodo 23372807（CC BY 4.0、MassSpecGym の 2 分割）。基盤モデルの重みは再配布していない（DATA.md）
- 凍結モデルのライセンス（本文に無く、配布元で確認）
  - DreaMS: コード MIT、重み Zenodo 10997887 は CC BY 4.0
  - MolDeBERTa（HF SaeedLab/MolDeBERTa-base-123M-mtr）: **CC BY-NC-ND 4.0**（非商用・改変禁止）
- 推論の重さは本文に無い。README では CUDA GPU が要る

## この競技で使えるか
- A の結論は、今日の項目 2（sanpier キット）の後押しになる。sanpier キットは、同組成式・近い質量の難しい負例を入れた候補ベースの InfoNCE（温度 0.1）で学習している（キットの train.py）。これは A の「対照損失は検索に強い」と同じ型。一方、縮小版の土台 FPNet は指紋を BCE で学習し、`fp[bits] @ z` で採点している。A の表 1 では、BCE（HR@1 53.01）から対照損失（63.11）への差は約 10 ポイントある。ただし A は MRR を報告していない。また、A の分割と候補（PubChem 同質量 256）はこの競技の pool とは違う。なので、これを v3 での改善量の見込みには使わない
- MSAlign: コードは MIT だが、MolDeBERTa は CC BY-NC-ND 4.0。競技の規則（754610: 最終提出は端から端まで再現できること）に加えて、非商用ライセンスのモデルを提出に使ってよいかどうかは**未確認**。MSAlign の配布重みは MassSpecGym で学習したもので、この競技の train と重なるかどうかも未確認。融合の重みを検証の R@1 で選ぶやり方は、この競技のハイパラ調整の禁止に当たるので、写さない
- Enveda-180: 主催者 Enveda のデータで、隠れテストと近い可能性がある。この競技では既に `enveda_v1` 検証集合（#14）の元として使っている。学習に使ってよいかどうかは、外部データの規則としては未確認

## v3 で測る案（書くだけ、実行しない）
- 案: 縮小版の score 段の採点式を、`fp[bits] @ z`（ロジットと指紋の内積）から `cos(σ(z), fp)`（予測確率と候補指紋の cosine）に替える。学習は要らない。A の表 6 の「cosine の方が良い」は二値化した予測指紋と候補の比較なので、ロジットの内積を cosine に替えたときに同じことが起きるかは分からない
- 事前登録すること: 替えるのは採点式 1 つだけ。prior（μ 0.15）と frag（λ 0.5）はそのまま。新しい定数は作らない
- 対照は score+prior+frag（v3_c1 0.8535 / v3_c2 0.0117 / enveda_v1 0.8376 / v3_c2n 0.6284）
- 採用規則（#48 と同じ）: 「新 − 対照」の 95% 区間（分子ブートストラップ 2,000 回）の下限が、v3_c1・v3_c2・enveda_v1・v3_c2n の 4 集合すべてで ≥ −0.02、かつ少なくとも 1 集合で下限 > 0。v3_c1 と v3_c2 の両方の判定を書く。集合が欠けたら採用しない
- 時間・資源: 実装とテストで 1 時間、測定は CPU で 4 集合 約 5 分（frag の測定は 4 集合で約 3 分、最大 1.81 GB だった）。メモリ 2 GB 以下、ディスク 10 MB 以下
- 期待は小さい。縮小版の上での改善は、0.479 を超える根拠にならない（縮小版の score+prior は LB 0.282、#37）
