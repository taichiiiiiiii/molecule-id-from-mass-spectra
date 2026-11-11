# 主催者への質問（756893 の後半）— 投稿用の下書き

投稿先: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/756893 へのコメント（または新しいトピック）
投稿者: オーナー（ブラウザでログインして投稿する。Kaggle CLI には投稿機能が無い）
状態: 下書き（未投稿）。オーナー許可 2026-11-11（#3 コメント 6796064117）

---

Thank you, David, for the clear answer on the first half.

A follow-up on the second half of the original question, since it decides how we prepare our final submission:

If a team regenerates such a table itself — with its own generator code published together with the solution (same column definitions as the inference code, a documented query list and seed, and only competition data plus already-permitted public datasets as inputs) — and the solution loads that self-generated table instead of the shipped one, is that considered reproducible? The rebuilt table would not be byte-identical to the shipped one, but it would be exactly reproducible from the published generator.

Put differently: is the requirement "the artifact can be regenerated from code that is part of (or published with) the solution", rather than "the artifact matches the original upstream file"?

Thanks again for keeping the rules clear before the deadline.

---

## 日本語の要旨（記録用）
- 前半（作者が生成コードを公開していない表をそのまま読むのは再現可能ではない）は回答済み（David Healey、2026-11-11 00:16 UTC）
- 後半: 自前の生成コードを解と一緒に公開し、その生成器から作った表を読むなら、元の表とバイト一致しなくても再現可能と認められるか
- 判断への影響: 認められれば #64 の「Engine 2 の 2 表を自前で作り直した 0.487 系統」を最終候補にできる。認められなければ、Engine 2 型の部品は推論時に生成するか、外す
