# 主催者への質問（756893 の後半）— 投稿用の下書き

投稿先: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/756893 への**コメント**（新しいトピックにはしない。「these tables」などが指すものを元のスレッドで読めるようにするため）
投稿者: オーナー（ブラウザでログインして投稿する。Kaggle CLI には投稿機能が無い）
状態: 下書き（未投稿）。オーナー許可 2026-11-11（#3 コメント 6796064117）

---

Thank you, David, for the clear answer.

Following up on Shishir's question above (I am a different team, asking because it decides how we prepare our final submission): the second half of the question — whether a rebuilt table is acceptable — is still open.

Concretely, for the two tables discussed here (`rank_train.npz` from prvsiyan/casmi26-ranker-features and `sim_rank_rows_nofp.npz` from megayak/casmi26-simulated-ranker-rows): if a team generates such a table itself, with its own generator code published together with the solution — producing the feature columns that the solution's own inference code expects, with a documented query list and random seed, and using only the competition data plus publicly available datasets allowed under the competition rules (including public model weights whose training code is itself published) — and the solution loads that self-generated table instead of the shipped one, is that considered reproducible? The table would be built to the team's own design rather than replicating the original (for example a different query set), so it would not match the shipped file; it would be reproducible from the published generator with the same code, data and seed.

Put differently: is the requirement "the artifact can be regenerated from code that is part of (or published with) the solution", rather than "the artifact matches the original upstream file"?

Thanks again for keeping the rules clear before the deadline.

---

## 日本語の要旨（記録用）
- 回答済みの部分（原文、David Healey、2026-11-11 00:16 UTC）: "Solutions that load intermediate data artifacts without being able to reproduce those artifacts are not reproducible."
  - こちらの解釈: 判定の基準は「その中間物を再現できるか」。生成コードの無い表をそのまま読む解はこれに当たる。後半も「再現できれば可」と読める余地があり、英文の最後の段はその確認
- 聞くこと: 自前の生成コードを解と一緒に公開し、その生成器から作った表を読むなら、元の表とバイト一致しなくても再現可能と認められるか
- 元の質問からの言い換え（承知の上）: 元の後半は「作者の公開設計どおりに作り直す」だったが、ここでは「自前の生成器を解と一緒に公開する」に広げている（#64 の方針に合わせた）
- 判断への影響: 認められれば #64 の「Engine 2 の 2 表を自前で作り直した 0.487 系統」を最終候補にできる。認められなければ、Engine 2 型の部品は推論時に生成するか、外す
