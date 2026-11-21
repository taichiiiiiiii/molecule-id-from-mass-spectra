# fpnet_kit（#82）

Engine 1 の FPNet（`fpnet_full1.pt` の代わり）を、公開の学習コードから自前で作り直すための道具。

- `train_fpnet_orig.py`: `dmitriigluzdov/casmi26-fold-safe-fpnet` の `train_fpnet.py` の写し（バイト一致、sha256 `46b4dceebb2badeac78e0d5e55849169f5bef8da70a3adf1a7431fe34e603312`、Apache-2.0、作者 Dmitrii Gluzdov）。変更しない
- `make_mps.py`: 原本から MPS 版を作る置き換えの一覧（どの置き換えも原本で 1 回だけ一致することを assert）
- `train_fpnet_mps.py`: `make_mps.py` の出力。原本との差分は印の付いた行だけ
  - `# [port:device]`: `--device`（既定 mps）、評価と学習の `collate`・`autocast` を選んだデバイスで fp32（autocast なし）、macOS で `Queue.qsize()` が `NotImplementedError` になるのでログの行だけ `_qsize`（-1 を返す）
  - `# [port:resume]`: `--ckpt_every_min`（既定 30）・`--resume`（既定 1）、`<tag>_resume.pt` への途中保存（tmp → `os.replace`、学習の終わりにも 1 回）と再開（モデル・AdamW・GradScaler・step・経過時間・履歴・損失の EMA・torch の CPU と MPS の RNG）、SIGTERM で worker を止めて終わる、親が消えた worker は自分で終わる、再開後の worker の RNG は `(seed*1000+wid+1, 再開した step)` の別の列（データの並びは近似の再開で、ビット単位の再現ではない）
- `fpnet_mps_loop.sh`: 本番の再開ループ（入力の sha256 の確認・vm_stat の空き待ち・memguard・再開は最大 5 回）
- `fpnet_mps_monitor.sh`: メモリの記録（vm_stat の空き、主プロセスの footprint と IOAccelerator (graphics)、プロセスの木の RSS の合計は記録だけ）

テスト: `tests/test_fpnet_mps_port.py`（差分の検査は CI、再開とループは torch・casmi のコードがある Mac だけ）。

## T4 の 2 セッション（#82、11/21 の事前登録）

- `make_t4.py`: 原本から T4 版 `train_fpnet_t4.py` を作る置き換えの一覧（どれも原本で 1 回だけ一致を assert）。足す行は `# [t4:resume]` の印つきだけ、消える行は worker の乱数の 1 行だけ。デバイス・精度・バッチ・損失・ハイパラ・lr の式は原本のまま
  - `--session`（1 / 2）・`--session_hours`（既定 10.5）・`--resume_from`。学習プロセスの経過が `session_hours` に達したら、その step の lr を決めた直後（バッチを取る前）に `<tag>_resume.pt` を保存して `SESSION_DONE` で正常終了。82,855 step に先に達したら原本どおり最終評価と `fpnet_final.pt`
  - 再開: 保存した引数と今の引数が（session・session_hours・resume_from・code・run を除き）同じで、保存したセッションが 1 つ前でなければ止まる。最初の step の lr が保存した lr とビット一致しなければ止まる
  - worker の乱数: 新しい学習はキットどおり `seed*1000+wid+1`、再開したセッションは `seed*1000+wid+1 + 7919*session`。データの流れは 1 セッション通しと同じにならない（「公開コードで作り直した」と書く）
- `fpnet_resume.py`: 途中保存（モデル・AdamW・GradScaler・step・経過時間・履歴・EMA・lr・torch CPU/CUDA・numpy・python の乱数、tmp → `os.replace`）・再開・ビット比較・CUDA の自己検査（呼び出し側の乱数の状態を戻す）。キットには lr スケジューラのオブジェクトが無い（lr は step と経過時間の式）ので、step・経過時間・lr を保存して照合する
- カーネル: `scripts/train/make_fpnet_full_kernel.py --session 1|2`（`kernels/fpnet_full_train` / `kernels/fpnet_full_train_s2`）
- テスト: `tests/test_fpnet_t4_resume.py`（(a) 保存 → 読み込みのビット一致、(b) worker なしの loss 列の一致と、派生版そのものを偽の casmi で CPU 実行して 1 セッションと 2 セッションの最終重みがビット一致、(c) worker の乱数の式、自己検査、偽の /kaggle でノート 2 本を通しで実行）
