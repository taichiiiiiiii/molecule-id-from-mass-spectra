# fpnet_kit（#82）

Engine 1 の FPNet（`fpnet_full1.pt` の代わり）を、公開の学習コードから自前で作り直すための道具。

- `train_fpnet_orig.py`: `dmitriigluzdov/casmi26-fold-safe-fpnet` の `train_fpnet.py` の写し（バイト一致、sha256 `46b4dceebb2badeac78e0d5e55849169f5bef8da70a3adf1a7431fe34e603312`、Apache-2.0、作者 Dmitrii Gluzdov）。変更しない
- `make_mps.py`: 原本から MPS 版を作る置き換えの一覧（どの置き換えも原本で 1 回だけ一致することを assert）
- `train_fpnet_mps.py`: `make_mps.py` の出力。原本との差分は印の付いた行だけ
  - `# [port:device]`: `--device`（既定 mps）、評価と学習の `collate`・`autocast` を選んだデバイスで fp32（autocast なし）、macOS で `Queue.qsize()` が `NotImplementedError` になるのでログの行だけ `_qsize`（-1 を返す）
  - `# [port:resume]`: `--ckpt_every_min`（既定 30）・`--resume`（既定 1）、`<tag>_resume.pt` への途中保存（tmp → `os.replace`、学習の終わりにも 1 回）と再開（モデル・AdamW・GradScaler・step・経過時間・履歴・損失の EMA・torch の CPU と MPS の RNG）、SIGTERM で worker を止めて終わる、親が消えた worker は自分で終わる、再開後の worker の RNG は `(seed*1000+wid+1, 再開した step)` の別の列（データの並びは近似の再開で、ビット単位の再現ではない）
  - `# [port:memory]`: `--accum`（既定 1 = 原本と同じ計算）。worker が送る bs 件を main で `accum` 個の連続した等分に切り、各部分の損失 / accum で backward し、unscale・clip 1.0・AdamW の step は原本どおり 1 回（`accum_backward`）。損失はすべて平均なので勾配は一括と同じ（丸めと dropout の引き方だけ違う）。fp32 の MPS で活性のメモリを減らすため（#82、11/18 昼の事前登録）
- `fpnet_mps_loop.sh`: 本番の再開ループ（入力の sha256 の確認・vm_stat の空きと swap の待ち・memguard・swap が開始時より 2,048 MB 増えたら止める・再開は最大 5 回）
- `fpnet_mps_speed.sh`: 速さとメモリの測定のドライバー（2 回。memguard の rc 99・早い終わり・途中保存の読み込みの失敗・swap の増え・主プロセスが見つからない・SIGTERM 後の rc が 143 でない、のどれかで以降の回を起動しない）
- `fpnet_mps_monitor.sh`: メモリの記録（vm_stat の空き、主プロセスの footprint と IOAccelerator (graphics)、プロセスの木の RSS の合計は記録だけ）

テスト: `tests/test_fpnet_mps_port.py`（差分の検査と測定のドライバーは CI、勾配の一致・再開は torch・casmi のコードがあるときだけ、ループは Mac だけ）。
