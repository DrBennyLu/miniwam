# MiniWAM

## Author： Benny Lu

Lightweight **Fast-WAM**-style World Action Model for LIBERO, designed for a single RTX GPU player.


- **Training**: joint flow matching on actions + future video latents (`L = L_act + λ L_vid`)
- **Inference**: single-pass world encoding from the current frame (no future video denoising)
- **Hybrid**: frozen SD-VAE + trainable small DiT (~80–200M)

## Setup

```bash
cd /PATH_TO_YOUR_MINIWAM/miniwam
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Download frozen VAE + CLIP once (5090 training):

```bash
# caches to ~/.cache/huggingface
python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"
```

Offline smoke (mock VAE, CPU): `bash scripts/smoke_train.sh`

LIBERO simulation (separate env):

```bash
conda activate libero
cd /PATH_TO_YOUR_LIBERO/LIBERO
python scripts/libero_ws_server.py --suite libero_object
```

## Quick start

```bash
# VAE reconstruction smoke test
python scripts/smoke_vae.py --config configs/libero_object_mini.yaml

# Train (single task, mini config)
bash scripts/train_single_task.sh

# Train full suite / video-only / ablation (λ=0)
bash scripts/train_suite.sh
bash scripts/train_video_only.sh
bash scripts/train_no_video.sh

# Closed-loop eval (WebSocket)
python -m miniwam.eval.libero_ws_eval --config configs/libero_object_mini.yaml \
  --checkpoint checkpoints/mini_wam/latest.pt
```

# train the network
```bash
python -m miniwam.train --config configs/libero_object_task2_large.yaml
```

# see the image generation result
```bash
python scripts/viz_video_pred.py \
  --config configs/libero_object_task2_20k.yaml \
  --checkpoint checkpoints/libero_object_task2_20k/latest.pt \
  --out-dir outputs/viz_video_pred/step20000
```

# Close loop eval
```bash
python -m miniwam.eval.libero_ws_eval \
  --config configs/libero_object_task2_large.yaml \
  --checkpoint checkpoints/libero_object_task2_large/latest.pt
```

## Configs

All YAML files inherit from `configs/default.yaml` via Hydra `defaults`. Set `data.lerobot_path` or `data.hdf5_root` in YAML (default LeRobot path points to ScriptedVLA `dada/libero-object`).

| Category | Config | Purpose |
|----------|--------|---------|
| Base | `default.yaml` | Shared data / model / training / eval defaults |
| Single task | `libero_object_mini.yaml` | Task 2 (cream cheese), small DiT, fast iteration |
| Single task | `libero_object_task2_20k.yaml` | Task 2, 20k steps on mini model |
| Single task | `libero_object_task2_medium.yaml` | Task 2, ~27M params, larger batch (5090 starter) |
| Single task | `libero_object_task2_large.yaml` | Task 2, ~90M params, 200k steps |
| Full suite | `libero_object_suite.yaml` | All 10 libero_object tasks, wrist camera |
| Ablation | `libero_object_video_only.yaml` | Full suite, video latent only (`train_action: false`) |
| Ablation | `libero_object_video_only_mini.yaml` | Single task video-only, 2k steps |
| Ablation | `libero_object_no_video.yaml` | Full suite, action-only (`lambda_vid: 0`) |
| Other suite | `libero_spatial_suite.yaml` | libero_spatial (10 tasks) |
| Other suite | `libero_goal_suite.yaml` | libero_goal (10 tasks) |
| Other suite | `libero_long_suite.yaml` | libero_10 long-horizon suite |
| Smoke | `libero_object_mini_smoke.yaml` | Offline / CI train smoke (mock VAE, CPU) |
| Smoke | `libero_object_no_video_smoke.yaml` | Action-only smoke |
| Smoke | `libero_object_video_only_smoke.yaml` | Video-only smoke |

---

# MiniWAM（日本語）

## 著者： Benny Lu

LIBERO 向けの軽量 **Fast-WAM** 型 World Action Model。1 枚の RTX GPU で動かすことを想定しています。

- **学習**: アクションと未来映像の潜在表現に対する joint flow matching（`L = L_act + λ L_vid`）
- **推論**: 現在フレームからの 1 パス world encoding（未来映像の denoising なし）
- **ハイブリッド**: 凍結 SD-VAE + 学習可能な小型 DiT（約 80–200M パラメータ）

## セットアップ

```bash
cd /PATH_TO_YOUR_MINIWAM/miniwam
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

凍結 VAE + CLIP の初回ダウンロード（5090 での学習向け）:

```bash
# ~/.cache/huggingface にキャッシュ
python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')"
```

オフライン smoke（mock VAE、CPU）: `bash scripts/smoke_train.sh`

LIBERO シミュレーション（別環境）:

```bash
conda activate libero
cd /PATH_TO_YOUR_LIBERO/LIBERO
python scripts/libero_ws_server.py --suite libero_object
```

## クイックスタート

```bash
# VAE 再構成 smoke テスト
python scripts/smoke_vae.py --config configs/libero_object_mini.yaml

# 学習（単一タスク、mini 設定）
bash scripts/train_single_task.sh

# 全 suite / video-only / 消融（λ=0）
bash scripts/train_suite.sh
bash scripts/train_video_only.sh
bash scripts/train_no_video.sh

# クローズドループ評価（WebSocket）
python -m miniwam.eval.libero_ws_eval --config configs/libero_object_mini.yaml \
  --checkpoint checkpoints/mini_wam/latest.pt
```

# ネットワークの学習
```bash
python -m miniwam.train --config configs/libero_object_task2_large.yaml
```

# 映像生成結果の確認
```bash
python scripts/viz_video_pred.py \
  --config configs/libero_object_task2_20k.yaml \
  --checkpoint checkpoints/libero_object_task2_20k/latest.pt \
  --out-dir outputs/viz_video_pred/step20000
```

# クローズドループ評価
```bash
python -m miniwam.eval.libero_ws_eval \
  --config configs/libero_object_task2_large.yaml \
  --checkpoint checkpoints/libero_object_task2_large/latest.pt
```

## 設定ファイル

すべての YAML は Hydra の `defaults` 経由で `configs/default.yaml` を継承します。YAML 内で `data.lerobot_path` または `data.hdf5_root` を設定してください（デフォルトの LeRobot パスは ScriptedVLA の `dada/libero-object` を指します）。

| カテゴリ | 設定ファイル | 用途 |
|----------|--------------|------|
| Base | `default.yaml` | データ / モデル / 学習 / 評価の共通デフォルト |
| Single task | `libero_object_mini.yaml` | Task 2（クリームチーズ）、小型 DiT、高速イテレーション |
| Single task | `libero_object_task2_20k.yaml` | Task 2、mini モデルで 20k ステップ |
| Single task | `libero_object_task2_medium.yaml` | Task 2、約 27M パラメータ、大きめ batch（5090 向け入門） |
| Single task | `libero_object_task2_large.yaml` | Task 2、約 90M パラメータ、200k ステップ |
| Full suite | `libero_object_suite.yaml` | libero_object 全 10 タスク、手首カメラあり |
| Ablation | `libero_object_video_only.yaml` | 全 suite、映像潜在のみ（`train_action: false`） |
| Ablation | `libero_object_video_only_mini.yaml` | 単一タスク video-only、2k ステップ |
| Ablation | `libero_object_no_video.yaml` | 全 suite、アクションのみ（`lambda_vid: 0`） |
| Other suite | `libero_spatial_suite.yaml` | libero_spatial（10 タスク） |
| Other suite | `libero_goal_suite.yaml` | libero_goal（10 タスク） |
| Other suite | `libero_long_suite.yaml` | libero_10 長期ホライズン suite |
| Smoke | `libero_object_mini_smoke.yaml` | オフライン / CI 学習 smoke（mock VAE、CPU） |
| Smoke | `libero_object_no_video_smoke.yaml` | アクションのみ smoke |
| Smoke | `libero_object_video_only_smoke.yaml` | video-only smoke |
