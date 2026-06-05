# MiniWAM

Lightweight **Fast-WAM**-style World Action Model for LIBERO, designed for a single RTX 5090 (32GB).

- **Training**: joint flow matching on actions + future video latents (`L = L_act + λ L_vid`)
- **Inference**: single-pass world encoding from the current frame (no future video denoising)
- **Hybrid**: frozen SD-VAE + trainable small DiT (~80–200M)

## Setup

```bash
cd /home/lxg/ai_models/miniwam
python -m venv .venv && source .venv/bin/activate
pip install -e .
# Or reuse ScriptedVLA env:
# export PYTHONPATH=/home/lxg/ai_models/miniwam
# /home/lxg/ai_models/ScriptedVLA/.venv/bin/python -m miniwam.train ...
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
cd /home/lxg/ai_models/LIBERO
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

## Configs

| Config | Purpose |
|--------|---------|
| `configs/libero_object_mini.yaml` | Single task, fast iteration |
| `configs/libero_object_suite.yaml` | All 10 libero_object tasks |
| `configs/libero_{spatial,goal,long}_suite.yaml` | Other LIBERO suites |

Set `data.lerobot_path` or `data.hdf5_root` in YAML. Default LeRobot path points to ScriptedVLA `dada/libero-object`.
