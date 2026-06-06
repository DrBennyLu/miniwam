#!/usr/bin/env python3
"""D6: decode predicted future latents vs GT and save frame strips."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from miniwam.data import build_dataset
from miniwam.models import MiniWAM
from miniwam.models.dit_blocks import patchify_latents, unpatchify_latents
from miniwam.models.flow_matching import build_noisy_sample, sample_timesteps, velocity_target
from miniwam.models.mini_wam import (
    TYPE_ACTION,
    TYPE_TEXT,
    TYPE_VIDEO_CLEAN,
    TYPE_VIDEO_FUTURE,
    build_fast_wam_attention_mask,
)
from miniwam.utils.config import load_config
from miniwam.utils.seed import set_seed


@torch.no_grad()
def predict_future_latents(model: MiniWAM, images: torch.Tensor, instructions: list[str]) -> torch.Tensor:
    """Flow-denoise future video latents conditioned on frame 0."""
    device = images.device
    b = images.shape[0]
    latents = model.vae.encode_sequence(images)
    f0 = latents[:, 0]
    lh, lw = latents.shape[-2], latents.shape[-1]
    n_future = model.video_frames - 1
    c = model._latent_ch

    future = torch.randn(b, n_future, c, lh, lw, device=device)
    steps = model.num_inference_steps
    text = model._encode_text(instructions, device)
    clean_tokens = model.latent_in(patchify_latents(f0, model.patch_size))
    n_text = text.shape[1]
    n_clean = clean_tokens.shape[1]
    n_fut = (lh // model.patch_size) * (lw // model.patch_size) * n_future

    dummy_action = torch.zeros(b, model.action_horizon, model.action_dim, device=device)

    for i in range(steps, 0, -1):
        t_vid = torch.full((b,), i / steps, device=device)
        t_act = torch.zeros(b, device=device)
        noisy_action = build_noisy_sample(
            dummy_action, t_act, torch.randn_like(dummy_action)
        )
        fut_patches = patchify_latents(
            future.reshape(b * n_future, c, lh, lw), model.patch_size
        ).reshape(b, n_fut, -1)
        future_tokens = model.latent_in(fut_patches)
        action_tokens = model.action_in(noisy_action)

        tokens = torch.cat([text, clean_tokens, future_tokens, action_tokens], dim=1)
        types = torch.cat(
            [
                torch.full((b, n_text), TYPE_TEXT, device=device, dtype=torch.long),
                torch.full((b, n_clean), TYPE_VIDEO_CLEAN, device=device, dtype=torch.long),
                torch.full((b, n_fut), TYPE_VIDEO_FUTURE, device=device, dtype=torch.long),
                torch.full((b, model.action_horizon), TYPE_ACTION, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        attn_mask = build_fast_wam_attention_mask(types[0], include_future_video=True)
        t_combined = torch.maximum(t_vid, t_act)
        hidden = model.backbone(tokens, t_combined, types, attn_mask)
        act_start = n_text + n_clean + n_fut
        pred_v = model.latent_out(hidden[:, n_text + n_clean : act_start])
        pred_v = pred_v.reshape(b, n_fut, -1)
        patches_per_frame = (lh // model.patch_size) * (lw // model.patch_size)
        for fi in range(n_future):
            sl = slice(fi * patches_per_frame, (fi + 1) * patches_per_frame)
            v_patch = pred_v[:, sl, :]
            v_latent = unpatchify_latents(v_patch, c, lh, lw, model.patch_size)
            future[:, fi] = future[:, fi] - (1.0 / steps) * v_latent

    return future


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="./outputs/viz_video_pred")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = build_dataset(cfg)
    sample = ds[args.sample_idx]
    images = sample["images"].unsqueeze(0).to(device)
    instruction = [sample["instruction"]]

    model = MiniWAM(cfg).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})")
    else:
        print("No checkpoint: showing untrained prediction vs GT")
    model.eval()

    with torch.no_grad():
        gt_latents = model.vae.encode_sequence(images)
        pred_future = predict_future_latents(model, images, instruction)

        gt_frames = images[0]
        pred_frames = [images[0, 0]]
        gt_future_frames = []
        for fi in range(pred_future.shape[1]):
            dec_pred = model.vae.decode(pred_future[:, fi])[0]
            dec_gt = model.vae.decode(gt_latents[:, fi + 1])[0]
            pred_frames.append(dec_pred)
            gt_future_frames.append(dec_gt)

        pred_strip = torch.stack(pred_frames, dim=0)
        gt_strip = torch.cat([images[0, :1], torch.stack(gt_future_frames, dim=0)], dim=0)

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_image(gt_strip, out_dir / "gt_frames.png", nrow=model.video_frames)
        save_image(pred_strip, out_dir / "pred_frames.png", nrow=model.video_frames)
        paired = torch.stack(
            [gt_strip[i] for i in range(model.video_frames)]
            + [pred_strip[i] for i in range(model.video_frames)],
            dim=0,
        )
        save_image(paired, out_dir / "gt_vs_pred.png", nrow=model.video_frames)

    print(f"Saved to {out_dir}/")
    print("  gt_frames.png   - GT decoded sequence")
    print("  pred_frames.png - predicted future (f0 real + f1..h predicted)")
    print("  gt_vs_pred.png  - top row GT, bottom row prediction")


if __name__ == "__main__":
    main()
