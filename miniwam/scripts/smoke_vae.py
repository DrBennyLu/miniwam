"""Smoke test: frozen SD-VAE reconstruction on LIBERO samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image

from miniwam.data import build_dataset
from miniwam.models.vae_wrapper import FrozenSDVAE
from miniwam.utils.config import load_config
from miniwam.utils.seed import set_seed


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-10:
        return 99.0
    return float(10 * torch.log10(torch.tensor(1.0 / mse)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="./outputs/smoke_vae")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(cfg)
    vae = FrozenSDVAE(cfg["model"].get("vae_id", "stabilityai/sd-vae-ft-mse")).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        frames = sample["images"].unsqueeze(0).to(device)
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        with torch.no_grad():
            z = vae.encode(flat)
            recon = vae.decode(z)
        scores.append(psnr(recon, flat))
        grid = torch.cat([flat[:1], recon[:1]], dim=0)
        save_image(grid, out_dir / f"sample_{i}_recon.png", nrow=2)

    print(f"VAE smoke test on {len(scores)} windows")
    print(f"Mean PSNR: {sum(scores) / len(scores):.2f} dB")
    print(f"Saved images to {out_dir}")


if __name__ == "__main__":
    main()
