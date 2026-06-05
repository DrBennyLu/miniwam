"""Frozen Stable Diffusion VAE for latent video tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenSDVAE(nn.Module):
    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse", dtype=torch.float32):
        super().__init__()
        self._is_mock = model_id in ("mock", "simple", "conv")
        if self._is_mock:
            from .conv_vae import SimpleConvVAE

            self.vae = SimpleConvVAE()
            self.scaling_factor = 1.0
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(model_id)
            self.scaling_factor = self.vae.config.scaling_factor
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, C, H, W) in [0,1] -> latents (B, C_lat, H/8, W/8)."""
        if self._is_mock:
            latents = self.vae.encoder(images)
            return latents * self.scaling_factor
        x = images * 2.0 - 1.0
        param = next(self.vae.parameters())
        x = x.to(dtype=param.dtype, device=param.device)
        latents = self.vae.encode(x).latent_dist.sample()
        return latents * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / self.scaling_factor
        if self._is_mock:
            return self.vae.decoder(latents).clamp(0, 1)
        param = next(self.vae.parameters())
        images = self.vae.decode(latents.to(dtype=param.dtype, device=param.device)).sample
        return (images / 2 + 0.5).clamp(0, 1)

    @torch.no_grad()
    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, C, H, W) -> (B, T, C_lat, h, w)."""
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        z = self.encode(flat)
        return z.reshape(b, t, *z.shape[1:])
