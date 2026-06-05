"""Lightweight conv VAE fallback when SD-VAE is unavailable (smoke tests / offline)."""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleConvVAE(nn.Module):
    """Small trainable-style VAE used frozen for MiniWAM dev without HF download."""

    def __init__(self, latent_channels: int = 4, scale: int = 8):
        super().__init__()
        self.latent_channels = latent_channels
        self.scale = scale
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 4, 2, 1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )
        for p in self.parameters():
            p.requires_grad = False

    @property
    def config(self):
        from types import SimpleNamespace

        return SimpleNamespace(latent_channels=self.latent_channels, scaling_factor=1.0)

    def encode(self, x: torch.Tensor):
        z = self.encoder(x)
        return type("obj", (), {"latent_dist": type("d", (), {"sample": lambda: z})()})()

    def decode(self, z: torch.Tensor):
        return type("obj", (), {"sample": self.decoder(z)})()
