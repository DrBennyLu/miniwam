"""Flow matching utilities (logit-normal t, linear interpolation path)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Logit-normal timesteps in (0, 1), per Fast-WAM / common FM practice."""
    u = torch.randn(batch_size, device=device)
    t = torch.sigmoid(u)
    return t


def flow_matching_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE between predicted velocity and (noise - target)."""
    target_vel = target
    loss = F.mse_loss(pred, target_vel, reduction="none")
    if mask is not None:
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def build_noisy_sample(
    clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """y_t = (1-t)*clean + t*noise with broadcast t."""
    t = t.view(-1, *([1] * (clean.ndim - 1)))
    return (1.0 - t) * clean + t * noise


def velocity_target(noise: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return noise - clean


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=t.device))
            * torch.arange(half, device=t.device).float()
            / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)
