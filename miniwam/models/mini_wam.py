"""MiniWAM: Fast-WAM-style joint video-action flow matching with single-pass inference."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .dit_blocks import MiniWAMBackbone, patchify_latents
from .flow_matching import (
    build_noisy_sample,
    flow_matching_loss,
    sample_timesteps,
    velocity_target,
)
from .text_encoder import CLIPTextEncoder
from .vae_wrapper import FrozenSDVAE

TYPE_TEXT = 0
TYPE_VIDEO_CLEAN = 1
TYPE_VIDEO_FUTURE = 2
TYPE_ACTION = 3


def build_fast_wam_attention_mask(
    token_types: torch.Tensor,
    include_future_video: bool,
) -> torch.Tensor:
    n = token_types.shape[0]
    types = token_types.tolist()
    mask = torch.zeros(n, n, dtype=torch.bool, device=token_types.device)
    idx_text = [i for i, t in enumerate(types) if t == TYPE_TEXT]
    idx_clean = [i for i, t in enumerate(types) if t == TYPE_VIDEO_CLEAN]
    idx_future = [i for i, t in enumerate(types) if t == TYPE_VIDEO_FUTURE]
    idx_action = [i for i, t in enumerate(types) if t == TYPE_ACTION]

    def allow_rows(cols, rows):
        for r in rows:
            for c in cols:
                mask[r, c] = True

    allow_rows(idx_text, idx_text)
    allow_rows(idx_clean, idx_text + idx_clean)
    if include_future_video:
        allow_rows(idx_future, idx_text + idx_clean + idx_future)
    allow_rows(idx_action, idx_text + idx_clean + idx_action)
    return mask


class MiniWAM(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.action_horizon = cfg["data"]["action_horizon"]
        self.video_frames = cfg["data"]["video_frames"]
        self.patch_size = m.get("patch_size", 2)
        self.hidden_dim = m["hidden_dim"]
        self.action_dim = m.get("action_dim", 7)
        self.lambda_vid = float(m.get("lambda_vid", 1.0))
        self.train_video = bool(m.get("train_video", True))
        self.train_action = bool(m.get("train_action", True))
        self.num_inference_steps = m.get("num_inference_steps", 10)

        self.vae = FrozenSDVAE(m.get("vae_id", "stabilityai/sd-vae-ft-mse"))
        latent_ch = self.vae.vae.config.latent_channels
        self._latent_ch = latent_ch
        patch_dim = latent_ch * self.patch_size * self.patch_size

        if m.get("text_encoder", "clip") == "none":
            self.text_encoder = None
        else:
            self.text_encoder = CLIPTextEncoder(
                m.get("clip_model", "openai/clip-vit-base-patch32"),
                proj_dim=self.hidden_dim,
            )

        self.latent_in = nn.Linear(patch_dim, self.hidden_dim)
        self.latent_out = nn.Linear(self.hidden_dim, patch_dim)
        self.action_in = nn.Linear(self.action_dim, self.hidden_dim)
        self.action_out = nn.Linear(self.hidden_dim, self.action_dim)

        self.backbone = MiniWAMBackbone(
            hidden_dim=self.hidden_dim,
            num_layers=m["num_layers"],
            num_heads=m["num_heads"],
            mlp_ratio=m.get("mlp_ratio", 4.0),
            dropout=m.get("dropout", 0.1),
        )

    def _encode_text(self, instructions: List[str], device: torch.device) -> torch.Tensor:
        if self.text_encoder is not None:
            return self.text_encoder(instructions)
        b = len(instructions)
        return torch.zeros(b, 1, self.hidden_dim, device=device)

    def _latents_to_tokens(self, latents: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latents.shape
        out = []
        for ti in range(t):
            patches = patchify_latents(latents[:, ti], self.patch_size)
            out.append(self.latent_in(patches))
        return torch.cat(out, dim=1), h, w

    def forward_train(self, batch: Dict) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        actions = batch["actions"]
        instructions = batch["instructions"]
        b = images.shape[0]
        device = images.device

        with torch.no_grad():
            latents = self.vae.encode_sequence(images)      # 原始图像-》vae压缩-》latents
        future_gt = latents[:, 1:]
        lh, lw = latents.shape[-2], latents.shape[-1]

        t_vid = sample_timesteps(b, device)    # 采样时间步
        t_act = sample_timesteps(b, device)
        noise_vid = torch.randn_like(future_gt)
        noise_act = torch.randn_like(actions)
        noisy_future = build_noisy_sample(future_gt, t_vid, noise_vid)    # 未来视频加噪音
        noisy_action = build_noisy_sample(actions, t_act, noise_act)    # 未来动作加噪音

        text = self._encode_text(instructions, device)      # 文本指令——》clip编码-》文本条件
        clean_tokens = self.latent_in(patchify_latents(latents[:, 0], self.patch_size))     # 当前帧-》分块——〉投影
        fut_patches = patchify_latents(
            noisy_future.reshape(b * (self.video_frames - 1), self._latent_ch, lh, lw),
            self.patch_size,
        ).reshape(b, -1, self.latent_in.in_features)
        future_tokens = self.latent_in(fut_patches)    # 加噪音后输出——》token
        action_tokens = self.action_in(noisy_action)

        n_text = text.shape[1]
        n_clean = clean_tokens.shape[1]
        n_fut = future_tokens.shape[1]

        tokens = torch.cat([text, clean_tokens, future_tokens, action_tokens], dim=1)    # 拼接所有token，文本+当前图像+未来视频+未来动作
        # 给每个token标注类型，文本+当前图像+未来视频+未来动作
        types = torch.cat(
            [
                torch.full((b, n_text), TYPE_TEXT, device=device, dtype=torch.long),
                torch.full((b, n_clean), TYPE_VIDEO_CLEAN, device=device, dtype=torch.long),
                torch.full((b, n_fut), TYPE_VIDEO_FUTURE, device=device, dtype=torch.long),
                torch.full((b, self.action_horizon), TYPE_ACTION, device=device, dtype=torch.long),
            ],
            dim=1,
        )
        attn_mask = build_fast_wam_attention_mask(types[0], include_future_video=True)    # 生成注意力掩码
        t_combined = torch.maximum(t_vid, t_act)
        hidden = self.backbone(tokens, t_combined, types, attn_mask)    # 送入miniwam backbone，主干网络向前，条件注入，特征建模

        # 拆分输出
        act_start = n_text + n_clean + n_fut
        pred_vid = self.latent_out(hidden[:, n_text + n_clean : act_start])
        pred_act = self.action_out(hidden[:, act_start :])

        target_vid = patchify_latents(
            velocity_target(noise_vid, future_gt).reshape(
                b * (self.video_frames - 1), self._latent_ch, lh, lw
            ),
            self.patch_size,
        ).reshape(b, n_fut, -1)
        target_act = velocity_target(noise_act, actions)
        # 计算流匹配损失
        loss_vid = flow_matching_loss(pred_vid, target_vid)
        loss_act = flow_matching_loss(pred_act, target_act)

        loss = torch.tensor(0.0, device=device)
        if self.train_action:
            loss = loss + loss_act
        if self.train_video and self.lambda_vid > 0:
            loss = loss + self.lambda_vid * loss_vid

        return {
            "loss": loss,
            "loss_act": loss_act.detach(),
            "loss_vid": loss_vid.detach(),
        }

    @torch.no_grad()
    def encode_world(self, images: torch.Tensor, instructions: List[str]) -> torch.Tensor:
        device = images.device
        latents = self.vae.encode_sequence(images[:, :1])
        text = self._encode_text(instructions, device)
        clean_tokens = self.latent_in(patchify_latents(latents[:, 0], self.patch_size))
        b = images.shape[0]
        tokens = torch.cat([text, clean_tokens], dim=1)
        types = torch.cat(
            [
                torch.full((b, text.shape[1]), TYPE_TEXT, device=device, dtype=torch.long),
                torch.full(
                    (b, clean_tokens.shape[1]), TYPE_VIDEO_CLEAN, device=device, dtype=torch.long
                ),
            ],
            dim=1,
        )
        attn_mask = build_fast_wam_attention_mask(types[0], include_future_video=False)
        self._infer_n_text = text.shape[1]
        self._infer_n_clean = clean_tokens.shape[1]
        return self.backbone(tokens, None, types, attn_mask)

    @torch.no_grad()
    def predict_action_chunk(
        self, images: torch.Tensor, instructions: List[str]
    ) -> torch.Tensor:
        world_hidden = self.encode_world(images, instructions)
        b = images.shape[0]
        device = images.device
        action = torch.randn(b, self.action_horizon, self.action_dim, device=device)
        steps = self.num_inference_steps
        n_world = world_hidden.shape[1]
        n_text = getattr(self, "_infer_n_text", 1)
        for i in range(steps, 0, -1):
            t = torch.full((b,), i / steps, device=device)
            act_tokens = self.action_in(action)
            tokens = torch.cat([world_hidden, act_tokens], dim=1)
            types = torch.zeros(b, n_world + self.action_horizon, dtype=torch.long, device=device)
            types[:, :n_text] = TYPE_TEXT
            types[:, n_text:n_world] = TYPE_VIDEO_CLEAN
            types[:, n_world:] = TYPE_ACTION
            attn_mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=device)
            hidden = self.backbone(tokens, t, types, attn_mask)
            pred_v = self.action_out(hidden[:, -self.action_horizon :])
            action = action - (1.0 / steps) * pred_v
        return action.clamp(-1.0, 1.0)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train(batch)
        actions = self.predict_action_chunk(batch["images"], batch["instructions"])
        return {"actions": actions, "loss": torch.tensor(0.0, device=actions.device)}
