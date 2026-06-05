"""Decode LIBERO WebSocket JPEG observations for MiniWAM."""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
import torch
from PIL import Image


def decode_b64_image(b64_str: str, flip_vertical: bool = True) -> np.ndarray:
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img)
    if flip_vertical:
        arr = arr[::-1].copy()
    return arr


def obs_to_model_frame(
    agentview_b64: str,
    wrist_b64: str | None,
    image_size: int,
    use_wrist: bool,
    concat_cameras: bool,
) -> torch.Tensor:
    agent = decode_b64_image(agentview_b64)
    agent_img = Image.fromarray(agent).resize((image_size, image_size), Image.BILINEAR)
    if use_wrist and wrist_b64:
        wrist = decode_b64_image(wrist_b64)
        wrist_img = Image.fromarray(wrist).resize((image_size, image_size), Image.BILINEAR)
        if concat_cameras:
            combo = Image.new("RGB", (image_size * 2, image_size))
            combo.paste(agent_img, (0, 0))
            combo.paste(wrist_img, (image_size, 0))
            arr = np.array(combo)
        else:
            arr = np.array(agent_img)
    else:
        arr = np.array(agent_img)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


def stack_obs_sequence(
    frames: list[torch.Tensor],
    video_frames: int,
) -> torch.Tensor:
    """Take last video_frames frames; pad by repeating first if short."""
    if len(frames) >= video_frames:
        sel = frames[-video_frames:]
    else:
        sel = [frames[0]] * (video_frames - len(frames)) + frames
    return torch.stack(sel, dim=0)
