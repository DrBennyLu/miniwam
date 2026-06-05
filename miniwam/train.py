"""MiniWAM training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from miniwam.data import build_dataloader
from miniwam.models import MiniWAM
from miniwam.utils.config import load_config
from miniwam.utils.seed import set_seed


def apply_training_mode(cfg: dict) -> dict:
    mode = cfg["training"].get("mode", "full")
    if mode == "video_only":
        cfg["model"]["train_action"] = False
        cfg["model"]["train_video"] = True
    elif mode == "action_only":
        cfg["model"]["train_action"] = True
        cfg["model"]["train_video"] = False
        cfg["model"]["lambda_vid"] = 0.0
    return cfg


def save_checkpoint(path: Path, model, optimizer, step: int, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": cfg,
        },
        path,
    )
    latest = path.parent / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(path.name)
    except OSError:
        import shutil

        shutil.copy2(path, latest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = apply_training_mode(load_config(args.config))
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloader(cfg, split="train")

    model = MiniWAM(cfg).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    optim = torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optim.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"Resumed from {args.resume} at step {start_step}")

    max_steps = cfg["training"]["max_steps"]
    accum = cfg["training"].get("gradient_accumulation", 1)
    save_dir = Path(cfg["training"]["save_dir"])
    log_dir = Path(cfg["training"].get("log_dir", save_dir / "tb"))
    writer = SummaryWriter(log_dir=str(log_dir))
    use_bf16 = cfg["training"].get("bf16", False)
    scaler = torch.cuda.amp.GradScaler(enabled=use_bf16)

    model.train()
    step = start_step
    pbar = tqdm(total=max_steps, initial=step, desc="train")
    optim.zero_grad(set_to_none=True)
    data_iter = iter(train_loader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        for k in ("images", "actions", "task_ids", "episode_ids"):
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model.forward_train(batch)
            loss = out["loss"] / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            if cfg["training"].get("grad_clip"):
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(trainable, cfg["training"]["grad_clip"])
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

        if step % cfg["training"].get("log_every", 50) == 0:
            writer.add_scalar("loss/total", out["loss"].item(), step)
            writer.add_scalar("loss/act", out["loss_act"].item(), step)
            writer.add_scalar("loss/vid", out["loss_vid"].item(), step)
            pbar.set_postfix(
                loss=f"{out['loss'].item():.4f}",
                act=f"{out['loss_act'].item():.4f}",
                vid=f"{out['loss_vid'].item():.4f}",
            )

        if step > 0 and step % cfg["training"].get("save_every", 2000) == 0:
            ckpt_path = save_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(ckpt_path, model, optim, step, cfg)

        step += 1
        pbar.update(1)

    ckpt_path = save_dir / f"checkpoint_step_{step}.pt"
    save_checkpoint(ckpt_path, model, optim, step, cfg)
    writer.close()
    pbar.close()
    print(f"Training done. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
