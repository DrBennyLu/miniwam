"""Closed-loop LIBERO evaluation via WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch

from miniwam.eval.libero_ws_client import LiberoWSClient
from miniwam.eval.obs_adapter import obs_to_model_frame, stack_obs_sequence
from miniwam.models import MiniWAM
from miniwam.utils.config import load_config
from miniwam.utils.seed import set_seed


async def run_rollout(
    client: LiberoWSClient,
    model: MiniWAM,
    cfg: dict,
    task_id: int,
    init_id: int,
    device: torch.device,
) -> dict:
    eval_cfg = cfg["eval"]
    data_cfg = cfg["data"]
    ep = await client.create_episode(
        task_id=task_id,
        init_id=init_id,
        suite=eval_cfg.get("suite", "libero_object"),
        max_steps=eval_cfg.get("max_steps", 600),
    )
    episode_id = ep["episode_id"]
    instruction = ep.get("instruction", "")
    frame_buffer: list[torch.Tensor] = []

    def ingest(obs_payload: dict):
        frame = obs_to_model_frame(
            obs_payload["agentview_b64"],
            obs_payload.get("wrist_b64"),
            data_cfg["image_size"],
            data_cfg.get("use_wrist", False),
            data_cfg.get("concat_cameras", True),
        )
        frame_buffer.append(frame)

    ingest(ep)
    done = False
    success = False
    steps = 0
    chunk_execute = eval_cfg.get("chunk_execute", 8)

    while not done and steps < eval_cfg.get("max_steps", 600):
        seq = stack_obs_sequence(frame_buffer, model.video_frames)
        images = seq.unsqueeze(0).to(device)
        with torch.no_grad():
            chunk = model.predict_action_chunk(images, [instruction])[0].cpu().numpy()
        for ai in range(min(chunk_execute, len(chunk))):
            result = await client.step(episode_id, chunk[ai].tolist())
            steps += 1
            done = result.get("done", False)
            success = result.get("success", False)
            if done:
                break
            ingest(result)
    await client.close_episode(episode_id)
    return {"task_id": task_id, "init_id": init_id, "success": success, "steps": steps}


async def async_main(cfg: dict, checkpoint: str):
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = MiniWAM(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    eval_cfg = cfg["eval"]
    num_trials = eval_cfg.get("num_trials", 10)
    task_id = eval_cfg.get("task_id", 0)
    successes = 0
    records = []

    async with LiberoWSClient(eval_cfg["ws_url"]) as client:
        for trial in range(num_trials):
            init_id = trial % 50
            rec = await run_rollout(client, model, cfg, task_id, init_id, device)
            successes += int(rec["success"])
            records.append(rec)
            print(
                f"trial {trial + 1}/{num_trials} success={rec['success']} steps={rec['steps']}"
            )

    rate = successes / max(1, num_trials)
    summary = {
        "checkpoint": checkpoint,
        "task_id": task_id,
        "num_trials": num_trials,
        "success_rate": rate,
        "records": records,
    }
    out_dir = Path(cfg["training"]["save_dir"]) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ws_eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Success rate: {rate * 100:.1f}% ({successes}/{num_trials})")
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    ckpt = args.checkpoint or str(Path(cfg["training"]["save_dir"]) / "latest.pt")
    asyncio.run(async_main(cfg, ckpt))


if __name__ == "__main__":
    main()
