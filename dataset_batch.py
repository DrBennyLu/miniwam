from miniwam.utils.config import load_config
from miniwam.data import build_dataset, build_dataloader
cfg = load_config("configs/libero_object_mini.yaml")
# 单条样本
ds = build_dataset(cfg)
s = ds[0]
print("=== 单条样本 ===")
print("images:", s["images"].shape)   # (5, 3, 128, 128)
print("actions:", s["actions"].shape) # (16, 7)
print("instruction:", s["instruction"])
print("len(dataset):", len(ds))
# 一个 batch
batch = next(iter(build_dataloader(cfg)))
print("\n=== batch ===")
print("images:", batch["images"].shape)   # (B, 5, 3, 128, 128)
print("actions:", batch["actions"].shape) # (B, 16, 7)