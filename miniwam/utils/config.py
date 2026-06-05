"""YAML config loader with simple inheritance via `defaults` key."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key == "defaults":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_config_path(config_dir: Path, entry: str) -> Path:
    if entry == "default":
        return config_dir / "default.yaml"
    if entry.endswith(".yaml"):
        return config_dir / entry
    return config_dir / f"{entry}.yaml"


def _expand_defaults(path: Path, visited: set[str]) -> Dict[str, Any]:
    key = str(path.resolve())
    if key in visited:
        raise ValueError(f"Circular config defaults: {path}")
    visited.add(key)
    cfg = _load_yaml(path)
    merged: Dict[str, Any] = {}
    for entry in cfg.get("defaults") or []:
        parent_path = _resolve_config_path(path.parent, entry)
        if not parent_path.exists():
            raise FileNotFoundError(f"Config default not found: {parent_path}")
        merged = _deep_merge(merged, _expand_defaults(parent_path, visited))
    merged = _deep_merge(merged, {k: v for k, v in cfg.items() if k != "defaults"})
    return merged


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    merged = _expand_defaults(path, set())
    merged["_config_path"] = str(path)
    return merged
