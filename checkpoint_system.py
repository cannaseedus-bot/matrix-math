
import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

@dataclass
class CheckpointMeta:
    step: int
    epoch: int
    loss: float
    best: bool = False
    tag: str = ""
    extra: Optional[Dict[str, Any]] = None

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_checkpoint(model, optimizer, meta: CheckpointMeta, out_dir: str) -> str:
    """Save a checkpoint (weights + optimizer + meta.json) into out_dir."""
    _ensure_dir(out_dir)
    ckpt_name = f"step_{meta.step:08d}.pt"
    ckpt_path = os.path.join(out_dir, ckpt_name)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "meta": asdict(meta),
    }
    torch.save(payload, ckpt_path)
    # write/update last + maybe best
    with open(os.path.join(out_dir, "last.json"), "w", encoding="utf-8") as f:
        json.dump(meta.__dict__, f, indent=2)
    if meta.best:
        best_path = os.path.join(out_dir, "best.pt")
        torch.save(payload, best_path)
        with open(os.path.join(out_dir, "best.json"), "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f, indent=2)
    return ckpt_path

def load_checkpoint(model, optimizer, ckpt_path: str, map_location: str = "cpu") -> CheckpointMeta:
    """Load checkpoint into model/optimizer and return meta."""
    payload = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    meta_dict = payload.get("meta") or {}
    return CheckpointMeta(**meta_dict)

def load_best_or_last(model, optimizer, ckpt_dir: str, map_location: str = "cpu") -> Optional[CheckpointMeta]:
    """Try best.pt, then last.json, else None."""
    best_path = os.path.join(ckpt_dir, "best.pt")
    last_json = os.path.join(ckpt_dir, "last.json")
    if os.path.isfile(best_path):
        return load_checkpoint(model, optimizer, best_path, map_location=map_location)
    if os.path.isfile(last_json):
        with open(last_json, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        # caller must still load weights manually if needed
        return CheckpointMeta(**meta_dict)
    return None
