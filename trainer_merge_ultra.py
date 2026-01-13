
"""Hybrid checkpoint merge: MX2LM + Qwen adapters + Cline @grams."""
import os
import json
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM

from .scxq2_engine import pack_scxq2

@dataclass
class MergeUltraConfig:
    base_model: str
    mx2lm_ckpt: str
    qwen_ckpt: str
    mx2lm_grams: str
    cline_grams: str
    out_dir: str
    alpha: float = 0.5

def _load_state(path: str) -> dict:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload

def merge_checkpoints(cfg: MergeUltraConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)

    s_mx = _load_state(cfg.mx2lm_ckpt)
    s_qw = _load_state(cfg.qwen_ckpt)

    sd = model.state_dict()
    alpha = cfg.alpha

    for k in sd.keys():
        if k in s_mx and k in s_qw and torch.is_tensor(sd[k]):
            sd[k] = (1.0 - alpha) * s_mx[k] + alpha * s_qw[k]
        elif k in s_mx:
            sd[k] = s_mx[k]
        elif k in s_qw:
            sd[k] = s_qw[k]

    model.load_state_dict(sd)

    out_model = os.path.join(cfg.out_dir, "mx2lm_hybrid.pt")
    torch.save(model.state_dict(), out_model)

    with open(cfg.mx2lm_grams, "r", encoding="utf-8") as f:
        g_mx = json.load(f)
    with open(cfg.cline_grams, "r", encoding="utf-8") as f:
        g_cl = json.load(f)

    brains = {
        "type": "mx2lm-hybrid-brain",
        "mx2lm_grams": g_mx,
        "cline_grams": g_cl,
    }
    with open(os.path.join(cfg.out_dir, "brains_hybrid.json"), "w", encoding="utf-8") as f:
        json.dump(brains, f)
    with open(os.path.join(cfg.out_dir, "brains_hybrid.scxq2"), "w", encoding="utf-8") as f:
        f.write(pack_scxq2(brains))

    print(f"[MERGE] wrote hybrid model to {out_model}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--mx2lm-ckpt", required=True)
    parser.add_argument("--qwen-ckpt", required=True)
    parser.add_argument("--mx2lm-grams", required=True)
    parser.add_argument("--cline-grams", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    cfg = MergeUltraConfig(
        base_model=args.base_model,
        mx2lm_ckpt=args.mx2lm_ckpt,
        qwen_ckpt=args.qwen_ckpt,
        mx2lm_grams=args.mx2lm_grams,
        cline_grams=args.cline_grams,
        out_dir=args.out,
        alpha=args.alpha,
    )
    merge_checkpoints(cfg)
