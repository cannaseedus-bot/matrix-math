
"""Ultra trainer for Qwen adapter (ASX patch)."""
import os
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from .device_utils import get_device
from .checkpoint_system import save_checkpoint, CheckpointMeta

@dataclass
class QwenUltraConfig:
    base_model: str
    lore_dirs: List[str]
    out_dir: str
    max_length: int = 2048
    batch_size: int = 1
    lr: float = 5e-5
    warmup_steps: int = 100
    max_steps: int = 800
    device: str = "auto"

class LoreDataset(Dataset):
    def __init__(self, texts, tok, max_length):
        self.texts = texts
        self.tok = tok
        self.max_length = max_length

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        return enc

def _load_lore(dirs: List[str]) -> List[str]:
    exts = (".txt", ".log", ".jsonl")
    buf = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for name in files:
                if name.lower().endswith(exts):
                    with open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as f:
                        buf.append(f.read())
    return buf

def train_qwen_ultra(cfg: QwenUltraConfig) -> None:
    device = get_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    texts = _load_lore(cfg.lore_dirs)
    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model.to(device)

    ds = LoreDataset(texts, tok, cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    step = 0
    best = 1e9

    for epoch in range(1000000):
        for batch in loader:
            step += 1
            model.train()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            model.zero_grad()

            if step % 50 == 0:
                print(f"[QWEN] step {step} loss={loss.item():.4f}")

            if step % 200 == 0:
                l = float(loss.detach().cpu().item())
                is_best = l < best
                if is_best:
                    best = l
                meta = CheckpointMeta(step=step, epoch=epoch, loss=l, best=is_best, tag="qwen_ultra")
                save_checkpoint(
                    model,
                    optim,
                    meta,
                    out_dir=os.path.join(cfg.out_dir, "checkpoints_qwen"),
                )

            if step >= cfg.max_steps:
                return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lore", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()

    cfg = QwenUltraConfig(
        base_model=args.model,
        lore_dirs=args.lore,
        out_dir=args.out,
        max_steps=args.max_steps,
        device=args.device,
    )
    train_qwen_ultra(cfg)
