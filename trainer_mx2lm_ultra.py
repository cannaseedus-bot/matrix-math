
"""Ultra trainer for MX2LM (primary brain)."""
import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

from .device_utils import get_device
from .grams_utils import build_ngrams, normalize_ngrams
from .checkpoint_system import save_checkpoint, CheckpointMeta
from .scxq2_engine import pack_scxq2
from .xcfe_engine import eval_if_block

@dataclass
class UltraConfig:
    model_name: str
    data_dirs: List[str]
    out_dir: str
    max_length: int = 2048
    batch_size: int = 1
    lr: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    device: str = "auto"

class TextDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
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

def load_corpus(dirs: List[str]) -> List[str]:
    texts: List[str] = []
    exts = (".txt", ".jsonl", ".xjson", ".khl", ".scx", ".html")
    for d in dirs:
        for root, _, files in os.walk(d):
            for name in files:
                if name.lower().endswith(exts):
                    path = os.path.join(root, name)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            txt = f.read()
                        texts.append(txt)
                    except Exception:
                        continue
    return texts

def build_grams_snapshot(texts: List[str], out_dir: str) -> Dict[int, Dict]:
    ngram_tables = build_ngrams(texts, n_values=(1, 2, 3))
    probs = normalize_ngrams(ngram_tables)
    snapshot = {
        "type": "@grams",
        "n": [1, 2, 3],
        "tables": {str(n): {" ".join(k): v for k, v in probs[n].items()} for n in probs},
    }
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "mx2lm_grams.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    scx_blob = pack_scxq2(snapshot)
    with open(os.path.join(out_dir, "mx2lm_grams.scxq2"), "w", encoding="utf-8") as f:
        f.write(scx_blob)
    return snapshot

def ultra_train_mx2lm(cfg: UltraConfig, curriculum_block: Dict[str, Any] | None = None) -> None:
    device = get_device(cfg.device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    texts = load_corpus(cfg.data_dirs)
    grams_snapshot = build_grams_snapshot(texts, os.path.join(cfg.out_dir, "brains"))
    ctx = {
        "stats": {
            "num_docs": len(texts),
            "num_tokens_est": sum(len(t.split()) for t in texts),
        },
        "loss": {"train": 1e9},
        "grams": grams_snapshot,
    }

    if curriculum_block is not None:
        branch = eval_if_block(curriculum_block, ctx)
        if branch == "else":
            cfg.max_steps = min(cfg.max_steps, 200)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    model.to(device)

    dataset = TextDataset(texts, tokenizer, cfg.max_length)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    step = 0
    best_loss = 1e9

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

            ctx["loss"]["train"] = float(loss.detach().cpu().item())

            if step % 50 == 0:
                print(f"[MX2LM] step {step} loss={ctx['loss']['train']:.4f}")

            if step % 200 == 0:
                is_best = ctx["loss"]["train"] < best_loss
                if is_best:
                    best_loss = ctx["loss"]["train"]
                meta = CheckpointMeta(
                    step=step,
                    epoch=epoch,
                    loss=ctx["loss"]["train"],
                    best=is_best,
                    tag="mx2lm_ultra",
                    extra={"num_docs": ctx["stats"]["num_docs"]},
                )
                save_checkpoint(
                    model,
                    optim,
                    meta,
                    out_dir=os.path.join(cfg.out_dir, "checkpoints"),
                )

            if step >= cfg.max_steps:
                return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    cfg = UltraConfig(
        model_name=args.model,
        data_dirs=args.data,
        out_dir=args.out,
        max_steps=args.max_steps,
        device=args.device,
    )
    ultra_train_mx2lm(cfg)
