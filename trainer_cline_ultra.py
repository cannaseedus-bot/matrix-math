
"""Trainer for Cline JAR-derived embeddings → n-gram + vector memory."""
import os
import json
from dataclasses import dataclass
from typing import List

from .grams_utils import build_ngrams, normalize_ngrams
from .scxq2_engine import pack_scxq2

@dataclass
class ClineUltraConfig:
    jar_lore_dirs: List[str]
    out_dir: str

def _gather_jar_lore(dirs: List[str]) -> List[str]:
    exts = (".java", ".log", ".txt", ".proto")
    buf = []
    for d in dirs:
        for root, _, files in os.walk(d):
            for name in files:
                if name.lower().endswith(exts):
                    try:
                        with open(os.path.join(root, name), "r", encoding="utf-8", errors="ignore") as f:
                            buf.append(f.read())
                    except Exception:
                        continue
    return buf

def train_cline_memory(cfg: ClineUltraConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    texts = _gather_jar_lore(cfg.jar_lore_dirs)
    tables = build_ngrams(texts, n_values=(1, 2, 3))
    probs = normalize_ngrams(tables)

    snapshot = {
        "type": "@grams",
        "agent": "cline",
        "tables": {str(n): {" ".join(k): v for k, v in probs[n].items()} for n in probs},
    }
    raw_path = os.path.join(cfg.out_dir, "brains", "cline_grams.json")
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f)
    blob = pack_scxq2(snapshot)
    with open(os.path.join(cfg.out_dir, "brains", "cline_grams.scxq2"), "w", encoding="utf-8") as f:
        f.write(blob)
    print(f"[CLINE] wrote @grams snapshot to {raw_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lore", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = ClineUltraConfig(jar_lore_dirs=args.lore, out_dir=args.out)
    train_cline_memory(cfg)
