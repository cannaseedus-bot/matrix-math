# matrix-math

## Overview
`matrix-math` is a small collection of Python utilities that sketch out a multi-stage
training workflow for language-model experiments. The repo includes lightweight trainers,
checkpoint helpers, and serialization helpers that together simulate a pipeline for:

- building n-gram “memory” snapshots from text corpora,
- training small language-model adapters on lore data, and
- merging multiple checkpoint sources into a hybrid model artifact.

The code is intentionally minimal: it does not provide a production-ready training stack,
but it does show how the data flows from raw text → n-grams → checkpoints → merged
artifacts. The modules can be used independently or chained together depending on what
part of the workflow you want to prototype.

## What’s included

### Trainers
- **MX2LM trainer (`trainer_mx2lm_ultra.py`)**
  - Loads text corpora from directories, tokenizes with a Hugging Face tokenizer,
    and fine-tunes a base causal LM.
  - Generates an `@grams` snapshot (1–3 grams) from the same corpus and saves it
    alongside checkpoints.
  - Uses the XCFE rule helper to optionally adjust training steps based on a
    curriculum block.

- **Qwen adapter trainer (`trainer_qwen_ultra.py`)**
  - Loads “lore” text files, fine-tunes a base model, and writes checkpoints.
  - Designed to act like a lightweight adapter training stage for later merging.

- **Cline memory trainer (`trainer_cline_ultra.py`)**
  - Scans source-like “lore” data (Java/log/text/proto) and produces an `@grams`
    snapshot representing a lightweight memory pack.

- **Hybrid merge (`trainer_merge_ultra.py`)**
  - Loads a base model and two checkpoint state dicts, then interpolates weights
    with a configurable alpha.
  - Packages a “hybrid brain” JSON document combining MX2LM and Cline n-grams.

### Engines & utilities
- **Checkpoint system (`checkpoint_system.py`)**
  - Saves model/optimizer state along with JSON metadata.
  - Tracks `last.json` and optional `best.pt` snapshots.

- **N-gram utilities (`grams_utils.py`)**
  - Builds frequency tables and normalized probability tables for n-grams.

- **SCXQ2 packer (`scxq2_engine.py`)**
  - Compresses JSON payloads (like `@grams`) into a base64/zlib blob.

- **XCFE evaluator (`xcfe_engine.py`)**
  - Evaluates simple `@if` control-flow blocks with comparison operators.

- **Device selection (`device_utils.py`)**
  - Chooses CPU, CUDA, or MPS depending on availability.

## How the pieces fit together
1. **Corpus ingestion**: trainers walk directories and load text-like files into
   a list of strings.
2. **Training loop**: MX2LM/Qwen trainers use Hugging Face tokenizers and
   `AutoModelForCausalLM` with a standard AdamW + linear warmup schedule.
3. **Memory snapshots**: n-grams are computed from the same corpora and saved in
   both JSON and compressed SCXQ2 formats.
4. **Checkpointing**: model weights + optimizer state are saved periodically,
   with metadata for resuming and “best” tracking.
5. **Merging**: the hybrid merge script interpolates two checkpoint sets and
   bundles the `@grams` memories into a final artifact.

## Requirements
- Python 3.10+
- PyTorch
- `transformers`

Example install (adjust to your environment):

```bash
pip install torch transformers
```

## Usage examples

### Train MX2LM
```bash
python -m trainer_mx2lm_ultra \
  --model gpt2 \
  --data ./data/corpus \
  --out ./runs/mx2lm
```

### Train Qwen adapter
```bash
python -m trainer_qwen_ultra \
  --model gpt2 \
  --lore ./data/lore \
  --out ./runs/qwen
```

### Build Cline memory
```bash
python -m trainer_cline_ultra \
  --lore ./data/cline_lore \
  --out ./runs/cline
```

### Merge checkpoints
```bash
python -m trainer_merge_ultra \
  --base-model gpt2 \
  --mx2lm-ckpt ./runs/mx2lm/checkpoints/step_00000200.pt \
  --qwen-ckpt ./runs/qwen/checkpoints_qwen/step_00000200.pt \
  --mx2lm-grams ./runs/mx2lm/brains/mx2lm_grams.json \
  --cline-grams ./runs/cline/brains/cline_grams.json \
  --out ./runs/hybrid \
  --alpha 0.5
```

## Outputs to expect
- **Checkpoints**: `.pt` files plus `last.json`/`best.json` metadata.
- **N-gram snapshots**: `.json` (readable) and `.scxq2` (compressed) files in
  `brains/` directories.
- **Hybrid bundles**: merged model weights plus combined `brains_hybrid.json` and
  `brains_hybrid.scxq2`.

## Repository layout
```
.
├── micronaut
│   ├── brains
│   ├── io
│   ├── micronaut.ps1
│   ├── micronaut.s7
│   ├── object.toml
│   ├── proof
│   ├── semantics.xjson
│   └── trace
├── checkpoint_system.py
├── device_utils.py
├── grams_utils.py
├── scxq2_engine.py
├── trainer_cline_ultra.py
├── trainer_merge_ultra.py
├── trainer_mx2lm_ultra.py
├── trainer_qwen_ultra.py
└── xcfe_engine.py
```

## Micronaut object layout
The `micronaut/` directory contains a file-centric SCO/1 object with sealed
brains, append-only IO files, and a PowerShell orchestrator that routes
REST/file interactions without host-side inference logic.

## Notes
- The trainers use minimal error handling and assume local filesystem data.
- The SCXQ2 format here is a simple zlib+base64 wrapper for prototyping.
- `matrix-math` is best viewed as a skeleton for experiments rather than a
  hardened training framework.
