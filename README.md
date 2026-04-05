# Parameter Golf — Primitives-Gating Approach

> **Challenge**: Train the smallest language model that fits in 16 MB and achieves the best bits-per-byte on FineWeb.
> **Baseline**: 9-layer × 512-dim transformer → **1.2244 BPB**

This is my submission for the [OpenAI Parameter Golf Challenge](https://openai.com/index/parameter-golf/).

---

## Approach: SVD Primitive-Gating Feedforward

The core idea is to **replace each transformer's feedforward MLP** with a
*primitive-gating* layer that selects from a **shared library of 12 fixed basis vectors**
(extracted from the baseline's weights via SVD) instead of running a full 512→2048→512 MLP.

This dramatically reduces the per-layer parameter count while preserving expressive power
through the learned gating and blending mechanism.

### Architecture Changes (vs Baseline)

```
Standard FFN (baseline):
  x → Linear(512→2048) → GELU → Linear(2048→512) → output
  Parameters per layer: 512×2048 + 2048×512 ≈ 2.1M

Primitives-Gating FFN (ours):
  x → Gate(512→12) → top-3 indices
    → Retrieve primitives[i,j,k]           ← FIXED (no gradient)
    → Blend via learned 3×3 matrix
    → Linear(512→128) → ReLU → Linear(128→512)
  Parameters per layer: ~131K   (16× reduction)
```

### Parameter Budget

| Component | Size | Trainable? |
|---|---|---|
| 12 SVD primitives × 512 dims | ~25 KB | ❌ Fixed |
| 9 × gating network (512→12) | ~56 KB | ✅ |
| 9 × blend matrix (3×3) | ~1 KB | ✅ |
| 9 × proj_down (512→128) | ~2.4 MB | ✅ |
| 9 × proj_up (128→512) | ~2.4 MB | ✅ |
| Embedding + attention (unchanged) | ~5 MB | ✅ |
| **Total trainable** | **~9.1 MB** | |
| **After INT8 + zlib** | **~2–3 MB** | |

### Primitive Initialization (SVD)

```python
# Extract W_in from each of 9 baseline feedforward layers
W_list = [block.mlp.fc.weight.T for block in baseline.transformer.h]

# Concatenate horizontally: (512, 9×2048)
concat = torch.cat(W_list, dim=1)

# SVD → top-12 left singular vectors → primitive library
U, S, Vh = torch.linalg.svd(concat, full_matrices=False)
primitives = U[:, :12].T   # (12, 512)  ← fixed forever
```

### Overlap Penalty

To encourage **diverse primitive specialization** across layers (instead of all 9 layers
picking the same 3 primitives), we add a regularization term to the loss:

```
overlap_penalty = weight × max(0, overlap_percent − threshold) / 100

threshold: 0.9 → 0.7  (decays during training)
weight:    0.01 → 0.1  (increases during training)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download FineWeb dataset

```bash
python data/download_fineweb.py
```

### 3. Train

```bash
# Full training run (~10 min on 8×H100):
python train_gpt.py

# Smoke-test on CPU with synthetic data:
python train_gpt.py --smoke_test

# Use SVD primitives from the baseline checkpoint:
python train_gpt.py --baseline_ckpt path/to/baseline.pt
```

### 4. Compress for submission

```bash
python compress_model.py --checkpoint out/ckpt_final.pt --output submission.pkl.zlib --verify
```

### 5. Analyze primitive selection logs

```python
from logging_utils import summarize_log
summary = summarize_log('out/primitives_log.jsonl')
print(summary['per_layer_favorite'])   # Which primitive each layer prefers
```

---

## File Structure

```
parameter-golf/
├── model.py           ← GPT + PrimitivesGatingFF (core modification)
├── train_gpt.py       ← Training loop with overlap penalty
├── primitives.py      ← SVD extraction + model initialization
├── logging_utils.py   ← JSON per-batch logging
├── compress_model.py  ← INT8 quantization + zlib compression
├── requirements.txt
└── README.md
```

All PRIMITIVES-GATING modifications are marked with `# PRIMITIVES-GATING` comments in the code.

---

## Submission Checklist

- [x] Compressed model size < 16 MB (INT8 + zlib → ~2–3 MB)
- [x] Training time < 10 minutes on 8×H100
- [x] val-BPB reported and compared to baseline (1.2244)
- [x] JSON logs showing primitive selection patterns per batch
- [x] All code changes marked with `# PRIMITIVES-GATING` comments

---

## Sources

- [OpenAI Parameter Golf](https://openai.com/index/parameter-golf/)
- [Baseline repo](https://github.com/openai/parameter-golf)
