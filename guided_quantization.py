"""
guided_quantization.py — Guided Weight-Wise Mixed-Precision Quantization
for the OpenAI Parameter Golf Challenge.

Instead of uniform INT8, we learn per-weight importance via
    importance = |weight| × |gradient|
and apply mixed-precision quantization:
  • Top BF16_FRAC  (default 15%) of rows → BFLOAT16  (16 bits/weight)
  • Middle INT8_FRAC (default 35%) rows  → INT8 per-row (8 bits/weight)
  • Bottom INT4_FRAC (default 50%) rows  → INT4 packed  (4 bits/weight)

Expected average ≈ 7.2 bits/weight vs 8 for uniform INT8 — ~10% savings —
while preserving quality on the weights that matter most.
"""

import os
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

# ── Tunable split fractions (must sum ≤ 1.0) ─────────────────────────────────
BF16_FRAC = float(os.environ.get("BF16_FRAC", "0.15"))   # top N%   → BF16
INT8_FRAC = float(os.environ.get("INT8_FRAC", "0.35"))   # middle N% → INT8
INT4_FRAC = float(os.environ.get("INT4_FRAC", "0.50"))   # bottom N% → INT4
# Expected bits/weight = 0.15*16 + 0.35*8 + 0.50*4 = 7.2  (<8 baseline INT8)

# Importance aggregation across columns within a row
IMPORTANCE_REDUCE = os.environ.get("IMPORTANCE_REDUCE", "mean")  # mean|max|norm

# Number of tokens used for importance forward-backward pass
IMPORTANCE_TOKENS = int(os.environ.get("IMPORTANCE_TOKENS", "4096"))

# Tensors smaller than this or matching name patterns → passthrough (fp16)
KEEP_FLOAT_MAX_NUMEL = 65_536
KEEP_FLOAT_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "k_gain",
    "skip_weight", "skip_weights", "primitives", "blend",
)

# INT4 packing constants
INT4_ZERO_POINT = 8    # stored = clipped_signed + 8  → unsigned [1, 15]
INT4_MAX_ABS    = 7    # symmetric clamp to [-7, 7]
SCALE_DTYPE     = torch.float16
INT8_CLIP_Q     = 0.9999984   # 99.99984th-pct clip (matches baseline)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTANCE SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_importance_scores(
    model: nn.Module,
    val_tokens: Tensor,
    seq_len: int,
    device: torch.device,
    vocab_size: int,
) -> Dict[str, Tensor]:
    """
    Run one forward-backward pass on a small token batch and return
    per-row importance scores for every large 2-D float weight matrix.

        importance[name][row] = reduce(|W[row, :]| × |G[row, :]|)

    where reduce is mean / max / norm (controlled by IMPORTANCE_REDUCE).
    Small tensors and control-scalar tensors are excluded (they will be
    stored as passthrough fp16 anyway).
    """
    model.eval()
    model.zero_grad()

    # Build a small batch from the beginning of val_tokens
    n_tok  = min(IMPORTANCE_TOKENS, val_tokens.numel() - seq_len - 1)
    n_tok  = (n_tok // seq_len) * seq_len   # round down to full sequences
    if n_tok < seq_len:
        raise ValueError(f"val_tokens too short for importance pass "
                         f"(need ≥{seq_len + 1}, have {val_tokens.numel()})")

    x = val_tokens[:n_tok].view(-1, seq_len).to(device)
    y = val_tokens[1:n_tok + 1].view(-1, seq_len).to(device)

    with torch.enable_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size).float(),
            y.reshape(-1).long(),
        )
        loss.backward()

    scores: Dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if param.ndim != 2:
            continue
        if param.numel() <= KEEP_FLOAT_MAX_NUMEL:
            continue
        if any(p in name for p in KEEP_FLOAT_PATTERNS):
            continue

        w = param.detach().cpu().float()
        g = param.grad.detach().cpu().float()
        raw = w.abs() * g.abs()                        # (R, C)

        if IMPORTANCE_REDUCE == "max":
            scores[name] = raw.amax(dim=1)
        elif IMPORTANCE_REDUCE == "norm":
            scores[name] = raw.norm(dim=1)
        else:                                          # "mean" (default)
            scores[name] = raw.mean(dim=1)

    model.zero_grad()
    model.train()
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION MAP
# ═══════════════════════════════════════════════════════════════════════════════

def create_row_quant_map(
    importance_scores: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """
    Given per-row importance scores, return a dict:
        quant_map[name] = int8 tensor, shape (n_rows,), values ∈ {4, 8, 16}
    Thresholds are computed globally so the overall BF16/INT8/INT4 split
    matches BF16_FRAC / INT8_FRAC / INT4_FRAC across all quantized rows.
    """
    all_scores = torch.cat([s.flatten() for s in importance_scores.values()]).float()

    # bf16_thresh: top BF16_FRAC rows must be ≥ this value
    bf16_thresh = torch.quantile(all_scores, 1.0 - BF16_FRAC)
    # int4_thresh: bottom INT4_FRAC rows must be < this value
    int4_thresh = torch.quantile(all_scores, INT4_FRAC)

    quant_map: Dict[str, Tensor] = {}
    for name, row_scores in importance_scores.items():
        bits = torch.full((row_scores.numel(),), 8, dtype=torch.int8)
        bits[row_scores >= bf16_thresh] = 16
        bits[row_scores <  int4_thresh] =  4
        quant_map[name] = bits

    return quant_map


# ═══════════════════════════════════════════════════════════════════════════════
# LOW-LEVEL QUANTIZATION PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_rows_int8(rows: Tensor) -> Tuple[Tensor, Tensor]:
    """
    rows : (R, C) float32
    Returns quantized (R, C) int8 and per-row scale (R,) float16.
    Uses 99.99984th-percentile clipping (identical to baseline).
    """
    rows = rows.float()
    clip_abs = torch.quantile(rows.abs(), INT8_CLIP_Q, dim=1)
    scale    = (clip_abs / 127.0).clamp(min=1e-12)
    clipped  = torch.clamp(rows, -clip_abs[:, None], clip_abs[:, None])
    q        = (clipped / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q, scale.to(SCALE_DTYPE)


def dequantize_rows_int8(q: Tensor, scale: Tensor) -> Tensor:
    """(R,C) int8, (R,) float16 → (R,C) float32"""
    return q.float() * scale.float()[:, None]


def quantize_rows_int4(rows: Tensor) -> Tuple[Tensor, Tensor]:
    """
    rows  : (R, C) float32
    Returns:
        packed : (R, ⌈C/2⌉) uint8  — high nibble = even col, low = odd col
        scale  : (R,)        float16 — per-row symmetric scale
    """
    rows = rows.float()
    R, C = rows.shape
    scale = (rows.abs().amax(dim=1) / INT4_MAX_ABS).clamp(min=1e-12)

    q_s  = (rows / scale[:, None]).round().clamp(-INT4_MAX_ABS, INT4_MAX_ABS).to(torch.int8)
    q_u  = (q_s + INT4_ZERO_POINT).to(torch.uint8)    # unsigned [1, 15]

    # Pad to even column count
    if C % 2 != 0:
        pad = torch.zeros(R, 1, dtype=torch.uint8)
        q_u = torch.cat([q_u, pad], dim=1)

    packed = (q_u[:, 0::2] << 4) | q_u[:, 1::2]      # pack two nibbles per byte
    return packed, scale.to(SCALE_DTYPE)


def dequantize_rows_int4(packed: Tensor, scale: Tensor, orig_cols: int) -> Tensor:
    """
    packed    : (R, ⌈C/2⌉) uint8
    scale     : (R,)        float16
    orig_cols : original C before packing
    Returns   : (R, orig_cols) float32
    """
    packed = packed.to(torch.int32)
    high   = (packed >> 4) & 0xF                          # even columns
    low    =  packed        & 0xF                          # odd columns
    q_u    = torch.stack([high, low], dim=2).reshape(packed.shape[0], -1)
    q_u    = q_u[:, :orig_cols]                           # trim padding
    q_s    = (q_u - INT4_ZERO_POINT).float()              # back to signed
    return q_s * scale.float()[:, None]


# ═══════════════════════════════════════════════════════════════════════════════
# MIXED-PRECISION STATE-DICT QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def guided_quantize_state_dict(
    state_dict: Dict[str, Tensor],
    quant_map:  Dict[str, Tensor],
) -> dict:
    """
    Apply per-row mixed-precision quantization to `state_dict`.
    Tensors not covered by quant_map (small, 1-D, control scalars) are stored
    as fp16 passthrough (matching baseline behaviour).

    Returns a serialisable dict for torch.save / guided_dequantize_state_dict.
    """
    result: dict = {"_version": 2, "_tensors": {}}

    for name, t in state_dict.items():
        t = t.cpu()
        is_float = t.dtype in {torch.float32, torch.float16, torch.bfloat16}
        is_small  = t.numel() <= KEEP_FLOAT_MAX_NUMEL
        is_ctrl   = any(p in name for p in KEEP_FLOAT_PATTERNS)
        use_mixed = (is_float and not is_small and not is_ctrl
                     and t.ndim == 2 and name in quant_map)

        if not use_mixed:
            # ── Passthrough ──────────────────────────────────────────────────
            if is_ctrl:
                stored = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                stored = t.to(dtype=torch.float16).contiguous()
            else:
                stored = t.contiguous()
            result["_tensors"][name] = {
                "kind":      "passthrough",
                "tensor":     stored,
                "orig_dtype": str(t.dtype).removeprefix("torch."),
            }
            continue

        # ── Mixed per-row quantization ────────────────────────────────────
        bits      = quant_map[name]          # (n_rows,) int8 ∈ {4, 8, 16}
        t_f32     = t.float()
        n_cols    = t_f32.shape[1]

        bf16_idx  = (bits == 16).nonzero(as_tuple=True)[0]
        int8_idx  = (bits ==  8).nonzero(as_tuple=True)[0]
        int4_idx  = (bits ==  4).nonzero(as_tuple=True)[0]

        entry: dict = {
            "kind":      "mixed",
            "shape":     tuple(t.shape),
            "orig_dtype": str(t.dtype).removeprefix("torch."),
            "n_cols":    n_cols,
        }

        if bf16_idx.numel() > 0:
            entry["bf16_rows"] = bf16_idx.to(torch.int16)
            entry["bf16_data"] = t_f32[bf16_idx].to(torch.bfloat16)

        if int8_idx.numel() > 0:
            q8, s8 = quantize_rows_int8(t_f32[int8_idx])
            entry["int8_rows"]  = int8_idx.to(torch.int16)
            entry["int8_quant"] = q8
            entry["int8_scale"] = s8

        if int4_idx.numel() > 0:
            p4, s4 = quantize_rows_int4(t_f32[int4_idx])
            entry["int4_rows"]   = int4_idx.to(torch.int16)
            entry["int4_packed"] = p4
            entry["int4_scale"]  = s4

        result["_tensors"][name] = entry

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DEQUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def guided_dequantize_state_dict(obj: dict) -> Dict[str, Tensor]:
    """Reconstruct a float state-dict from a guided_quantize_state_dict output."""
    state_dict: Dict[str, Tensor] = {}

    for name, entry in obj["_tensors"].items():
        orig = entry.get("orig_dtype", "float32")

        if entry["kind"] == "passthrough":
            t = entry["tensor"]
            state_dict[name] = t.to(getattr(torch, orig))
            continue

        # Mixed-precision reconstruction
        shape  = entry["shape"]
        n_cols = entry["n_cols"]
        recon  = torch.zeros(shape, dtype=torch.float32)

        if "bf16_rows" in entry:
            rows = entry["bf16_rows"].long()
            recon[rows] = entry["bf16_data"].float()

        if "int8_rows" in entry:
            rows = entry["int8_rows"].long()
            recon[rows] = dequantize_rows_int8(entry["int8_quant"],
                                               entry["int8_scale"])

        if "int4_rows" in entry:
            rows = entry["int4_rows"].long()
            recon[rows] = dequantize_rows_int4(entry["int4_packed"],
                                               entry["int4_scale"], n_cols)

        state_dict[name] = recon.to(getattr(torch, orig))

    return state_dict


# ═══════════════════════════════════════════════════════════════════════════════
# BYTE-BUDGET ESTIMATION  (fast, no actual quantization)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_mixed_precision_bytes(
    state_dict: Dict[str, Tensor],
    quant_map:  Dict[str, Tensor],
) -> dict:
    """
    Fast estimate of raw serialised bytes before zlib compression.
    Useful for tuning BF16_FRAC / INT8_FRAC / INT4_FRAC before committing
    to the full quantization pass.
    """
    bf16_b = int8_b = int4_b = pass_b = 0

    for name, t in state_dict.items():
        t = t.cpu()
        is_float  = t.dtype in {torch.float32, torch.float16, torch.bfloat16}
        is_small  = t.numel() <= KEEP_FLOAT_MAX_NUMEL
        is_ctrl   = any(p in name for p in KEEP_FLOAT_PATTERNS)
        use_mixed = (is_float and not is_small and not is_ctrl
                     and t.ndim == 2 and name in quant_map)

        if not use_mixed:
            pass_b += t.numel() * 2          # stored as fp16
            continue

        bits   = quant_map[name]
        n_cols = t.shape[1]
        n_bf16 = int((bits == 16).sum())
        n_int8 = int((bits ==  8).sum())
        n_int4 = int((bits ==  4).sum())

        bf16_b += n_bf16 * n_cols * 2                   # data (bfloat16)
        bf16_b += n_bf16 * 2                            # row indices (int16)

        int8_b += n_int8 * n_cols * 1                   # data (int8)
        int8_b += n_int8 * 2                            # per-row scale (fp16)
        int8_b += n_int8 * 2                            # row indices (int16)

        int4_b += n_int4 * ((n_cols + 1) // 2)          # packed data (uint8)
        int4_b += n_int4 * 2                            # per-row scale (fp16)
        int4_b += n_int4 * 2                            # row indices (int16)

    total_raw = bf16_b + int8_b + int4_b + pass_b
    return {
        "bf16_bytes":     bf16_b,
        "int8_bytes":     int8_b,
        "int4_bytes":     int4_b,
        "pass_bytes":     pass_b,
        "total_raw_bytes": total_raw,
        "total_raw_MB":   total_raw / 1e6,
    }
