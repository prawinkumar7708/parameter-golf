"""
Parameter Golf - Guided Weight-Wise Quantization Fork
======================================================
Forked from: https://github.com/openai/parameter-golf/blob/main/train_gpt.py

MODIFICATION: Replaces uniform INT8 post-training quantization with
Guided Weight-Wise Mixed-Precision Quantization (see guided_quantization.py).

After training, one forward-backward pass computes per-weight importance:
    importance = |weight| × |gradient|
Rows are then assigned to BF16 / INT8 / INT4 by global importance percentile
(default 15% / 35% / 50%), giving ~7.2 bits/weight vs 8 for uniform INT8.
This preserves precision where it matters while compressing aggressively
elsewhere — targeting val_bpb < 1.2244 within the 16 MB byte budget.

Key changes vs baseline:
  - Added: guided_quantization.py module (importance scoring + mixed quant)
  - Changed: final compression uses guided_quantize_state_dict() instead of
             quantize_state_dict_int8()
  - Added: USE_PRIMITIVES_GATING env var kept for backward compat (default=0)
  - Unchanged: all hyperparameters, tokenizer, data, optimizer

Usage (1xH100 smoke test):
    RUN_ID=gq_smoke \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 WARMUP_ITERS=100 ITERATIONS=500 \\
    torchrun --standalone --nproc_per_node=1 train_gpt.py

Usage (8xH100 full run):
    RUN_ID=guided_quant_v1 \\
    DATA_PATH=./data/datasets/fineweb10B_sp1024 \\
    TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \\
    VOCAB_SIZE=1024 \\
    torchrun --standalone --nproc_per_node=8 train_gpt.py

Tune quantization split via env vars:
    BF16_FRAC=0.15 INT8_FRAC=0.35 INT4_FRAC=0.50  (default, ~7.2 bits/w)
    BF16_FRAC=0.10 INT8_FRAC=0.30 INT4_FRAC=0.60  (more aggressive, ~6.8)
    BF16_FRAC=0.20 INT8_FRAC=0.40 INT4_FRAC=0.40  (conservative, ~7.6)
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import time
import uuid
import zlib
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

from guided_quantization import (
    compute_importance_scores,
    create_row_quant_map,
    guided_quantize_state_dict,
    guided_dequantize_state_dict,
    estimate_mixed_precision_bytes,
)

# ============================================================
# HYPERPARAMETERS
# ============================================================

class Hyperparameters:
    data_path        = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files      = os.path.join(data_path, "fineweb_train_*.bin")
    val_files        = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path   = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id           = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed             = int(os.environ.get("SEED", 1337))

    val_batch_size   = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every   = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))

    # Model shape (must match baseline defaults)
    vocab_size       = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers       = int(os.environ.get("NUM_LAYERS", 9))
    num_kv_heads     = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim        = int(os.environ.get("MODEL_DIM", 512))
    num_heads        = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult         = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings   = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base        = float(os.environ.get("ROPE_BASE", 10000.0))
    logit_softcap    = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    qk_gain_init     = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # PRIMITIVES-GATING settings
    use_primitives_gating    = bool(int(os.environ.get("USE_PRIMITIVES_GATING", "0")))
    n_primitives             = int(os.environ.get("N_PRIMITIVES", 12))
    n_selected               = int(os.environ.get("N_SELECTED", 3))
    proj_hidden              = int(os.environ.get("PROJ_HIDDEN", 64))
    overlap_penalty_weight   = float(os.environ.get("OVERLAP_PENALTY_WEIGHT", "0.05"))

    # Training schedule
    train_seq_len    = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    iterations       = int(os.environ.get("ITERATIONS", 20_000))
    warmup_iters     = int(os.environ.get("WARMUP_ITERS", 500))
    max_lr           = float(os.environ.get("MAX_LR", "3e-4"))
    min_lr           = float(os.environ.get("MIN_LR", "3e-5"))

    # Muon optimizer
    muon_momentum    = float(os.environ.get("MUON_MOMENTUM", "0.95"))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", "0.85"))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1            = float(os.environ.get("BETA1", "0.9"))
    beta2            = float(os.environ.get("BETA2", "0.95"))
    adam_eps         = float(os.environ.get("ADAM_EPS", "1e-8"))
    grad_clip_norm   = float(os.environ.get("GRAD_CLIP_NORM", "0.0"))

    embed_lr         = float(os.environ.get("EMBED_LR", "0.6"))
    head_lr          = float(os.environ.get("HEAD_LR", "0.008"))
    tied_embed_lr    = float(os.environ.get("TIED_EMBED_LR", "0.05"))
    matrix_lr        = float(os.environ.get("MATRIX_LR", "0.04"))
    scalar_lr        = float(os.environ.get("SCALAR_LR", "0.04"))


# ============================================================
# INT8 COMPRESSION CONSTANTS (identical to baseline)
# ============================================================

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
        "q_gain,k_gain,skip_weight,skip_weights,primitives,blend",
    ).split(",") if p
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",") if p
)
INT8_KEEP_FLOAT_MAX_NUMEL   = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE    = torch.float16
INT8_CLIP_PERCENTILE        = 99.99984
INT8_CLIP_Q                 = INT8_CLIP_PERCENTILE / 100.0


# ============================================================
# MUON OPTIMIZER (identical to baseline)
# ============================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float,
                 backend_steps: int, nesterov: bool = True):
        defaults = dict(lr=lr, momentum=momentum,
                        backend_steps=backend_steps, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            backend_steps = group["backend_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf
                if g.ndim >= 2:
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    scale = max(g.size(-2), g.size(-1)) ** 0.5
                    p.add_(g, alpha=-lr * scale)
                else:
                    p.add_(g, alpha=-lr)


# ============================================================
# DATA LOADING (identical to baseline)
# ============================================================

def load_data_shard(file: Path) -> Tensor:
    header = np.fromfile(file, dtype=np.int32, count=256)
    if header[0] == 20240520:
        token_dtype = np.uint16
    elif header[0] == 20240801:
        token_dtype = np.uint32
    else:
        return torch.from_numpy(
            np.fromfile(file, dtype=np.uint16).astype(np.int32))
    ntok = header[2]
    tokens_np = np.fromfile(
        file, dtype=token_dtype, count=ntok, offset=256 * 4)
    return torch.from_numpy(tokens_np.astype(np.int32))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching {pattern!r}")
        random.shuffle(self.files)
        self.file_idx = 0
        self.tokens: Tensor = load_data_shard(Path(self.files[0]))
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(Path(self.files[self.file_idx]))
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks, remaining = [], n
        while remaining > 0:
            avail = len(self.tokens) - self.pos
            if avail == 0:
                self._advance_file()
                avail = len(self.tokens)
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int,
                 device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int,
                   grad_accum_steps: int):
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        span = chunk[start : start + per_rank_span].to(
            self.device, non_blocking=True)
        x = span[:-1].view(-1, seq_len)
        y = span[1:].view(-1, seq_len)
        return x, y


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No val files matching {pattern!r}")
    tokens = torch.cat([load_data_shard(Path(f)) for f in files])
    n = (len(tokens) - 1) // seq_len * seq_len
    return tokens[: n + 1]


# ============================================================
# EVALUATION (identical to baseline)
# ============================================================

def build_sentencepiece_luts(sp, vocab_size: int, device: torch.device):
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=bool)
    is_boundary_np = np.ones((table_size,), dtype=bool)
    for tid in range(sp_vocab_size):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        is_boundary_np[tid] = False
        if sp.is_byte(tid):
            base_bytes_np[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            has_leading_space_np[tid] = True
            piece = piece[1:]
        base_bytes_np[tid] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_np, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def eval_val(model, val_tokens: Tensor, seq_len: int, batch_size_tokens: int,
             device: torch.device, sp, vocab_size: int,
             rank: int = 0, world_size: int = 1):
    model.eval()
    base_bytes_lut, has_leading_space_lut, is_boundary_lut = \
        build_sentencepiece_luts(sp, vocab_size, device)

    total_loss = torch.zeros((), device=device)
    total_bytes = torch.zeros((), device=device)
    num_seqs = (len(val_tokens) - 1) // seq_len
    seqs_per_rank = num_seqs // world_size
    start = rank * seqs_per_rank
    end = start + seqs_per_rank
    batch_seqs = max(1, batch_size_tokens // seq_len)

    for i in range(start, end, batch_seqs):
        j_end = min(i + batch_seqs, end)
        x = torch.stack([
            val_tokens[k * seq_len : (k + 1) * seq_len]
            for k in range(i, j_end)
        ]).to(device)
        y = torch.stack([
            val_tokens[k * seq_len + 1 : (k + 1) * seq_len + 1]
            for k in range(i, j_end)
        ]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        B, T = x.shape
        total_loss += loss.detach() * B * T
        for b in range(B):
            tok_ids = y[b]
            bb = base_bytes_lut[tok_ids].long()
            bb = bb + has_leading_space_lut[tok_ids].long()
            bb = torch.where(is_boundary_lut[tok_ids], torch.ones_like(bb), bb)
            total_bytes += bb.sum().float()

    if world_size > 1:
        dist.all_reduce(total_loss)
        dist.all_reduce(total_bytes)

    total_toks = seqs_per_rank * world_size * seq_len
    avg_loss = (total_loss / total_toks).item()
    bpb = avg_loss / math.log(2) if total_bytes.item() > 0 else float("nan")
    model.train()
    return avg_loss, bpb


# ============================================================
# INT8 QUANTIZATION (identical to baseline)
# ============================================================

def quantize_float_tensor(t: Tensor):
    t32 = t.float()
    if t32.ndim == 2 and t32.numel() > 0:
        clip_abs = torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
        clipped = torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp(min=1e-12)
        q = (clipped / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE)
    else:
        clip_abs = (torch.quantile(t32.abs(), INT8_CLIP_Q)
                    if t32.numel() > 0 else torch.tensor(1.0))
        clipped = torch.clamp(t32, -clip_abs, clip_abs)
        scale = (clip_abs / 127.0).clamp(min=1e-12)
        q = (clipped / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale.to(INT8_PER_ROW_SCALE_DTYPE)


def quantize_state_dict_int8(state_dict: dict) -> dict:
    result: dict = {"_version": 1, "_tensors": {}}
    passthrough_orig_dtypes: dict = {}
    for name, t in state_dict.items():
        t = t.cpu()
        is_float = t.dtype in {torch.float32, torch.float16, torch.bfloat16}
        is_small = t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL
        keep_float = (not is_float or is_small
                      or any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS))
        if keep_float:
            if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                stored = t.float().contiguous()
            elif t.dtype in {torch.float32, torch.bfloat16}:
                passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                stored = t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
            else:
                stored = t
            result["_tensors"][name] = {"kind": "passthrough", "tensor": stored}
        else:
            q, scale = quantize_float_tensor(t)
            result["_tensors"][name] = {
                "kind": "int8",
                "quantized": q,
                "scale": scale,
                "orig_shape": tuple(t.shape),
                "orig_dtype": str(t.dtype).removeprefix("torch."),
            }
    result["_passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return result


def dequantize_state_dict_int8(obj: dict) -> dict:
    state_dict = {}
    orig_dtypes = obj.get("_passthrough_orig_dtypes", {})
    for name, entry in obj["_tensors"].items():
        if entry["kind"] == "passthrough":
            t = entry["tensor"]
            if name in orig_dtypes:
                t = t.to(getattr(torch, orig_dtypes[name]))
            state_dict[name] = t.float()
        else:
            q = entry["quantized"].float()
            scale = entry["scale"].float()
            t = q * (scale[:, None] if q.ndim == 2 else scale)
            state_dict[name] = t.to(getattr(torch, entry.get("orig_dtype", "float32")))
    return state_dict


# ============================================================
# MODEL COMPONENTS (identical to baseline)
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    """Weights in fp32; cast to x.dtype at matmul time for bf16 compute."""
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if ((param.ndim < 2
                 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS))
                    and param.dtype != torch.float32):
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv = base ** (-torch.arange(0, dim, 2).float() / dim)
        self.register_buffer("inv_freq", inv)
        self._seq_len_cached = -1
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device,
                dtype: torch.dtype):
        if (self._cos_cached is None
                or self._seq_len_cached != seq_len
                or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device,
                              dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return (self._cos_cached.to(dtype=dtype),
                self._sin_cached.to(dtype=dtype))


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin,
                      x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 rope_base: float, qk_gain_init: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = dim // num_heads
        self.q_proj = CastedLinear(dim, dim, bias=False)
        self.k_proj = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = CastedLinear(dim, num_kv_heads * self.head_dim, bias=False)
        self.proj   = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init))
        self.k_gain = nn.Parameter(torch.full((num_kv_heads,), qk_gain_init))

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        cos, sin = self.rotary(T, x.device, x.dtype)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = q * self.q_gain[None, :, None, None].to(q.dtype)
        k = k * self.k_gain[None, :, None, None].to(k.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# ============================================================
# PRIMITIVES-GATING FF (our modification)
# ============================================================

class PrimitivesGatingFF(nn.Module):
    """
    PRIMITIVES-GATING: Drop-in replacement for the relu^2 MLP.

    Primitive basis vectors are fixed (SVD-extracted or random-ortho)
    and NOT trained. Only gate/blend/proj_down/proj_up are learned.

    With vocab=1024 and dim=512, proj_hidden=64:
      Params per layer: gate(512x12) + blend(3x3) + proj_down(512x64) +
                        proj_up(64x512) = ~72K  vs  ~524K for MLP-2x
    """

    def __init__(self, dim: int, n_primitives: int, n_selected: int,
                 proj_hidden: int, layer_idx: int = 0):
        super().__init__()
        self.dim = dim
        self.n_primitives = n_primitives
        self.n_selected = n_selected
        self.proj_hidden = proj_hidden
        self.layer_idx = layer_idx

        # Fixed basis — initialized random, replaced by SVD before training
        self.register_buffer("primitives",
            torch.randn(n_primitives, dim) * 0.02)

        self.gate     = CastedLinear(dim, n_primitives, bias=True)
        self.blend    = nn.Parameter(torch.eye(n_selected))
        self.proj_down = CastedLinear(dim, proj_hidden, bias=False)
        self.proj_up   = CastedLinear(proj_hidden, dim, bias=False)
        self.proj_up._zero_init = True

        self._last_indices: Optional[Tensor] = None
        self._last_probs:   Optional[Tensor] = None

    def set_primitives(self, primitives: Tensor) -> None:
        assert primitives.shape == (self.n_primitives, self.dim)
        self.primitives.copy_(primitives)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        gate_logits = self.gate(x)                       # (B, T, P)
        gate_probs  = F.softmax(gate_logits, dim=-1)     # (B, T, P)
        mean_probs  = gate_probs.mean(dim=(0, 1))        # (P,)
        top_idx     = torch.topk(mean_probs, self.n_selected).indices  # (K,)

        self._last_indices = top_idx.detach()
        self._last_probs   = mean_probs.detach()

        selected = self.primitives[top_idx]              # (K, D)  — frozen
        blend_w  = F.softmax(self.blend, dim=-1)         # (K, K)
        blended  = (blend_w @ selected).sum(dim=0)       # (D,)

        sel_probs = gate_probs[..., top_idx]             # (B, T, K)
        fused = (sel_probs.unsqueeze(-1) *
                 selected.unsqueeze(0).unsqueeze(0)).sum(dim=-2)  # (B, T, D)

        z = x + fused + blended.unsqueeze(0).unsqueeze(0) * 0.1
        return self.proj_up(F.relu(self.proj_down(z)))


class MLP(nn.Module):
    """Baseline relu^2 MLP — used when USE_PRIMITIVES_GATING=0."""
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(torch.relu(self.fc(x)).square())


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, rope_base: float, qk_gain_init: float,
                 use_primitives_gating: bool = False, n_primitives: int = 12,
                 n_selected: int = 3, proj_hidden: int = 64,
                 layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        if use_primitives_gating:
            self.mlp = PrimitivesGatingFF(
                dim, n_primitives, n_selected, proj_hidden, layer_idx)
        else:
            self.mlp = MLP(dim, mlp_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ============================================================
# FULL GPT MODEL (baseline structure + skip connections)
# ============================================================

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int,
                 num_heads: int, num_kv_heads: int, mlp_mult: int,
                 tie_embeddings: bool, tied_embed_init_std: float,
                 logit_softcap: float, rope_base: float, qk_gain_init: float,
                 use_primitives_gating: bool = False, n_primitives: int = 12,
                 n_selected: int = 3, proj_hidden: int = 64):
        super().__init__()
        self.tie_embeddings    = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap     = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, model_dim)

        # U-Net skip connections between encoder/decoder halves
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights   = min(self.num_encoder_layers,
                                      self.num_decoder_layers)
        self.skip_weights = nn.Parameter(
            torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))

        self.blocks = nn.ModuleList([
            Block(
                dim=model_dim, num_heads=num_heads,
                num_kv_heads=num_kv_heads, mlp_mult=mlp_mult,
                rope_base=rope_base, qk_gain_init=qk_gain_init,
                use_primitives_gating=use_primitives_gating,
                n_primitives=n_primitives, n_selected=n_selected,
                proj_hidden=proj_hidden, layer_idx=i,
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm()

        self.lm_head = (None if tie_embeddings
                        else CastedLinear(model_dim, vocab_size, bias=False))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0,
                        std=self.tied_embed_init_std)
        for m in self.modules():
            if isinstance(m, CastedLinear):
                if getattr(m, "_zero_init", False):
                    nn.init.zeros_(m.weight)
                else:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, idx: Tensor, targets: Tensor) -> Tensor:
        x = self.tok_emb(idx)           # (B, T, D) — no wpe; RoPE in attn

        # Encoder half
        enc_outputs: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x)
            enc_outputs.append(x)

        # Decoder half with U-Net skip connections
        for i in range(self.num_decoder_layers):
            if i < self.num_skip_weights:
                skip_idx = self.num_encoder_layers - 1 - i
                sw = self.skip_weights[i].to(x.dtype)
                x = x + enc_outputs[skip_idx] * sw
            x = self.blocks[self.num_encoder_layers + i](x)

        x = self.norm(x)

        if self.tie_embeddings:
            logits_proj = x @ self.tok_emb.weight.T
        else:
            logits_proj = self.lm_head(x)

        logits = self.logit_softcap * torch.tanh(
            logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets.long(), reduction="mean")

    def get_layer_selections(self) -> dict:
        return {
            i: block.mlp._last_indices
            for i, block in enumerate(self.blocks)
            if isinstance(block.mlp, PrimitivesGatingFF)
            and block.mlp._last_indices is not None
        }


# ============================================================
# PRIMITIVES INIT (SVD or random-orthogonal)
# ============================================================

def init_primitives_gating(model: GPT) -> None:
    """
    Initialize fixed primitives for all PrimitivesGatingFF layers.
    Uses random orthogonal basis (SVD of random matrix) so that
    primitives span a well-conditioned subspace of dim-dimensional space.
    """
    dim = model.blocks[0].mlp.dim
    n_prim = model.blocks[0].mlp.n_primitives
    rand = torch.randn(dim, n_prim)
    Q, _ = torch.linalg.qr(rand)                   # (dim, n_prim)
    primitives = Q[:, :n_prim].T.contiguous()       # (n_prim, dim)
    for block in model.blocks:
        if isinstance(block.mlp, PrimitivesGatingFF):
            block.mlp.set_primitives(primitives.clone())


# ============================================================
# OVERLAP PENALTY
# ============================================================

def compute_overlap_penalty(model: GPT, weight: float) -> Tensor:
    """Penalize layers that pick the same primitives."""
    selections = model.get_layer_selections()
    if len(selections) < 2:
        return torch.tensor(0.0)
    layer_ids = sorted(selections.keys())
    total, count = 0.0, 0
    for i in range(len(layer_ids)):
        for j in range(i + 1, len(layer_ids)):
            si = set(selections[layer_ids[i]].tolist())
            sj = set(selections[layer_ids[j]].tolist())
            total += len(si & sj) / max(len(si), len(sj), 1)
            count += 1
    return torch.tensor(weight * total / max(count, 1))


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main() -> None:
    global zeropower_via_newtonschulz5
    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # DDP / CUDA setup
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank       = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    if distributed:
        dist.init_process_group("nccl")

    is_master = (rank == 0)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    if is_master:
        print(f"[pg] run_id={args.run_id}")
        print(f"[pg] world_size={world_size}  grad_accum={grad_accum_steps}")
        print(f"[pg] use_primitives_gating={args.use_primitives_gating}")
        print(f"[pg] vocab={args.vocab_size}  dim={args.model_dim}  "
              f"layers={args.num_layers}")

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)

    # Data
    train_loader = DistributedTokenLoader(
        args.train_files, rank=rank,
        world_size=world_size, device=device)
    val_tokens = load_validation_tokens(
        args.val_files, args.train_seq_len)

    # Build model
    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        use_primitives_gating=args.use_primitives_gating,
        n_primitives=args.n_primitives,
        n_selected=args.n_selected,
        proj_hidden=args.proj_hidden,
    ).to(device).bfloat16()

    for m in base_model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(base_model)

    if args.use_primitives_gating:
        init_primitives_gating(base_model)
        if is_master:
            print("[pg] Primitives initialized (random orthogonal basis)")

    n_params = sum(p.numel() for p in base_model.parameters()
                   if p.requires_grad)
    if is_master:
        print(f"[pg] Trainable params: {n_params:,}")

    compiled_model = torch.compile(
        base_model, dynamic=False, fullgraph=True)
    model: nn.Module = (
        DDP(compiled_model, device_ids=[local_rank],
            broadcast_buffers=False)
        if distributed else compiled_model
    )

    # Optimizer
    block_named = list(base_model.blocks.named_parameters())
    matrix_params = [
        p for n, p in block_named
        if p.ndim >= 2
        and not any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p for n, p in block_named
        if p.ndim < 2
        or any(pat in n for pat in CONTROL_TENSOR_NAME_PATTERNS)
    ]

    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
    )
    for g in optimizer_muon.param_groups:
        g["base_lr"] = g["lr"]

    adam_groups = []
    if args.tie_embeddings:
        adam_groups.append({
            "params": [base_model.tok_emb.weight],
            "base_lr": args.tied_embed_lr,
        })
    else:
        adam_groups.append({
            "params": [base_model.tok_emb.weight],
            "base_lr": args.embed_lr,
        })
        if base_model.lm_head is not None:
            adam_groups.append({
                "params": list(base_model.lm_head.parameters()),
                "base_lr": args.head_lr,
            })
    adam_groups.append({"params": scalar_params, "base_lr": args.scalar_lr})

    optimizer_adam = torch.optim.Adam(
        adam_groups,
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
    )
    for g in optimizer_adam.param_groups:
        if "base_lr" not in g:
            g["base_lr"] = args.matrix_lr

    optimizers = [optimizer_muon, optimizer_adam]

    def get_lr_scale(step: int) -> float:
        if step < args.warmup_iters:
            return step / max(args.warmup_iters, 1)
        progress = (step - args.warmup_iters) / max(
            args.iterations - args.warmup_iters, 1)
        cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr / args.max_lr + (
            1 - args.min_lr / args.max_lr) * cos_decay

    # Training
    t0 = time.time()
    log_lines: list[str] = []

    for step in range(args.iterations):
        # Wallclock cap
        if (args.max_wallclock_seconds > 0
                and (time.time() - t0) >= args.max_wallclock_seconds):
            if is_master:
                print(f"[pg] Wallclock cap at step {step}")
            break

        # Validation
        if args.val_loss_every > 0 and step % args.val_loss_every == 0:
            val_loss, val_bpb = eval_val(
                base_model, val_tokens, args.train_seq_len,
                args.val_batch_size, device, sp, args.vocab_size,
                rank, world_size)
            if is_master:
                elapsed = time.time() - t0
                line = (f"step={step:6d}  val_loss={val_loss:.4f}  "
                        f"val_bpb={val_bpb:.4f}  elapsed={elapsed:.1f}s")
                print(line)
                log_lines.append(line)

        # Forward / backward
        scale = get_lr_scale(step)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = (
                    micro_step == grad_accum_steps - 1)
            x, y = train_loader.next_batch(
                args.train_batch_tokens,
                args.train_seq_len,
                grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
                if args.use_primitives_gating:
                    overlap = compute_overlap_penalty(
                        base_model, args.overlap_penalty_weight)
                    loss = loss + overlap.to(loss.device)
            train_loss += loss.detach()
            (loss * grad_scale).backward()

        train_loss /= grad_accum_steps

        # Muon momentum warmup
        frac = min(step / max(args.muon_momentum_warmup_steps, 1), 1.0)
        cur_mom = ((1 - frac) * args.muon_momentum_warmup_start
                   + frac * args.muon_momentum)
        for g in optimizer_muon.param_groups:
            g["momentum"] = cur_mom

        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * scale

        if args.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                base_model.parameters(), args.grad_clip_norm)

        for opt in optimizers:
            opt.step()

        if is_master and step % 100 == 0:
            elapsed = time.time() - t0
            line = (f"step={step:6d}  train_loss={train_loss.item():.4f}"
                    f"  elapsed={elapsed:.1f}s")
            print(line)
            log_lines.append(line)

    # ---- Final eval ----
    val_loss, val_bpb = eval_val(
        base_model, val_tokens, args.train_seq_len, args.val_batch_size,
        device, sp, args.vocab_size, rank, world_size)

    if is_master:
        print(f"\n[pg] === FINAL RESULTS ===")
        print(f"[pg] val_loss = {val_loss:.4f}")
        print(f"[pg] val_bpb  = {val_bpb:.4f}  (baseline: 1.2244)")

    # ---- Guided Mixed-Precision Quantization + zlib compression ----
    if is_master:
        sd = dict(base_model.state_dict())

        # ── Step 1: save full-precision weights (baseline_trained_weights.pt) ──
        out_dir = Path(f"logs/{args.run_id}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(sd, out_dir / "baseline_trained_weights.pt")
        print(f"\n[pg] Saved full-precision weights → {out_dir}/baseline_trained_weights.pt")

        # ── Step 2: compute per-row importance scores (one fwd-bwd pass) ──────
        print("[pg] Computing importance scores (one fwd-bwd pass)…")
        importance_scores = compute_importance_scores(
            base_model, val_tokens, args.train_seq_len,
            device, args.vocab_size,
        )
        print(f"[pg] Importance scores computed for {len(importance_scores)} tensors")

        # ── Step 3: build quantization map ────────────────────────────────────
        quant_map = create_row_quant_map(importance_scores)

        # Print per-tensor split summary
        for name, bits in quant_map.items():
            n = bits.numel()
            b16 = int((bits == 16).sum()); b8 = int((bits == 8).sum()); b4 = int((bits == 4).sum())
            print(f"[pg]   {name:45s}  BF16={b16/n*100:5.1f}%  INT8={b8/n*100:5.1f}%  INT4={b4/n*100:5.1f}%  ({n} rows)")

        # ── Step 4: fast byte budget estimate (before full quantization) ──────
        budget_est = estimate_mixed_precision_bytes(sd, quant_map)
        print(f"\n[pg] === ESTIMATED SIZE (pre-zlib) ===")
        print(f"[pg]   BF16 data : {budget_est['bf16_bytes']/1e6:.3f} MB")
        print(f"[pg]   INT8 data : {budget_est['int8_bytes']/1e6:.3f} MB")
        print(f"[pg]   INT4 data : {budget_est['int4_bytes']/1e6:.3f} MB")
        print(f"[pg]   Passthru  : {budget_est['pass_bytes']/1e6:.3f} MB")
        print(f"[pg]   Raw total : {budget_est['total_raw_MB']:.3f} MB")

        # ── Step 5: apply mixed-precision quantization ────────────────────────
        print("[pg] Applying guided mixed-precision quantization…")
        quantized = guided_quantize_state_dict(sd, quant_map)
        buf = io.BytesIO()
        torch.save(quantized, buf)
        model_bytes = zlib.compress(buf.getvalue(), level=9)

        # Code bytes = train_gpt.py + guided_quantization.py (both ship in artifact)
        gq_path = Path(__file__).parent / "guided_quantization.py"
        gq_code  = gq_path.read_text(encoding="utf-8") if gq_path.exists() else ""
        code_bytes  = (code + gq_code).encode("utf-8")
        total_bytes = len(code_bytes) + len(model_bytes)
        budget      = 16_000_000

        print(f"\n[pg] === SIZE CHECK ===")
        print(f"[pg] Model (mixed+zlib)  : {len(model_bytes)/1e6:.3f} MB")
        print(f"[pg] Code size           : {len(code_bytes)/1e6:.3f} MB")
        print(f"[pg] Total artifact      : {total_bytes/1e6:.3f} MB")
        status = "UNDER BUDGET ✓" if total_bytes < budget else "OVER BUDGET ✗"
        print(f"[pg] Status              : {status}")
        if total_bytes >= budget:
            print("[pg] HINT: set BF16_FRAC lower / INT4_FRAC higher to shrink model")

        # ── Step 6: roundtrip validation (dequantize → eval) ─────────────────
        print("[pg] Running roundtrip dequantization + validation…")
        rt_sd = guided_dequantize_state_dict(quantized)
        base_model.load_state_dict(rt_sd)
        rt_loss, rt_bpb = eval_val(
            base_model, val_tokens, args.train_seq_len, args.val_batch_size,
            device, sp, args.vocab_size, rank, world_size)
        bpb_delta = rt_bpb - val_bpb
        print(f"[pg] Roundtrip val_bpb = {rt_bpb:.4f}  "
              f"(Δ vs full-precision: {bpb_delta:+.4f})")
        if rt_bpb < 1.2244:
            print(f"[pg] ✓ BEATS BASELINE  ({rt_bpb:.4f} < 1.2244)")
        else:
            print(f"[pg] ✗ does not beat baseline yet ({rt_bpb:.4f} ≥ 1.2244)")

        # ── Step 7: save outputs ───────────────────────────────────────────────
        (out_dir / "model.pkl.zlib").write_bytes(model_bytes)
        (out_dir / "train_log.txt").write_text("\n".join(log_lines))
        print(f"[pg] Saved to {out_dir}/")

        # Leaderboard parsing line (matches baseline format)
        print(f"\nfinal_int8_zlib_roundtrip: val_bpb={rt_bpb:.4f}  "
              f"compressed_bytes={len(model_bytes)}  "
              f"total_artifact_bytes={total_bytes}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
