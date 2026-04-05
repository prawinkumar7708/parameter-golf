"""
Parameter Golf - Primitives-Gating Transformer
================================================
Baseline: 9-layer, 512-dim GPT (9x512) trained on FineWeb — 1.2244 BPB
This file implements the modified feedforward sub-layer using SVD-extracted
primitive basis vectors + learned per-layer gating/blending/projection.

All PRIMITIVES-GATING modifications are marked with # PRIMITIVES-GATING comments.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 50304          # GPT-2 vocab rounded to nearest 64
    n_layer: int = 9
    n_head: int = 8
    n_kv_head: int = 4               # GQA: 4 KV heads, 8 query heads
    n_embd: int = 512
    block_size: int = 1024           # context length
    dropout: float = 0.0
    bias: bool = False
    # PRIMITIVES-GATING: feedforward modification settings
    use_primitives_gating: bool = True
    n_primitives: int = 12           # total primitive basis vectors
    n_selected: int = 3              # top-k primitives selected per forward pass
    proj_hidden: int = 128           # bottleneck projection dimension


# ---------------------------------------------------------------------------
# Standard attention (unchanged from baseline)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head self-attention with grouped query attention (GQA).
    This is the UNMODIFIED baseline attention layer.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Q projection: full heads; K,V projections: KV heads only (GQA)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Repeat K,V heads to match Q heads for GQA
        groups = self.n_head // self.n_kv_head
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        # Flash attention (efficient SDPA)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


# ---------------------------------------------------------------------------
# PRIMITIVES-GATING: Modified Feedforward Sub-layer
# ---------------------------------------------------------------------------

class PrimitivesGatingFF(nn.Module):
    """
    PRIMITIVES-GATING: Replaces the standard MLP feedforward in each transformer block.

    Architecture:
        Input (512)
            → Gating network (512 → 12 logits → softmax → top-3 indices)
            → Retrieve primitives[i], primitives[j], primitives[k]   (each 512-dim, FIXED)
            → Blend via learned 3×3 weight matrix → blended primitive (512)
            → Project down: 512 → 128 (learned)
            → ReLU
            → Project up:  128 → 512 (learned)
        Output (512)

    Primitives are initialized via SVD of the baseline feedforward weights and
    remain FROZEN throughout training.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_primitives = config.n_primitives   # 12
        self.n_selected = config.n_selected       # 3
        self.proj_hidden = config.proj_hidden     # 128
        self.layer_idx = layer_idx

        # PRIMITIVES-GATING: Fixed primitive library — initialized from SVD, NOT trained.
        # Shape: (n_primitives, n_embd) = (12, 512)
        # Registered as a buffer so it's part of state_dict but has no gradient.
        self.register_buffer(
            "primitives",
            torch.randn(config.n_primitives, config.n_embd) * 0.02,
        )

        # PRIMITIVES-GATING: Gating network — small linear producing 12 logits.
        # Input: token embedding (512), Output: 12 logits over primitive library.
        self.gate = nn.Linear(config.n_embd, config.n_primitives, bias=True)

        # PRIMITIVES-GATING: Blend-weight matrix — learned 3×3 matrix combining
        # the 3 selected primitives into a single blended vector.
        self.blend = nn.Parameter(torch.eye(config.n_selected))  # init as identity

        # PRIMITIVES-GATING: Bottleneck projections — compress blended primitive
        # to 128 dims, apply nonlinearity, expand back to 512.
        self.proj_down = nn.Linear(config.n_embd, config.proj_hidden, bias=config.bias)
        self.proj_up   = nn.Linear(config.proj_hidden, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

        # Storage for logging (filled during forward, read by training loop)
        self._last_selected_indices: Optional[torch.Tensor] = None   # (n_selected,)
        self._last_gate_probs: Optional[torch.Tensor] = None         # (n_primitives,)

    def set_primitives(self, primitives: torch.Tensor):
        """
        PRIMITIVES-GATING: Called once after SVD initialization to set the fixed
        primitive library.  primitives shape: (n_primitives, n_embd).
        """
        assert primitives.shape == (self.n_primitives, self.n_embd), (
            f"Expected ({self.n_primitives}, {self.n_embd}), got {primitives.shape}"
        )
        self.primitives.copy_(primitives)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)  where C = n_embd = 512
        Returns: (B, T, C)
        """
        B, T, C = x.shape

        # --- Gating ---
        # PRIMITIVES-GATING: Compute per-token logits over primitive library.
        # Average-pool over tokens for a single gate decision per position
        # (alternatively gate per token; here we gate per-token for expressiveness).
        gate_logits = self.gate(x)                      # (B, T, n_primitives)
        gate_probs  = F.softmax(gate_logits, dim=-1)    # (B, T, n_primitives)

        # PRIMITIVES-GATING: Select top-3 primitives by mean gate probability
        # across the batch and time dims (so selection is consistent per layer).
        mean_probs = gate_probs.mean(dim=(0, 1))        # (n_primitives,)
        top_indices = torch.topk(mean_probs, self.n_selected).indices  # (3,)

        # Store for logging
        self._last_selected_indices = top_indices.detach()
        self._last_gate_probs       = mean_probs.detach()

        # --- Primitive retrieval & blending ---
        # PRIMITIVES-GATING: Retrieve the 3 selected fixed primitives.
        selected = self.primitives[top_indices]          # (3, 512)

        # PRIMITIVES-GATING: Blend using learned 3×3 matrix.
        # blend @ selected → (3, 512), then sum along primitive axis.
        blend_w = F.softmax(self.blend, dim=-1)          # (3, 3) row-softmax for stability
        blended = blend_w @ selected                     # (3, 512)
        blended = blended.sum(dim=0)                     # (512,)

        # Broadcast blended primitive across batch and time
        blended = blended.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # (B, T, 512)

        # PRIMITIVES-GATING: Fuse input context with blended primitive via element-wise
        # gating: the per-token gate probabilities for selected primitives weight the blend.
        sel_probs = gate_probs[..., top_indices]         # (B, T, 3)
        fused = (sel_probs.unsqueeze(-1) * selected.unsqueeze(0).unsqueeze(0)).sum(-2)
        # fused: (B, T, 512) — context-sensitive blend

        # Residual mix: original input + primitive-guided signal
        z = x + fused + blended * 0.1

        # --- Bottleneck projection ---
        # PRIMITIVES-GATING: Down-project → nonlinearity → up-project.
        h = self.proj_down(z)        # (B, T, 128)
        h = F.relu(h)
        h = self.proj_up(h)          # (B, T, 512)
        h = self.dropout(h)

        return h


# ---------------------------------------------------------------------------
# Standard MLP (used only when use_primitives_gating=False)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Baseline feedforward MLP — kept for ablation comparison."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.drop(self.proj(self.act(self.fc(x))))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """
    Standard transformer block.
    PRIMITIVES-GATING: The feedforward (mlp) is replaced by PrimitivesGatingFF
    when config.use_primitives_gating is True.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)

        # PRIMITIVES-GATING: Select feedforward implementation
        if config.use_primitives_gating:
            self.mlp = PrimitivesGatingFF(config, layer_idx=layer_idx)
        else:
            self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# Full GPT Model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    GPT language model with optional Primitives-Gating feedforward modification.
    Baseline: 9 layers, 512 dims, 8 heads (4 KV), tied embeddings.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, bias=config.bias),
        ))

        # Tied embeddings: lm_head shares weights with token embeddings
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(T, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    # PRIMITIVES-GATING: Helper to collect per-layer primitive selections for logging
    def get_layer_selections(self):
        """Returns dict: layer_idx -> selected primitive indices (list of ints)."""
        selections = {}
        for i, block in enumerate(self.transformer.h):
            if hasattr(block.mlp, '_last_selected_indices') and \
               block.mlp._last_selected_indices is not None:
                selections[i] = block.mlp._last_selected_indices.cpu().tolist()
        return selections

    def get_gate_probs(self):
        """Returns dict: layer_idx -> gate probability distribution (list of floats)."""
        probs = {}
        for i, block in enumerate(self.transformer.h):
            if hasattr(block.mlp, '_last_gate_probs') and \
               block.mlp._last_gate_probs is not None:
                probs[i] = block.mlp._last_gate_probs.cpu().tolist()
        return probs

    def configure_optimizers(self, weight_decay: float, lr: float, device_type: str):
        """
        Separate parameters into weight-decay and no-decay groups.
        PRIMITIVES-GATING: Primitive buffers are not parameters so excluded automatically.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params     = [p for n, p in param_dict.items() if p.dim() >= 2]
        no_decay_params  = [p for n, p in param_dict.items() if p.dim() < 2]

        groups = [
            {'params': decay_params,    'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in torch.optim.AdamW.__init__.__doc__
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
            fused=use_fused if use_fused else False,
        )
        return optimizer
