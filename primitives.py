"""
primitives.py
=============
PRIMITIVES-GATING: SVD-based primitive basis extraction.

Extracts the top-12 singular vectors from the concatenated feedforward
weight matrices of the 9 baseline transformer layers. These become the
fixed primitive library used in PrimitivesGatingFF.

Usage:
    from primitives import extract_primitives, init_model_primitives
    primitives = extract_primitives(baseline_model, n_primitives=12)
    init_model_primitives(new_model, primitives)
"""

import torch
import torch.nn as nn
from typing import Optional


def extract_primitives(
    model: nn.Module,
    n_primitives: int = 12,
    use_win: bool = True,
) -> torch.Tensor:
    """
    PRIMITIVES-GATING: Extract primitive basis vectors via truncated SVD.

    Algorithm:
      1. Collect W_in matrices from each of the 9 feedforward layers.
         W_in shape: (n_embd, 4*n_embd) = (512, 2048)
      2. Concatenate all 9 matrices horizontally → (512, 9*2048) = (512, 18432)
      3. Run truncated SVD: U, S, Vh = svd(concat)
      4. Top-k left singular vectors (U[:, :k]) form the primitive library.
         Each vector: (512,) — one primitive basis direction.

    Args:
        model:        The baseline (or any) GPT model whose feedforward weights to use.
        n_primitives: How many SVD components to keep (default 12).
        use_win:      If True, use proj_down / fc weights; else use proj_up weights.

    Returns:
        primitives: Tensor of shape (n_primitives, n_embd) — the primitive library.
    """
    weight_matrices = []

    for i, block in enumerate(model.transformer.h):
        ff = block.mlp

        # Support both baseline MLP and PrimitivesGatingFF layouts
        if hasattr(ff, 'fc'):
            # Baseline MLP: weight shape (4*n_embd, n_embd) → transpose to (n_embd, 4*n_embd)
            W = ff.fc.weight.data.T.float()          # (512, 2048)
        elif hasattr(ff, 'proj_down'):
            # PrimitivesGatingFF: use proj_down weight (proj_hidden, n_embd)
            W = ff.proj_down.weight.data.T.float()   # (512, 128)
        else:
            raise ValueError(f"Unknown feedforward layer type at block {i}: {type(ff)}")

        weight_matrices.append(W)

    # PRIMITIVES-GATING: Concatenate W_in matrices along the column axis
    # Result: (n_embd, total_columns) e.g., (512, 18432) for 9 layers × 2048
    concat = torch.cat(weight_matrices, dim=1)   # (512, 9*col_dim)

    # PRIMITIVES-GATING: Truncated SVD — top-k left singular vectors
    # torch.linalg.svd returns U (512, 512), S, Vh
    # We only need the first n_primitives columns of U
    U, S, Vh = torch.linalg.svd(concat, full_matrices=False)
    # U shape: (512, min(512, 18432)) = (512, 512)
    # Take top-k columns → (512, n_primitives) → transpose to (n_primitives, 512)
    primitives = U[:, :n_primitives].T.contiguous()  # (n_primitives, 512)

    print(f"[primitives] Extracted {n_primitives} primitives from concatenated weight matrix")
    print(f"  Concatenated shape: {concat.shape}")
    print(f"  Top-{n_primitives} singular values: {S[:n_primitives].tolist()}")
    print(f"  Primitives shape: {primitives.shape}")

    return primitives


def extract_primitives_from_checkpoint(
    checkpoint_path: str,
    n_primitives: int = 12,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    PRIMITIVES-GATING: Load a baseline checkpoint and extract primitives from it.

    This is the typical workflow:
      1. Load baseline model checkpoint
      2. Extract SVD-based primitives
      3. Use them to initialize the new model's primitive library

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        n_primitives:    Number of primitives to extract.
        device:          Device to load the checkpoint on.

    Returns:
        primitives: Tensor of shape (n_primitives, n_embd)
    """
    from model import GPT, GPTConfig

    print(f"[primitives] Loading baseline checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    config_dict = ckpt.get('config', {})
    config = GPTConfig(**{k: v for k, v in config_dict.items()
                          if k in GPTConfig.__dataclass_fields__})
    # Force baseline MLP (not primitives-gating) to extract from standard weights
    config.use_primitives_gating = False

    baseline_model = GPT(config)
    # Strip any "module." prefix (DDP wrapping)
    clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    baseline_model.load_state_dict(clean_state, strict=False)
    baseline_model.eval()

    return extract_primitives(baseline_model, n_primitives=n_primitives)


def init_model_primitives(model: nn.Module, primitives: torch.Tensor):
    """
    PRIMITIVES-GATING: Copy the extracted SVD primitives into every
    PrimitivesGatingFF layer in the model.

    The primitives buffer is NOT a trainable parameter — it is fixed
    after this initialization step.

    Args:
        model:      GPT model with PrimitivesGatingFF feedforward layers.
        primitives: Tensor of shape (n_primitives, n_embd).
    """
    initialized = 0
    for i, block in enumerate(model.transformer.h):
        ff = block.mlp
        if hasattr(ff, 'set_primitives'):
            prim_device = next(model.parameters()).device
            ff.set_primitives(primitives.to(prim_device))
            initialized += 1
            print(f"  [init_primitives] Layer {i}: primitives initialized ✓")

    if initialized == 0:
        print("[init_primitives] WARNING: No PrimitivesGatingFF layers found!")
    else:
        print(f"[init_primitives] Initialized {initialized} layers with SVD primitives.")


def random_primitives(n_primitives: int = 12, n_embd: int = 512) -> torch.Tensor:
    """
    PRIMITIVES-GATING: Fallback — generate random orthonormal primitives via QR
    decomposition when no baseline checkpoint is available.
    """
    print(f"[primitives] Generating {n_primitives} random orthonormal primitives "
          f"(no baseline checkpoint provided)")
    raw = torch.randn(n_embd, n_primitives)
    Q, _ = torch.linalg.qr(raw)              # (n_embd, n_primitives)
    return Q.T.contiguous()                  # (n_primitives, n_embd)
