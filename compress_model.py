"""
compress_model.py
=================
PRIMITIVES-GATING: Model compression for submission.

Pipeline:
  1. Load trained model checkpoint
  2. Apply INT8 quantization to all weight tensors
  3. Serialize to bytes
  4. Compress with zlib
  5. Save .pkl.zlib file
  6. Report compressed size vs 16 MB budget

Usage:
    python compress_model.py --checkpoint out/model_final.pt --output submission.pkl.zlib
"""

import argparse
import io
import os
import pickle
import struct
import zlib
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# INT8 quantization helpers
# ---------------------------------------------------------------------------

def quantize_tensor_int8(tensor: torch.Tensor) -> Tuple[bytes, float, float]:
    """
    PRIMITIVES-GATING: Symmetric INT8 quantization of a single tensor.

    Quantization:
        scale = max(|tensor|) / 127
        q = clamp(round(tensor / scale), -128, 127).int8()

    Returns:
        (quantized_bytes, scale, zero_point)
    """
    tensor = tensor.float()
    abs_max = tensor.abs().max().item()
    if abs_max == 0:
        scale = 1.0
    else:
        scale = abs_max / 127.0

    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q.numpy().tobytes(), scale, 0.0  # zero_point=0 for symmetric


def dequantize_tensor_int8(
    data: bytes,
    shape: tuple,
    scale: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    PRIMITIVES-GATING: Reconstruct a float tensor from INT8 bytes.
    """
    import numpy as np
    arr = np.frombuffer(data, dtype=np.int8).reshape(shape)
    return torch.from_numpy(arr.copy()).float() * scale


# ---------------------------------------------------------------------------
# Model quantization
# ---------------------------------------------------------------------------

def quantize_state_dict(
    state_dict: Dict[str, torch.Tensor],
    skip_keys: Tuple[str, ...] = ('primitives',),
) -> Dict:
    """
    PRIMITIVES-GATING: Quantize all parameter tensors to INT8.

    Buffers listed in skip_keys are kept as-is (e.g., the fixed primitives
    are already compact and skipping avoids losing their orthonormality).

    Returns a dict of:
        key → {
            'type': 'int8' | 'float32',
            'data': bytes,
            'shape': tuple,
            'scale': float,        # only for int8
            'zero_point': float,   # only for int8
        }
    """
    quantized = {}
    total_original = 0
    total_quantized = 0

    for key, tensor in state_dict.items():
        tensor = tensor.cpu()
        original_bytes = tensor.nelement() * tensor.element_size()
        total_original += original_bytes

        # Skip non-float tensors or small buffers
        skip = any(s in key for s in skip_keys)
        if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16) and not skip:
            q_bytes, scale, zp = quantize_tensor_int8(tensor)
            quantized[key] = {
                'type':       'int8',
                'data':       q_bytes,
                'shape':      tuple(tensor.shape),
                'scale':      scale,
                'zero_point': zp,
                'orig_dtype': str(tensor.dtype),
            }
            total_quantized += len(q_bytes)
        else:
            raw = pickle.dumps(tensor)
            quantized[key] = {
                'type': 'float32',
                'data': raw,
            }
            total_quantized += len(raw)

    print(f"[compress] Original size : {total_original / 1024 / 1024:.2f} MB")
    print(f"[compress] After INT8    : {total_quantized / 1024 / 1024:.2f} MB")
    return quantized


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_model(
    checkpoint_path: str,
    output_path: str,
    compression_level: int = 9,
) -> int:
    """
    PRIMITIVES-GATING: Full compression pipeline.

    1. Load checkpoint state dict
    2. Quantize to INT8
    3. Pickle the quantized dict
    4. zlib compress
    5. Write to output_path
    6. Return compressed size in bytes
    """
    print(f"\n[compress] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Extract state dict and config
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
        config     = ckpt.get('config', {})
        step       = ckpt.get('step', -1)
    else:
        state_dict = ckpt
        config     = {}
        step       = -1

    print(f"[compress] Checkpoint step: {step}")
    print(f"[compress] Parameters:      {sum(t.nelement() for t in state_dict.values()):,}")

    # INT8 quantize
    quantized = quantize_state_dict(state_dict)

    # Bundle with config metadata
    bundle = {
        'quantized_state': quantized,
        'config':          config,
        'step':            step,
        'format_version':  1,
    }

    # Pickle → zlib compress
    raw_bytes = pickle.dumps(bundle, protocol=4)
    compressed = zlib.compress(raw_bytes, level=compression_level)

    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(compressed)

    size_mb = len(compressed) / 1024 / 1024
    budget_mb = 16.0
    status = "✅ UNDER BUDGET" if size_mb < budget_mb else "❌ OVER BUDGET"

    print(f"\n[compress] Compressed size : {size_mb:.3f} MB  {status}")
    print(f"[compress] Budget          : {budget_mb:.1f} MB")
    print(f"[compress] Output          : {output_path}")

    if size_mb > budget_mb:
        print("\n⚠️  WARNING: Submission exceeds 16 MB budget!")

    return len(compressed)


def decompress_model(compressed_path: str) -> Tuple[Dict[str, torch.Tensor], dict]:
    """
    PRIMITIVES-GATING: Decompress and dequantize a compressed submission.

    Returns:
        (state_dict, config)
    """
    with open(compressed_path, 'rb') as f:
        data = f.read()

    raw = zlib.decompress(data)
    bundle = pickle.loads(raw)

    quantized_state = bundle['quantized_state']
    config          = bundle.get('config', {})

    state_dict = {}
    for key, entry in quantized_state.items():
        if entry['type'] == 'int8':
            tensor = dequantize_tensor_int8(
                entry['data'],
                entry['shape'],
                entry['scale'],
            )
        else:
            tensor = pickle.loads(entry['data'])
        state_dict[key] = tensor

    return state_dict, config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compress trained model for submission")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--output', type=str, default='submission.pkl.zlib',
                        help='Output path for compressed model')
    parser.add_argument('--level', type=int, default=9,
                        help='zlib compression level (1-9, default 9 = max)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify decompression after compression')
    args = parser.parse_args()

    size = compress_model(args.checkpoint, args.output, args.level)

    if args.verify:
        print("\n[compress] Verifying decompression...")
        state_dict, config = decompress_model(args.output)
        print(f"[compress] Decompressed keys: {len(state_dict)}")
        print(f"[compress] Config: {config}")
        print("[compress] Verification ✅")


if __name__ == '__main__':
    main()
