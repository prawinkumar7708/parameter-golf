"""
logging_utils.py
================
PRIMITIVES-GATING: Structured JSON logging for primitive selection patterns,
gating entropy, overlap penalties, and training metrics.

Log format per batch:
{
  "step": int,
  "train_loss": float,
  "val_loss": float | null,
  "val_bpb": float | null,
  "overlap_penalty_weight": float,
  "overlap_threshold": float,
  "layer_selections": {"0": [i, j, k], "1": [...], ...},
  "layer_pair_overlaps": {"0-1": float, "0-2": float, ...},
  "gating_entropy": {"0": float, "1": float, ...},
  "mean_overlap": float,
  "max_overlap": float,
}
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------

def compute_layer_pair_overlaps(
    selections: Dict[int, List[int]],
) -> Dict[str, float]:
    """
    PRIMITIVES-GATING: For each pair of layers, compute the overlap percentage
    between their selected primitive indices.

    overlap(L1, L2) = |selected(L1) ∩ selected(L2)| / |selected(L1)| * 100

    Args:
        selections: dict mapping layer_idx → list of selected primitive indices.

    Returns:
        dict mapping "L1-L2" → overlap percentage (0-100).
    """
    overlaps = {}
    layer_ids = sorted(selections.keys())
    for i, l1 in enumerate(layer_ids):
        for l2 in layer_ids[i + 1:]:
            s1 = set(selections[l1])
            s2 = set(selections[l2])
            if not s1:
                overlap_pct = 0.0
            else:
                overlap_pct = len(s1 & s2) / len(s1) * 100.0
            overlaps[f"{l1}-{l2}"] = round(overlap_pct, 2)
    return overlaps


def compute_gating_entropy(
    gate_probs: Dict[int, List[float]],
) -> Dict[str, float]:
    """
    PRIMITIVES-GATING: Compute Shannon entropy of each layer's gating distribution.

    High entropy → layer is uncertain / spreading attention across many primitives.
    Low entropy  → layer consistently favors a small set of primitives.

    H = -sum(p * log2(p))   (base-2, in bits)
    """
    entropies = {}
    for layer_idx, probs in gate_probs.items():
        H = 0.0
        for p in probs:
            if p > 1e-9:
                H -= p * math.log2(p)
        entropies[str(layer_idx)] = round(H, 4)
    return entropies


# ---------------------------------------------------------------------------
# Overlap penalty schedule
# ---------------------------------------------------------------------------

class OverlapPenaltySchedule:
    """
    PRIMITIVES-GATING: Manages the overlap penalty threshold and weight
    across training.

    Threshold: linear decay from threshold_start → threshold_end over training.
    Weight:    linear increase from weight_start → weight_end over training.
    """

    def __init__(
        self,
        total_steps: int,
        threshold_start: float = 0.9,
        threshold_end: float   = 0.7,
        weight_start: float    = 0.01,
        weight_end: float      = 0.1,
    ):
        self.total_steps     = max(total_steps, 1)
        self.threshold_start = threshold_start
        self.threshold_end   = threshold_end
        self.weight_start    = weight_start
        self.weight_end      = weight_end

    def get(self, step: int) -> Tuple[float, float]:
        """
        Returns (threshold, weight) at the given training step.
        """
        progress = min(step / self.total_steps, 1.0)
        threshold = self.threshold_start + (self.threshold_end - self.threshold_start) * progress
        weight    = self.weight_start    + (self.weight_end    - self.weight_start)    * progress
        return threshold, weight


def compute_overlap_penalty(
    selections: Dict[int, List[int]],
    threshold: float,
    weight: float,
) -> float:
    """
    PRIMITIVES-GATING: Compute the overlap regularization penalty.

    For each pair of layers:
        pair_penalty = max(0, (overlap_pct/100 - threshold))
    Total = weight * mean(pair_penalties)

    Args:
        selections: layer_idx → selected primitive indices
        threshold:  current overlap threshold (fraction, 0-1)
        weight:     current penalty weight

    Returns:
        scalar penalty value (Python float, not a tensor)
    """
    pair_overlaps = compute_layer_pair_overlaps(selections)
    if not pair_overlaps:
        return 0.0

    penalties = []
    for overlap_pct in pair_overlaps.values():
        overlap_frac = overlap_pct / 100.0
        pen = max(0.0, overlap_frac - threshold)
        penalties.append(pen)

    mean_penalty = sum(penalties) / len(penalties)
    return weight * mean_penalty


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class PrimitivesLogger:
    """
    PRIMITIVES-GATING: Writes per-batch JSON log entries to a JSONL file.

    Each line of the output file is a valid JSON object (one per batch/step).
    After training, the log shows which primitives were favored by which
    layers across the full training run.
    """

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode so restarts don't erase history
        self._fh = open(self.log_path, 'a')
        print(f"[logger] Logging to {self.log_path}")

    def log(
        self,
        step: int,
        train_loss: float,
        selections: Dict[int, List[int]],
        gate_probs: Dict[int, List[float]],
        overlap_threshold: float,
        overlap_weight: float,
        val_loss: Optional[float] = None,
        val_bpb: Optional[float] = None,
    ):
        """Write one JSON log entry to the JSONL file."""
        pair_overlaps = compute_layer_pair_overlaps(selections)
        gating_entropy = compute_gating_entropy(gate_probs)

        overlap_values = list(pair_overlaps.values())
        mean_overlap = (sum(overlap_values) / len(overlap_values)) if overlap_values else 0.0
        max_overlap  = max(overlap_values) if overlap_values else 0.0

        entry = {
            "step":                 step,
            "train_loss":           round(train_loss, 6),
            "val_loss":             round(val_loss, 6) if val_loss is not None else None,
            "val_bpb":              round(val_bpb, 6)  if val_bpb  is not None else None,
            "overlap_penalty_weight": round(overlap_weight, 6),
            "overlap_threshold":    round(overlap_threshold, 4),
            "layer_selections":     {str(k): v for k, v in selections.items()},
            "layer_pair_overlaps":  pair_overlaps,
            "gating_entropy":       gating_entropy,
            "mean_overlap_pct":     round(mean_overlap, 2),
            "max_overlap_pct":      round(max_overlap, 2),
        }
        self._fh.write(json.dumps(entry) + '\n')
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Summary analysis
# ---------------------------------------------------------------------------

def summarize_log(log_path: str) -> Dict:
    """
    PRIMITIVES-GATING: Read the JSONL log and produce a summary of
    primitive selection patterns across training.

    Returns dict with:
      - primitive_usage: how often each primitive was selected (across all layers/steps)
      - per_layer_favorite: most-selected primitive per layer
      - mean_overlap_trend: list of mean overlaps over training
    """
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if not entries:
        return {}

    # Aggregate primitive usage counts
    primitive_usage: Dict[int, int] = {}
    per_layer_counts: Dict[str, Dict[int, int]] = {}
    mean_overlaps = []

    for entry in entries:
        for layer_str, indices in entry.get('layer_selections', {}).items():
            for idx in indices:
                primitive_usage[idx] = primitive_usage.get(idx, 0) + 1
                if layer_str not in per_layer_counts:
                    per_layer_counts[layer_str] = {}
                per_layer_counts[layer_str][idx] = per_layer_counts[layer_str].get(idx, 0) + 1
        mean_overlaps.append(entry.get('mean_overlap_pct', 0.0))

    per_layer_favorite = {
        layer: max(counts, key=counts.get)
        for layer, counts in per_layer_counts.items()
    }

    return {
        "total_steps":          len(entries),
        "primitive_usage":      primitive_usage,
        "per_layer_favorite":   per_layer_favorite,
        "mean_overlap_trend":   mean_overlaps,
        "final_val_bpb":        entries[-1].get('val_bpb'),
        "final_train_loss":     entries[-1].get('train_loss'),
    }
