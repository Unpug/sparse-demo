"""
evaluate.py — IoU and accuracy metrics against a ground-truth JSON.

Ground-truth JSON format (exported from world.html console snippet):
    [{"x": 0, "y": 0, "z": 0, "type": 1}, ...]
"""

from __future__ import annotations
import json


def load_ground_truth(path: str) -> set[tuple]:
    with open(path) as f:
        data = json.load(f)
    return {(int(b["x"]), int(b["y"]), int(b["z"])) for b in data}


def compute_iou(
    predicted_set: set[tuple],
    ground_truth_set: set[tuple],
    verbose: bool = True,
) -> tuple[float, float, float]:
    """
    Compute Intersection-over-Union, Precision, and Recall.

    Returns (iou, precision, recall).
    """
    intersection = len(predicted_set & ground_truth_set)
    union = len(predicted_set | ground_truth_set)

    iou       = intersection / union        if union            > 0 else 0.0
    precision = intersection / len(predicted_set)    if predicted_set    else 0.0
    recall    = intersection / len(ground_truth_set) if ground_truth_set else 0.0

    if verbose:
        print("\n── Reconstruction Accuracy ─────────────────────────────────")
        print(f"  Predicted blocks  : {len(predicted_set):>6}")
        print(f"  Ground truth      : {len(ground_truth_set):>6}")
        print(f"  Intersection      : {intersection:>6}")
        print(f"  IoU               : {iou:.4f}")
        print(f"  Precision         : {precision:.4f}")
        print(f"  Recall            : {recall:.4f}")
        print("────────────────────────────────────────────────────────────\n")

    return iou, precision, recall
