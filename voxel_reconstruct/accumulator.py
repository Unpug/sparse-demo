"""
accumulator.py — Multi-frame evidence accumulation using grid_fit.

Each GLB contributes block-position votes.  A block is confirmed when
it appears in at least `min_votes` independent files.
"""

from __future__ import annotations

import numpy as np
from loader import load_glb_as_pointcloud, subsample_pointcloud
from grid_fit import fit_grid, estimate_block_size


def accumulate_evidence(
    glb_paths: list[str],
    min_votes: int = 1,
    max_points_per_cloud: int = 500_000,
    block_size: float | None = None,   # None → auto-estimate per cloud then average
    face_threshold: float = 0.15,
    cell_min_votes: int = 3,           # per-cloud vote threshold inside fit_grid
    x_bounds: tuple = (0, 50),
    y_bounds: tuple = (0, 20),
    z_bounds: tuple = (0, 50),
    estimate_normals: bool = True,
    verbose: bool = True,
) -> tuple[set, dict, np.ndarray]:
    """
    Process one or more GLB files and accumulate block-detection votes.

    Returns
    -------
    predicted_world  set of (x, y, z) tuples with >= min_votes file-votes
    evidence         dict (x,y,z) → file-vote count
    combined_points  stacked raw point cloud for visualisation
    """
    evidence: dict[tuple, int] = {}
    all_points: list[np.ndarray] = []

    # If block_size not given, estimate it from the first cloud and reuse
    shared_block_size = block_size

    for i, glb_path in enumerate(glb_paths):
        if verbose:
            print(f"  [{i+1}/{len(glb_paths)}] Loading {glb_path} ...", flush=True)

        points, _colors = load_glb_as_pointcloud(glb_path)
        points, _colors = subsample_pointcloud(points, _colors,
                                               max_points=max_points_per_cloud)
        all_points.append(points)

        if verbose:
            print(f"         {len(points):,} points", flush=True)

        if shared_block_size is None:
            shared_block_size = estimate_block_size(points)
            if verbose:
                print(f"         auto block_size = {shared_block_size:.4f}", flush=True)

        frame_blocks, _ = fit_grid(
            points,
            block_size=shared_block_size,
            face_threshold=face_threshold,
            min_votes=cell_min_votes,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            estimate_normals=estimate_normals,
            verbose=verbose,
        )

        if verbose:
            print(f"         blocks this file: {len(frame_blocks)}", flush=True)

        for block in frame_blocks:
            evidence[block] = evidence.get(block, 0) + 1

    predicted_world = {pos for pos, votes in evidence.items() if votes >= min_votes}
    combined_points = np.vstack(all_points) if all_points else np.zeros((0, 3))

    return predicted_world, evidence, combined_points
