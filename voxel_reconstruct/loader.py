"""
loader.py — GLB loading and point cloud extraction.

VGGT-Ω outputs point clouds in OpenCV convention (Y-down, Z-forward).
We convert to Three.js / world.html convention (Y-up, Z-backward) via:
    [x, y, z]  →  [x, -y, -z]
"""

import numpy as np
import trimesh


def load_glb_as_pointcloud(glb_path: str):
    """
    Load a .glb exported from VGGT-Ω and return all vertex positions as a
    unified (N, 3) float32 array, plus an optional (N, 3) uint8 color array.

    The coordinate system is converted from OpenCV → Three.js here so that
    every downstream stage works in world-space (Y-up, Z-backward).
    """
    scene = trimesh.load(glb_path, force="scene")

    all_points = []
    all_colors = []

    if isinstance(scene, trimesh.Scene):
        geometries = scene.geometry.values()
    else:
        # Single-mesh GLB
        geometries = [scene]

    for geom in geometries:
        verts = np.asarray(geom.vertices, dtype=np.float32)
        all_points.append(verts)

        if hasattr(geom, "visual") and hasattr(geom.visual, "vertex_colors"):
            vc = np.asarray(geom.visual.vertex_colors, dtype=np.uint8)
            all_colors.append(vc[:, :3])  # drop alpha

    if not all_points:
        raise ValueError(f"No geometry found in {glb_path}")

    points = np.vstack(all_points)  # (N, 3)
    colors = np.vstack(all_colors) if all_colors else None

    # ── coordinate system conversion ──────────────────────────────────────
    # OpenCV: x-right, y-down, z-forward
    # Three.js: x-right, y-up,   z-backward
    points[:, 1] *= -1  # flip Y
    points[:, 2] *= -1  # flip Z

    return points, colors


def subsample_pointcloud(points, colors=None, max_points: int = 500_000):
    """
    Random sub-sample to keep memory and RANSAC runtime under control.
    """
    if len(points) <= max_points:
        return points, colors

    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx], (colors[idx] if colors is not None else None)
