"""
reconstruct.py — Main entry point for the voxel inverse reconstruction pipeline.

The reconstruction is grid-first:
  1. Estimate (or accept) a uniform block size from the point cloud.
  2. Find the sub-voxel lattice origin offset that maximises point-to-face alignment.
  3. For each point near an axis-aligned face boundary, vote for the block on the
     solid side (using local surface normals where available).
  4. Threshold votes → occupied set.
  5. Optionally accumulate evidence across multiple GLB files.
  6. Output a clean, uniform, equidistant block world (predicted_world.json)
     and an interactive Three.js visualisation.

Usage
-----
    python reconstruct.py output.glb
    python reconstruct.py frame1.glb frame2.glb frame3.glb
    python reconstruct.py frame1.glb frame2.glb ground_truth.json
    python reconstruct.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main(
    glb_paths: list[str],
    ground_truth_path: str | None = None,
    output_dir: str = ".",
    # grid-fit params
    block_size: float | None = None,
    face_threshold: float = 0.15,
    cell_min_votes: int = 3,
    min_votes: int = 1,
    estimate_normals: bool = True,
    # world bounds
    x_bounds: tuple = (0, 50),
    y_bounds: tuple = (0, 20),
    z_bounds: tuple = (0, 50),
    # visualisation
    max_cloud_points: int = 80_000,
) -> None:
    from accumulator import accumulate_evidence
    from evaluate import compute_iou, load_ground_truth
    from visualize import generate_html_viewer

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    print(f"\n── Grid-fit reconstruction: {len(glb_paths)} GLB file(s) ──────────────")

    predicted_set, evidence, combined_points = accumulate_evidence(
        glb_paths,
        min_votes=min_votes,
        block_size=block_size,
        face_threshold=face_threshold,
        cell_min_votes=cell_min_votes,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        estimate_normals=estimate_normals,
        verbose=True,
    )

    print(f"\n  Total predicted blocks : {len(predicted_set)}")

    # ── save predicted_world.json ─────────────────────────────────────────
    pred_path = os.path.join(output_dir, "predicted_world.json")
    predicted_list = [{"x": x, "y": y, "z": z} for x, y, z in sorted(predicted_set)]
    with open(pred_path, "w") as f:
        json.dump(predicted_list, f, indent=2)
    print(f"  Saved {pred_path}  ({len(predicted_set)} blocks)")

    # ── IoU evaluation ────────────────────────────────────────────────────
    if ground_truth_path:
        gt_set = load_ground_truth(ground_truth_path)
        compute_iou(predicted_set, gt_set, verbose=True)

    # ── HTML visualisation ────────────────────────────────────────────────
    viz_path = os.path.join(output_dir, "reconstruction_visualization.html")
    print(f"\n── Generating visualisation ─────────────────────────────────────────")
    generate_html_viewer(
        predicted_set,
        combined_points,
        viz_path,
        max_cloud_points=max_cloud_points,
    )
    print(f"  Saved {viz_path}")

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description="Grid-first voxel world inverse reconstruction from VGGT-Ω GLB output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python reconstruct.py output.glb
  python reconstruct.py frame1.glb frame2.glb frame3.glb
  python reconstruct.py frame1.glb frame2.glb ground_truth.json
  python reconstruct.py output.glb --block-size 1.0 --face-threshold 0.12
""",
    )
    parser.add_argument("inputs", nargs="+",
                        help=".glb file(s) and optionally a trailing ground_truth.json")
    parser.add_argument("--output-dir", default=".", metavar="DIR")

    g = parser.add_argument_group("grid parameters")
    g.add_argument("--block-size", type=float, default=None, metavar="F",
                   help="Known block size in raw point-cloud units. "
                        "Auto-estimated from the cloud if omitted.")
    g.add_argument("--face-threshold", type=float, default=0.15, metavar="F",
                   help="A point is 'on a face' if its fractional lattice coordinate "
                        "is within this distance of 0 or 1.  "
                        "Units: fraction of one block. (default: 0.15)")
    g.add_argument("--cell-min-votes", type=int, default=3, metavar="N",
                   help="Min face-point votes for a cell to be accepted within one cloud. "
                        "(default: 3)")
    g.add_argument("--min-votes", type=int, default=1, metavar="N",
                   help="Min number of GLB files that must detect a block. (default: 1)")
    g.add_argument("--no-normals", action="store_true",
                   help="Skip local normal estimation (faster, votes both sides of each face).")

    b = parser.add_argument_group("world bounds")
    b.add_argument("--x-bounds", type=int, nargs=2, default=[0, 50], metavar="N")
    b.add_argument("--y-bounds", type=int, nargs=2, default=[0, 20], metavar="N")
    b.add_argument("--z-bounds", type=int, nargs=2, default=[0, 50], metavar="N")

    parser.add_argument("--max-cloud-points", type=int, default=80_000, metavar="N",
                        help="Max points embedded in the HTML viewer. (default: 80000)")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])

    glb_paths, ground_truth_path = [], None
    for p in args.inputs:
        if p.endswith(".json"):
            ground_truth_path = p
        elif p.endswith(".glb"):
            glb_paths.append(p)
        else:
            print(f"Warning: unrecognised input '{p}' — expected .glb or .json, skipping.")

    if not glb_paths:
        print("Error: at least one .glb file is required.")
        sys.exit(1)

    main(
        glb_paths=glb_paths,
        ground_truth_path=ground_truth_path,
        output_dir=args.output_dir,
        block_size=args.block_size,
        face_threshold=args.face_threshold,
        cell_min_votes=args.cell_min_votes,
        min_votes=args.min_votes,
        estimate_normals=not args.no_normals,
        x_bounds=tuple(args.x_bounds),
        y_bounds=tuple(args.y_bounds),
        z_bounds=tuple(args.z_bounds),
        max_cloud_points=args.max_cloud_points,
    )
