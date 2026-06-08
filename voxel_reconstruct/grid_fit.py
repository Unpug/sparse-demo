"""
grid_fit.py — Grid-first voxel reconstruction.

Instead of detecting planes and snapping them, we commit up-front to a
uniform voxel lattice and ask: "which cells of that lattice are occupied?"

Pipeline
--------
1.  Estimate block size (voxel scale) from the point cloud via 1-D
    auto-correlation of projected coordinates.
2.  Find the sub-voxel origin offset on each axis independently via a
    fractional-coordinate histogram peak — this is the δ ∈ [0,1) that
    makes the most points land on block-face boundaries.
3.  Transform the cloud into lattice coordinates: divide by block_size,
    subtract origin offset → every block face should now be near an integer.
4.  For each point, find the nearest integer boundary on each axis.
    If the point is within `face_threshold` of that boundary it is
    "on a face" for that axis.  Estimate the face normal from local
    neighbourhood to decide which side is solid.
5.  Vote for the block on the solid side.  A block is accepted when it
    exceeds `min_votes` face-point votes.
6.  Return the set of (x, y, z) integer lattice coordinates.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


# ── 1. Block-size estimation via auto-correlation ─────────────────────────────

def _axis_autocorr(coords: np.ndarray, n_bins: int = 512, max_lag_frac: float = 0.5) -> float:
    """
    Estimate the dominant period in a 1-D coordinate sequence via
    histogram auto-correlation.  Returns the estimated period in the
    same units as `coords`.
    """
    lo, hi = float(coords.min()), float(coords.max())
    span = hi - lo
    if span < 1e-6:
        return 1.0

    hist, edges = np.histogram(coords, bins=n_bins, range=(lo, hi))
    hist = hist.astype(float)
    hist -= hist.mean()

    # circular auto-correlation via FFT
    fft = np.fft.rfft(hist)
    acorr = np.fft.irfft(fft * np.conj(fft))
    acorr = acorr[: len(hist) // 2]   # keep positive lags only

    bin_size = span / n_bins
    max_lag_bins = int(n_bins * max_lag_frac)

    # find first strong peak (ignore lag-0)
    peaks, props = find_peaks(acorr[1:max_lag_bins], height=0, prominence=0)
    if len(peaks) == 0:
        return 1.0

    # weight peaks by height × prominence, pick highest
    heights = props["prominences"]
    best = peaks[np.argmax(heights)]
    period_bins = best + 1   # +1 because we sliced away lag-0
    return float(period_bins * bin_size)


def estimate_block_size(points: np.ndarray, percentile_clip: float = 2.0) -> float:
    """
    Estimate the uniform block size (voxel edge length) from the point cloud.

    We compute the dominant period independently for X, Y, Z and return the
    median — Y is usually most reliable because horizontal floor layers give
    the clearest signal.
    """
    # clip outliers before analysis
    lo = np.percentile(points, percentile_clip, axis=0)
    hi = np.percentile(points, 100 - percentile_clip, axis=0)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    pts = points[mask]
    if len(pts) < 100:
        pts = points

    periods = []
    for axis in range(3):
        p = _axis_autocorr(pts[:, axis])
        if p > 0.05:
            periods.append(p)

    if not periods:
        return 1.0

    return float(np.median(periods))


# ── 2. Sub-voxel origin offset estimation ────────────────────────────────────

def estimate_origin_offset(
    lattice_coords: np.ndarray,
    n_bins: int = 200,
) -> np.ndarray:
    """
    Given coordinates already divided by block_size, find the sub-voxel
    offset δ ∈ [0,1) per axis that maximises how many points land near
    an integer (i.e. on a block face).

    Returns a (3,) array of offsets.
    """
    offsets = np.zeros(3)
    for axis in range(3):
        frac = lattice_coords[:, axis] % 1.0          # fractional part in [0,1)
        hist, bin_edges = np.histogram(frac, bins=n_bins, range=(0.0, 1.0))
        # Block faces are at integer boundaries, so the fractional part of
        # face-points clusters near 0 (or 1, same thing).  Find the peak
        # nearest to 0 or 1 in the circular histogram.
        # We roll the histogram so the tallest edge is centred.
        peak_bin = int(np.argmax(hist))
        bin_centre = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0
        # δ is the fractional amount we need to subtract so that the peak
        # moves to 0 (i.e. points land on integer boundaries)
        offsets[axis] = bin_centre % 1.0

    return offsets


# ── 3. Local normal estimation ────────────────────────────────────────────────

def _estimate_normals(points: np.ndarray, k: int = 15) -> np.ndarray:
    """
    PCA-based normal estimation for each point using its k nearest neighbours.
    Returns (N, 3) unit normals.
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        return np.asarray(pcd.normals, dtype=np.float32)
    except Exception:
        # fallback: all normals pointing up — coarse but safe
        n = np.zeros((len(points), 3), dtype=np.float32)
        n[:, 1] = 1.0
        return n


# ── 4–5. Face voting → occupancy grid ────────────────────────────────────────

def fit_grid(
    points: np.ndarray,
    block_size: float | None = None,
    face_threshold: float = 0.15,
    min_votes: int = 3,
    x_bounds: tuple = (0, 50),
    y_bounds: tuple = (0, 20),
    z_bounds: tuple = (0, 50),
    estimate_normals: bool = True,
    verbose: bool = True,
) -> tuple[set, dict]:
    """
    Main grid-fitting entry point.

    Parameters
    ----------
    points          (N, 3) float array in world-ish coordinates (already
                    coord-converted from OpenCV → Three.js by loader.py).
    block_size      If None, auto-estimated from the point cloud.
    face_threshold  A point is "on a face" if its fractional lattice
                    coordinate (after offset correction) is within this
                    distance of 0 or 1.  Units: fractions of one block.
    min_votes       Minimum face-point votes for a cell to be accepted.
    x/y/z_bounds    Clip output to world bounds.
    estimate_normals Use local PCA normals to decide which side is solid.
                    If False, vote for both sides (noisier but faster).

    Returns
    -------
    occupied_set    set of (x, y, z) integer lattice coordinates
    vote_map        dict (x,y,z) → int vote count (for debugging)
    """
    # ── step 1: block size ────────────────────────────────────────────────
    if block_size is None:
        block_size = estimate_block_size(points)
    if verbose:
        print(f"  block_size = {block_size:.4f} raw units")

    # ── step 2: convert to lattice coordinates ────────────────────────────
    lc = points / block_size   # lattice coords (real-valued)

    # ── step 3: origin offset ─────────────────────────────────────────────
    offsets = estimate_origin_offset(lc)
    if verbose:
        print(f"  grid offsets = {offsets}")
    lc = lc - offsets          # shift so block faces → near integers

    # ── step 4: (optional) local normals ─────────────────────────────────
    if estimate_normals:
        normals = _estimate_normals(points)
    else:
        normals = None

    # ── step 5: vote ──────────────────────────────────────────────────────
    vote_map: dict[tuple, int] = {}

    frac = lc % 1.0                        # fractional part per axis, [0,1)
    near_zero = frac < face_threshold      # close to integer boundary below
    near_one  = frac > (1.0 - face_threshold)  # close to integer boundary above

    ix = np.floor(lc).astype(int)         # integer cell index

    for i in range(len(lc)):
        pt_lc = lc[i]
        pt_ix = ix[i]
        n = normals[i] if normals is not None else None

        for axis in range(3):
            # face below (boundary at integer ix[axis])
            if near_zero[i, axis]:
                _vote_face(vote_map, pt_ix, axis, side=-1,
                           normal=n, x_bounds=x_bounds,
                           y_bounds=y_bounds, z_bounds=z_bounds,
                           vote_both=(normals is None))

            # face above (boundary at integer ix[axis]+1)
            if near_one[i, axis]:
                _vote_face(vote_map, pt_ix, axis, side=+1,
                           normal=n, x_bounds=x_bounds,
                           y_bounds=y_bounds, z_bounds=z_bounds,
                           vote_both=(normals is None))

    # ── step 6: threshold ─────────────────────────────────────────────────
    occupied_set = {pos for pos, votes in vote_map.items() if votes >= min_votes}
    if verbose:
        print(f"  total candidates: {len(vote_map)}  accepted (≥{min_votes} votes): {len(occupied_set)}")

    return occupied_set, vote_map


def _vote_face(
    vote_map: dict,
    pt_ix: np.ndarray,
    axis: int,
    side: int,       # -1 = block below/behind boundary, +1 = block above/ahead
    normal: np.ndarray | None,
    x_bounds: tuple,
    y_bounds: tuple,
    z_bounds: tuple,
    vote_both: bool = False,
) -> None:
    """
    Cast a vote for the block on the solid side of this face.

    If `normal` is available, use the dot product with the axis direction to
    decide which side is solid (normal points *away* from solid, so the block
    is on the opposite side from where the normal points).

    If `vote_both`, increment both sides (used when no normal information).
    """
    AXES = [(1,0,0), (0,1,0), (0,0,1)]
    bounds = [x_bounds, y_bounds, z_bounds]

    # The face at boundary `pt_ix[axis] + (1 if side==+1 else 0)` divides:
    #   block A = pt_ix  (below/behind)
    #   block B = pt_ix with axis incremented by 1 (above/ahead)
    bx_a = int(pt_ix[0])
    by_a = int(pt_ix[1])
    bz_a = int(pt_ix[2])

    bx_b, by_b, bz_b = bx_a, by_a, bz_a
    if axis == 0:   bx_b += 1
    elif axis == 1: by_b += 1
    else:           bz_b += 1

    candidates = []
    if vote_both:
        candidates = [(bx_a, by_a, bz_a), (bx_b, by_b, bz_b)]
    elif normal is not None:
        # normal points away from solid
        ax_dir = np.array(AXES[axis], dtype=float)
        dot = float(np.dot(normal, ax_dir))
        if side == -1:
            # face at ix[axis]: normal pointing in +axis → solid is block A (below)
            block = (bx_a, by_a, bz_a) if dot > 0 else (bx_b, by_b, bz_b)
        else:
            # face at ix[axis]+1: normal pointing in -axis → solid is block B (above)
            block = (bx_b, by_b, bz_b) if dot < 0 else (bx_a, by_a, bz_a)
        candidates = [block]
    else:
        candidates = [(bx_a, by_a, bz_a)]

    for (bx, by, bz) in candidates:
        if (bounds[0][0] <= bx <= bounds[0][1] and
                bounds[1][0] <= by <= bounds[1][1] and
                bounds[2][0] <= bz <= bounds[2][1]):
            key = (bx, by, bz)
            vote_map[key] = vote_map.get(key, 0) + 1
