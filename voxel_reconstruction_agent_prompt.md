# Voxel World Inverse Reconstruction — Coding Agent Prompt

## Project Overview

This is a research pipeline that reconstructs a discrete voxel world model from first-person RGB observations alone — no coordinate access, no privileged API data. The goal is to produce not a point cloud representation, but an **actual predicted block grid** (`world[x][y][z]`): the minimal discrete model that is maximally consistent with all visual observations and the world's known generation rules.

This pipeline is part of a larger research project benchmarking the gap between vision-derived spatial understanding and privileged world-state access in voxel environments — directly analogous to the gap between a robot navigating from cameras vs. GPS.

---

## What Has Been Built So Far

### Stage 1 — The Voxel World (Complete)
A custom 3D voxel world implemented as a single self-contained `world.html` file using Three.js and WebGL. Features:
- Procedurally placed blocks on a flat grid (gray terrain blocks + one red target block)
- First-person player controller with gravity, POV camera, WASD movement
- The ground truth world state is stored as a JavaScript array `world[x][y][z]` accessible at runtime
- The Three.js `PerspectiveCamera` has known fixed intrinsics: FOV=75°, aspect=width/height, near=0.1, far=1000

### Stage 2 — Visual 3D Reconstruction (Complete)
**VGGT-Ω** (Visual Geometry Grounded Transformer Omega, CVPR 2026 Oral, `arxiv:2605.15195`) takes multiple first-person RGB screenshots from the voxel world and produces:
- Per-pixel depth maps (shape: `(1, N_frames, H, W, 1)`)
- Camera pose estimates (extrinsics + intrinsics)
- Spatial register embeddings (compact learned spatial representations)
- A **3D point cloud exported as a `.glb` file** via the Gradio demo at `github.com/facebookresearch/vggt-omega`

The depth maps on voxel world screenshots already show **block-level geometric structure zero-shot** — depth discontinuities align with block edges without any fine-tuning on voxel data.

### Stage 3 — Inverse Reconstruction (THIS IS WHAT YOU ARE BUILDING)
Take the `.glb` point cloud output from VGGT-Ω and reconstruct the actual discrete block grid through inverse procedural reasoning.

---

## The Core Insight

A raw point cloud is a representation — millions of 3D coordinates with no semantic structure. The voxel world has hard prior knowledge baked in:

1. **All block faces are axis-aligned** — face normals must be `[1,0,0]`, `[0,1,0]`, or `[0,0,1]`
2. **All block boundaries fall on integer coordinates** — a surface at `y=3.73` is actually `y=4`
3. **Block size is exactly 1 unit** — face extents are integer multiples
4. **Ground plane is at `y=0`** — known constant
5. **Blocks are cubic and solid** — no partial blocks, no non-cubic geometry

These five priors reduce the reconstruction from an ill-posed continuous problem to a tractable discrete one. The output is not an approximation — it is a **hypothesis about the exact block grid** that generated the observations.

---

## What You Are Building

A Python program (NOT HTML — this is a standalone data processing pipeline) that:

1. **Loads a `.glb` file** output from VGGT-Ω
2. **Extracts the point cloud** from the GLB mesh/geometry
3. **Runs iterative RANSAC plane detection** to find all planar surfaces
4. **Filters to axis-aligned planes only** using known world priors
5. **Snaps each detected plane to the integer grid**
6. **Maps each snapped plane face to the block(s) it belongs to**
7. **Accumulates evidence across multiple frames/GLB files** into a 3D occupancy grid
8. **Outputs a predicted `world[x][y][z]` dictionary** — the reconstructed discrete model
9. **Visualizes both the input point cloud and the reconstructed block grid** side by side
10. **Computes reconstruction accuracy (IoU)** if a ground truth JSON is provided

---

## Technical Specification

### Input
- One or more `.glb` files exported from VGGT-Ω Gradio demo (each from a different set of viewpoints)
- Optional: `ground_truth.json` — the exported `world[x][y][z]` array from `world.html` for accuracy evaluation

### Output
- `predicted_world.json` — dictionary of `"x,y,z": block_type` for all detected blocks
- `reconstruction_visualization.html` — interactive Three.js viewer showing both the raw point cloud and the reconstructed block grid overlaid
- Console output: per-stage statistics and final IoU score (if ground truth provided)

### Dependencies
```
open3d          # point cloud loading, RANSAC plane detection, visualization
numpy           # linear algebra, grid operations
trimesh         # GLB file loading and mesh-to-pointcloud conversion
scipy           # spatial operations
matplotlib      # 2D diagnostic plots
```

Install with:
```bash
pip install open3d trimesh numpy scipy matplotlib
```

---

## Algorithm — Stage by Stage

### Stage 1: GLB Loading and Point Cloud Extraction

Load the `.glb` file using `trimesh`. A GLB from VGGT-Ω contains a scene with one or more mesh objects. Extract all vertex positions as a unified point cloud. Also extract vertex colors if present (useful for block type identification later).

```python
import trimesh
import numpy as np

def load_glb_as_pointcloud(glb_path):
    scene = trimesh.load(glb_path)
    all_points = []
    all_colors = []
    
    if isinstance(scene, trimesh.Scene):
        for name, geometry in scene.geometry.items():
            all_points.append(geometry.vertices)
            if hasattr(geometry.visual, 'vertex_colors'):
                all_colors.append(geometry.visual.vertex_colors[:, :3])
    else:
        all_points.append(scene.vertices)
    
    points = np.vstack(all_points)
    colors = np.vstack(all_colors) if all_colors else None
    return points, colors
```

### Stage 2: Coordinate System Normalization

VGGT-Ω outputs point clouds in camera space. The world space of `world.html` uses Three.js conventions (Y-up, right-handed). Apply a coordinate transform to align the reconstructed cloud with the world grid before snapping. The transform can be estimated from the ground plane detection (Stage 3) — the detected horizontal plane at the lowest Y extent is `y=0`.

### Stage 3: Iterative RANSAC Plane Detection

Use Open3D's `segment_plane` in a loop. Each iteration extracts the largest remaining plane, removes its inliers, and repeats until no plane has more than `min_inliers` points remaining.

```python
import open3d as o3d

def extract_all_planes(points, distance_threshold=0.12, min_inliers=50, max_planes=200):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    planes = []
    remaining = pcd
    
    for _ in range(max_planes):
        if len(remaining.points) < min_inliers:
            break
        
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=2000
        )
        
        if len(inliers) < min_inliers:
            break
        
        inlier_cloud = remaining.select_by_index(inliers)
        planes.append({
            'model': plane_model,      # [a, b, c, d]
            'points': np.asarray(inlier_cloud.points),
            'inlier_count': len(inliers)
        })
        
        remaining = remaining.select_by_index(inliers, invert=True)
    
    return planes
```

**Key parameter:** `distance_threshold=0.12` — set to roughly 1/8 of a block unit. In your world blocks are 1.0 unit, so 0.12 gives tolerance for VGGT-Ω depth estimation noise while still distinguishing adjacent block faces.

### Stage 4: Axis-Alignment Filter

For each detected plane, check whether its normal is within `axis_tolerance` degrees of one of the three world axes. Discard any plane that isn't axis-aligned — in a voxel world, non-axis-aligned planes are exclusively noise or depth estimation artifacts.

```python
def filter_axis_aligned(planes, axis_tolerance_deg=20.0):
    axis_vectors = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    aligned = []
    for plane in planes:
        a, b, c, d = plane['model']
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        dots = np.abs(axis_vectors @ normal)
        best_axis_dot = np.max(dots)
        best_axis_idx = np.argmax(dots)
        
        angle_deg = np.degrees(np.arccos(np.clip(best_axis_dot, -1, 1)))
        
        if angle_deg <= axis_tolerance_deg:
            plane['axis'] = best_axis_idx // 2        # 0=X, 1=Y, 2=Z
            plane['direction'] = best_axis_idx % 2    # 0=positive, 1=negative
            plane['axis_normal'] = axis_vectors[best_axis_idx]
            aligned.append(plane)
    
    return aligned
```

### Stage 5: Integer Grid Snapping

For each axis-aligned plane, compute the plane's position along its dominant axis and snap to the nearest integer. This is the core prior step — it converts continuous geometry back into the discrete grid.

```python
def snap_planes_to_grid(planes):
    snapped = []
    for plane in planes:
        a, b, c, d = plane['model']
        axis = plane['axis']
        
        axis_components = [a, b, c]
        normal_component = axis_components[axis]
        
        # position along axis = -d / normal_component
        raw_position = -d / (normal_component + 1e-8)
        snapped_position = round(raw_position)
        
        plane['raw_position'] = raw_position
        plane['snapped_position'] = snapped_position
        plane['snap_error'] = abs(raw_position - snapped_position)
        
        snapped.append(plane)
    
    # discard planes where snap error is too large (not a real block face)
    snapped = [p for p in snapped if p['snap_error'] < 0.4]
    
    return snapped
```

### Stage 6: Plane Face to Block Identity

Each snapped plane face is the boundary between two block positions. Determine which side has the block by the face direction, then enumerate all `(x,z)` or `(x,y)` or `(y,z)` grid cells covered by the plane's inlier point footprint.

```python
def plane_to_blocks(plane, world_bounds=(-50, 50)):
    axis = plane['axis']
    direction = plane['direction']
    pos = plane['snapped_position']
    points = plane['points']
    
    # determine which side the block is on
    # direction=0 means normal points positive, so block is on negative side
    block_offset = -1 if direction == 0 else 0
    
    blocks = set()
    
    for pt in points:
        if axis == 0:    # YZ plane — X face
            by = int(np.floor(pt[1]))
            bz = int(np.floor(pt[2]))
            bx = pos + block_offset
            blocks.add((bx, by, bz))
        elif axis == 1:  # XZ plane — Y face (floor/ceiling)
            bx = int(np.floor(pt[0]))
            bz = int(np.floor(pt[2]))
            by = pos + block_offset
            blocks.add((bx, by, bz))
        elif axis == 2:  # XY plane — Z face
            bx = int(np.floor(pt[0]))
            by = int(np.floor(pt[1]))
            bz = pos + block_offset
            blocks.add((bx, by, bz))
    
    # filter to world bounds
    blocks = {(x, y, z) for x, y, z in blocks
              if world_bounds[0] <= x <= world_bounds[1]
              and 0 <= y <= 20
              and world_bounds[0] <= z <= world_bounds[1]}
    
    return blocks
```

### Stage 7: Multi-Frame Evidence Accumulation

Process multiple GLB files. For each block position, count how many independent frame sets detected it. Require at least `min_votes` detections before confirming a block exists.

```python
def accumulate_evidence(glb_paths, min_votes=1):
    evidence = {}
    
    for glb_path in glb_paths:
        points, colors = load_glb_as_pointcloud(glb_path)
        planes = extract_all_planes(points)
        planes = filter_axis_aligned(planes)
        planes = snap_planes_to_grid(planes)
        
        frame_blocks = set()
        for plane in planes:
            frame_blocks.update(plane_to_blocks(plane))
        
        for block in frame_blocks:
            evidence[block] = evidence.get(block, 0) + 1
    
    predicted_world = {pos for pos, votes in evidence.items()
                       if votes >= min_votes}
    
    return predicted_world, evidence
```

### Stage 8: IoU Evaluation (if ground truth provided)

```python
def compute_iou(predicted_set, ground_truth_set):
    intersection = len(predicted_set & ground_truth_set)
    union = len(predicted_set | ground_truth_set)
    iou = intersection / union if union > 0 else 0.0
    
    precision = intersection / len(predicted_set) if predicted_set else 0.0
    recall = intersection / len(ground_truth_set) if ground_truth_set else 0.0
    
    print(f"Predicted blocks:    {len(predicted_set)}")
    print(f"Ground truth blocks: {len(ground_truth_set)}")
    print(f"Intersection:        {intersection}")
    print(f"IoU:                 {iou:.4f}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    
    return iou, precision, recall
```

### Stage 9: Visualization

Produce an interactive HTML viewer using Three.js that shows:
- **Left panel**: the raw VGGT-Ω point cloud (colored by depth)
- **Right panel**: the reconstructed block grid (each predicted block rendered as a colored cube)
- **Toggle**: overlay mode showing both simultaneously with block grid in wireframe

The HTML viewer should be self-contained — embed the point cloud data and block grid as JSON directly in the HTML file.

---

## Full Entry Point

```python
import json
import sys

def main(glb_paths, ground_truth_path=None, output_dir="."):
    print(f"Processing {len(glb_paths)} GLB file(s)...")
    
    predicted_set, evidence = accumulate_evidence(glb_paths, min_votes=1)
    
    # save predicted world
    predicted_list = [{"x": x, "y": y, "z": z} for x, y, z in predicted_set]
    with open(f"{output_dir}/predicted_world.json", "w") as f:
        json.dump(predicted_list, f, indent=2)
    print(f"Saved predicted_world.json ({len(predicted_set)} blocks)")
    
    # evaluate if ground truth provided
    if ground_truth_path:
        with open(ground_truth_path) as f:
            gt_data = json.load(f)
        ground_truth_set = {(b["x"], b["y"], b["z"]) for b in gt_data}
        compute_iou(predicted_set, ground_truth_set)
    
    # generate visualization
    generate_html_viewer(predicted_set, glb_paths[0], f"{output_dir}/reconstruction_visualization.html")
    print("Saved reconstruction_visualization.html")

if __name__ == "__main__":
    glb_files = sys.argv[1:-1] if len(sys.argv) > 2 else [sys.argv[1]]
    gt_file = sys.argv[-1] if sys.argv[-1].endswith(".json") else None
    main(glb_files, gt_file)
```

---

## File Structure to Produce

```
voxel_reconstruct/
├── reconstruct.py          # main entry point (the full pipeline)
├── loader.py               # GLB loading and point cloud extraction
├── planes.py               # RANSAC, axis filter, grid snapping, block mapping
├── accumulator.py          # multi-frame evidence accumulation
├── evaluate.py             # IoU and accuracy metrics
├── visualize.py            # HTML Three.js viewer generation
├── requirements.txt        # open3d, trimesh, numpy, scipy, matplotlib
└── README.md               # usage instructions
```

---

## Important Implementation Notes

- **Open3D RANSAC** is the correct tool — do NOT use scikit-learn's RANSAC, it's not optimized for plane fitting in 3D at this scale
- **Coordinate system**: VGGT-Ω outputs in OpenCV convention (Y-down, Z-forward). Three.js uses Y-up, Z-backward. Apply the transform `[x, y, z] → [x, -y, -z]` before snapping
- **Scale ambiguity**: VGGT-Ω estimates relative depth, not metric depth. The point cloud may be scaled differently from your world units. Resolve by detecting the ground plane first and using the known block size (1.0 unit) to calibrate the scale factor before snapping
- **Multiple planes at same snapped position**: merge them — they're the same face detected twice due to point density variation
- **The HTML visualization file** should be completely self-contained with Three.js loaded from CDN and all data embedded as inline JSON — no server required, opens directly in a browser
- **Do not use any game coordinate data** from `world.html` during reconstruction — the only input to the pipeline is the `.glb` file. Ground truth JSON is only used for the evaluation step, never during reconstruction

---

## Exporting Ground Truth from world.html

To produce the `ground_truth.json` for the evaluation step, add this snippet to `world.html` (run in browser console after the world loads):

```javascript
const blocks = [];
for (let x = -25; x <= 25; x++)
  for (let y = 0; y <= 10; y++)
    for (let z = -25; z <= 25; z++)
      if (world[x] && world[x][y] && world[x][y][z])
        blocks.push({x, y, z, type: world[x][y][z]});

const blob = new Blob([JSON.stringify(blocks, null, 2)], {type: 'application/json'});
const a = document.createElement('a');
a.href = URL.createObjectURL(blob);
a.download = 'ground_truth.json';
a.click();
```

---

## Usage

```bash
# single GLB file, no evaluation
python reconstruct.py output.glb

# multiple GLB files (multi-frame accumulation)
python reconstruct.py frame1.glb frame2.glb frame3.glb

# with ground truth evaluation
python reconstruct.py frame1.glb frame2.glb ground_truth.json
```

---

## Research Context

This pipeline is the evaluation backbone for a novel benchmark measuring **vision-derived spatial reconstruction accuracy in discrete voxel environments**. The key metric is the IoU between the predicted block grid and ground truth — the first time this metric has been computed for a VGGT-class model on a voxel-world domain. The pipeline will later be extended to full Minecraft environments where block type classification (color → block type via a trained MLP) and learned terrain priors replace the hardcoded generation rules used here.
