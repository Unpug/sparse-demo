# voxel_reconstruct

Grid-first inverse reconstruction: takes `.glb` point cloud output from **VGGT-Ω**
and recovers the discrete voxel block grid that best matches the observed geometry.

The output is always a **clean, uniform, equidistant block world** — the same kind
that `world.html` generates — not a point cloud or an irregular mesh.

## Install

```bash
pip install open3d trimesh numpy scipy matplotlib
```

## Usage

```bash
# Single GLB, no ground truth
python reconstruct.py output.glb

# Multiple GLBs — block votes accumulated across all files
python reconstruct.py frame1.glb frame2.glb frame3.glb

# With IoU evaluation against exported ground truth
python reconstruct.py frame1.glb frame2.glb ground_truth.json

# Provide the known block size to skip auto-estimation
python reconstruct.py output.glb --block-size 1.0 --face-threshold 0.12
```

## Outputs

| File | Description |
|------|-------------|
| `predicted_world.json` | `[{x,y,z}, ...]` — one entry per detected block |
| `reconstruction_visualization.html` | Self-contained Three.js viewer (open in any browser) |

## How it works (grid-first)

Instead of detecting planes first and trying to snap them, we commit to a
uniform lattice up-front and ask "which cells are occupied?":

1. **Block-size estimation** — auto-correlate projected coordinates on each axis
   to find the dominant periodic spacing (the voxel edge length in raw cloud units).
   You can override this with `--block-size`.

2. **Origin offset estimation** — divide all coordinates by the block size to get
   lattice coordinates, then find the sub-voxel shift δ ∈ [0,1) per axis that
   maximises how many points land near an integer (i.e. on a face boundary).
   This corrects for the unknown position of the world grid within the camera frame.

3. **Face voting** — for every point within `face_threshold` of an axis-aligned
   integer boundary, cast a vote for the block on the solid side.
   Local PCA surface normals (via Open3D) tell us which side is solid.

4. **Thresholding** — a cell is accepted as occupied if it receives ≥ `cell-min-votes`
   face-point votes.

5. **Multi-file accumulation** — repeat across all GLB files; require ≥ `min-votes`
   files to agree before confirming a block.

The result is always a clean `world[x][y][z]`-style boolean lattice of
uniform-size, uniformly-spaced cubes — exactly what `world.html` generates.

## Export ground truth from world.html

Paste in the browser console after the world loads:

```js
const blocks = [];
for (let x = 0; x < 30; x++)
  for (let y = 0; y < 5; y++)
    for (let z = 0; z < 30; z++)
      if (world[x]?.[y]?.[z])
        blocks.push({x, y, z, type: world[x][y][z]});

const blob = new Blob([JSON.stringify(blocks, null, 2)], {type:'application/json'});
const a = document.createElement('a');
a.href = URL.createObjectURL(blob);
a.download = 'ground_truth.json';
a.click();
```

## Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--block-size F` | auto | Voxel edge length in raw cloud units |
| `--face-threshold F` | 0.15 | Point counts as "on a face" if within this fraction of one block from a boundary |
| `--cell-min-votes N` | 3 | Face-point votes required per cloud to accept a cell |
| `--min-votes N` | 1 | GLB files that must agree on a block |
| `--no-normals` | — | Skip normal estimation; vote both sides of each face (faster, noisier) |
| `--x/y/z-bounds N N` | 0–50, 0–20, 0–50 | Clip output to world bounds |

## File structure

```
voxel_reconstruct/
├── reconstruct.py   CLI entry point
├── loader.py        GLB → point cloud; OpenCV → Three.js coord flip
├── grid_fit.py      Block-size estimation, origin fitting, face voting
├── accumulator.py   Multi-GLB loop and vote accumulation
├── evaluate.py      IoU / precision / recall vs ground truth
├── visualize.py     Self-contained Three.js HTML viewer
└── requirements.txt open3d trimesh numpy scipy matplotlib
```
