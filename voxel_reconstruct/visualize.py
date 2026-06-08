"""
visualize.py — Generate a self-contained Three.js HTML viewer.

Layout:
  Left panel  → raw VGGT-Ω point cloud (colored by depth)
  Right panel → reconstructed block grid (solid cubes)
  Toggle      → overlay mode (both at once; grid in wireframe)

All data is embedded as inline JSON — no server required.
"""

from __future__ import annotations

import json
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _depth_color(points: np.ndarray) -> list[str]:
    """Map depth (Z axis) to a hex color string for each point."""
    z = points[:, 2].astype(float)
    lo, hi = z.min(), z.max()
    if hi == lo:
        return ["#4488ff"] * len(points)

    t = (z - lo) / (hi - lo)  # 0 = near (blue), 1 = far (red)
    colors = []
    for v in t:
        r = int(255 * v)
        g = int(255 * (1 - abs(2 * v - 1)))
        b = int(255 * (1 - v))
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def _subsample_for_html(points: np.ndarray, max_pts: int = 80_000) -> np.ndarray:
    if len(points) <= max_pts:
        return points
    idx = np.random.choice(len(points), max_pts, replace=False)
    return points[idx]


# ── main entry point ──────────────────────────────────────────────────────────

def generate_html_viewer(
    predicted_set: set[tuple],
    points: np.ndarray,
    output_path: str,
    max_cloud_points: int = 80_000,
) -> None:
    """
    Write a self-contained HTML file with an interactive Three.js viewer.

    Args:
        predicted_set   — set of (x, y, z) tuples for reconstructed blocks
        points          — (N, 3) numpy array of the raw point cloud
        output_path     — file path to write to
        max_cloud_points— cap on embedded point count (keep HTML size reasonable)
    """
    pts = _subsample_for_html(points, max_cloud_points)
    colors = _depth_color(pts)

    # Serialise as flat arrays for compact embedding
    pts_flat = pts.flatten().tolist()
    colors_list = colors  # list of "#rrggbb" strings

    blocks_list = [{"x": x, "y": y, "z": z} for x, y, z in sorted(predicted_set)]

    pts_json    = json.dumps(pts_flat)
    colors_json = json.dumps(colors_list)
    blocks_json = json.dumps(blocks_list)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Voxel Reconstruction Viewer</title>
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ background:#0d0d1a; font-family:monospace; color:#ccc; overflow:hidden; }}
    #ui {{
      position:fixed; top:12px; left:50%; transform:translateX(-50%);
      display:flex; gap:10px; z-index:10;
    }}
    .btn {{
      padding:7px 16px; border-radius:6px; font-family:monospace; font-size:13px;
      cursor:pointer; border:1px solid #555; background:rgba(0,0,0,0.6); color:#ccc;
    }}
    .btn:hover {{ background:rgba(40,40,80,0.9); }}
    .btn.active {{ background:#1a3a6a; border-color:#4a8adf; color:#8cf; }}
    #info {{
      position:fixed; bottom:12px; left:50%; transform:translateX(-50%);
      font-size:12px; color:#666; text-align:center;
    }}
    #labels {{
      position:fixed; top:12px; display:flex; width:100%;
      pointer-events:none;
    }}
    .panel-label {{
      flex:1; text-align:center; font-size:12px; color:#555;
      padding-top:12px; letter-spacing:1px;
    }}
    canvas {{ display:block; }}
  </style>
</head>
<body>
  <div id="labels">
    <div class="panel-label" id="lbl-left">POINT CLOUD</div>
    <div class="panel-label" id="lbl-right">RECONSTRUCTED GRID</div>
  </div>
  <div id="ui">
    <button class="btn active" id="btn-split">Split View</button>
    <button class="btn" id="btn-overlay">Overlay</button>
    <button class="btn" id="btn-cloud">Cloud Only</button>
    <button class="btn" id="btn-grid">Grid Only</button>
  </div>
  <div id="info">Drag to orbit · Scroll to zoom · Right-drag to pan</div>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }}
  }}
  </script>

  <script type="module">
    import * as THREE from 'three';
    import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

    // ── embedded data ──────────────────────────────────────────────────────
    const PTS_FLAT  = {pts_json};
    const COLORS    = {colors_json};
    const BLOCKS    = {blocks_json};

    // ── modes ──────────────────────────────────────────────────────────────
    // 'split' | 'overlay' | 'cloud' | 'grid'
    let mode = 'split';

    // ── scene ──────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({{ antialias:true }});
    renderer.setPixelRatio(devicePixelRatio);
    renderer.setSize(innerWidth, innerHeight);
    renderer.autoClear = false;
    document.body.appendChild(renderer.domElement);

    function makeScene() {{
      const s = new THREE.Scene();
      s.background = new THREE.Color(0x0d0d1a);
      s.fog = new THREE.Fog(0x0d0d1a, 80, 160);
      s.add(new THREE.AmbientLight(0xffffff, 0.7));
      const dl = new THREE.DirectionalLight(0xffffff, 1.0);
      dl.position.set(20, 40, 20);
      s.add(dl);
      return s;
    }}

    const sceneL = makeScene();
    const sceneR = makeScene();

    function makeCamera() {{
      const cam = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 400);
      cam.position.set(25, 20, 50);
      return cam;
    }}

    const camL = makeCamera();
    const camR = makeCamera();

    const ctrlL = new OrbitControls(camL, renderer.domElement);
    ctrlL.target.set(15, 5, 15);
    ctrlL.enableDamping = true;
    ctrlL.update();

    // Right panel shares the same orbit controls as left in split mode
    // (both cameras are synced in the render loop)

    // ── point cloud geometry ───────────────────────────────────────────────
    (function buildCloud() {{
      const n = PTS_FLAT.length / 3;
      const positions = new Float32Array(PTS_FLAT);
      const colorArr  = new Float32Array(n * 3);

      for (let i = 0; i < n; i++) {{
        const hex = COLORS[i] || '#4488ff';
        const r = parseInt(hex.slice(1,3),16)/255;
        const g = parseInt(hex.slice(3,5),16)/255;
        const b = parseInt(hex.slice(5,7),16)/255;
        colorArr[i*3]   = r;
        colorArr[i*3+1] = g;
        colorArr[i*3+2] = b;
      }}

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('color',    new THREE.BufferAttribute(colorArr,  3));

      const mat = new THREE.PointsMaterial({{ size:0.08, vertexColors:true }});
      sceneL.add(new THREE.Points(geo, mat));

      // also add a dim copy to the overlay scene (sceneR)
      const matDim = new THREE.PointsMaterial({{ size:0.06, vertexColors:true, opacity:0.4, transparent:true }});
      sceneR.add(new THREE.Points(geo, matDim));
    }})();

    // ── block grid geometry ────────────────────────────────────────────────
    (function buildGrid() {{
      const geo     = new THREE.BoxGeometry(1,1,1);
      const matSolid = new THREE.MeshLambertMaterial({{ color:0x88aacc }});
      const matWire  = new THREE.MeshBasicMaterial({{ color:0x44aaff, wireframe:true }});

      const solidGroup = new THREE.Group();
      const wireGroup  = new THREE.Group();

      for (const b of BLOCKS) {{
        const sm = new THREE.Mesh(geo, matSolid);
        sm.position.set(b.x+0.5, b.y+0.5, b.z+0.5);
        solidGroup.add(sm);

        const wm = new THREE.Mesh(geo, matWire);
        wm.position.set(b.x+0.5, b.y+0.5, b.z+0.5);
        wireGroup.add(wm);
      }}

      sceneR.add(solidGroup);  // right panel: solid blocks (+ dim cloud added above)
      sceneR.add(wireGroup);   // overlay: wireframe (toggled by visibility)
      wireGroup.visible = false;

      window._solidGroup = solidGroup;
      window._wireGroup  = wireGroup;
    }})();

    // ── UI buttons ─────────────────────────────────────────────────────────
    const btns = {{
      split:   document.getElementById('btn-split'),
      overlay: document.getElementById('btn-overlay'),
      cloud:   document.getElementById('btn-cloud'),
      grid:    document.getElementById('btn-grid'),
    }};
    const lblLeft  = document.getElementById('lbl-left');
    const lblRight = document.getElementById('lbl-right');

    function setMode(m) {{
      mode = m;
      Object.values(btns).forEach(b => b.classList.remove('active'));
      btns[m].classList.add('active');
    }}

    btns.split.addEventListener('click',   () => setMode('split'));
    btns.overlay.addEventListener('click', () => setMode('overlay'));
    btns.cloud.addEventListener('click',   () => setMode('cloud'));
    btns.grid.addEventListener('click',    () => setMode('grid'));

    // ── resize ─────────────────────────────────────────────────────────────
    window.addEventListener('resize', () => {{
      renderer.setSize(innerWidth, innerHeight);
    }});

    // ── render loop ────────────────────────────────────────────────────────
    function animate() {{
      requestAnimationFrame(animate);
      ctrlL.update();

      // Sync right camera to left so split view feels coherent
      camR.position.copy(camL.position);
      camR.quaternion.copy(camL.quaternion);

      renderer.clear();

      const W = innerWidth, H = innerHeight;
      const HW = Math.floor(W / 2);

      if (mode === 'split') {{
        lblLeft.style.display  = '';
        lblRight.style.display = '';
        _wireGroup.visible = false;

        // left: cloud
        renderer.setScissorTest(true);
        renderer.setScissor(0, 0, HW, H);
        renderer.setViewport(0, 0, HW, H);
        camL.aspect = HW / H;
        camL.updateProjectionMatrix();
        renderer.render(sceneL, camL);

        // right: grid
        renderer.setScissor(HW, 0, W - HW, H);
        renderer.setViewport(HW, 0, W - HW, H);
        camR.aspect = (W - HW) / H;
        camR.updateProjectionMatrix();
        renderer.render(sceneR, camR);

        renderer.setScissorTest(false);

      }} else if (mode === 'overlay') {{
        lblLeft.style.display  = 'none';
        lblRight.style.display = 'none';
        _wireGroup.visible = true;
        _solidGroup.visible = false;

        renderer.setViewport(0, 0, W, H);
        camL.aspect = W / H;
        camL.updateProjectionMatrix();
        renderer.render(sceneR, camL);
        _solidGroup.visible = true;

      }} else if (mode === 'cloud') {{
        lblLeft.style.display  = 'none';
        lblRight.style.display = 'none';
        renderer.setViewport(0, 0, W, H);
        camL.aspect = W / H;
        camL.updateProjectionMatrix();
        renderer.render(sceneL, camL);

      }} else {{ // grid
        lblLeft.style.display  = 'none';
        lblRight.style.display = 'none';
        _wireGroup.visible = false;
        renderer.setViewport(0, 0, W, H);
        camR.aspect = W / H;
        camR.updateProjectionMatrix();
        renderer.render(sceneR, camR);
      }}
    }}

    animate();
  </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
