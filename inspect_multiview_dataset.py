"""
inspect_multiview_dataset.py

This script performs geometric sanity checks for a Kubric-generated multi-view dataset.

The goal of this tool is to verify whether the following components are mutually
consistent:

    - camera intrinsics
    - camera extrinsics
    - depth maps
    - multi-view images

Specifically, the script checks whether pixels from one view can be correctly
reprojected into another view using the depth map and camera parameters.

------------------------------------------------------------
Dataset Assumption
------------------------------------------------------------

Each scene contains multiple views with the following files:

    view_xxx/
        A.png          # frame at time t
        B.png          # frame at time t+1
        Y.png          # motion-magnified frame
        depth.exr      # depth map
        mask.png       # foreground mask
        camera.json    # camera intrinsics and extrinsics

The depth values follow Kubric's convention where extremely large values
(e.g., ~1e10) represent background / infinity and should be ignored.

------------------------------------------------------------
Geometric Verification Pipeline
------------------------------------------------------------

For each scene, the script performs the following checks:

1. Camera Matrix Consistency
   Verify that the world-to-camera (w2c) and camera-to-world (c2w) matrices
   are valid inverses.

2. Depth Statistics
   Report depth distribution and compute the ratio of valid foreground pixels.

3. Pixel Roundtrip Test
   Check that a pixel projected to 3D and then reprojected back to the same
   camera returns the original pixel location.

4. Self Reprojection
   Reproject view0 pixels back into view0 using depth and camera parameters.
   This verifies depth correctness and projection implementation.

5. Cross-view Reprojection
   Reproject pixels from view0 to view1:

        pixel(view0)
        → depth
        → 3D point in camera0
        → transform to world
        → transform to camera1
        → project to image plane

   The result is compared with the ground-truth image in view1.

------------------------------------------------------------
Evaluation Metrics
------------------------------------------------------------

The following metrics are reported:

    MAE
        Mean absolute RGB error between the reprojected image
        and the ground-truth target image.

    Coverage
        Fraction of pixels that successfully project into the target image.

    Mask IoU
        Overlap between the projected foreground mask and the
        ground-truth mask.

    Centroid Distance
        Distance between projected mask centroid and ground-truth centroid.

These metrics indicate whether the multi-view geometry in the dataset is
correct.

------------------------------------------------------------
Important Implementation Notes
------------------------------------------------------------

1. Kubric depth values represent z-depth (distance along camera z-axis),
   not ray length.

2. Background pixels often have extremely large depth values (~1e10).
   These pixels must be masked out before reprojection.

3. Kubric / Blender cameras use an OpenGL-style coordinate convention,
   which may require conversion to the standard computer vision
   camera convention (+Z forward) before projection.

------------------------------------------------------------
Outputs
------------------------------------------------------------

The script generates:

    summary.json
        numerical statistics for each scene

    *_reprojection_check.png
        visualization of reprojection alignment

    *_absdiff.png
        RGB difference maps

These outputs help diagnose geometry errors in synthetic datasets.

------------------------------------------------------------
Typical Usage
------------------------------------------------------------

python inspect_multiview_dataset.py \
    --root output_test \
    --out output_test/_inspect \
    --max_scenes 5

Optional flags:

    --depth_mode {z, ray}
    --use_pixel_centers
    --do_bilinear_splat
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_png(p):
    return np.array(Image.open(p).convert('RGB'))

def load_mask(p):
    m = np.array(Image.open(p))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8) * 255

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_depth(depth):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[2] == 1:
            depth = depth[..., 0]
        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW or HxWx1, got {depth.shape}")
    return depth.astype(np.float32)

def make_depth_valid_mask(depth, max_depth=100.0):
    depth = normalize_depth(depth)
    return np.isfinite(depth) & (depth > 0) & (depth < max_depth)

def project_points(K, T_w2c, pts_w):
    pts_h = np.concatenate(
        [pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    pts_c = (T_w2c @ pts_h.T).T[:, :3]
    z = pts_c[:, 2]
    valid = z > 1e-6

    uv = np.zeros((pts_w.shape[0], 2), dtype=np.float32)
    uv[valid, 0] = K[0, 0] * (pts_c[valid, 0] / z[valid]) + K[0, 2]
    uv[valid, 1] = K[1, 1] * (pts_c[valid, 1] / z[valid]) + K[1, 2]
    return uv, z, valid

def convert_extrinsics_gl_to_cv(T_c2w_gl, T_w2c_gl):
    CV_FROM_GL = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)

    # X_world = T_c2w_gl * X_gl = T_c2w_gl * CV_FROM_GL * X_cv
    T_c2w_cv = T_c2w_gl @ CV_FROM_GL

    # X_cv = CV_FROM_GL * X_gl = CV_FROM_GL * T_w2c_gl * X_world
    T_w2c_cv = CV_FROM_GL @ T_w2c_gl

    return T_c2w_cv, T_w2c_cv

def depth_to_camera_points_zdepth(depth, K, use_pixel_centers=True):
    depth = normalize_depth(depth)
    H, W = depth.shape

    offset = 0.5 if use_pixel_centers else 0.0
    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32) + offset,
        np.arange(H, dtype=np.float32) + offset
    )

    z = depth.astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z

    pts_c = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts_c


def depth_to_camera_points_raydepth(depth, K, use_pixel_centers=True):
    """
    If depth is ray length instead of z-depth, use this conversion.
    """
    depth = normalize_depth(depth)
    H, W = depth.shape

    offset = 0.5 if use_pixel_centers else 0.0
    xs, ys = np.meshgrid(
        np.arange(W, dtype=np.float32) + offset,
        np.arange(H, dtype=np.float32) + offset
    )

    d = depth.astype(np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    rx = (xs - cx) / fx
    ry = (ys - cy) / fy
    rz = np.ones_like(rx)

    norm = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-12
    x = d * rx / norm
    y = d * ry / norm
    z = d * rz / norm

    pts_c = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    return pts_c


def camera_points_to_world(pts_c, T_c2w):
    pts_h = np.concatenate(
        [pts_c, np.ones((pts_c.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    pts_w = (T_c2w @ pts_h.T).T[:, :3]
    return pts_w


def depth_to_world(depth, K, T_c2w, depth_mode="z", use_pixel_centers=True):
    if depth_mode == "z":
        pts_c = depth_to_camera_points_zdepth(depth, K, use_pixel_centers=use_pixel_centers)
    elif depth_mode == "ray":
        pts_c = depth_to_camera_points_raydepth(depth, K, use_pixel_centers=use_pixel_centers)
    else:
        raise ValueError(f"Unsupported depth_mode: {depth_mode}")

    pts_w = camera_points_to_world(pts_c, T_c2w)
    return pts_w


def reproject_rgb(
    src_rgb,
    src_depth,
    src_cam,
    tgt_cam,
    depth_mode="z",
    use_pixel_centers=True,
    do_bilinear_splat=True,
):
    K_src = np.array(src_cam['intrinsic_K'], dtype=np.float32)
    T_src_c2w_gl = np.array(src_cam['extrinsic_camera_to_world'], dtype=np.float32)
    T_src_w2c_gl = np.array(src_cam['extrinsic_world_to_camera'], dtype=np.float32)

    K_tgt = np.array(tgt_cam['intrinsic_K'], dtype=np.float32)
    T_tgt_c2w_gl = np.array(tgt_cam['extrinsic_camera_to_world'], dtype=np.float32)
    T_tgt_w2c_gl = np.array(tgt_cam['extrinsic_world_to_camera'], dtype=np.float32)

    T_src_c2w, _ = convert_extrinsics_gl_to_cv(T_src_c2w_gl, T_src_w2c_gl)
    _, T_tgt_w2c = convert_extrinsics_gl_to_cv(T_tgt_c2w_gl, T_tgt_w2c_gl)

    src_depth = normalize_depth(src_depth)
    H, W = src_depth.shape
    src_valid_depth = make_depth_valid_mask(src_depth, max_depth=100.0).reshape(-1)

    pts_w = depth_to_world(
        src_depth,
        K_src,
        T_src_c2w,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
    )
    uv, z_tgt, valid_z = project_points(K_tgt, T_tgt_w2c, pts_w)

    if use_pixel_centers:
        uv = uv - 0.5

    out_acc = np.zeros((H, W, 3), dtype=np.float32)
    w_acc = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.uint8)

    src_flat = src_rgb.reshape(-1, 3).astype(np.float32)
    u = uv[:, 0]
    v = uv[:, 1]

    if do_bilinear_splat:
        valid = (
                valid_z &
                src_valid_depth &
                (u >= 0) & (u < W - 1) &
                (v >= 0) & (v < H - 1)
        )
    else:
        valid = (
                valid_z &
                src_valid_depth &
                (u >= 0) & (u < W) &
                (v >= 0) & (v < H)
        )

    idxs = np.nonzero(valid)[0]

    if do_bilinear_splat:
        for idx in idxs:
            uu = u[idx]
            vv = v[idx]
            zz = z_tgt[idx]
            color = src_flat[idx]

            u0 = int(np.floor(uu))
            v0 = int(np.floor(vv))
            du = uu - u0
            dv = vv - v0

            neighbors = [
                (u0,     v0,     (1 - du) * (1 - dv)),
                (u0 + 1, v0,     du * (1 - dv)),
                (u0,     v0 + 1, (1 - du) * dv),
                (u0 + 1, v0 + 1, du * dv),
            ]

            for xpix, ypix, w in neighbors:
                if w <= 0:
                    continue
                if xpix < 0 or xpix >= W or ypix < 0 or ypix >= H:
                    continue

                if zz < zbuf[ypix, xpix] + 1e-4:
                    if zz < zbuf[ypix, xpix]:
                        zbuf[ypix, xpix] = zz
                        out_acc[ypix, xpix] = 0.0
                        w_acc[ypix, xpix] = 0.0
                    out_acc[ypix, xpix] += w * color
                    w_acc[ypix, xpix] += w
                    coverage[ypix, xpix] = 255
    else:
        u_int = np.rint(u[idxs]).astype(np.int32)
        v_int = np.rint(v[idxs]).astype(np.int32)
        for k, idx in enumerate(idxs):
            uu = u_int[k]
            vv = v_int[k]
            zz = z_tgt[idx]
            if 0 <= uu < W and 0 <= vv < H and zz < zbuf[vv, uu]:
                zbuf[vv, uu] = zz
                out_acc[vv, uu] = src_flat[idx]
                w_acc[vv, uu] = 1.0
                coverage[vv, uu] = 255

    out = np.zeros((H, W, 3), dtype=np.uint8)
    m = w_acc > 1e-8
    out[m] = np.clip(out_acc[m] / w_acc[m, None], 0, 255).astype(np.uint8)

    stats = {
        "num_total_points": int(len(u)),
        "num_positive_z": int(valid_z.sum()),
        "num_valid_in_image": int(valid.sum()),
        "valid_ratio_total": float(valid.sum() / max(len(u), 1)),
        "coverage_ratio": float((coverage > 0).mean()),
        "u_min": float(np.min(u[valid])) if np.any(valid) else None,
        "u_max": float(np.max(u[valid])) if np.any(valid) else None,
        "v_min": float(np.min(v[valid])) if np.any(valid) else None,
        "v_max": float(np.max(v[valid])) if np.any(valid) else None,
        "z_tgt_min": float(np.min(z_tgt[valid])) if np.any(valid) else None,
        "z_tgt_max": float(np.max(z_tgt[valid])) if np.any(valid) else None,
    }

    return out, coverage, stats

def reproject_mask(src_mask, src_depth, src_cam, tgt_cam,
                   depth_mode="z", use_pixel_centers=True):
    K_src = np.array(src_cam['intrinsic_K'], dtype=np.float32)
    T_src_c2w_gl = np.array(src_cam['extrinsic_camera_to_world'], dtype=np.float32)
    T_src_w2c_gl = np.array(src_cam['extrinsic_world_to_camera'], dtype=np.float32)

    K_tgt = np.array(tgt_cam['intrinsic_K'], dtype=np.float32)
    T_tgt_c2w_gl = np.array(tgt_cam['extrinsic_camera_to_world'], dtype=np.float32)
    T_tgt_w2c_gl = np.array(tgt_cam['extrinsic_world_to_camera'], dtype=np.float32)

    T_src_c2w, _ = convert_extrinsics_gl_to_cv(T_src_c2w_gl, T_src_w2c_gl)
    _, T_tgt_w2c = convert_extrinsics_gl_to_cv(T_tgt_c2w_gl, T_tgt_w2c_gl)

    src_depth = normalize_depth(src_depth)
    H, W = src_depth.shape
    src_valid_depth = make_depth_valid_mask(src_depth, max_depth=100.0).reshape(-1)

    pts_w = depth_to_world(
        src_depth,
        K_src,
        T_src_c2w,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
    )
    uv, z_tgt, valid_z = project_points(K_tgt, T_tgt_w2c, pts_w)

    if use_pixel_centers:
        uv = uv - 0.5

    out = np.zeros((H, W), dtype=np.uint8)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    src_flat = (src_mask.reshape(-1) > 0)
    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)

    valid = (
        valid_z &
        src_valid_depth &
        src_flat &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H)
    )

    idxs = np.nonzero(valid)[0]
    for idx in idxs:
        uu, vv, zz = u[idx], v[idx], z_tgt[idx]
        if zz < zbuf[vv, uu]:
            zbuf[vv, uu] = zz
            out[vv, uu] = 255

    return out

def masked_mae(a, b, mask):
    m = mask > 0
    if m.sum() == 0:
        return np.nan
    return np.abs(a.astype(np.float32) - b.astype(np.float32))[m].mean()

def mask_iou(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return np.nan
    return inter / union


def mask_centroid(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def centroid_distance(mask_a, mask_b):
    ca = mask_centroid(mask_a)
    cb = mask_centroid(mask_b)
    if ca is None or cb is None:
        return np.nan
    dx = ca[0] - cb[0]
    dy = ca[1] - cb[1]
    return float(np.sqrt(dx * dx + dy * dy))

def check_camera_consistency(cam, name="cam"):
    T_w2c = np.array(cam['extrinsic_world_to_camera'], dtype=np.float32)
    T_c2w = np.array(cam['extrinsic_camera_to_world'], dtype=np.float32)

    I1 = T_w2c @ T_c2w
    I2 = T_c2w @ T_w2c
    err1 = np.abs(I1 - np.eye(4, dtype=np.float32)).max()
    err2 = np.abs(I2 - np.eye(4, dtype=np.float32)).max()

    print(f"[{name}] inverse check max error: w2c*c2w={err1:.6e}, c2w*w2c={err2:.6e}")


def print_intrinsics(cam, name="cam"):
    K = np.array(cam['intrinsic_K'], dtype=np.float32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"[{name}] fx={fx:.6f}, fy={fy:.6f}, cx={cx:.6f}, cy={cy:.6f}")


def print_depth_stats(depth, name="depth"):
    depth = normalize_depth(depth)
    finite = np.isfinite(depth)
    valid = finite & (depth > 0)
    print(
        f"[{name}] shape={depth.shape}, "
        f"min={float(depth[valid].min()) if np.any(valid) else None:.6f}, "
        f"max={float(depth[valid].max()) if np.any(valid) else None:.6f}, "
        f"mean={float(depth[valid].mean()) if np.any(valid) else None:.6f}, "
        f"median={float(np.median(depth[valid])) if np.any(valid) else None:.6f}, "
        f"positive_ratio={float(valid.mean()):.6f}"
    )

def print_depth_valid_stats(depth, name="depth", max_depth=100.0):
    depth = normalize_depth(depth)
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    print(f"[{name}] valid(<{max_depth}m) ratio = {valid.mean():.6f}")


def debug_sample_points(depth, K, pts=None, name="sample"):
    if pts is None:
        pts = [(256, 256), (100, 100), (400, 400)]

    depth = normalize_depth(depth)
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print(f"[{name}] sample point debug:")
    for v, u in pts:
        if not (0 <= v < H and 0 <= u < W):
            continue
        z = float(depth[v, u])
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        r = np.sqrt(x * x + y * y + z * z)
        print(
            f"  (u,v)=({u},{v}) depth={z:.6f}, "
            f"xyz_zdepth=({x:.6f},{y:.6f},{z:.6f}), "
            f"ray_len={r:.6f}"
        )


def roundtrip_pixel_test(depth, K, T_c2w, T_w2c, samples=None, use_pixel_centers=True, name="roundtrip"):
    if samples is None:
        samples = [(256, 256), (100, 100), (400, 400)]

    depth = normalize_depth(depth)
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    offset = 0.5 if use_pixel_centers else 0.0

    print(f"[{name}] pixel roundtrip test:")
    for v, u in samples:
        if not (0 <= v < H and 0 <= u < W):
            continue

        z = float(depth[v, u])
        x = ((u + offset) - cx) / fx * z
        y = ((v + offset) - cy) / fy * z

        pt_c = np.array([x, y, z, 1.0], dtype=np.float32)
        pt_w = T_c2w @ pt_c
        pt_c2 = T_w2c @ pt_w

        x2, y2, z2 = pt_c2[:3]
        u2 = fx * (x2 / z2) + cx - offset
        v2 = fy * (y2 / z2) + cy - offset

        du = u2 - u
        dv = v2 - v

        print(
            f"  src=({u},{v}) -> reproj=({u2:.6f},{v2:.6f}), "
            f"du={du:.6e}, dv={dv:.6e}, z={z:.6f}, z2={z2:.6f}"
        )


def save_absdiff_image(a, b, out_path, title="abs diff"):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    vmax = max(1.0, float(np.percentile(diff, 99)))
    plt.figure(figsize=(6, 6))
    plt.imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_masked_absdiff_image(a, b, mask, out_path, title="masked abs diff"):
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)
    show = np.zeros_like(diff)
    m = mask > 0
    show[m] = diff[m]
    vmax = max(1.0, float(np.percentile(show[m], 99))) if np.any(m) else 1.0
    plt.figure(figsize=(6, 6))
    plt.imshow(show, cmap='hot', vmin=0, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def find_scenes(root):
    root = Path(root)
    scenes = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith('_'):
            continue
        if any((p / f'view_{i:03d}').exists() for i in range(10)):
            scenes.append(p)
            continue
        views = [q for q in p.iterdir() if q.is_dir() and q.name.startswith('view_')]
        if len(views) >= 2:
            scenes.append(p)
    return sorted(scenes)


def process_scene(scene_dir, out_dir, depth_mode="z", use_pixel_centers=True, do_bilinear_splat=True):
    print("=" * 100)
    print(f"[scene] {scene_dir}")

    views = sorted([p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith('view_')])
    if len(views) < 2:
        return None

    v0, v1 = views[0], views[1]

    A0 = load_png(v0 / 'A.png')
    A1 = load_png(v1 / 'A.png')
    Y0 = load_png(v0 / 'Y.png')
    Y1 = load_png(v1 / 'Y.png')

    segA0 = load_mask(v0 / 'seg_A.png')
    segA1 = load_mask(v1 / 'seg_A.png')
    segY0 = load_mask(v0 / 'seg_Y.png')
    segY1 = load_mask(v1 / 'seg_Y.png')

    D0A = np.load(v0 / 'depth_A.npy')
    D0Y = np.load(v0 / 'depth_Y.npy')
    cam0 = load_json(v0 / 'camera.json')
    cam1 = load_json(v1 / 'camera.json')
    segA1 = np.array(Image.open(v1 / 'seg_A.png'))
    segY1 = np.array(Image.open(v1 / 'seg_Y.png'))
    if segA1.ndim == 3:
        segA1 = segA1[..., 0]
    if segY1.ndim == 3:
        segY1 = segY1[..., 0]

    fgA1 = segA1 > 0
    fgY1 = segY1 > 0

    K0 = np.array(cam0['intrinsic_K'], dtype=np.float32)
    K1 = np.array(cam1['intrinsic_K'], dtype=np.float32)
    T0_w2c = np.array(cam0['extrinsic_world_to_camera'], dtype=np.float32)
    T0_c2w = np.array(cam0['extrinsic_camera_to_world'], dtype=np.float32)
    T1_w2c = np.array(cam1['extrinsic_world_to_camera'], dtype=np.float32)
    T1_c2w = np.array(cam1['extrinsic_camera_to_world'], dtype=np.float32)

    print_intrinsics(cam0, "cam0")
    print_intrinsics(cam1, "cam1")
    check_camera_consistency(cam0, "cam0")
    check_camera_consistency(cam1, "cam1")
    print_depth_stats(D0A, "D0A")
    print_depth_stats(D0Y, "D0Y")
    print_depth_valid_stats(D0A, "D0A", max_depth=100.0)
    print_depth_valid_stats(D0Y, "D0Y", max_depth=100.0)
    debug_sample_points(D0A, K0, name="D0A")
    roundtrip_pixel_test(D0A, K0, T0_c2w, T0_w2c, name="cam0 self roundtrip")
    roundtrip_pixel_test(D0A, K1, T1_c2w, T1_w2c, name="cam1 matrix roundtrip with cam1 intrinsics")

    # self reprojection
    reproj_self_A, cov_self_A, stats_self_A = reproject_rgb(
        A0, D0A, cam0, cam0,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
        do_bilinear_splat=do_bilinear_splat,
    )

    mask_self_A = reproject_mask(
        segA0, D0A, cam0, cam0,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
    )
    iou_self_A = mask_iou(mask_self_A, segA0)
    cd_self_A = centroid_distance(mask_self_A, segA0)
    print(f"[self mask A] IoU={iou_self_A:.6f}, centroid_dist={cd_self_A:.6f}")


    self_mae_A = masked_mae(reproj_self_A, A0, cov_self_A)
    print(
        f"[self reproj A] mae={self_mae_A}, "
        f"coverage={float((cov_self_A > 0).mean()):.6f}, "
        f"stats={stats_self_A}"
    )

    # cross reprojection
    reproj_A, cov_A, stats_A = reproject_rgb(
        A0, D0A, cam0, cam1,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
        do_bilinear_splat=do_bilinear_splat,
    )
    reproj_Y, cov_Y, stats_Y = reproject_rgb(
        Y0, D0Y, cam0, cam1,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
        do_bilinear_splat=do_bilinear_splat,
    )

    eval_mask_A = ((cov_A > 0) & fgA1).astype(np.uint8) * 255
    eval_mask_Y = ((cov_Y > 0) & fgY1).astype(np.uint8) * 255

    mae_A = masked_mae(reproj_A, A1, eval_mask_A)
    mae_Y = masked_mae(reproj_Y, Y1, eval_mask_Y)

    print(
        f"[cross reproj A] mae={mae_A}, "
        f"coverage={float((cov_A > 0).mean()):.6f}, "
        f"stats={stats_A}"
    )
    print(
        f"[cross reproj Y] mae={mae_Y}, "
        f"coverage={float((cov_Y > 0).mean()):.6f}, "
        f"stats={stats_Y}"
    )

    mask_cross_A = reproject_mask(
        segA0, D0A, cam0, cam1,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
    )
    iou_cross_A = mask_iou(mask_cross_A, segA1)
    cd_cross_A = centroid_distance(mask_cross_A, segA1)
    print(f"[cross mask A] IoU={iou_cross_A:.6f}, centroid_dist={cd_cross_A:.6f}")

    mask_cross_Y = reproject_mask(
        segY0, D0Y, cam0, cam1,
        depth_mode=depth_mode,
        use_pixel_centers=use_pixel_centers,
    )
    iou_cross_Y = mask_iou(mask_cross_Y, segY1)
    cd_cross_Y = centroid_distance(mask_cross_Y, segY1)
    print(f"[cross mask Y] IoU={iou_cross_Y:.6f}, centroid_dist={cd_cross_Y:.6f}")

    # Optional alternative interpretation: ray depth
    reproj_A_ray, cov_A_ray, stats_A_ray = reproject_rgb(
        A0, D0A, cam0, cam1,
        depth_mode="ray",
        use_pixel_centers=use_pixel_centers,
        do_bilinear_splat=do_bilinear_splat,
    )
    mae_A_ray = masked_mae(reproj_A_ray, A1, cov_A_ray)
    print(
        f"[cross reproj A | assume ray depth] mae={mae_A_ray}, "
        f"coverage={float((cov_A_ray > 0).mean()):.6f}, "
        f"stats={stats_A_ray}"
    )

    baseline = cam1.get('baseline_to_anchor', None)
    if baseline is None:
        p0 = np.array(cam0['camera_position'], dtype=np.float32)
        p1 = np.array(cam1['camera_position'], dtype=np.float32)
        baseline = float(np.linalg.norm(p1 - p0))

    # main figure
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    axes[0, 0].imshow(A0)
    axes[0, 0].set_title('view0 A')

    axes[0, 1].imshow(A1)
    axes[0, 1].set_title('view1 A')

    axes[0, 2].imshow(reproj_A)
    axes[0, 2].set_title('A reproj 0→1')

    axes[0, 3].imshow(cov_A, cmap='gray')
    axes[0, 3].set_title(f'A coverage\nMAE={mae_A:.2f}')

    axes[1, 0].imshow(Y0)
    axes[1, 0].set_title('view0 Y')

    axes[1, 1].imshow(Y1)
    axes[1, 1].set_title('view1 Y')

    axes[1, 2].imshow(reproj_Y)
    axes[1, 2].set_title('Y reproj 0→1')

    axes[1, 3].imshow(cov_Y, cmap='gray')
    axes[1, 3].set_title(f'Y coverage\nMAE={mae_Y:.2f}')

    axes[2, 0].imshow(reproj_self_A)
    axes[2, 0].set_title(f'self reproj A\nMAE={self_mae_A:.2f}')

    axes[2, 1].imshow(cov_self_A, cmap='gray')
    axes[2, 1].set_title(f'self coverage\n{(cov_self_A > 0).mean():.4f}')

    axes[2, 2].imshow(reproj_A_ray)
    axes[2, 2].set_title(f'A reproj 0→1 (ray)\nMAE={mae_A_ray:.2f}')

    axes[2, 3].imshow(cov_A_ray, cmap='gray')
    axes[2, 3].set_title(f'A coverage (ray)\n{(cov_A_ray > 0).mean():.4f}')

    for ax in axes.ravel():
        ax.axis('off')

    fig.suptitle(
        f'{scene_dir.name} | baseline={baseline:.4f} m | views={v0.name},{v1.name}\n'
        f'depth_mode={depth_mode} | pixel_centers={use_pixel_centers} | bilinear_splat={do_bilinear_splat}'
    )
    fig.tight_layout()
    out_path = out_dir / f'{scene_dir.name}_reprojection_check.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # extra debug figures
    save_absdiff_image(A1, reproj_A, out_dir / f'{scene_dir.name}_A_absdiff.png', title='A abs diff')
    save_masked_absdiff_image(A1, reproj_A, cov_A, out_dir / f'{scene_dir.name}_A_absdiff_masked.png', title='A masked abs diff')
    save_absdiff_image(A0, reproj_self_A, out_dir / f'{scene_dir.name}_A_self_absdiff.png', title='A self abs diff')
    save_absdiff_image(A1, reproj_A_ray, out_dir / f'{scene_dir.name}_A_ray_absdiff.png', title='A ray abs diff')

    return {
        'scene': scene_dir.name,
        'view0': v0.name,
        'view1': v1.name,
        'baseline_m': baseline,
        'depth_mode': depth_mode,
        'use_pixel_centers': use_pixel_centers,
        'do_bilinear_splat': do_bilinear_splat,

        'A_mae': None if np.isnan(mae_A) else float(mae_A),
        'Y_mae': None if np.isnan(mae_Y) else float(mae_Y),
        'A_coverage_ratio': float((cov_A > 0).mean()),
        'Y_coverage_ratio': float((cov_Y > 0).mean()),

        'self_A_mae': None if np.isnan(self_mae_A) else float(self_mae_A),
        'self_A_coverage_ratio': float((cov_self_A > 0).mean()),

        'A_mae_ray_assumption': None if np.isnan(mae_A_ray) else float(mae_A_ray),
        'A_coverage_ratio_ray_assumption': float((cov_A_ray > 0).mean()),

        'stats_A': stats_A,
        'stats_Y': stats_Y,
        'stats_self_A': stats_self_A,
        'stats_A_ray': stats_A_ray,

        'figure': str(out_path),
        'A_absdiff': str(out_dir / f'{scene_dir.name}_A_absdiff.png'),
        'A_absdiff_masked': str(out_dir / f'{scene_dir.name}_A_absdiff_masked.png'),
        'A_self_absdiff': str(out_dir / f'{scene_dir.name}_A_self_absdiff.png'),
        'A_ray_absdiff': str(out_dir / f'{scene_dir.name}_A_ray_absdiff.png'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to generated dataset root, e.g. /kubric/output')
    ap.add_argument('--out', default=None, help='Where to save inspection figures and summary')
    ap.add_argument('--max_scenes', type=int, default=5)
    ap.add_argument('--depth_mode', default='z', choices=['z', 'ray'], help='Interpretation of stored depth')
    ap.add_argument('--no_pixel_centers', action='store_true', help='Disable +0.5 pixel center convention')
    ap.add_argument('--nearest_only', action='store_true', help='Use nearest forward mapping instead of bilinear splatting')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / '_inspect'
    out_dir.mkdir(parents=True, exist_ok=True)

    use_pixel_centers = not args.no_pixel_centers
    do_bilinear_splat = not args.nearest_only

    scenes = find_scenes(root)[:args.max_scenes]
    rows = []

    print("#" * 100)
    print(f"root = {root}")
    print(f"out_dir = {out_dir}")
    print(f"depth_mode = {args.depth_mode}")
    print(f"use_pixel_centers = {use_pixel_centers}")
    print(f"do_bilinear_splat = {do_bilinear_splat}")
    print(f"num_scenes = {len(scenes)}")
    print("#" * 100)

    for s in scenes:
        try:
            r = process_scene(
                s,
                out_dir,
                depth_mode=args.depth_mode,
                use_pixel_centers=use_pixel_centers,
                do_bilinear_splat=do_bilinear_splat,
            )
            if r:
                rows.append(r)
        except Exception as e:
            print(f"[ERROR] scene={s.name}: {e}")
            rows.append({'scene': s.name, 'error': str(e)})

    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    print(f'Saved summary: {summary_path}')
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()