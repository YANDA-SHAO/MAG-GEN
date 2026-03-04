# kb_min.py
# Stable A/B/Y generation (single-scene, single-physics-run) + multi-view cameras.
#
# Core pipeline (kept):
# - Build ONE scene
# - Run ONE PyBullet simulation to get true states at A (frame_start) and B (frame_end)
# - Compute rigid delta per object, extrapolate to Y with factor alpha (no second physics run)
# - Render A, B, Y with EXACT SAME camera (same renderer, same scene)
#
# Extensions (carefully added, minimal disruption):
# - Multi-view camera: same scene/physics, render multiple views by orbiting ONE camera.
# - Pixel-displacement driven motion: jointly sample (alpha, speed, dt) so that
#   A->B is tiny (0.1~1 px) and A->Y is visible (2~10 px) for a "typical" view.
# - Optional per-view quality guards to reject unusable views (invisible, too small depth variation, etc.)
#
# Stability rules (kept):
# - DO NOT touch bpy objects until AFTER Blender(scene, ...) is created
# - Avoid numpy types in json
# - Bake A/B/Y transforms into Blender objects right after renderer is created, before any render.

import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender

import bpy  # only used after Blender renderer is created


# -----------------------
# Flags / Hyperparameters
# -----------------------
parser = kb.ArgumentParser()

# Dataset split control (deterministic asset filtering)
parser.add_argument("--objects_split", choices=["train", "test"], default="train",
                    help="Split for selecting object assets deterministically.")
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train",
                    help="Split for selecting HDRI backgrounds deterministically.")

# How many objects per scene
parser.add_argument("--min_num_objects", type=int, default=1,
                    help="Minimum number of objects in the scene.")
parser.add_argument("--max_num_objects", type=int, default=3,
                    help="Maximum number of objects in the scene.")

# Asset manifests
parser.add_argument("--kubasic_assets", type=str, default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str, default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str, default="gs://kubric-public/assets/GSO/GSO.json")

# Camera motion mode (kept for compatibility; we still keep camera fixed over time A/B/Y)
parser.add_argument("--camera", choices=["fixed_random", "linear_movement"], default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)  # kept for compatibility

# Multi-view cameras (speed-up: one scene -> many views)
parser.add_argument("--num_views", type=int, default=15,
                    help="Number of camera views to render per scene. One scene/physics, many viewpoints.")
parser.add_argument("--view_mode", choices=["uniform", "random"], default="uniform",
                    help="How to sample azimuth angles for views. uniform: evenly spaced; random: i.i.d.")
parser.add_argument("--view_seed_offset", type=int, default=100000,
                    help="Offset added to base seed to make view sampling deterministic but separate.")
parser.add_argument("--view_skip_bad", action="store_true",
                    help="If set, skip saving a view if it fails per-view quality checks (still keep scene).")

# Camera orbit ranges around the object-group center (controls framing / keeping objects in view)
parser.add_argument("--cam_r_min", type=float, default=2.0,
                    help="Min camera radius around group center (meters-ish). Smaller -> closer, higher risk of clipping/out-of-frame.")
parser.add_argument("--cam_r_max", type=float, default=4.0,
                    help="Max camera radius around group center.")
parser.add_argument("--cam_el_min_deg", type=float, default=15.0,
                    help="Min elevation angle in degrees. Too high elevation can cause 'objects come from top'.")
parser.add_argument("--cam_el_max_deg", type=float, default=55.0,
                    help="Max elevation angle in degrees.")
parser.add_argument("--cam_lookat_jitter", type=float, default=0.10,
                    help="Look-at point jitter (meters). Small jitter adds diversity without losing target.")

# Camera intrinsics (distance feeling) - sampled per view (optional)
parser.add_argument("--focal_mm_min", type=float, default=24.0,
                    help="Min focal length in mm. Smaller -> wider FOV (objects less likely out-of-frame).")
parser.add_argument("--focal_mm_max", type=float, default=50.0,
                    help="Max focal length in mm. Larger -> narrower FOV (objects more likely out-of-frame).")
parser.add_argument("--sensor_width_mm", type=float, default=32.0,
                    help="Sensor width in mm. Keep fixed for stable intrinsics across views.")

# Spawn region around origin (controls where objects appear)
parser.add_argument("--spawn_xy_min", type=float, default=0.4,
                    help="Min absolute XY extent for spawn region box.")
parser.add_argument("--spawn_xy_max", type=float, default=1.0,
                    help="Max absolute XY extent for spawn region box.")
parser.add_argument("--spawn_z_min", type=float, default=0.4,
                    help="Min spawn height.")
parser.add_argument("--spawn_z_max", type=float, default=1.2,
                    help="Max spawn height.")

# Object scale
parser.add_argument("--scale_min", type=float, default=0.4,
                    help="Min raw scale factor before normalizing by bounds.")
parser.add_argument("--scale_max", type=float, default=1.6,
                    help="Max raw scale factor before normalizing by bounds.")

# Motion targets in PIXELS (this is the key coupling of alpha+speed+camera)
parser.add_argument("--px_ab_min", type=float, default=0.015,
                    help="Target pixel displacement magnitude for A->B (tiny, almost invisible).")
parser.add_argument("--px_ab_max", type=float, default=0.5,
                    help="Max target pixel displacement for A->B.")
parser.add_argument("--px_ay_min", type=float, default=1.0,
                    help="Target pixel displacement magnitude for A->Y (visible).")
parser.add_argument("--px_ay_max", type=float, default=5.0,
                    help="Max target pixel displacement for A->Y.")
parser.add_argument("--alpha_min", type=float, default=1.0,
                    help="Min amplification factor alpha used for Y extrapolation.")
parser.add_argument("--alpha_max", type=float, default=5.0,
                    help="Max amplification factor alpha used for Y extrapolation.")

# Timing
parser.add_argument("--dt_frames_min", type=int, default=1,
                    help="Minimum dt in frames where B = A + dt.")
parser.add_argument("--dt_frames_max", type=int, default=3,
                    help="Maximum dt in frames where B = A + dt.")
parser.add_argument("--y_gap_frames", type=int, default=10,
                    help="Y_frame = B_frame + y_gap_frames (index only; physics is not rerun).")

# World-speed guard (prevents huge motion causing fly-out even if pixel targets fail)
parser.add_argument("--speed_world_max", type=float, default=2.0,
                    help="Clamp world speed (m/s-ish) to reduce out-of-frame risk.")

# Rendering
parser.add_argument("--samples_per_pixel", type=int, default=64,
                    help="Blender samples per pixel (quality vs speed).")

# Defaults for Kubric setup
parser.set_defaults(frame_start=5, frame_end=6, frame_rate=60, resolution=518)

# Debug / Diagnostics
parser.add_argument("--save_debug", action="store_true",
                    help="If set, run extra deep debug per scene (single-frame vs multi-frame checks, flow saves, etc.).")

# Optional quality guards (scene-level)
parser.add_argument("--reject_bad", action="store_true",
                    help="If set, reject the entire scene if it fails basic quality checks (useful for batch generation).")
parser.add_argument("--min_rgb_nz_ab", type=float, default=0.001,
                    help="Min nonzero ratio for A-B RGB difference. Too low => A and B almost identical.")
parser.add_argument("--min_depth_std", type=float, default=0.02,
                    help="Min std of depth map. Too low => depth nearly constant (bad for point cloud).")
parser.add_argument("--max_depth_max", type=float, default=1e6,
                    help="Sanity guard to avoid depth explosions / infs.")

FLAGS = parser.parse_args()


# -----------------------
# Helpers
# -----------------------
def to_serializable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def md5_bytes(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()


def _md5_arr_f32(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x.astype(np.float32))
    return hashlib.md5(x.tobytes()).hexdigest()


def _write_rgb_png_from_rgba(rgba: np.ndarray, path_png: Path) -> None:
    rgba = rgba.astype(np.float32)
    if rgba.max() > 1.5:
        rgba = rgba / 255.0
    rgb01 = np.clip(rgba[..., :3], 0.0, 1.0)
    kb.write_png(rgb01, str(path_png))


def _ensure_blender_image_loaded(filepath: str):
    for img in bpy.data.images:
        if img.filepath == filepath:
            return img
    return bpy.data.images.load(filepath)


def _pick_ids(asset_source: kb.AssetSource, split: str, rng: np.random.RandomState):
    # Use sorted asset keys for determinism
    ids = sorted(list(asset_source._assets.keys()))
    if split == "test":
        ids = [aid for aid in ids if (hash(aid) % 10) == 0]
    else:
        ids = [aid for aid in ids if (hash(aid) % 10) != 0]
    return ids, rng.choice(ids)


def _spawn_region_from_flags() -> List[Tuple[float, float, float]]:
    # box corners: [min_xyz, max_xyz]
    xy0 = float(FLAGS.spawn_xy_min)
    xy1 = float(FLAGS.spawn_xy_max)
    z0 = float(FLAGS.spawn_z_min)
    z1 = float(FLAGS.spawn_z_max)
    return [(-xy1, -xy1, z0), (xy1, xy1, z1)]


def _group_center(objs: List[kb.Object3D]) -> np.ndarray:
    ps = []
    for o in objs:
        p = np.array(o.position, dtype=np.float64)
        ps.append(p)
    if len(ps) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.mean(np.stack(ps, axis=0), axis=0)


def _sample_camera_orbit(rng: np.random.RandomState,
                         center: np.ndarray,
                         az: float,
                         r: float,
                         el_deg: float) -> Tuple[Tuple[float, float, float], np.ndarray]:
    el = np.deg2rad(float(el_deg))
    x = float(r * np.cos(az) * np.cos(el))
    y = float(r * np.sin(az) * np.cos(el))
    z = float(r * np.sin(el))
    pos = center + np.array([x, y, z], dtype=np.float64)

    # slight look-at jitter for diversity, but keep it small
    jitter = rng.uniform(-FLAGS.cam_lookat_jitter, FLAGS.cam_lookat_jitter, size=(3,))
    jitter[2] = rng.uniform(-0.5 * FLAGS.cam_lookat_jitter, 0.5 * FLAGS.cam_lookat_jitter)
    lookat = center + jitter
    return (float(pos[0]), float(pos[1]), float(pos[2])), lookat.astype(np.float64)


def _compute_fpx(res: int, focal_mm: float, sensor_width_mm: float) -> float:
    # Simple pinhole approx: f_px = (f_mm / sensor_mm) * image_width_px
    # Use width==height==resolution in your setup.
    return float((focal_mm / sensor_width_mm) * float(res))


# -----------------------
# Quaternion math (xyzw) - kept
# -----------------------
def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q) + 1e-12
    return q / n


def quat_mul(q1, q2):
    # xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def quat_inv(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64) / (np.dot(q, q) + 1e-12)


def quat_to_axis_angle(q):
    q = quat_normalize(q)
    x, y, z, w = q
    w = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = np.array([x, y, z], dtype=np.float64) / s
    return axis, angle


def axis_angle_to_quat(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = 0.5 * angle
    s = np.sin(half)
    x, y, z = axis * s
    w = np.cos(half)
    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))


def quat_pow(delta_q, alpha):
    delta_q = quat_normalize(delta_q)
    axis, angle = quat_to_axis_angle(delta_q)
    return axis_angle_to_quat(axis, angle * float(alpha))


# -----------------------
# Scene build (kept structure, just made controllable)
# -----------------------
def build_scene_once(seed: int):
    scene, _rng_unused, output_dir, scratch_dir = kb.setup(FLAGS)
    rng = np.random.RandomState(seed)

    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
    hdri = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # Background HDRI
    _, hdri_id = _pick_ids(hdri, FLAGS.backgrounds_split, rng)
    background_hdri = hdri.create(asset_id=hdri_id)

    # Dome
    dome = kubasic.create(
        asset_id="dome",
        name="dome",
        friction=0.3,
        restitution=0.5,
        static=True,
        background=True,
    )
    scene += dome

    # Camera (we will set per-view later; still create a camera object here)
    # Keep sensor width fixed, focal will be set per view.
    scene.camera = kb.PerspectiveCamera(focal_length=float(35.0), sensor_width=float(FLAGS.sensor_width_mm))

    # Objects
    gso_ids, _ = _pick_ids(gso, FLAGS.objects_split, rng)
    num_objects = int(rng.randint(int(FLAGS.min_num_objects), int(FLAGS.max_num_objects) + 1))
    obj_ids = [rng.choice(gso_ids) for _ in range(num_objects)]

    simulator_tmp = PyBullet(scene, scratch_dir)
    objs = []

    spawn_region = _spawn_region_from_flags()

    for aid in obj_ids:
        obj = gso.create(asset_id=aid)

        # Scale sampling
        scale_raw = float(rng.uniform(float(FLAGS.scale_min), float(FLAGS.scale_max)))
        obj.scale = scale_raw / np.max(obj.bounds[1] - obj.bounds[0])
        obj.metadata["scale_raw"] = scale_raw

        scene += obj

        # place with no-overlap
        kb.move_until_no_overlap(obj, simulator_tmp, spawn_region=spawn_region, rng=rng)

        # velocity will be set AFTER we decide (alpha, dt, pixel targets) in main()
        obj.velocity = (0.0, 0.0, 0.0)
        obj.angular_velocity = (0.0, 0.0, 0.0)

        objs.append(obj)

    return scene, objs, dome, background_hdri.filename, hdri_id, obj_ids, Path(output_dir), Path(scratch_dir)


def pose_from_keyframes(o, frame: int):
    kf = o.keyframes
    if "position" not in kf or "quaternion" not in kf:
        raise KeyError("Object has no position/quaternion keyframes")
    if frame not in kf["position"] or frame not in kf["quaternion"]:
        raise KeyError(
            f"Frame {frame} not found in keyframes. Available frames: {sorted(kf['position'].keys())}"
        )
    p = np.array(kf["position"][frame], dtype=np.float64)
    q = np.array(kf["quaternion"][frame], dtype=np.float64)  # xyzw
    return p, q


# -----------------------
# Pixel-target motion sampler (couples alpha + dt + world speed)
# -----------------------
def sample_alpha_dt_and_speed_world(rng: np.random.RandomState, fpx: float, z_ref: float, frame_rate: float) -> Dict[str, float]:
    """
    Goal:
      - A->B pixel disp ~ [px_ab_min, px_ab_max]
      - A->Y pixel disp ~ [px_ay_min, px_ay_max]
      - alpha in [alpha_min, alpha_max]
      - dt in [dt_frames_min, dt_frames_max]
    Using pinhole approx: px ≈ fpx * (world_disp / z_ref)
      => world_disp ≈ px * z_ref / fpx
      => speed_world ≈ world_disp / dt_seconds
    """
    dt_frames = int(rng.randint(int(FLAGS.dt_frames_min), int(FLAGS.dt_frames_max) + 1))
    dt_seconds = float(dt_frames) / float(frame_rate)

    # sample alpha first
    alpha = float(rng.uniform(float(FLAGS.alpha_min), float(FLAGS.alpha_max)))

    # AB px must be compatible with AY px range: AY = alpha * AB
    # so AB in [px_ay_min/alpha, px_ay_max/alpha] AND [px_ab_min, px_ab_max]
    ab_lo = max(float(FLAGS.px_ab_min), float(FLAGS.px_ay_min) / alpha)
    ab_hi = min(float(FLAGS.px_ab_max), float(FLAGS.px_ay_max) / alpha)

    # if incompatible, adjust alpha toward feasible region (lightweight fix)
    # fallback: clamp alpha so that feasible interval exists
    if ab_lo >= ab_hi:
        # try to choose AB in [px_ab_min, px_ab_max], then set alpha to hit AY range
        ab_px = float(rng.uniform(float(FLAGS.px_ab_min), float(FLAGS.px_ab_max)))
        # alpha should satisfy: alpha*ab_px in [px_ay_min, px_ay_max]
        a_lo = float(FLAGS.px_ay_min) / max(1e-6, ab_px)
        a_hi = float(FLAGS.px_ay_max) / max(1e-6, ab_px)
        alpha = float(np.clip(alpha, a_lo, a_hi))
        alpha = float(np.clip(alpha, float(FLAGS.alpha_min), float(FLAGS.alpha_max)))
        # recompute feasible AB range
        ab_lo = max(float(FLAGS.px_ab_min), float(FLAGS.px_ay_min) / alpha)
        ab_hi = min(float(FLAGS.px_ab_max), float(FLAGS.px_ay_max) / alpha)

    # final AB px
    if ab_lo < ab_hi:
        ab_px = float(rng.uniform(ab_lo, ab_hi))
    else:
        # still infeasible: pick safest tiny AB
        ab_px = float(FLAGS.px_ab_min)

    world_disp = float(ab_px) * float(z_ref) / max(1e-6, float(fpx))
    speed_world = float(world_disp) / max(1e-6, dt_seconds)
    speed_world = float(min(speed_world, float(FLAGS.speed_world_max)))

    return {
        "alpha": alpha,
        "dt_frames": float(dt_frames),
        "ab_px": ab_px,
        "z_ref": float(z_ref),
        "fpx": float(fpx),
        "speed_world": speed_world,
    }


# -----------------------
# Quality checks (safe, lightweight)
# -----------------------
def depth_stats(d: np.ndarray) -> Dict[str, float]:
    d = d.astype(np.float32)
    finite = np.isfinite(d)
    if not finite.any():
        return {"finite_ratio": 0.0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    v = d[finite]
    return {
        "finite_ratio": float(finite.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "std": float(v.std()),
    }


def rgb_diff_nonzero_ratio(a: np.ndarray, b: np.ndarray, thr: float = 1e-6) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return float((diff > thr).mean())


# -----------------------
# Main
# -----------------------
def main():
    seed = int(FLAGS.seed or 0)
    rng = np.random.RandomState(seed)

    # Build scene once
    scene, objs, dome, hdri_file, hdri_id, obj_ids, out_root, scratch_dir = build_scene_once(seed)

    # Decide reference center and a "typical" camera configuration for motion calibration
    center0 = _group_center(objs)

    # Choose a typical view for calibration: median radius, median elevation, median focal
    res = int(FLAGS.resolution)
    r_typ = float(0.5 * (float(FLAGS.cam_r_min) + float(FLAGS.cam_r_max)))
    el_typ = float(0.5 * (float(FLAGS.cam_el_min_deg) + float(FLAGS.cam_el_max_deg)))
    focal_typ = float(0.5 * (float(FLAGS.focal_mm_min) + float(FLAGS.focal_mm_max)))
    fpx_typ = _compute_fpx(res, focal_typ, float(FLAGS.sensor_width_mm))

    # Use z_ref approx as camera radius (good enough for scaling speed)
    z_ref = max(0.5, r_typ)

    # Sample alpha, dt, speed_world jointly so pixels are in desired ranges
    motion_pack = sample_alpha_dt_and_speed_world(
        rng=rng,
        fpx=fpx_typ,
        z_ref=z_ref,
        frame_rate=float(FLAGS.frame_rate),
    )
    alpha = float(motion_pack["alpha"])
    dt_frames = int(motion_pack["dt_frames"])
    speed_world = float(motion_pack["speed_world"])

    print("[MOTION] alpha =", alpha, " dt_frames =", dt_frames,
          " target_AB_px ~", motion_pack["ab_px"],
          " fpx_typ =", motion_pack["fpx"], " z_ref =", motion_pack["z_ref"],
          " speed_world =", speed_world)

    # Set A/B/Y frames
    A_frame = int(FLAGS.frame_start)
    B_frame = A_frame + int(dt_frames)
    Y_frame = B_frame + int(FLAGS.y_gap_frames)

    # Set per-object initial planar velocities (same magnitude, random direction)
    # Keep z velocity 0 (your previous behavior)
    for o in objs:
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        vx = float(speed_world * np.cos(theta))
        vy = float(speed_world * np.sin(theta))
        o.velocity = (vx, vy, 0.0)
        o.angular_velocity = (0.0, 0.0, 0.0)

    # Run PyBullet once (enough to cover B_frame)
    simulator = PyBullet(scene, str(scratch_dir))
    simulator.run(frame_start=0, frame_end=int(B_frame) + 1)

    # Read A/B poses from Kubric keyframes
    states_A = [pose_from_keyframes(o, A_frame) for o in objs]
    states_B = [pose_from_keyframes(o, B_frame) for o in objs]

    # Debug magnitude for obj0
    pA0, qA0 = states_A[0]
    pB0, qB0 = states_B[0]
    d_world_ab = float(np.linalg.norm(pB0 - pA0))
    print("||pB - pA|| (world) =", d_world_ab, "  alpha*||pB-pA|| =", alpha * d_world_ab)

    # Compute Y and insert keyframes (rigid extrapolation)
    for o, (pA, qA), (pB, qB) in zip(objs, states_A, states_B):
        pY = pA + alpha * (pB - pA)

        qA_n = quat_normalize(qA)
        qB_n = quat_normalize(qB)
        dq = quat_mul(qB_n, quat_inv(qA_n))
        dq_a = quat_pow(dq, alpha)
        qY = quat_mul(qA_n, dq_a)

        o.position = (float(pY[0]), float(pY[1]), float(pY[2]))
        o.quaternion = (float(qY[0]), float(qY[1]), float(qY[2]), float(qY[3]))
        o.keyframe_insert("position", Y_frame)
        o.keyframe_insert("quaternion", Y_frame)

    # Keyframe sanity (obj0)
    o0 = objs[0]
    pA_kf = np.array(o0.keyframes["position"][A_frame], dtype=np.float64)
    pB_kf = np.array(o0.keyframes["position"][B_frame], dtype=np.float64)
    pY_kf = np.array(o0.keyframes["position"][Y_frame], dtype=np.float64)
    print("KF ||pB-pA|| =", float(np.linalg.norm(pB_kf - pA_kf)))
    print("KF ||pY-pB|| =", float(np.linalg.norm(pY_kf - pB_kf)))

    # Create renderer (after keyframes exist) - kept
    renderer = Blender(scene, str(scratch_dir), samples_per_pixel=int(FLAGS.samples_per_pixel))

    # Bake A/B/Y into Blender objects right after renderer creation - kept (critical stability)
    def _bake_kubric_kf_to_blender(renderer, objs, frames):
        scn = bpy.context.scene
        for f in frames:
            scn.frame_set(int(f))
            bpy.context.view_layer.update()
            for o in objs:
                bo = o.linked_objects[renderer]
                p = np.array(o.keyframes["position"][int(f)], dtype=np.float64)
                q = np.array(o.keyframes["quaternion"][int(f)], dtype=np.float64)  # xyzw
                qwxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))

                bo.location = (float(p[0]), float(p[1]), float(p[2]))
                bo.rotation_mode = "QUATERNION"
                bo.rotation_quaternion = qwxyz

                bo.keyframe_insert(data_path="location", frame=int(f))
                bo.keyframe_insert(data_path="rotation_quaternion", frame=int(f))
        bpy.context.view_layer.update()

    _bake_kubric_kf_to_blender(renderer, objs, [A_frame, B_frame, Y_frame])

    # Set HDRI only after renderer created - kept
    if hasattr(renderer, "_set_ambient_light_hdri"):
        renderer._set_ambient_light_hdri(hdri_file)

    dome_blender = dome.linked_objects[renderer]
    mat = dome_blender.data.materials[0]
    nodes = mat.node_tree.nodes
    if "Image Texture" in nodes:
        nodes["Image Texture"].image = _ensure_blender_image_loaded(hdri_file)

    # Output dir per scene
    out_scene = out_root / f"{seed:06d}"
    out_scene.mkdir(parents=True, exist_ok=True)

    # Save scene-level meta (shared across views)
    meta_scene = {
        "seed": seed,
        "alpha": alpha,
        "frame_A": A_frame,
        "frame_B": B_frame,
        "frame_Y": Y_frame,
        "dt_frames": dt_frames,
        "speed_world": speed_world,
        "hdri_id": str(hdri_id),
        "obj_ids": [str(x) for x in obj_ids],
        "spawn_region": _spawn_region_from_flags(),
        "camera_orbit_ranges": {
            "cam_r_min": float(FLAGS.cam_r_min),
            "cam_r_max": float(FLAGS.cam_r_max),
            "cam_el_min_deg": float(FLAGS.cam_el_min_deg),
            "cam_el_max_deg": float(FLAGS.cam_el_max_deg),
        },
        "pixel_targets": {
            "px_ab_min": float(FLAGS.px_ab_min),
            "px_ab_max": float(FLAGS.px_ab_max),
            "px_ay_min": float(FLAGS.px_ay_min),
            "px_ay_max": float(FLAGS.px_ay_max),
        },
        "calibration_typical": {
            "r_typ": r_typ,
            "el_typ": el_typ,
            "focal_typ": focal_typ,
            "fpx_typ": fpx_typ,
            "z_ref": z_ref,
        },
        "center0": [float(x) for x in center0],
    }
    with open(out_scene / "meta_scene.json", "w", encoding="utf-8") as f:
        json.dump(to_serializable(meta_scene), f, indent=2)

    # -------------------------
    # Multi-view rendering loop
    # -------------------------
    rng_view = np.random.RandomState(seed + int(FLAGS.view_seed_offset))
    num_views = int(FLAGS.num_views)

    # azimuth list
    if FLAGS.view_mode == "uniform":
        az_list = [2.0 * np.pi * (i / max(1, num_views)) for i in range(num_views)]
        # tiny random phase shift so uniform isn't always aligned
        phase = float(rng_view.uniform(0.0, 2.0 * np.pi))
        az_list = [float(a + phase) for a in az_list]
    else:
        az_list = [float(rng_view.uniform(0.0, 2.0 * np.pi)) for _ in range(num_views)]

    def set_camera_for_view(view_id: int, az: float):
        # radius + elevation per view (small jitter, but keep in range)
        r = float(rng_view.uniform(float(FLAGS.cam_r_min), float(FLAGS.cam_r_max)))
        el = float(rng_view.uniform(float(FLAGS.cam_el_min_deg), float(FLAGS.cam_el_max_deg)))

        # update group center at A_frame (more correct than center0)
        # NOTE: this is safe; it reads kubric keyframes, not Blender.
        centerA = _group_center(objs)

        cam_pos, lookat = _sample_camera_orbit(rng_view, centerA, az, r, el)

        # intrinsics per view (small changes in "distance feeling")
        focal = float(rng_view.uniform(float(FLAGS.focal_mm_min), float(FLAGS.focal_mm_max)))
        scene.camera.focal_length = focal
        scene.camera.sensor_width = float(FLAGS.sensor_width_mm)

        scene.camera.position = cam_pos
        scene.camera.look_at(tuple(float(x) for x in lookat))

        # Keyframe camera at A/B/Y to avoid interpolation/caching surprises
        for f_ in [A_frame, B_frame, Y_frame]:
            scene.camera.keyframe_insert("position", f_)
            scene.camera.keyframe_insert("quaternion", f_)

        cam_info = kb.get_camera_info(scene.camera)
        cam_info["focal_length_mm"] = float(focal)
        cam_info["sensor_width_mm"] = float(FLAGS.sensor_width_mm)
        cam_info["view_id"] = int(view_id)
        cam_info["orbit"] = {"az": float(az), "r": float(r), "el_deg": float(el)}
        return cam_info

    # Optional deep debug (kept, runs only when --save_debug)
    def _debug_project_bbox(renderer, obj, frame_id: int, tag: str):
        from mathutils import Vector
        scn = bpy.context.scene
        cam = scn.camera
        scn.frame_set(int(frame_id))
        bpy.context.view_layer.update()

        bo = obj.linked_objects[renderer]

        def world_to_cam(pt_world):
            cam_inv = cam.matrix_world.inverted()
            return cam_inv @ pt_world

        origin_cam = world_to_cam(bo.matrix_world.translation)
        print(f"[{tag}] cam_space(origin)=", [float(origin_cam.x), float(origin_cam.y), float(origin_cam.z)])

        xs, ys, zs = [], [], []
        for v in bo.bound_box:
            vw = bo.matrix_world @ Vector(v)
            vc = world_to_cam(vw)
            xs.append(float(vc.x))
            ys.append(float(vc.y))
            zs.append(float(vc.z))

        print(
            f"[{tag}] cam_space(bbox) x[min,max]=", [min(xs), max(xs)],
            " y[min,max]=", [min(ys), max(ys)],
            " z[min,max]=", [min(zs), max(zs)],
        )

    # Render each view
    kept_views = 0
    for vid, az in enumerate(az_list):
        cam_info = set_camera_for_view(vid, az)

        # Ensure Blender evaluates the camera change
        bpy.context.scene.frame_set(int(A_frame))
        bpy.context.view_layer.update()

        # Render A/B/Y
        data = renderer.render(frames=[A_frame, B_frame, Y_frame], return_layers=("rgba", "depth"))

        A_rgba = data["rgba"][0]
        B_rgba = data["rgba"][1]
        Y_rgba = data["rgba"][2]
        depth_A = data["depth"][0].astype(np.float32)
        depth_B = data["depth"][1].astype(np.float32)
        depth_Y = data["depth"][2].astype(np.float32)

        # Basic diagnostics
        nz_ab = rgb_diff_nonzero_ratio(A_rgba, B_rgba)
        dsA = depth_stats(depth_A)
        dsB = depth_stats(depth_B)
        dsY = depth_stats(depth_Y)

        # Depth std check (point cloud usefulness)
        depth_ok = (dsA["finite_ratio"] > 0.999 and dsA["std"] >= float(FLAGS.min_depth_std) and dsA["max"] <= float(FLAGS.max_depth_max))

        # Scene-level reject? (apply per view too if view_skip_bad)
        view_ok = True
        if float(nz_ab) < float(FLAGS.min_rgb_nz_ab):
            view_ok = False
        if not depth_ok:
            view_ok = False

        if FLAGS.view_skip_bad and (not view_ok):
            print(f"[VIEW {vid:03d}] skipped (nz_ab={nz_ab:.6f}, depth_std={dsA['std']:.6f}, finite={dsA['finite_ratio']:.6f})")
            continue

        # Save
        out_view = out_scene / f"view_{vid:03d}"
        out_view.mkdir(parents=True, exist_ok=True)

        _write_rgb_png_from_rgba(A_rgba, out_view / "A.png")
        _write_rgb_png_from_rgba(B_rgba, out_view / "B.png")
        _write_rgb_png_from_rgba(Y_rgba, out_view / "Y.png")

        np.save(out_view / "depth_A.npy", depth_A)
        np.save(out_view / "depth_B.npy", depth_B)
        np.save(out_view / "depth_Y.npy", depth_Y)

        # quick diff vis for sanity
        # quick diff vis for sanity (AB / AY / BY)
        AB = np.abs(A_rgba.astype(np.float32) - B_rgba.astype(np.float32))[..., :3]
        AY = np.abs(A_rgba.astype(np.float32) - Y_rgba.astype(np.float32))[..., :3]
        BY = np.abs(B_rgba.astype(np.float32) - Y_rgba.astype(np.float32))[..., :3]

        def _save_diff(diff3: np.ndarray, path_png: Path) -> None:
            m = float(diff3.max()) + 1e-12
            kb.write_png(np.clip(diff3 / m, 0.0, 1.0), str(path_png))

        _save_diff(AB, out_view / "diff_AB.png")
        _save_diff(AY, out_view / "diff_AY.png")
        _save_diff(BY, out_view / "diff_BY.png")


        strip = np.concatenate([A_rgba[..., :3], B_rgba[..., :3], Y_rgba[..., :3]], axis=1).astype(np.float32)
        if strip.max() > 1.5:
            strip = strip / 255.0
        kb.write_png(np.clip(strip, 0.0, 1.0), str(out_view / "strip_ABY.png"))

        # camera info per view (for point cloud reconstruction)
        with open(out_view / "camera.json", "w", encoding="utf-8") as f:
            json.dump(to_serializable(cam_info), f, indent=2)

        # meta per view
        meta_view = {
            "seed": seed,
            "view_id": int(vid),
            "nz_ab": float(nz_ab),
            "depth_A_stats": dsA,
            "depth_B_stats": dsB,
            "depth_Y_stats": dsY,
            "md5": {
                "A": md5_file(out_view / "A.png"),
                "B": md5_file(out_view / "B.png"),
                "Y": md5_file(out_view / "Y.png"),
            },
        }
        with open(out_view / "meta_view.json", "w", encoding="utf-8") as f:
            json.dump(to_serializable(meta_view), f, indent=2)

        kept_views += 1
        print(f"[VIEW {vid:03d}] saved (nz_ab={nz_ab:.6f}, depth_std={dsA['std']:.6f})")

        # deep debug only if requested
        if FLAGS.save_debug:
            _debug_project_bbox(renderer, objs[0], A_frame, f"V{vid:03d}_A")
            _debug_project_bbox(renderer, objs[0], B_frame, f"V{vid:03d}_B")
            _debug_project_bbox(renderer, objs[0], Y_frame, f"V{vid:03d}_Y")

    print("[DONE] scene =", f"{seed:06d}", " kept_views =", kept_views, " out_dir =", str(out_scene))


if __name__ == "__main__":
    main()