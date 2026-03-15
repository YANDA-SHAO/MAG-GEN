# generator_vibration.py
# Minimal-risk vibration version for MAG-GEN:
# - no PyBullet
# - suspended rigid object
# - tiny translation / rotation / mixed vibration
# - A / B / Y triplets
# - multi-view rendering
# - simple Kubric primitives only
#
# Notes:
# - Background hookup follows MAG-GEN kb_min.py:
#   renderer._set_ambient_light_hdri(hdri_file)
#   + dome.linked_objects[renderer]
#   + nodes["Image Texture"].image = _ensure_blender_image_loaded(hdri_file)
# - Only small bug fixes are applied. Unrelated logic is kept as-is.

"""
Generate synthetic tiny vibration data for motion magnification training.

Unlike the physics generator, this script uses analytic motion models
instead of a physics engine. It directly simulates small structural
motions including:

    translation
    rotation
    mixed translation + rotation

The script produces triplets:
    A : reference frame
    B : small vibration frame
    Y : magnified vibration frame

This generator is designed for structural vibration datasets where
motions are extremely small.
"""

import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import kubric as kb
from kubric.renderer import Blender


# =========================================================
# Args
# =========================================================
parser = kb.ArgumentParser()

# Rendering
parser.add_argument("--samples_per_pixel", type=int, default=64)

parser.add_argument("--num_static_min", type=int, default=1)
parser.add_argument("--num_static_max", type=int, default=3)

# Views
parser.add_argument("--num_views", type=int, default=20)
parser.add_argument("--view_mode", choices=["uniform", "random"], default="uniform")
parser.add_argument("--cam_r_min", type=float, default=1.4)
parser.add_argument("--cam_r_max", type=float, default=3.0)
parser.add_argument("--cam_el_min_deg", type=float, default=18.0)
parser.add_argument("--cam_el_max_deg", type=float, default=45.0)
parser.add_argument("--focal_mm_min", type=float, default=28.0)
parser.add_argument("--focal_mm_max", type=float, default=45.0)
parser.add_argument("--sensor_width_mm", type=float, default=32.0)
parser.add_argument("--view_skip_bad", action="store_true")

# Scene
parser.add_argument("--num_static_objects", type=int, default=0)

# Motion mode
parser.add_argument("--motion_mode", choices=["translation", "rotation", "mixed"], default="mixed")

# Base pose
parser.add_argument("--center_x", type=float, default=0.0)
parser.add_argument("--center_y", type=float, default=0.0)
parser.add_argument("--center_z", type=float, default=0.9)

# Translation vibration
parser.add_argument("--trans_amp_min", type=float, default=0.005)
parser.add_argument("--trans_amp_max", type=float, default=0.030)
parser.add_argument("--trans_axis_mode", choices=["x", "y", "z", "random"], default="random")

# Rotation vibration
parser.add_argument("--rot_amp_deg_min", type=float, default=0.2)
parser.add_argument("--rot_amp_deg_max", type=float, default=2.0)
parser.add_argument("--rot_axis_mode", choices=["x", "y", "z", "random"], default="random")

# Temporal
parser.add_argument("--num_frames", type=int, default=16)
parser.add_argument("--freq_hz_min", type=float, default=0.8)
parser.add_argument("--freq_hz_max", type=float, default=3.0)

# A/B sampling
parser.add_argument("--frame_A", type=int, default=4)
parser.add_argument("--dt_frames_min", type=int, default=1)
parser.add_argument("--dt_frames_max", type=int, default=3)

# Magnification
parser.add_argument("--alpha_min", type=float, default=5.0)
parser.add_argument("--alpha_max", type=float, default=30.0)

# Safety limits
parser.add_argument("--max_trans_world", type=float, default=0.05)
parser.add_argument("--max_rot_deg_world", type=float, default=5.0)
parser.add_argument("--world_x_min", type=float, default=-0.25)
parser.add_argument("--world_x_max", type=float, default=0.25)
parser.add_argument("--world_y_min", type=float, default=-0.25)
parser.add_argument("--world_y_max", type=float, default=0.25)
parser.add_argument("--world_z_min", type=float, default=0.60)
parser.add_argument("--world_z_max", type=float, default=1.20)

# =========================================================
# Assets / background / object source
# =========================================================
parser.add_argument("--objects_split", choices=["train", "test"], default="train")
parser.add_argument("--backgrounds_split", choices=["train", "test"], default="train")

parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")

parser.add_argument("--use_floor", action="store_true")
parser.add_argument("--use_dome", action="store_true")
parser.add_argument("--use_hdri", action="store_true")

parser.add_argument("--object_source", choices=["primitive", "gso"], default="primitive")
parser.add_argument("--static_source", choices=["primitive", "gso"], default="primitive")

# Optional extra primitive categories
parser.add_argument("--object_kind", choices=["sphere", "cube", "torus", "random"], default="random")
parser.add_argument("--static_kind", choices=["sphere", "cube", "torus", "random"], default="random")

# Visibility
parser.add_argument("--min_foreground_ratio", type=float, default=0.001)

FLAGS = parser.parse_args()

logging.basicConfig(level=logging.INFO)


# =========================================================
# Small math utils
# =========================================================
def _to_float_list(x):
    return [float(v) for v in x]


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        if v.shape[0] == 3:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.zeros_like(v, dtype=np.float32)
    return v / n


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(3,))
    return normalize(v.astype(np.float32))


def choose_axis(mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if mode == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if mode == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return random_unit_vector(rng)


def quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Quaternion order: [w, x, y, z]
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float32)


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float32)


def axis_angle_to_quat(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = normalize(axis)
    half = 0.5 * angle_rad
    s = math.sin(half)
    return quat_normalize(np.array([math.cos(half), axis[0]*s, axis[1]*s, axis[2]*s], dtype=np.float32))


def quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    q = quat_normalize(q)
    w = float(np.clip(q[0], -1.0, 1.0))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w*w, 0.0))
    if s < 1e-8:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = np.array([q[1]/s, q[2]/s, q[3]/s], dtype=np.float32)
        axis = normalize(axis)
    return axis, angle


def quat_power(q_delta: np.ndarray, alpha: float) -> np.ndarray:
    axis, angle = quat_to_axis_angle(q_delta)
    return axis_angle_to_quat(axis, alpha * angle)


def angle_deg_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
    qd = quat_mul(q2, quat_inv(q1))
    _, angle = quat_to_axis_angle(qd)
    angle_deg = math.degrees(angle)
    if angle_deg > 180.0:
        angle_deg = 360.0 - angle_deg
    return float(angle_deg)


def safe_scene_name_from_seed(seed: int) -> str:
    return f"{seed:06d}"


def rgba_to_rgb_uint8(rgba: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgba)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        return arr[..., :3]
    return arr


def write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def make_strip(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.concatenate([a, b, y], axis=1)


def abs_diff_vis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    return d


def _pick_ids(asset_source, split: str, rng: np.random.Generator):
    ids = sorted(list(asset_source._assets.keys()))
    if split == "test":
        ids = [aid for aid in ids if (hash(aid) % 10) == 0]
    else:
        ids = [aid for aid in ids if (hash(aid) % 10) != 0]
    if len(ids) == 0:
        raise RuntimeError(f"No asset ids found for split={split}")
    return ids, ids[int(rng.integers(0, len(ids)))]


def random_xy_away_from_center(rng: np.random.Generator,
                               min_r: float = 0.35,
                               max_r: float = 0.95):
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    rad = float(rng.uniform(min_r, max_r))
    x = rad * math.cos(ang)
    y = rad * math.sin(ang)
    return x, y


# =========================================================
# Scene helpers
# =========================================================
def parse_resolution(res):
    if isinstance(res, (tuple, list)):
        return (int(res[0]), int(res[1]))
    if isinstance(res, int):
        return (res, res)
    if isinstance(res, str):
        s = res.lower().replace(" ", "")
        if "x" in s:
            w, h = s.split("x")
            return (int(w), int(h))
        v = int(s)
        return (v, v)
    raise ValueError(f"Unsupported resolution format: {res}")


def random_color(rng: np.random.Generator) -> Tuple[float, float, float]:
    return tuple(float(v) for v in rng.uniform(0.1, 0.95, size=(3,)))


def random_object_scale(rng: np.random.Generator, base: float = 0.28) -> Tuple[float, float, float]:
    sx = float(rng.uniform(0.8, 1.2)) * base
    sy = float(rng.uniform(0.8, 1.2)) * base
    sz = float(rng.uniform(0.8, 1.2)) * base
    return (sx, sy, sz)


def build_primitive(kind: str,
                    name: str,
                    position: Tuple[float, float, float],
                    quaternion: np.ndarray,
                    scale: Tuple[float, float, float],
                    color: Tuple[float, float, float],
                    rng: np.random.Generator):
    material = kb.PrincipledBSDFMaterial(color=color)

    if kind == "random":
        kind = str(rng.choice(["sphere", "cube", "torus"]))

    try:
        if kind == "sphere":
            obj = kb.Sphere(name=name, scale=max(scale), position=position, material=material)
        elif kind == "cube":
            obj = kb.Cube(name=name, scale=scale, position=position, material=material)
        elif kind == "torus":
            obj = kb.Torus(name=name, scale=scale, position=position, material=material)
        else:
            raise ValueError(f"Unknown primitive kind: {kind}")
    except Exception as e:
        logging.warning("Primitive '%s' failed, fallback to cube. Error: %s", kind, e)
        obj = kb.Cube(name=name, scale=scale, position=position, material=material)

    obj.quaternion = tuple(float(v) for v in quaternion)
    return obj


def build_gso_object(gso_source,
                     asset_id: str,
                     name: str,
                     position: Tuple[float, float, float],
                     quaternion: np.ndarray,
                     scale_raw: float):
    obj = gso_source.create(asset_id=asset_id)

    try:
        bounds = np.array(obj.bounds, dtype=np.float32)
        extent = np.max(bounds[1] - bounds[0])
        if extent > 1e-8:
            obj.scale = float(scale_raw) / float(extent)
        else:
            obj.scale = float(scale_raw)
    except Exception:
        obj.scale = float(scale_raw)

    obj.position = tuple(float(v) for v in position)
    obj.quaternion = tuple(float(v) for v in quaternion)
    return obj


def build_target_object(scene,
                        rng: np.random.Generator,
                        gso_source,
                        gso_ids: List[str],
                        target_quat0: np.ndarray,
                        target_scale: Tuple[float, float, float],
                        target_color: Tuple[float, float, float]):
    if FLAGS.object_source == "gso" and gso_source is not None and len(gso_ids) > 0:
        try:
            aid = gso_ids[int(rng.integers(0, len(gso_ids)))]
            obj = build_gso_object(
                gso_source=gso_source,
                asset_id=aid,
                name="target",
                position=(FLAGS.center_x, FLAGS.center_y, FLAGS.center_z),
                quaternion=target_quat0,
                scale_raw=float(rng.uniform(0.6, 1.4)),
            )
            scene += obj
            return obj
        except Exception as e:
            logging.warning("GSO target failed, fallback to primitive. Error: %s", e)

    obj = build_primitive(
        kind=FLAGS.object_kind,
        name="target",
        position=(FLAGS.center_x, FLAGS.center_y, FLAGS.center_z),
        quaternion=target_quat0,
        scale=target_scale,
        color=target_color,
        rng=rng,
    )
    scene += obj
    return obj


def place_static_objects(scene,
                         rng: np.random.Generator,
                         n: int,
                         gso_source=None,
                         gso_ids=None):
    gso_ids = gso_ids or []

    for i in range(n):
        px, py = random_xy_away_from_center(rng)
        pz = float(rng.uniform(0.05, 0.55))
        q = axis_angle_to_quat(random_unit_vector(rng), float(rng.uniform(0, math.pi)))
        sc = random_object_scale(rng, base=0.18)
        color = random_color(rng)

        obj = None

        if FLAGS.static_source == "gso" and gso_source is not None and len(gso_ids) > 0:
            try:
                aid = gso_ids[int(rng.integers(0, len(gso_ids)))]
                obj = build_gso_object(
                    gso_source=gso_source,
                    asset_id=aid,
                    name=f"static_{i:03d}",
                    position=(px, py, pz),
                    quaternion=q,
                    scale_raw=float(rng.uniform(0.4, 1.0)),
                )
            except Exception as e:
                logging.warning("GSO static failed, fallback to primitive. Error: %s", e)

        if obj is None:
            obj = build_primitive(
                kind=FLAGS.static_kind,
                name=f"static_{i:03d}",
                position=(px, py, pz),
                quaternion=q,
                scale=sc,
                color=color,
                rng=rng,
            )

        scene += obj


def _ensure_blender_image_loaded(filepath: str):
    import bpy

    filepath = str(filepath)
    abspath = str(Path(filepath).resolve())

    for img in bpy.data.images:
        try:
            if bpy.path.abspath(img.filepath) == abspath:
                return img
        except Exception:
            pass

    return bpy.data.images.load(abspath, check_existing=True)


def attach_hdri_to_renderer_and_dome(renderer, dome, hdri_file: Optional[str]) -> None:
    if not hdri_file:
        return

    try:
        if hasattr(renderer, "_set_ambient_light_hdri"):
            renderer._set_ambient_light_hdri(hdri_file)

        if dome is not None:
            dome_blender = dome.linked_objects[renderer]
            if dome_blender is None:
                return

            if not getattr(dome_blender.data, "materials", None):
                return
            if len(dome_blender.data.materials) == 0:
                return

            mat = dome_blender.data.materials[0]
            if mat is None or mat.node_tree is None:
                return

            nodes = mat.node_tree.nodes
            if "Image Texture" in nodes:
                nodes["Image Texture"].image = _ensure_blender_image_loaded(hdri_file)
    except Exception as e:
        logging.warning("Failed to attach HDRI to renderer/dome: %s", e)


def create_scene(seed: int):
    rng = np.random.default_rng(seed)

    scene = kb.Scene(
        resolution=parse_resolution(FLAGS.resolution),
        frame_start=1,
        frame_end=1,
        frame_rate=FLAGS.frame_rate,
    )

    # -----------------------------
    # Asset sources (best-effort)
    # -----------------------------
    kubasic = None
    hdri = None
    gso = None
    hdri_path = None
    hdri_id = None
    gso_ids = []
    dome = None

    try:
        kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    except Exception as e:
        logging.warning("Failed to load KuBasic manifest: %s", e)

    try:
        gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
        gso_ids, _ = _pick_ids(gso, FLAGS.objects_split, rng)
    except Exception as e:
        logging.warning("Failed to load GSO manifest: %s", e)
        gso = None
        gso_ids = []

    try:
        hdri = kb.AssetSource.from_manifest(FLAGS.hdri_assets)
        _, hdri_id = _pick_ids(hdri, FLAGS.backgrounds_split, rng)
        background_hdri = hdri.create(asset_id=hdri_id)
        hdri_path = getattr(background_hdri, "filename", None)
    except Exception as e:
        logging.warning("Failed to load HDRI manifest: %s", e)
        hdri = None
        hdri_path = None
        hdri_id = None

    # -----------------------------
    # Background dome
    # -----------------------------
    if FLAGS.use_dome and kubasic is not None:
        try:
            dome = kubasic.create(
                asset_id="dome",
                name="dome",
                static=True,
                background=True,
            )
            scene += dome
        except Exception as e:
            logging.warning("Failed to create dome background: %s", e)
            dome = None

    # -----------------------------
    # Floor
    # -----------------------------
    if FLAGS.use_floor:
        try:
            floor = kb.Cube(
                name="floor",
                scale=(3.0, 3.0, 0.05),
                position=(0.0, 0.0, 0.0),
                material=kb.PrincipledBSDFMaterial(color=(0.65, 0.65, 0.68)),
                static=True,
            )
            scene += floor
        except Exception as e:
            logging.warning("Failed to create floor: %s", e)

    # -----------------------------
    # Lighting
    # -----------------------------
    scene += kb.DirectionalLight(
        name="sun",
        position=(2.0, -2.0, 4.0),
        look_at=(0.0, 0.0, float(FLAGS.center_z)),
        intensity=2.0,
    )

    scene += kb.DirectionalLight(
        name="fill",
        position=(-2.0, 2.5, 3.0),
        look_at=(0.0, 0.0, float(FLAGS.center_z)),
        intensity=1.0,
    )

    # -----------------------------
    # Target object
    # -----------------------------
    target_quat0 = axis_angle_to_quat(random_unit_vector(rng), float(rng.uniform(0.0, math.pi)))
    target_scale = random_object_scale(rng, base=0.22)
    target_color = random_color(rng)

    target = build_target_object(
        scene=scene,
        rng=rng,
        gso_source=gso,
        gso_ids=gso_ids,
        target_quat0=target_quat0,
        target_scale=target_scale,
        target_color=target_color,
    )

    # -----------------------------
    # Static distractors
    # -----------------------------
    n_static = int(rng.integers(FLAGS.num_static_min, FLAGS.num_static_max + 1))

    if n_static > 0:
        place_static_objects(
            scene=scene,
            rng=rng,
            n=n_static,
            gso_source=gso,
            gso_ids=gso_ids,
        )

    renderer = Blender(scene, samples_per_pixel=FLAGS.samples_per_pixel)

    # -----------------------------
    # HDRI hookup: follow MAG-GEN kb_min.py
    # -----------------------------
    if FLAGS.use_hdri and hdri_path is not None:
        attach_hdri_to_renderer_and_dome(renderer, dome, hdri_path)

    return scene, renderer, target, dome, rng, hdri_id, hdri_path


# =========================================================
# Motion generation
# =========================================================
def generate_motion_params(rng: np.random.Generator) -> Dict:
    motion_mode = FLAGS.motion_mode
    alpha = float(rng.uniform(FLAGS.alpha_min, FLAGS.alpha_max))
    freq_hz = float(rng.uniform(FLAGS.freq_hz_min, FLAGS.freq_hz_max))
    phase_trans = float(rng.uniform(0.0, 2.0 * math.pi))
    phase_rot = float(rng.uniform(0.0, 2.0 * math.pi))

    trans_axis = choose_axis(FLAGS.trans_axis_mode, rng)
    rot_axis = choose_axis(FLAGS.rot_axis_mode, rng)

    trans_amp = float(rng.uniform(FLAGS.trans_amp_min, FLAGS.trans_amp_max))
    rot_amp_deg = float(rng.uniform(FLAGS.rot_amp_deg_min, FLAGS.rot_amp_deg_max))
    rot_amp_rad = math.radians(rot_amp_deg)

    return {
        "motion_mode": motion_mode,
        "alpha": alpha,
        "freq_hz": freq_hz,
        "phase_trans": phase_trans,
        "phase_rot": phase_rot,
        "trans_axis": trans_axis,
        "rot_axis": rot_axis,
        "trans_amp": trans_amp,
        "rot_amp_deg": rot_amp_deg,
        "rot_amp_rad": rot_amp_rad,
    }


def pose_at_frame(base_pos: np.ndarray,
                  base_quat: np.ndarray,
                  t_idx: int,
                  params: Dict) -> Tuple[np.ndarray, np.ndarray]:
    t = float(t_idx) / float(FLAGS.frame_rate)
    w = 2.0 * math.pi * params["freq_hz"]

    pos = base_pos.copy()
    quat = base_quat.copy()

    if params["motion_mode"] in ["translation", "mixed"]:
        offset = params["trans_amp"] * math.sin(w * t + params["phase_trans"]) * params["trans_axis"]
        pos = pos + offset

    if params["motion_mode"] in ["rotation", "mixed"]:
        angle = params["rot_amp_rad"] * math.sin(w * t + params["phase_rot"])
        q_delta = axis_angle_to_quat(params["rot_axis"], angle)
        quat = quat_mul(q_delta, base_quat)

    return pos.astype(np.float32), quat_normalize(quat)


def compute_amplified_pose(pA: np.ndarray, qA: np.ndarray,
                           pB: np.ndarray, qB: np.ndarray,
                           alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    pY = pA + alpha * (pB - pA)

    q_delta = quat_mul(qB, quat_inv(qA))
    qY = quat_mul(quat_power(q_delta, alpha), qA)
    qY = quat_normalize(qY)
    return pY.astype(np.float32), qY


def within_world_bounds(p: np.ndarray) -> bool:
    return (
        FLAGS.world_x_min <= p[0] <= FLAGS.world_x_max and
        FLAGS.world_y_min <= p[1] <= FLAGS.world_y_max and
        FLAGS.world_z_min <= p[2] <= FLAGS.world_z_max
    )


def sample_A_B_indices(rng: np.random.Generator) -> Tuple[int, int]:
    tA = int(np.clip(FLAGS.frame_A, 0, FLAGS.num_frames - 2))
    max_dt = min(FLAGS.dt_frames_max, FLAGS.num_frames - 1 - tA)
    min_dt = min(FLAGS.dt_frames_min, max_dt)
    if max_dt < 1:
        raise ValueError("Not enough frames for A/B sampling.")
    dt = int(rng.integers(min_dt, max_dt + 1))
    tB = tA + dt
    return tA, tB


# =========================================================
# Camera / render
# =========================================================
def sample_view_params(view_idx: int, num_views: int, rng: np.random.Generator):
    if FLAGS.view_mode == "uniform":
        az_deg = 360.0 * float(view_idx) / float(max(num_views, 1))
    else:
        az_deg = float(rng.uniform(0.0, 360.0))

    el_deg = float(rng.uniform(FLAGS.cam_el_min_deg, FLAGS.cam_el_max_deg))
    radius = float(rng.uniform(FLAGS.cam_r_min, FLAGS.cam_r_max))
    focal_mm = float(rng.uniform(FLAGS.focal_mm_min, FLAGS.focal_mm_max))
    return az_deg, el_deg, radius, focal_mm


def camera_position_from_spherical(center: np.ndarray, az_deg: float, el_deg: float, radius: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    x = center[0] + radius * math.cos(el) * math.cos(az)
    y = center[1] + radius * math.cos(el) * math.sin(az)
    z = center[2] + radius * math.sin(el)
    return np.array([x, y, z], dtype=np.float32)


def set_camera(scene, cam_pos: np.ndarray, look_at: np.ndarray, focal_mm: float):
    cam = kb.PerspectiveCamera(
        name="camera",
        position=tuple(float(v) for v in cam_pos),
        look_at=tuple(float(v) for v in look_at),
        focal_length=float(focal_mm),
        sensor_width=float(FLAGS.sensor_width_mm),
    )
    scene.camera = cam


def set_target_pose(target, pos: np.ndarray, quat: np.ndarray):
    target.position = tuple(float(v) for v in pos)
    target.quaternion = tuple(float(v) for v in quat)


def render_still_rgb_depth_seg(renderer) -> Dict[str, np.ndarray]:
    frame = renderer.render_still()
    return {
        "rgba": frame["rgba"],
        "depth": frame["depth"],
        "segmentation": frame["segmentation"],
    }


def extract_segmentation_ids(seg: np.ndarray) -> np.ndarray:
    seg = np.asarray(seg)
    if seg.ndim == 3 and seg.shape[-1] == 1:
        seg = seg[..., 0]
    return seg


def foreground_ratio_for_id(seg: np.ndarray, object_id: int) -> float:
    seg2d = extract_segmentation_ids(seg)
    mask = (seg2d == int(object_id))
    return float(np.count_nonzero(mask)) / float(seg2d.shape[0] * seg2d.shape[1])


# =========================================================
# Main
# =========================================================
def main():
    scene_name = safe_scene_name_from_seed(FLAGS.seed)
    out_root = Path(FLAGS.job_dir) / scene_name
    out_root.mkdir(parents=True, exist_ok=True)

    scene, renderer, target, dome, rng, hdri_id, hdri_path = create_scene(FLAGS.seed)

    base_pos = np.array([FLAGS.center_x, FLAGS.center_y, FLAGS.center_z], dtype=np.float32)
    base_quat = np.array(target.quaternion, dtype=np.float32)

    params = generate_motion_params(rng)
    tA, tB = sample_A_B_indices(rng)

    pA, qA = pose_at_frame(base_pos, base_quat, tA, params)
    pB, qB = pose_at_frame(base_pos, base_quat, tB, params)
    pY, qY = compute_amplified_pose(pA, qA, pB, qB, params["alpha"])

    trans_AB = float(np.linalg.norm(pB - pA))
    rot_AB_deg = angle_deg_between_quats(qA, qB)

    if trans_AB > FLAGS.max_trans_world:
        raise RuntimeError(f"AB translation too large: {trans_AB:.6f} > {FLAGS.max_trans_world}")
    if rot_AB_deg > FLAGS.max_rot_deg_world:
        raise RuntimeError(f"AB rotation too large: {rot_AB_deg:.4f} > {FLAGS.max_rot_deg_world}")
    if not within_world_bounds(pA):
        raise RuntimeError("A pose out of world bounds.")
    if not within_world_bounds(pB):
        raise RuntimeError("B pose out of world bounds.")
    if not within_world_bounds(pY):
        raise RuntimeError("Y pose out of world bounds.")

    center_for_views = base_pos.copy()
    target_segmentation_id = getattr(target, "segmentation_id", None)

    meta_scene = {
        "scene_id": scene_name,
        "seed": int(FLAGS.seed),
        "generator": "vibgen.py",
        "motion_mode": params["motion_mode"],
        "alpha": float(params["alpha"]),
        "freq_hz": float(params["freq_hz"]),
        "frame_rate": int(FLAGS.frame_rate),
        "num_frames": int(FLAGS.num_frames),
        "frame_A": int(tA),
        "frame_B": int(tB),
        "trans_axis": _to_float_list(params["trans_axis"]),
        "rot_axis": _to_float_list(params["rot_axis"]),
        "trans_amp_m": float(params["trans_amp"]),
        "rot_amp_deg": float(params["rot_amp_deg"]),
        "base_pos": _to_float_list(base_pos),
        "base_quat_wxyz": _to_float_list(base_quat),
        "pose_A": {"position": _to_float_list(pA), "quat_wxyz": _to_float_list(qA)},
        "pose_B": {"position": _to_float_list(pB), "quat_wxyz": _to_float_list(qB)},
        "pose_Y": {"position": _to_float_list(pY), "quat_wxyz": _to_float_list(qY)},
        "AB_translation_m": float(trans_AB),
        "AB_rotation_deg": float(rot_AB_deg),
        "target_segmentation_id": None if target_segmentation_id is None else int(target_segmentation_id),
        "hdri_path": hdri_path,
        "hdri_id": None if hdri_id is None else str(hdri_id),
        "notes": "No PyBullet. Analytic tiny vibration. A/B/Y rendered with same scene, same view per triplet.",
    }
    save_json(out_root / "meta_scene.json", meta_scene)

    for view_idx in range(FLAGS.num_views):
        view_rng = np.random.default_rng(FLAGS.seed + 100000 + view_idx)
        az_deg, el_deg, radius, focal_mm = sample_view_params(view_idx, FLAGS.num_views, view_rng)
        cam_pos = camera_position_from_spherical(center_for_views, az_deg, el_deg, radius)

        set_camera(scene, cam_pos, center_for_views, focal_mm)

        view_dir = out_root / f"view_{view_idx:03d}"
        view_dir.mkdir(parents=True, exist_ok=True)

        # A
        set_target_pose(target, pA, qA)
        frame_A = render_still_rgb_depth_seg(renderer)

        # B
        set_target_pose(target, pB, qB)
        frame_B = render_still_rgb_depth_seg(renderer)

        # Y
        set_target_pose(target, pY, qY)
        frame_Y = render_still_rgb_depth_seg(renderer)

        if target_segmentation_id is not None:
            fgA = foreground_ratio_for_id(frame_A["segmentation"], target_segmentation_id)
            fgB = foreground_ratio_for_id(frame_B["segmentation"], target_segmentation_id)
            fgY = foreground_ratio_for_id(frame_Y["segmentation"], target_segmentation_id)
        else:
            segA = extract_segmentation_ids(frame_A["segmentation"])
            segB = extract_segmentation_ids(frame_B["segmentation"])
            segY = extract_segmentation_ids(frame_Y["segmentation"])
            fgA = float(np.count_nonzero(segA)) / float(segA.shape[0] * segA.shape[1])
            fgB = float(np.count_nonzero(segB)) / float(segB.shape[0] * segB.shape[1])
            fgY = float(np.count_nonzero(segY)) / float(segY.shape[0] * segY.shape[1])

        if FLAGS.view_skip_bad and min(fgA, fgB, fgY) < FLAGS.min_foreground_ratio:
            logging.info("Skipping bad view %03d (low fg ratio).", view_idx)
            continue

        rgbA = rgba_to_rgb_uint8(frame_A["rgba"])
        rgbB = rgba_to_rgb_uint8(frame_B["rgba"])
        rgbY = rgba_to_rgb_uint8(frame_Y["rgba"])

        kb.write_png(rgbA, str(view_dir / "A.png"))
        kb.write_png(rgbB, str(view_dir / "B.png"))
        kb.write_png(rgbY, str(view_dir / "Y.png"))

        kb.write_palette_png(frame_A["segmentation"], str(view_dir / "seg_A.png"))
        kb.write_palette_png(frame_B["segmentation"], str(view_dir / "seg_B.png"))
        kb.write_palette_png(frame_Y["segmentation"], str(view_dir / "seg_Y.png"))

        write_npy(view_dir / "depth_A.npy", frame_A["depth"])
        write_npy(view_dir / "depth_B.npy", frame_B["depth"])
        write_npy(view_dir / "depth_Y.npy", frame_Y["depth"])

        diff_AB = abs_diff_vis(rgbA, rgbB)
        diff_AY = abs_diff_vis(rgbA, rgbY)
        diff_BY = abs_diff_vis(rgbB, rgbY)
        strip_ABY = make_strip(rgbA, rgbB, rgbY)

        kb.write_png(diff_AB, str(view_dir / "diff_AB.png"))
        kb.write_png(diff_AY, str(view_dir / "diff_AY.png"))
        kb.write_png(diff_BY, str(view_dir / "diff_BY.png"))
        kb.write_png(strip_ABY, str(view_dir / "strip_ABY.png"))

        meta_view = {
            "view_id": int(view_idx),
            "camera_position": _to_float_list(cam_pos),
            "look_at": _to_float_list(center_for_views),
            "azimuth_deg": float(az_deg),
            "elevation_deg": float(el_deg),
            "radius": float(radius),
            "focal_mm": float(focal_mm),
            "sensor_width_mm": float(FLAGS.sensor_width_mm),
            "foreground_ratio_A": float(fgA),
            "foreground_ratio_B": float(fgB),
            "foreground_ratio_Y": float(fgY),
        }
        save_json(view_dir / "camera.json", meta_view)

    renderer.save_state(str(out_root / "scene.blend"))
    logging.info("Saved scene to %s", out_root)


if __name__ == "__main__":
    main()