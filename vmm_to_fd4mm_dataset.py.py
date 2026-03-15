"""
vmm_to_fd4mm_dataset.py

Convert a VMM dataset into the FD4MM training format.

This script reorganizes frames and metadata from scene/view folders into
the flat structure required by the FD4MM training pipeline:

    frameA/
    frameB/
    frameC/
    amplified/
    meta/
    train_mf.txt

The magnification factor is read from meta_scene.json (alpha field).
"""

from pathlib import Path
import shutil
import json

# ====== 改这两个路径 ======
SRC_ROOT = Path(r"C:\Users\285261K\kubric_run\dataset\train")
DST_ROOT = Path(r"C:\Users\285261K\kubric_run\vmm_train")
# ==========================

# 输出目录
dir_frameA = DST_ROOT / "frameA"
dir_frameB = DST_ROOT / "frameB"
dir_frameC = DST_ROOT / "frameC"
dir_amp    = DST_ROOT / "amplified"
dir_meta   = DST_ROOT / "meta"

for d in [dir_frameA, dir_frameB, dir_frameC, dir_amp, dir_meta]:
    d.mkdir(parents=True, exist_ok=True)

mf_lines = []
sample_idx = 1

scene_dirs = sorted([p for p in SRC_ROOT.iterdir() if p.is_dir()])

for scene_dir in scene_dirs:
    meta_scene_path = scene_dir / "meta_scene.json"
    if not meta_scene_path.exists():
        print(f"[WARN] missing meta_scene.json in {scene_dir}")
        continue

    with open(meta_scene_path, "r", encoding="utf-8") as f:
        meta_scene = json.load(f)

    # 这里要按你的 meta_scene.json 实际字段名改
    # 先尝试几个常见名字
    if "alpha" not in meta_scene:
        raise KeyError(f"Missing 'alpha' in {meta_scene_path}")

    mf = meta_scene["alpha"]

    if mf is None:
        raise KeyError(
            f"Cannot find magnification factor in {meta_scene_path}. "
            f"Available keys: {list(meta_scene.keys())}"
        )

    view_dirs = sorted([p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("view_")])

    for view_dir in view_dirs:
        src_A = view_dir / "A.png"
        src_B = view_dir / "B.png"
        src_Y = view_dir / "Y.png"
        src_meta_view = view_dir / "meta_view.json"

        if not (src_A.exists() and src_B.exists() and src_Y.exists()):
            print(f"[WARN] missing A/B/Y in {view_dir}, skip")
            continue

        out_name = f"{sample_idx:06d}.png"
        out_json = f"{sample_idx:06d}.json"

        # A -> frameA
        shutil.copy2(src_A, dir_frameA / out_name)

        # B -> frameB
        shutil.copy2(src_B, dir_frameB / out_name)

        # Y -> amplified
        shutil.copy2(src_Y, dir_amp / out_name)

        # 为兼容旧代码，Y 再复制到 frameC
        shutil.copy2(src_Y, dir_frameC / out_name)

        # meta_view -> meta
        if src_meta_view.exists():
            shutil.copy2(src_meta_view, dir_meta / out_json)
        else:
            # 没有 meta_view.json 就写一个最小占位
            with open(dir_meta / out_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "scene_id": scene_dir.name,
                        "view_id": view_dir.name,
                        "magnification_factor": mf
                    },
                    f,
                    indent=2
                )

        mf_lines.append(str(mf))
        sample_idx += 1

# 写 train_mf.txt
with open(DST_ROOT / "train_mf.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(mf_lines) + "\n")

print(f"Done. Total samples: {sample_idx - 1}")
print(f"Output folder: {DST_ROOT}")