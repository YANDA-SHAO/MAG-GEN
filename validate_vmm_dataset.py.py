"""
validate_vmm_dataset.py

Verify the integrity of a VMM training dataset.

Checks include:
    - frameA/frameB/frameC/amplified/meta contain the same sample IDs
    - number of samples is consistent across folders
    - train_mf.txt length matches dataset size

This script helps detect missing frames or mismatched metadata before
training.
"""

from pathlib import Path

ROOT = Path(r"C:\Users\285261K\kubric_run\vmm_train_local_data5")

dirs = {
    "frameA": ROOT / "frameA",
    "frameB": ROOT / "frameB",
    "frameC": ROOT / "frameC",
    "amplified": ROOT / "amplified",
    "meta": ROOT / "meta",
}

def get_ids(folder, suffix):
    return sorted([p.stem for p in folder.glob(f"*{suffix}")])

ids = {}
ids["frameA"] = get_ids(dirs["frameA"], ".png")
ids["frameB"] = get_ids(dirs["frameB"], ".png")
ids["frameC"] = get_ids(dirs["frameC"], ".png")
ids["amplified"] = get_ids(dirs["amplified"], ".png")
ids["meta"] = get_ids(dirs["meta"], ".json")

for k, v in ids.items():
    print(f"{k}: {len(v)}")

same = (
    ids["frameA"] == ids["frameB"] ==
    ids["frameC"] == ids["amplified"] ==
    ids["meta"]
)

mf_path = ROOT / "train_mf.txt"
if not mf_path.exists():
    print("[ERROR] train_mf.txt does not exist")
    raise SystemExit(1)

with open(mf_path, "r", encoding="utf-8") as f:
    mf_lines = [line.strip() for line in f if line.strip()]

print(f"train_mf.txt lines: {len(mf_lines)}")

if not same:
    print("[ERROR] file id sets are not identical")
else:
    print("[OK] all file id sets match")

if len(mf_lines) != len(ids["frameA"]):
    print("[ERROR] train_mf.txt line count mismatch")
else:
    print("[OK] train_mf.txt line count matches")