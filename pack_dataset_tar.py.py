"""
pack_dataset_tar.py

Create a tar archive of a dataset directory.

This is useful for transferring large training datasets to remote servers
or cluster environments.

Example output:
    dataset_folder.tar
"""

import tarfile
from pathlib import Path

SRC = Path(r"C:\Users\285261K\kubric_run\vmm_train_local_data8")
TAR_PATH = Path(r"C:\Users\285261K\kubric_run\vmm_train_local_data8.tar")

if not SRC.exists():
    raise FileNotFoundError(f"{SRC} does not exist")

with tarfile.open(TAR_PATH, "w") as tar:
    tar.add(SRC, arcname=SRC.name)

print(f"Done: {TAR_PATH}")