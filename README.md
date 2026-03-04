# Kubric Tiny-Motion Dataset Generator

This repository provides a **Kubric-based synthetic dataset generator** for producing **tiny-motion A/B/Y triplets** with **RGB + depth + multi-view cameras**.

The generator is designed for research on:

- tiny motion detection
- motion magnification
- multi-view geometry
- 3D reconstruction from RGB-D

The pipeline uses:

- **Kubric** for scene generation
- **PyBullet** for physics simulation
- **Blender** for rendering

Each scene runs **one physics simulation** and renders **multiple camera views**.

---

# Dataset Concept

For each generated scene we produce three frames:

| Frame | Meaning |
|------|------|
| **A** | reference frame |
| **B** | small physical motion |
| **Y** | amplified motion |

Amplified translation:

```
p_Y = p_A + α (p_B − p_A)
```

Amplified rotation:

```
q_Y = q_A ⊗ (q_B q_A^{-1})^α
```

Therefore:

- **A → B** = tiny real motion  
- **A → Y** = amplified motion

---

# Output Structure

Example dataset output:

```
output/
 └── 000001/
      ├── meta_scene.json
      │
      ├── view_000/
      │    ├── A.png
      │    ├── B.png
      │    ├── Y.png
      │    ├── depth_A.npy
      │    ├── depth_B.npy
      │    ├── depth_Y.npy
      │    ├── camera.json
      │    ├── meta_view.json
      │    ├── diff_AB.png
      │    ├── diff_AY.png
      │    ├── diff_BY.png
      │    └── strip_ABY.png
      │
      ├── view_001/
      └── ...
```

Each scene contains multiple camera views.

---

# Repository Structure

Example repository layout:

```
kubric_run/
│
├── kb_min.py
├── main.py
├── dataset_3frames.py
│
└── output/
```

### kb_min.py

Main generator.

Responsibilities:

- build scene
- spawn objects
- run physics simulation
- compute amplified poses
- render RGB + depth

---

### main.py

Batch runner.

Responsibilities:

- run `kb_min.py` across many seeds
- monitor progress
- estimate runtime
- continue if a scene fails

---

# System Requirements

You only need:

- **Docker Desktop**
- **PowerShell (Windows)**

Kubric, Blender and PyBullet are already inside the Docker image.

---

# Setup (Windows + Docker)

## 1 Install Docker Desktop

Download:

https://www.docker.com/products/docker-desktop

Verify installation:

```powershell
docker --version
docker run hello-world
```

---

## 2 Pull the Kubric Docker image

```
docker pull kubricdockerhub/kubruntu
```

Test Kubric inside container:

```powershell
docker run --rm -it kubricdockerhub/kubruntu python3 -c "import kubric as kb; print('kubric ok')"
```

Expected output:

```
kubric ok
```

---

## 3 Create working directory

Example location:

```powershell
mkdir C:\Users\285261K\kubric_run -ErrorAction SilentlyContinue
cd C:\Users\285261K\kubric_run
```

Place the following files inside:

```
kb_min.py
main.py
dataset_3frames.py (optional)
```

Verify:

```powershell
dir C:\Users\285261K\kubric_run
```

Expected:

```
kb_min.py
main.py
dataset_3frames.py
```

---

# Running the Generator

Workflow:

1. Mount local folder into Docker container
2. Run scripts from `/kubric`
3. Save output to `/kubric/output`

This ensures data is saved on Windows.

---

# Generate One Scene

```powershell
docker run --rm -it `
  -v "C:\Users\285261K\kubric_run:/kubric" `
  kubricdockerhub/kubruntu `
  python3 /kubric/main.py --n 1 --seed0 54 --verbose --kb_path /kubric/kb_min.py `
  --extra "--job-dir /kubric/output"
```

After completion:

```powershell
dir C:\Users\285261K\kubric_run\output
```

---

# Generate Multiple Scenes

Example generating **200 scenes**:

```powershell
docker run --rm -it `
  -v "C:\Users\285261K\kubric_run:/kubric" `
  kubricdockerhub/kubruntu `
  python3 /kubric/main.py --n 200 --seed0 0 --verbose --kb_path /kubric/kb_min.py `
  --extra "--job-dir /kubric/output"
```

---

# Useful Options

Extra parameters can be passed via `--extra`.

### Increase camera views

```
--extra "--job-dir /kubric/output --num_views 20"
```

### Skip invalid views

```
--extra "--job-dir /kubric/output --num_views 20 --view_skip_bad"
```

### Faster rendering

```
--extra "--job-dir /kubric/output --samples_per_pixel 16"
```

### Adjust motion magnitude

```
--extra "--job-dir /kubric/output --px_ab_min 0.02 --px_ab_max 0.3 --px_ay_min 1.0 --px_ay_max 6.0"
```

---

# Verify Output

Check output directory:

```powershell
dir C:\Users\285261K\kubric_run\output
```

Example scene:

```
output/000054
```

---

# Common Issues

## Output not saved

Always specify:

```
--job-dir /kubric/output
```

Otherwise output may be written inside the container and disappear.

---

## Argument name error

Correct:

```
--job-dir
```

Incorrect:

```
--job_dir
```

---

## Failed object placement

Sometimes Kubric cannot place objects without overlap.

Example error:

```
RuntimeError: Failed to place
```

The batch script continues automatically.

To reduce failures:

```
--spawn_xy_max 1.2
--max_num_objects 2
--scale_max 1.2
```

---

# Reproducibility

Scene generation is deterministic.

Controlled by:

```
--seed
```

Output format:

```
output/{seed:06d}
```

---

# Recommended Batch Command

```
docker run --rm -it `
  -v "C:\Users\285261K\kubric_run:/kubric" `
  kubricdockerhub/kubruntu `
  python3 /kubric/main.py --n 200 --seed0 0 --verbose --kb_path /kubric/kb_min.py `
  --extra "--job-dir /kubric/output --num_views 15 --samples_per_pixel 64"
```

---

# References

Kubric official project:

https://github.com/google-research/kubric

If you need more details about scene configuration, asset sources, or renderer settings, please refer to the **official Kubric documentation**.
