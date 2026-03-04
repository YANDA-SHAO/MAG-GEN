# main.py
# Batch generator: run kb_min.py for many seeds, measure time, estimate ETA.

import argparse
import subprocess
import time
from pathlib import Path


def run_one(cmd, verbose: bool = True) -> int:
    if verbose:
        print("\n[CMD]", " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="How many scenes (seeds) to generate.")
    ap.add_argument("--seed0", type=int, default=0, help="Starting seed.")
    ap.add_argument("--kb_path", type=str, default="/kubric/kb_min.py", help="Path to kb_min.py inside container.")
    ap.add_argument("--python", type=str, default="python3", help="Python executable.")
    ap.add_argument("--extra", type=str, default="", help="Extra args passed to kb_min.py (as a single string).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    extra_args = args.extra.strip().split() if args.extra.strip() else []

    t0 = time.time()
    times = []

    for i in range(args.n):
        seed = args.seed0 + i

        cmd = [
            args.python, args.kb_path,
            "--seed", str(seed),
        ] + extra_args

        t1 = time.time()
        rc = run_one(cmd, verbose=args.verbose)
        t2 = time.time()

        dt = t2 - t1
        times.append(dt)

        # running stats
        done = i + 1
        avg = sum(times) / max(1, len(times))
        elapsed = t2 - t0
        remain = (args.n - done) * avg

        print(f"[PROGRESS] {done}/{args.n} seed={seed} rc={rc} "
              f"dt={fmt_time(dt)} avg={fmt_time(avg)} elapsed={fmt_time(elapsed)} ETA={fmt_time(remain)}")

        if rc != 0:
            print("[WARN] non-zero return code; continuing...")

    total = time.time() - t0
    avg = total / max(1, args.n)
    print("\n[FINISH] scenes =", args.n, " total =", fmt_time(total), " avg_per_scene =", fmt_time(avg))


if __name__ == "__main__":
    main()