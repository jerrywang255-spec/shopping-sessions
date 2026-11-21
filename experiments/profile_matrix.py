#!/usr/bin/env python3
"""A/B profile 8-core Shopping Sessions pipeline on the same generated input.

Report stage deltas and median wall times; this is not a competition score.
"""
import argparse
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument("--candidate", type=Path, required=True)
ap.add_argument("--data", type=Path, required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--threads", default="2,4,8")
ap.add_argument("--prefault", default="0,1")
a = ap.parse_args()

variants = [(int(n), int(pre)) for n in a.threads.split(",") for pre in a.prefault.split(",")]
runs = variants * a.repeats
random.Random(20261121).shuffle(runs)
results = {v: [] for v in variants}
stages = {v: {} for v in variants}
pat = re.compile(r"^\s*(\d+(?:\.\d+)?) ms  (.+)$", re.M)

with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp) / "output"
    for iteration, (n, pre) in enumerate(runs):
        shutil.rmtree(out, ignore_errors=True)
        env = os.environ.copy()
        env.update(
            DATA_DIR=str(a.data.resolve()),
            OUTPUT_DIR=str(out),
            NCPU=str(n),
            SS_TIMING="1",
        )
        if pre:
            env["SS_PREFAULT"] = "1"
        else:
            env.pop("SS_PREFAULT", None)
        t0 = time.perf_counter()
        result = subprocess.run([str(a.candidate.resolve())], env=env, text=True, capture_output=True)
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(f"{(n,pre)} run {iteration} failed: {result.stderr}")
        if not list((out/"user_daily").glob("*.parquet")):
            raise RuntimeError("Missing user_daily output")
        if not list((out/"campaign_daily").glob("*.parquet")):
            raise RuntimeError("Missing campaign_daily output")
        results[(n,pre)].append(elapsed)
        marks = [(label, float(ms)) for ms, label in pat.findall(result.stderr)]
        prev = 0.
        for label, ms in marks:
            stages[(n,pre)].setdefault(label,[]).append(ms-prev)
            prev = ms
        print(f"run={iteration:02d} threads={n} prefault={pre} seconds={elapsed:.4f} "+
              " stages="+",".join(f"{label}:{ms:.1f}ms" for label,ms in marks), flush=True)

print("\n=== Results, medians on this runner and this dataset ===",flush=True)
for variant in sorted(variants,key=lambda v: statistics.median(results[v])):
    med = statistics.median(results[variant])
    detail = ", ".join(f"{label}:{statistics.median(samples):.1f}ms"
                       for label,samples in stages[variant].items())
    print(f"threads={variant[0]} prefault={variant[1]} median={med:.4f}s ; {detail}",flush=True)
