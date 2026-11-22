#!/usr/bin/env python3
"""Randomized, interleaved paired binary benchmark with separate cache policies."""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
import time


def inputs(data):
    files = [data / "manifest.json", data / "products.csv", data / "campaigns.csv"]
    for folder in ("events", "clicks", "links"):
        files.extend(sorted((data / folder).glob("*.csv")))
    return files


def warm(files):
    subprocess.run(["cat", *map(str, files)], stdout=subprocess.DEVNULL, check=True)


def evict(files):
    for path in files:
        with path.open("rb") as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def check_cache(files, warm_cache):
    """Verify Linux input page residency outside the measured interval."""
    libc = ctypes.CDLL(None, use_errno=True)
    for path in files:
        size = path.stat().st_size
        if not size:
            continue
        with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) as mapping:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
            pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vec):
                raise OSError(ctypes.get_errno(), "mincore")
            resident = bytes(vec).count(b"\x01")
            if resident != (pages if warm_cache else 0):
                raise RuntimeError(f"{path}: {resident}/{pages} input pages resident")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--cpuset", default="0-7")
    p.add_argument("--cache", choices=("warm", "cold"), default="warm")
    p.add_argument("--seed", type=int, default=20261122)
    p.add_argument("--stages", action="store_true")
    p.add_argument("--baseline-prefault", action="store_true")
    p.add_argument("--candidate-prefault", action="store_true")
    a = p.parse_args()
    if a.repeats < 2 or a.repeats % 2:
        p.error("--repeats must be even and at least two for balanced order")
    files = inputs(a.data.resolve())
    binaries = [a.baseline.resolve(), a.candidate.resolve()]
    orders = [0, 1] * (a.repeats // 2)
    random.Random(a.seed).shuffle(orders)
    records = []
    peaks = [0, 0]
    with tempfile.TemporaryDirectory(prefix="ss-paired-") as tmp:
        output = Path(tmp) / "output"
        rss_file = Path(tmp) / "rss"
        for round_index, first in enumerate(orders, 1):
            order = (first, 1 - first)
            times = {}
            stage_output = {}
            for which in order:
                if output.exists():
                    shutil.rmtree(output)
                if a.cache == "warm":
                    warm(files)
                else:
                    evict(files)
                check_cache(files, a.cache == "warm")
                env = os.environ.copy()
                env.pop("BENCH", None)
                env.update(DATA_DIR=str(a.data.resolve()), OUTPUT_DIR=str(output), NCPU=str(a.threads))
                if a.stages:
                    env["SS_TIMING"] = "1"
                else:
                    env.pop("SS_TIMING", None)
                if (which == 0 and a.baseline_prefault) or (which == 1 and a.candidate_prefault):
                    env["SS_PREFAULT"] = "1"
                else:
                    env.pop("SS_PREFAULT", None)
                start = time.perf_counter()
                proc = subprocess.run(
                    ["/usr/bin/time", "-f", "%M", "-o", str(rss_file),
                     "taskset", "-c", a.cpuset, str(binaries[which])],
                    env=env, capture_output=True, text=True,
                )
                elapsed = time.perf_counter() - start
                if proc.returncode:
                    raise RuntimeError(f"{binaries[which]} failed in round {round_index}:\n{proc.stderr}")
                peaks[which] = max(peaks[which], int(rss_file.read_text().strip()))
                if not list((output / "user_daily").glob("*.parquet")) or not list((output / "campaign_daily").glob("*.parquet")):
                    raise RuntimeError(f"{binaries[which]} did not produce both tables")
                times[which] = elapsed
                stage_output[which] = proc.stderr
            record = {
                "round": round_index,
                "order": "BC" if first == 0 else "CB",
                "baseline_s": times[0],
                "candidate_s": times[1],
                "delta_ms": (times[1] - times[0]) * 1000,
            }
            records.append(record)
            print(
                f"round {round_index:02d} {record['order']} "
                f"baseline={times[0]:.4f}s candidate={times[1]:.4f}s delta={record['delta_ms']:+.1f}ms",
                flush=True,
            )
            if a.stages:
                for which, name in enumerate(("baseline", "candidate")):
                    print(f"-- round {round_index} {name}\n{stage_output[which]}", end="")
    deltas = [r["delta_ms"] for r in records]
    rng = random.Random(a.seed + 1)
    boots = sorted(statistics.median(rng.choices(deltas, k=len(deltas))) for _ in range(20000))
    result = {
        "binary_sha256": [hashlib.sha256(binary.read_bytes()).hexdigest() for binary in binaries],
        "cache": a.cache,
        "data": str(a.data.resolve()),
        "threads": a.threads,
        "cpuset": a.cpuset,
        "seed": a.seed,
        "baseline_prefault": a.baseline_prefault,
        "candidate_prefault": a.candidate_prefault,
        "rounds": a.repeats,
        "baseline_median_s": statistics.median(r["baseline_s"] for r in records),
        "candidate_median_s": statistics.median(r["candidate_s"] for r in records),
        "paired_delta_median_ms": statistics.median(deltas),
        "paired_delta_variance_ms2": statistics.variance(deltas),
        "paired_delta_stdev_ms": statistics.stdev(deltas),
        "paired_delta_iqr_ms": statistics.quantiles(deltas, n=4)[2] - statistics.quantiles(deltas, n=4)[0],
        "paired_delta_bootstrap_95ci_ms": [boots[500], boots[19500]],
        "candidate_faster_rounds": sum(d < 0 for d in deltas),
        "baseline_peak_rss_mib": peaks[0] / 1024,
        "candidate_peak_rss_mib": peaks[1] / 1024,
        "records": records,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
