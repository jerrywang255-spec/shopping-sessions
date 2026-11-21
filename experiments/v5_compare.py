"""Benchmark three binaries in position-balanced order and check every measured output."""
from itertools import permutations
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

import pyarrow.parquet as pq


def tables(directory):
    for name in ("user_daily", "campaign_daily"):
        files = sorted((directory / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"missing {name} in {directory}")
        table = pq.read_table(files)
        yield table.sort_by([(column, "ascending") for column in table.column_names])


def execute(name, output):
    env = dict(os.environ, OUTPUT_DIR=str(output), SS_TIMING="1")
    start = time.perf_counter()
    run = subprocess.run([Path("variants") / name], env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if run.returncode:
        raise RuntimeError(f"{name}: {run.stderr}")
    return elapsed, run.stderr.strip()


def main():
    names = os.environ["VARIANTS"].split()
    rng = random.Random(int(os.environ["SEED"]))
    scores = {name: [] for name in names}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for name in names:
            execute(name, root / f"warm-{name}")
        orders = list(permutations(names)) * (3 if len(names) == 2 else 1)
        rng.shuffle(orders)
        measured = []
        for block, order in enumerate(orders):
            for name in order:
                output = root / f"{block}-{name}"
                secs, stages = execute(name, output)
                scores[name].append(secs)
                measured.append((name, output))
                print(f"block={block} name={name} elapsed={secs:.4f}s\n{stages}", flush=True)
        reference = list(tables(root / "warm-baseline"))
        for name, output in measured:
            for actual, expected in zip(tables(output), reference):
                if not actual.schema.equals(expected.schema) or not actual.equals(expected):
                    raise ValueError(f"output mismatch in {output}")
        print(f"PASS: all {len(measured)} measured outputs agree on schema and rows")
        for name, times in scores.items():
            print(f"RESULT seed={os.environ['SEED']} {name} median={statistics.median(times):.4f}s samples={times}")


if __name__ == "__main__":
    main()
