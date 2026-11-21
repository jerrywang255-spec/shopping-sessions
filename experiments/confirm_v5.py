"""Compare release-image binaries in balanced pairs and verify every output."""
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time

import pyarrow.parquet as pq


def tables(folder):
    for name in ("user_daily", "campaign_daily"):
        files = sorted((folder / name).glob("*.parquet"))
        if not files:
            raise ValueError(f"missing {name}")
        table = pq.read_table(files)
        yield table.sort_by([(col, "ascending") for col in table.column_names])


def execute(name, folder):
    start = time.perf_counter()
    subprocess.run([Path("variants") / name], env=dict(os.environ, OUTPUT_DIR=str(folder)), check=True)
    return time.perf_counter() - start


def main():
    names = ("v4", "candidate")
    scores = {name: [] for name in names}
    orders = [names, names[::-1]] * 3
    random.Random(int(os.environ["SEED"])).shuffle(orders)
    with tempfile.TemporaryDirectory() as warm:
        for name in names:
            execute(name, Path(warm) / name)
        expected = list(tables(Path(warm) / "v4"))
    for block, order in enumerate(orders):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in order:
                elapsed = execute(name, root / name)
                scores[name].append(elapsed)
                print(f"block={block} name={name} elapsed={elapsed:.4f}s", flush=True)
            for name in names:
                for actual, reference in zip(tables(root / name), expected):
                    if not actual.schema.equals(reference.schema) or not actual.equals(reference):
                        raise ValueError(f"schema or row mismatch in block {block}, {name}")
    print("PASS: both complete tables agree for all 12 measured runs")
    for name, times in scores.items():
        print(f"RESULT seed={os.environ['SEED']} {name} median={statistics.median(times):.4f}s samples={times}")


if __name__ == "__main__":
    main()
