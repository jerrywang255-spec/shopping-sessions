#!/usr/bin/env python3
"""Compare partition bit pairs on one generated dataset, with output equality."""
import hashlib
import itertools
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
import time

import pyarrow.parquet as pq

data=Path(os.environ["DATA_DIR"]).resolve()
binary_dir=Path("variants").resolve()
pairs=list(itertools.product((8,10,12), repeat=2))
trials=[pair for pair in pairs for _ in range(3)]
random.Random(20261121).shuffle(trials)
results={pair: [] for pair in pairs}
reference=None

def signature(folder):
    h=hashlib.sha256()
    for name in ("user_daily","campaign_daily"):
        vals=[]
        for path in (folder/name).glob("*.parquet"):
            vals.extend(pq.read_table(path).to_pylist())
        # Sorting ignores Parquet file ordering and column ordering.
        rows=sorted(repr(sorted(row.items())) for row in vals)
        for row in rows:
            h.update(row.encode())
            h.update(b"\n")
    return h.hexdigest()

with tempfile.TemporaryDirectory() as d:
    output=Path(d)/"output"
    for idx,(p1,p2) in enumerate(trials):
        binary=binary_dir/f"ss-p1{p1}-p2{p2}"
        assert binary.is_file(),binary
        shutil.rmtree(output,ignore_errors=True)
        env=os.environ.copy()
        env.update(DATA_DIR=str(data),OUTPUT_DIR=str(output),NCPU="4")
        for name in ("SS_RADIX_EVENTS","SS_SHORT_SIMD","SS_SCALAR_CSV","SS_PREFAULT","SS_TIMING"):
            env.pop(name,None)
        start=time.perf_counter()
        run=subprocess.run([str(binary)],env=env,capture_output=True,text=True)
        elapsed=time.perf_counter()-start
        assert run.returncode==0,(p1,p2,run.stderr)
        check=signature(output)
        if reference is None:reference=check
        assert check==reference,f"Correctness mismatch for p1={p1}, p2={p2}"
        results[(p1,p2)].append(elapsed)
        print(f"run={idx:02d} p1={p1} p2={p2} seconds={elapsed:.4f}",flush=True)

print("\nPASS: all nine configurations gave identical Parquet output.")
print("Median runtimes (GitHub runner, not official):",flush=True)
for p in sorted(pairs,key=lambda p: statistics.median(results[p])):
    print(f"p1={p[0]} p2={p[1]} median={statistics.median(results[p]):.4f}s",flush=True)
