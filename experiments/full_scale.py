#!/usr/bin/env python3
"""Compare P2=10 and P2=8 on one 20-million-event generator dataset.

A GitHub-hosted runner is NOT the official scoring hardware.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import pyarrow.parquet as pq

data=Path(os.environ["DATA_DIR"]).resolve()
binaries={"original":Path("variants/ss-original").resolve(),
          "p2_8":Path("variants/ss-p2-8").resolve()}
repeat=3
plan=list(binaries)*repeat
random.Random(2256).shuffle(plan)
scores={name:[] for name in binaries}
parts={name:{} for name in binaries}
signatures={}
pat=re.compile(r"^\s*(\d+(?:\.\d+)?) ms  (.+)$",re.M)

def canonical(folder):
    hasher=hashlib.sha256()
    counts={}
    for name in ("user_daily","campaign_daily"):
        paths=list((folder/name).glob("*.parquet"))
        assert paths,f"missing {name}"
        records=[]
        for path in paths:
            records.extend(pq.read_table(path).to_pylist())
        counts[name]=len(records)
        # Produce exactly identical canonical byte strings independently of row/file ordering.
        encoded=sorted(json.dumps(r,sort_keys=True,separators=(",",":"),default=str)
                       for r in records)
        for e in encoded:
            hasher.update(e.encode("utf-8"))
            hasher.update(b"\n")
    return counts,hasher.hexdigest()

with tempfile.TemporaryDirectory() as temp:
    output=Path(temp)/"output"
    for i,name in enumerate(plan):
        shutil.rmtree(output,ignore_errors=True)
        env=os.environ.copy()
        env.update(DATA_DIR=str(data),OUTPUT_DIR=str(output),NCPU="4",SS_TIMING="1")
        for k in ("SS_PREFAULT","SS_SCALAR_CSV","SS_SHORT_SIMD","SS_RADIX_EVENTS"):
            env.pop(k,None)
        start=time.perf_counter()
        run=subprocess.run([str(binaries[name])],env=env,capture_output=True,text=True)
        secs=time.perf_counter()-start
        if run.returncode!=0:raise RuntimeError(f"{name} error: {run.stderr}")
        scores[name].append(secs)
        prev=0.0
        marks=[]
        for millis,label in pat.findall(run.stderr):
            now=float(millis)
            parts[name].setdefault(label,[]).append(now-prev)
            marks.append(f"{label}={now-prev:.1f}ms")
            prev=now
        if name not in signatures:
            signatures[name]=canonical(output)
            if len(signatures)==len(binaries):
                assert len({repr(x) for x in signatures.values()})==1,f"OUTPUT MISMATCH: {signatures}"
        print(f"run={i:02d} name={name} time={secs:.4f}s ; "+", ".join(marks),flush=True)

assert signatures["original"]==signatures["p2_8"],"Parquet result mismatch"
print("PASS: both output tables match exactly (canonicalized SHA-256).",flush=True)
for name in binaries:
    print(f"{name} median={statistics.median(scores[name]):.4f}s ; "+
          ", ".join(f"{key}={statistics.median(val):.1f}ms" for key,val in parts[name].items()),flush=True)
print(f"ratio original/tuned={statistics.median(scores['original'])/statistics.median(scores['p2_8']):.4f}x",flush=True)
