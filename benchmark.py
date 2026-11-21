#!/usr/bin/env python3
"""Benchmark a Shopping Sessions candidate using reproducible wall-clock runs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

p = argparse.ArgumentParser()
p.add_argument('--data', type=Path, required=True)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--baseline', type=Path)
p.add_argument('--repeats', type=int, default=5)
p.add_argument('--threads', type=int, default=0)
p.add_argument('--stages', action='store_true')
a = p.parse_args()
assert (a.data / 'manifest.json').is_file(), 'Missing input manifest'
assert a.repeats >= 1

def run(name, binary):
    samples = []
    with tempfile.TemporaryDirectory(prefix='shopping-bench-') as root:
        for k in range(a.repeats):
            output = Path(root) / 'output'
            shutil.rmtree(output, ignore_errors=True)
            env = os.environ.copy()
            env['DATA_DIR'] = str(a.data.resolve())
            env['OUTPUT_DIR'] = str(output)
            if a.threads:
                env['NCPU'] = str(a.threads)
            if a.stages:
                env['SS_TIMING'] = '1'
            start = time.perf_counter()
            proc = subprocess.run([str(binary.resolve())], env=env, capture_output=True, text=True)
            elapsed = time.perf_counter() - start
            if proc.returncode:
                raise RuntimeError(f'{name} failed on run {k+1}:\n{proc.stderr}')
            assert list((output / 'user_daily').glob('*.parquet')), 'Missing user_daily Parquet'
            assert list((output / 'campaign_daily').glob('*.parquet')), 'Missing campaign_daily Parquet'
            samples.append(elapsed)
            print(f'{name} run {k+1}: {elapsed:.4f}s')
            if a.stages:
                print(proc.stderr)
    median = statistics.median(samples)
    return dict(samples_s=samples, median_s=median, min_s=min(samples), under_0_3_median=median < .3)

result = {}
if a.baseline:
    result['baseline'] = run('baseline', a.baseline)
result['candidate'] = run('candidate', a.candidate)
print(json.dumps(result, indent=2))
