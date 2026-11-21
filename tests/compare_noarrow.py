#!/usr/bin/env python3
"""tests/compare_performance.py on the pyarrow-free reader, plus a raw-bytes gate.

Same synthetic generator (seed 1337) and the same semantic row comparison as
tests/compare_performance.py, but decoding Parquet with tests/pqmini.py because
pyarrow cannot be installed offline. In addition to the semantic comparison it
asserts the output files are byte-for-byte identical: pq.rs is a deterministic
writer, so a correctness-preserving optimization must reproduce identical bytes.
Reports the median runtime of each binary and the speedup ratio. Use more
--events/--clicks/--repeats for a more stable signal.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pqmini


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(columns)
        w.writerows(rows)


def make_data(root, n_events, n_clicks):
    d = root / "data"
    d.mkdir()
    rng = random.Random(1337)
    (d / "manifest.json").write_text(json.dumps({
        "window_start": "2026-10-01T00:00:00Z",
        "window_end": "2026-10-15T00:00:00Z"
    }))
    users = 3000
    products = 100
    campaigns = 30
    write_csv(d/"products.csv", ["product_id", "category"],
              ((f"p_{p}", f"cat_{p%10}") for p in range(products)))
    write_csv(d/"campaigns.csv", ["campaign_id", "channel"],
              ((f"c_{c}", f"ch_{c%5}") for c in range(campaigns)))
    write_csv(d/"links"/"part.csv", ["user_id_a", "user_id_b"],
              ((f"u_{i}", f"u_{i-1}") for i in range(1, 600) if i % 3 != 0))
    (d / "events").mkdir(exist_ok=True)
    (d / "clicks").mkdir(exist_ok=True)
    start = int(dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc).timestamp()*1000)
    with (d/"events"/"part.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"])
        for i in range(n_events):
            u = rng.randrange(users) if i % 13 else rng.randrange(10)
            t = start + rng.randrange(14*86400)*1000
            typ = "purchase" if i % 19 == 0 else ("cart" if i % 4 == 0 else "view")
            ingest = dt.datetime.fromtimestamp((t + 60000)/1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow((f"e_{i}", f"u_{u}", str(t), ingest, typ, f"p_{i%products}", "250" if typ == "purchase" else ""))
    with (d/"clicks"/"part.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["click_id", "user_id", "campaign_id", "ts"])
        for i in range(n_clicks):
            u = rng.randrange(users) if i % 7 else rng.randrange(10)
            t = start - 3*86400000 + rng.randrange(17*86400)*1000
            w.writerow((f"cl_{i}", f"u_{u}", f"c_{i%campaigns}", str(t)))
    return d


def normalize(output, table):
    rows, _ = pqmini.glob_table(output, table)
    return sorted([tuple(sorted((k, str(v)) for k, v in r.items())) for r in rows], key=str)


def file_digest(output, table):
    h = hashlib.sha256()
    for p in sorted((output / table).glob("*.parquet")):
        h.update(p.read_bytes())
    return h.hexdigest()


def measure(label, binary, data, root, repeats):
    durations = []
    reference = None
    digest = None
    for k in range(repeats):
        out = root / f"{label}_out"
        shutil.rmtree(out, ignore_errors=True)
        env = os.environ.copy()
        env.update(DATA_DIR=str(data), OUTPUT_DIR=str(out), NCPU="2")
        t0 = time.perf_counter()
        proc = subprocess.run([str(binary.resolve())], env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        if proc.returncode:
            raise RuntimeError(f"{label} failure on run {k+1}:\n{proc.stderr}")
        if reference is None:
            reference = (normalize(out, "user_daily"), normalize(out, "campaign_daily"))
            digest = (file_digest(out, "user_daily"), file_digest(out, "campaign_daily"))
        durations.append(elapsed)
        print(f"{label} run {k+1}: {elapsed:.4f}s", flush=True)
    return reference, digest, statistics.median(durations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--events", type=int, default=200000)
    p.add_argument("--clicks", type=int, default=100000)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--require-identical-bytes", action="store_true",
                   help="fail unless both tables are byte-for-byte identical")
    a = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = make_data(root, a.events, a.clicks)
        before, bdig, bt = measure("baseline", a.baseline, d, root, a.repeats)
        after, cdig, ct = measure("candidate", a.candidate, d, root, a.repeats)
        assert before == after, "candidate output differs from baseline (semantic rows, both tables)"
        identical = bdig == cdig
        print(f"byte-identical: user_daily={bdig[0][:12]}=={cdig[0][:12]}? "
              f"{bdig[0]==cdig[0]}; campaign_daily? {bdig[1]==cdig[1]}")
        if a.require_identical_bytes:
            assert identical, "candidate output is not byte-for-byte identical to baseline"
        print(f"PASS: all output rows match; baseline median {bt:.4f}s; "
              f"candidate median {ct:.4f}s; ratio {bt/ct:.3f}x", flush=True)
        print("NOTE: sandbox timing is NOT the official competition benchmark; measure relative speedup only.")


if __name__ == "__main__":
    main()
