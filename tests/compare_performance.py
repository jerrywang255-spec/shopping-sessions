#!/usr/bin/env python3
"""Comparable host benchmark: generate data, require equal outputs, report medians."""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import tempfile
import time

import pyarrow.parquet as pq


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
    write_csv(d/"products.csv", ["product_id","category"],
              ((f"p_{p}",f"cat_{p%10}") for p in range(products)))
    write_csv(d/"campaigns.csv", ["campaign_id","channel"],
              ((f"c_{c}",f"ch_{c%5}") for c in range(campaigns)))
    write_csv(d/"links"/"part.csv", ["user_id_a","user_id_b"],
              ((f"u_{i}", f"u_{i-1}") for i in range(1,600) if i%3 != 0))
    (d / "events").mkdir(exist_ok=True)
    (d / "clicks").mkdir(exist_ok=True)
    start = int(dt.datetime(2026,10,1,tzinfo=dt.timezone.utc).timestamp()*1000)
    with (d/"events"/"part.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["event_id","user_id","ts","ingest_ts","event_type","product_id","price_cents"])
        for i in range(n_events):
            u = rng.randrange(users) if i%13 else rng.randrange(10)
            t = start + rng.randrange(14*86400)*1000
            typ = "purchase" if i%19 == 0 else ("cart" if i%4 == 0 else "view")
            ingest = dt.datetime.fromtimestamp((t + 60000)/1000,dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow((f"e_{i}",f"u_{u}",str(t),ingest,typ,f"p_{i%products}","250" if typ=="purchase" else ""))
    with (d/"clicks"/"part.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n")
        w.writerow(["click_id","user_id","campaign_id","ts"])
        for i in range(n_clicks):
            u=rng.randrange(users) if i%7 else rng.randrange(10)
            t=start - 3*86400000 + rng.randrange(17*86400)*1000
            w.writerow((f"cl_{i}",f"u_{u}",f"c_{i%campaigns}",str(t)))
    return d


def normalize(output, table):
    paths = list((output/table).glob("*.parquet"))
    assert paths, f"Missing {table}"
    rows = pq.read_table(paths).to_pylist()
    return sorted([tuple(sorted(r.items())) for r in rows], key=str)


def measure(label, binary, data, root, repeats):
    durations=[]
    reference=None
    for k in range(repeats):
        out=root/f"{label}_out"
        shutil.rmtree(out,ignore_errors=True)
        env=os.environ.copy()
        env.update(DATA_DIR=str(data),OUTPUT_DIR=str(out),NCPU="2")
        t0=time.perf_counter()
        proc=subprocess.run([str(binary.resolve())],env=env,capture_output=True,text=True)
        elapsed=time.perf_counter()-t0
        if proc.returncode:
            raise RuntimeError(f"{label} failure on run {k+1}:\n{proc.stderr}")
        if reference is None:
            reference=(normalize(out,"user_daily"),normalize(out,"campaign_daily"))
        durations.append(elapsed)
        print(f"{label} run {k+1}: {elapsed:.4f}s",flush=True)
    return reference,statistics.median(durations)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--events",type=int,default=200000)
    p.add_argument("--clicks",type=int,default=100000)
    p.add_argument("--repeats",type=int,default=3)
    a=p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        d=make_data(root,a.events,a.clicks)
        before,bt=measure("baseline",a.baseline,d,root,a.repeats)
        after,ct=measure("candidate",a.candidate,d,root,a.repeats)
        assert before==after,"V3 output differs from baseline (both Parquet tables)"
        print(f"PASS: all output rows match; baseline median {bt:.4f}s; V3 median {ct:.4f}s; ratio {bt/ct:.3f}x",flush=True)
        print("NOTE: GitHub-hosted CI timing is not the official competition benchmark.")

if __name__=="__main__":
    main()
