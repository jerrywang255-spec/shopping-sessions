#!/usr/bin/env python3
"""Small end-to-end Shopping Sessions correctness check, including Parquet schema."""
import csv
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def setup(root):
    data = root / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"window_start":"2026-10-01T00:00:00Z","window_end":"2026-10-15T00:00:00Z"}')
    write_csv(data / "products.csv", ["product_id", "category"],
              [("p1", "cat1"), ("p2", "cat2")])
    write_csv(data / "campaigns.csv", ["campaign_id", "channel"],
              [("c_1", "email"), ("c_2", "social")])
    write_csv(data / "links" / "part.csv", ["user_id_a", "user_id_b"],
              [("u_a", "u_b"), ("A0", "u_a")])
    write_csv(data / "clicks" / "part.csv",
              ["click_id", "user_id", "campaign_id", "ts"],
              [("c1", "u_b", "c_1", "2026-09-30T23:59:00Z"),
               ("c2", "u_a", "c_2", "2026-10-01T00:44:00Z"),
               ("c3", "u_b", "c_1", "2026-10-01T00:45:00Z"),
               ("c4", "u_a", "c_2", "2026-10-01T00:45:00Z"),
               ("c5", "u_c", "c_2", "2026-10-01T00:00:00Z")])
    write_csv(data / "events" / "part.csv",
              ["event_id", "user_id", "ts", "ingest_ts", "event_type", "product_id", "price_cents"],
              [("e1", "u_a", "2026-10-01T00:00:00Z", "2026-10-01T00:10:00Z", "view", "p1", ""),
               ("e2", "u_b", "2026-10-01T00:10:00Z", "2026-10-01T00:12:00Z", "purchase", "p1", "250"),
               ("e3", "u_a", "2026-10-01T00:45:00Z", "2026-10-01T00:46:00Z", "cart", "p2", ""),
               ("e4", "u_a", "2026-10-01T00:45:00Z", "2026-10-01T00:46:00Z", "purchase", "p2", "100"),
               ("e5", "u_b", "2026-10-02T01:00:00Z", "2026-10-02T01:01:00Z", "purchase", "p2", "300"),
               ("e6", "u_c", "2026-10-01T00:00:00Z", "2026-10-01T00:01:00Z", "view", "p1", ""),
               ("e7", "u_a", "invalid", "2026-10-01T00:01:00Z", "view", "p1", ""),
               ("e7", "u_a", "2026-10-01T00:01:00Z", "2026-10-01T00:02:00Z", "view", "p1", ""),
               ("e8", "u_d", "2026-10-01T05:00:00Z", "2026-10-01T05:01:00Z", "view", "no_such_product", "")])
    return data


def rows(out, table):
    files = list((out / table).glob("*.parquet"))
    assert files, "No Parquet in " + table
    t = pq.read_table(files)
    assert t.schema.field("day").type == __import__("pyarrow").date32(), t.schema
    return sorted(t.to_pylist(), key=lambda x: (x.get("user_id", x.get("campaign_id")), x["day"]))


def check(candidate):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = setup(root)
        out = root / "output"
        env = os.environ.copy()
        env.update(DATA_DIR=str(data), OUTPUT_DIR=str(out), NCPU="2")
        result = subprocess.run([str(candidate.resolve())], env=env, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Executable crashed ({result.returncode})\n{result.stderr}")
        day1 = dt.date(2026, 10, 1)
        day2 = dt.date(2026, 10, 2)
        expected_users = [
            dict(user_id="A0", day=day1, sessions=2, events=4, revenue_cents=350, top_category="cat1"),
            dict(user_id="A0", day=day2, sessions=1, events=1, revenue_cents=300, top_category="cat2"),
            dict(user_id="u_c", day=day1, sessions=1, events=1, revenue_cents=0, top_category=None),
            dict(user_id="u_d", day=day1, sessions=1, events=1, revenue_cents=0, top_category=None),
        ]
        expected_campaigns = [
            dict(campaign_id="c_1", channel="email", day=day1, sessions=2, purchasing_sessions=2, revenue_cents=350, assisted_sessions=2),
            dict(campaign_id="c_1", channel="email", day=day2, sessions=1, purchasing_sessions=1, revenue_cents=300, assisted_sessions=1),
            dict(campaign_id="c_2", channel="social", day=day1, sessions=1, purchasing_sessions=0, revenue_cents=0, assisted_sessions=2),
            dict(campaign_id="c_2", channel="social", day=day2, sessions=0, purchasing_sessions=0, revenue_cents=0, assisted_sessions=1),
            dict(campaign_id="organic", channel="organic", day=day1, sessions=1, purchasing_sessions=0, revenue_cents=0, assisted_sessions=0),
        ]
        actual_users = rows(out, "user_daily")
        actual_campaigns = rows(out, "campaign_daily")
        assert actual_users == expected_users, f"user_daily mismatch:\nactual={actual_users}\nexpected={expected_users}"
        assert actual_campaigns == expected_campaigns, f"campaign_daily mismatch:\nactual={actual_campaigns}\nexpected={expected_campaigns}"
        print("PASS: linked users, session gaps, event dedup, tie attribution, assists, organic, parquet DATE")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
