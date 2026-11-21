# Shopping Sessions V3 — less sorting, fewer page faults

**Unverified experimental replacement**, built from V2 and designed to preserve the challenge semantics. `main.rs` retains the existing input parsing, deduplication, linked-identity resolution, sessionization, and Parquet writer.

## Changes relative to V2

1. **Click attribution:** sort each person's clicks by timestamp only; choose the byte-wise smallest click ID once per equal-timestamp group. This removes expensive ID comparisons from the sort comparator and preserves tie behavior.
2. **Campaign assists:** remove the comparison-sort of the same clicks by campaign. For people with more than 32 clicks, two stable 8-bit radix passes group by campaign in O(clicks + 512), preserving the previous timestamp order within each campaign. For smaller groups, keep comparison sort to avoid counting overhead.
3. **Memory:** disable eager `MADV_POPULATE_WRITE` by default so unused arena capacity isn't immediately faulted in. Set `SS_PREFAULT=1` to A/B test the old behavior.

## Reproducible build and benchmark

```bash
cargo build --release --offline
python3 benchmark.py --data /path/to/test-data --candidate ./target/release/ss --repeats 7 --threads 8 --stages
```

These changes are **hypotheses**, not a demonstrated speedup. They must be compiled and checked against the official reference on multiple seeds before submission. Benchmark on the *same CPU allocation and dataset* as your leaderboard evaluation. Run with and without `SS_PREFAULT=1` to select the faster page fault policy.

## What to profile next

- `parse, identity-aware clicks`: timestamp parsing, SIMD scanner, identity lookup and memory-shard pressure
- `dedup => final person shards`: hash-table probes, memory access, spills
- `stage4 sessions`: per-person event sorts and click/assist processing
- `done`: Parquet writing and output filesystem

The top performers' source code was not identified in public GitHub search. These changes are based on analysis of **your uploaded code** and generator distribution, not claims about other teams' exact techniques.