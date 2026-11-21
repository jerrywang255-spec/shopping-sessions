# Shopping Sessions performance engineering

You are the performance engineer for this checkout. Produce a small, reviewable change that improves elapsed time from container start to exit and preserves exact results on new data. Keep experiments local and preserve the user's existing edits. The team handles publishing and submissions.

## Read before editing

Consult the organizers' [pipeline specification](https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/rules.md?inline=1) and [scoring and download page](https://www.joinbuilders.org/buildathon/tracks/shopping-sessions). The specification controls any disagreement with a starter. Use the supplied generator for representative training inputs and its documented interface. Derive all results from the mounted data at runtime.

Read the root `Dockerfile`, `Cargo.toml`, `main.rs`, `pq.rs`, `benchmark.py` and relevant tests before selecting an experiment. The Docker build copies the two root Rust files into `src/`, uses Rust 1.91 with `x86-64-v3` and launches `/app/ss`. Match those settings in a temporary Linux build directory for binary tests. Preserve runtime dispatch when changing optional CPU instructions.

## Measure the scored workload

- Record the baseline revision and any local source changes. Preserve a baseline binary or image before editing. Treat the user's leaderboard time as a reference until the scored revision is identified. CI pins an older comparison commit.
- Use `benchmark.py --data DIR --candidate BIN --threads 8 --stages` for diagnosis. Its `SS_TIMING` lines are cumulative. Subtract adjacent timestamps. The interval ending at `stage4 sessions` includes writing user Parquet. The last interval covers campaign aggregation and output. Unset `BENCH` for every full-pipeline run.
- For the decision, alternate baseline and candidate in fresh containers on the same input, CPU set, cache conditions and storage. Time only container start through exit. Prepare data, build images and decode outputs outside that interval. Use at least three samples per version at practice scale (about 20M events). Use another seed and a final-scale input (about 60M) for the surviving change as resources permit. Record raw seconds, median, spread and peak memory. Label binary timings and reduced-scale runs clearly.
- Reproduce the scorer's 8 pinned logical CPUs, 24 GB RAM, no swap, uid 1000, read-only root and input, offline runtime, and writable output/scratch. The latter two share a 50 GB limit. Use a 64 MB tmpfs for `/tmp`. The competition caps each final run at 180 s. Keep generated inputs, answers and profiles outside version control.

## Establish correctness

Run the release unit tests and `python3 tests/smoke.py <linux-binary>`. Use the track's starter harness in a separate directory for all 36 fixtures and the official 1M-event seed-1 dev answers. Its `make check-dev` must read the candidate's actual outputs. `benchmark.py` checks for files only. `tests/compare_performance.py` is a quick differential check with two threads and synthetic data. A matching baseline can share a bug.

Compare the decoded rows of both tables, including multiplicities, nulls, column names and logical types. Row and column order, file boundaries and permitted integer widths may vary. Protect the boundaries relevant to each change:

- CSV quotes, literal carriage returns, spaces and empty or malformed rows. Keep earliest ingested well-formed event-ID copies ahead of validation, even when the earliest copy fails or a later copy names another user.
- Keep events exactly 24 hours late. Split sessions only above a 30-minute gap and assign the entire session to its UTC start day. Resolve linked identities by the bytewise minimum across the whole component, including link-only IDs.
- Include both endpoints of the seven-day click window. Break latest-click ties by bytewise click ID. Count each assisting campaign once per session, including the credited campaign, and retain assist-only rows. Organic has zero assists.
- Preserve purchase revenue, nullable top category, bytewise category ties, unknown products and the Parquet DATE type. Exercise header-only parts, uneven shards and shared-device users when changing parallel work distribution.

Keep changes with a repeatable end-to-end gain beyond local run-to-run variation. Report any checks that could not run, the exact blocker, and the strongest result the evidence supports.
