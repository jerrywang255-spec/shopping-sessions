# Shopping Sessions pipeline (Rust, no dependencies)

A multi-threaded Rust implementation of `rules.md` for the Databricks Shopping Sessions track.

- Reads every CSV in `/data` with memory maps and a SIMD (AVX-512 / AVX2) line splitter.
- Links: union-find over all link lines; each person is named by the byte-wise smallest ID.
- Events: dedup on `event_id` (smallest `ingest_ts` wins, before validation), validation, late-event filter.
- Sessions per person (30-minute gaps), last-click attribution within 7 days, assists per campaign.
- Writes both Parquet tables with a small built-in Parquet writer (`day` is a Parquet DATE).

Files sit flat at the repository root; the Dockerfile copies `main.rs` and `pq.rs` into `src/`.
