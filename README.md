# Shopping Sessions pipeline (Rust, no dependencies)

A multi-threaded Rust implementation of `rules.md` for the Databricks Shopping Sessions track.

- Reads every CSV in `/data` with memory maps and an AVX-512 / AVX2 line splitter (scalar fallback).
- Links: union-find over all link lines; each person is named by the byte-wise smallest ID.
- Events: dedup on `event_id` (smallest `ingest_ts` wins, before validation), validation, late-event filter.
- Sessions per person (30-minute gaps, equal timestamps share a session), last-click attribution
  within 7 days (tie: smallest `click_id`), assists per campaign.
- Writes `/output/user_daily/*.parquet` and `/output/campaign_daily/part-000.parquet` with a small
  built-in Parquet writer (`day` is a Parquet DATE).

Build and run like the scorer:

    docker build -t shopping-sessions .
    docker run --rm --network none --read-only --user 1000:1000 \
      -v "$PWD/dev":/data:ro -v "$PWD/out":/output -v /tmp/scratch:/scratch shopping-sessions

Set `SS_TIMING=1` to print stage timings to stderr.
