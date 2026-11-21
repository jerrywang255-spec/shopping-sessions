# syntax=docker/dockerfile:1.6
FROM rust:1.91-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml ./
COPY main.rs pq.rs ./src/
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release --offline && strip target/release/ss

FROM build AS training
RUN rustup component add llvm-tools-preview && \
    apt-get update && apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:13724cadbac5a99d847e27cea911cd2e338ff972b46c2039da87452011bb101e https://www.joinbuilders.org/buildathon/tracks/shopping-sessions/files/generator.zip /tmp/generator.zip
RUN unzip -q /tmp/generator.zip -d /tmp/training && \
    cargo build --manifest-path /tmp/training/generator/Cargo.toml --release --offline && \
    /tmp/training/generator/target/release/generate --seed 11 --events 800000 --clicks 200000 --output /tmp/train-data
RUN mkdir -p /tmp/profiles && \
    RUSTFLAGS="$RUSTFLAGS -C profile-generate=/tmp/profiles" cargo build --release --offline && \
    DATA_DIR=/tmp/train-data OUTPUT_DIR=/tmp/train-result NCPU=4 target/release/ss && \
    "$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/llvm-profdata" merge -o /tmp/ss.profdata /tmp/profiles && \
    RUSTFLAGS="$RUSTFLAGS -C profile-use=/tmp/ss.profdata" cargo build --release --offline && \
    strip target/release/ss

FROM debian:bookworm-slim AS runtime
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
USER 1000
CMD ["/app/ss"]

FROM runtime AS plain
COPY --from=build /src/target/release/ss /app/ss

FROM runtime
COPY --from=training /src/target/release/ss /app/ss
