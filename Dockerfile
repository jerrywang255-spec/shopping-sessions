# Shopping Sessions: std-only Rust pipeline (no crates: the build downloads nothing but the base images).
FROM rust:1.91-slim-bookworm AS build
WORKDIR /src
COPY main.rs pq.rs ./src/
COPY src ./src
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release --offline && strip target/release/ss

FROM debian:bookworm-slim
RUN useradd --uid 1000 --no-create-home --home-dir /scratch app
COPY --from=build /src/target/release/ss /app/ss
USER 1000
CMD ["/app/ss"]
