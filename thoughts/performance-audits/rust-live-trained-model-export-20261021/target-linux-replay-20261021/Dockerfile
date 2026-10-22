FROM rust:1.99.1-bookworm AS rust-toolchain
FROM ubuntu:24.04
COPY --from=rust-toolchain /usr/local/cargo /usr/local/cargo
COPY --from=rust-toolchain /usr/local/rustup /usr/local/rustup
ENV PATH="/usr/local/cargo/bin:${PATH}" \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*
