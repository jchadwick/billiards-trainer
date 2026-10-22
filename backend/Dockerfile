# Build the retained browser client and Rust server separately so the runtime
# image contains only static HTML/JS/CSS, Rust executables, model libraries,
# and the native GStreamer runtime needed for camera/stream capture.

FROM node:22-bookworm-slim AS web-builder
WORKDIR /src/frontend/web
COPY frontend/web/package.json frontend/web/package-lock.json ./
RUN npm ci
COPY frontend/web/ ./
RUN npm run build

FROM rust:1.99-trixie AS rust-builder
ENV CARGO_HOME=/usr/local/cargo \
    CARGO_TARGET_DIR=/build/target
WORKDIR /build

# The GStreamer feature is deliberately compiled into the deployment binaries;
# actual camera/stream capture remains selected through the mounted config.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY perception/ ./perception/
# Depending on the Linux target bundle, ort-sys may statically embed ONNX
# Runtime or provide a shared object. Copy any emitted shared library; static
# builds require no additional ONNX runtime artifact in the final image.
RUN cargo build --release --locked \
    --manifest-path perception/Cargo.toml \
    --features camera-gstreamer-source \
    --bin billiards-server \
    --bin billiards-perception-live \
    --bin billiards-perception-gstreamer-probe
RUN mkdir -p /runtime/bin /runtime/lib \
    && install -m 0755 /build/target/release/billiards-server /runtime/bin/ \
    && install -m 0755 /build/target/release/billiards-perception-live /runtime/bin/ \
    && install -m 0755 /build/target/release/billiards-perception-gstreamer-probe /runtime/bin/ \
    && find /build/target/release /root/.cache/ort.pyke.io \
        -name 'libonnxruntime*.so*' -exec cp -L '{}' /runtime/lib/ \;

FROM debian:trixie-slim AS runtime
ARG BUILD_DATE
ARG BUILD_VERSION=dev
ARG BUILD_COMMIT=unknown

LABEL org.opencontainers.image.title="Billiards Trainer" \
      org.opencontainers.image.description="Rust billiards localization server and static browser client" \
      org.opencontainers.image.version=${BUILD_VERSION} \
      org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.revision=${BUILD_COMMIT}

ENV ENVIRONMENT=production \
    HTTP_BIND=0.0.0.0 \
    HTTP_PORT=8000 \
    SCENE_BIND=127.0.0.1 \
    SCENE_PORT=49321 \
    CONFIG_FILE=/app/config/config.json \
    PROJECT_ROOT=/app \
    WEB_ROOT=/app/web \
    LIVE_EXECUTABLE=/app/bin/billiards-perception-live \
    CAPTURE_HELPER_EXECUTABLE=/app/bin/billiards-perception-gstreamer-probe \
    START_LIVE=false \
    LD_LIBRARY_PATH=/app/lib:/usr/local/lib

# Soup HTTP, multipart demux, JPEG decode, V4L2, and GStreamer base elements
# are all required by the live capture helper. OpenVINO stays an optional
# externally supplied runtime and is not coupled to the ONNX Runtime CPU path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /usr/sbin/nologin --create-home appuser \
    && mkdir -p /app/bin /app/lib /app/config /app/data /app/logs /app/temp /app/web /app/models \
    && chown -R appuser:appuser /app \
    && rm -rf /var/lib/apt/lists/*

COPY --from=rust-builder --chown=appuser:appuser /runtime/bin/ /app/bin/
COPY --from=rust-builder --chown=appuser:appuser /runtime/lib/ /app/lib/
COPY --from=web-builder --chown=appuser:appuser /src/backend/static/ /app/web/
COPY --chown=root:root backend/entrypoint.sh /app/entrypoint.sh
RUN chmod 0755 /app/entrypoint.sh

WORKDIR /app
USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:${HTTP_PORT}/api/v1/health/live >/dev/null || exit 1

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
