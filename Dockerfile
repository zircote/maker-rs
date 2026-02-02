# syntax=docker/dockerfile:1

# =============================================================================
# MAKER Docker Image
# Multi-stage build for minimal container with maker-cli and maker-mcp
# Supports: linux/amd64, linux/arm64 (native builds per platform)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build environment (native build per platform)
# -----------------------------------------------------------------------------
FROM rust:latest AS builder

# Determine target triple based on current platform
RUN case "$(uname -m)" in \
      "x86_64")  echo "x86_64-unknown-linux-musl" > /target_triple ;; \
      "aarch64") echo "aarch64-unknown-linux-musl" > /target_triple ;; \
      *)         echo "x86_64-unknown-linux-musl" > /target_triple ;; \
    esac

# Install musl toolchain for static linking
RUN apt-get update && apt-get install -y --no-install-recommends \
    musl-tools \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

# Add rust target
RUN rustup target add $(cat /target_triple)

# Set up cargo for static linking
ENV RUSTFLAGS="-C target-feature=+crt-static"

WORKDIR /build

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy source files to build dependencies
RUN mkdir -p src/bin && \
    echo 'fn main() {}' > src/bin/maker-mcp.rs && \
    echo 'fn main() {}' > src/bin/maker-cli.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --target $(cat /target_triple) 2>/dev/null || true

# Copy actual source code
COPY src ./src
COPY tests ./tests
COPY benches ./benches

# Touch source files to invalidate the dummy build
RUN touch src/lib.rs src/bin/*.rs

# Build release binaries
RUN cargo build --release \
    --target $(cat /target_triple) \
    --bin maker-mcp \
    --bin maker-cli

# Strip binaries for minimal size
RUN mkdir -p /out && \
    cp /build/target/$(cat /target_triple)/release/maker-mcp /out/ && \
    cp /build/target/$(cat /target_triple)/release/maker-cli /out/ && \
    strip /out/maker-mcp /out/maker-cli

# -----------------------------------------------------------------------------
# Stage 2: Minimal runtime image
# -----------------------------------------------------------------------------
FROM scratch

# Copy CA certificates for HTTPS connections to LLM providers
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy binaries from build output
COPY --from=builder /out/maker-mcp /usr/local/bin/
COPY --from=builder /out/maker-cli /usr/local/bin/

# Log level (API keys should be passed at runtime)
ENV RUST_LOG="info"

# Default to MCP server (override with --entrypoint for CLI)
ENTRYPOINT ["/usr/local/bin/maker-mcp"]

# Labels for container registry
LABEL org.opencontainers.image.source="https://github.com/zircote/maker-rs"
LABEL org.opencontainers.image.description="MAKER Framework - Zero-error LLM agent execution"
LABEL org.opencontainers.image.licenses="MIT"
