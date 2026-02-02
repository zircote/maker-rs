# syntax=docker/dockerfile:1

# =============================================================================
# MAKER Docker Image
# Multi-stage build for minimal container with maker-cli and maker-mcp
# Supports: linux/amd64, linux/arm64
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build environment
# -----------------------------------------------------------------------------
FROM --platform=$BUILDPLATFORM rust:1.85-alpine AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install build dependencies for static linking and cross-compilation
RUN apk add --no-cache musl-dev pkgconfig openssl-dev openssl-libs-static

# Determine target triple based on platform
RUN case "$TARGETPLATFORM" in \
      "linux/amd64") echo "x86_64-unknown-linux-musl" > /target_triple ;; \
      "linux/arm64") echo "aarch64-unknown-linux-musl" > /target_triple ;; \
      *) echo "x86_64-unknown-linux-musl" > /target_triple ;; \
    esac

# Install cross-compilation toolchain if needed
RUN if [ "$BUILDPLATFORM" != "$TARGETPLATFORM" ]; then \
      case "$TARGETPLATFORM" in \
        "linux/arm64") apk add --no-cache gcc-aarch64-none-elf ;; \
      esac \
    fi

# Add rust target
RUN rustup target add $(cat /target_triple)

# Set up cargo for static linking
ENV RUSTFLAGS="-C target-feature=+crt-static"
ENV PKG_CONFIG_ALL_STATIC=1
ENV OPENSSL_STATIC=1

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

# Build release binaries with all features
RUN cargo build --release --features "code-matcher,prometheus" \
    --target $(cat /target_triple) \
    --bin maker-mcp \
    --bin maker-cli

# Strip binaries for minimal size and move to known location
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

# Log level (API keys should be passed at runtime, not set here)
ENV RUST_LOG="info"

# Default to MCP server (override with --entrypoint for CLI)
ENTRYPOINT ["/usr/local/bin/maker-mcp"]

# Labels for container registry
LABEL org.opencontainers.image.source="https://github.com/zircote/maker-rs"
LABEL org.opencontainers.image.description="MAKER Framework - Zero-error LLM agent execution"
LABEL org.opencontainers.image.licenses="MIT"
