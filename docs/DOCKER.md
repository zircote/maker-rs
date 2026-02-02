# Docker

MAKER is available as a minimal Docker container from GitHub Container Registry.

## Quick Start

```bash
# Pull the latest image
docker pull ghcr.io/zircote/maker-rs:latest

# Run the MCP server
docker run -it --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/zircote/maker-rs:latest

# Run the CLI
docker run -it --rm \
  --entrypoint /usr/local/bin/maker-cli \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/zircote/maker-rs:latest --help
```

## Image Details

| Property | Value |
|----------|-------|
| Registry | `ghcr.io/zircote/maker-rs` |
| Base | `scratch` (no OS, minimal attack surface) |
| Architectures | `linux/amd64`, `linux/arm64` |
| Size | ~15-20 MB |
| Binaries | `maker-mcp`, `maker-cli` |

## Tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest build from `main` branch |
| `main` | Latest build from `main` branch |
| `v0.3.0` | Specific version release |
| `0.3` | Latest patch of minor version |
| `0` | Latest minor/patch of major version |
| `<sha>` | Specific commit SHA |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For OpenAI | API key for OpenAI provider |
| `ANTHROPIC_API_KEY` | For Anthropic | API key for Anthropic provider |
| `RUST_LOG` | No | Log level: `error`, `warn`, `info` (default), `debug`, `trace` |

At least one provider API key is required for voting operations.

## Running the MCP Server

The default entrypoint is `maker-mcp`, which runs the MCP server using stdio transport:

```bash
# Basic usage
docker run -it --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/zircote/maker-rs:latest

# With Anthropic provider
docker run -it --rm \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  ghcr.io/zircote/maker-rs:latest

# With both providers (for ensemble voting)
docker run -it --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
  ghcr.io/zircote/maker-rs:latest

# With debug logging
docker run -it --rm \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e RUST_LOG=debug \
  ghcr.io/zircote/maker-rs:latest
```

## Running the CLI

Override the entrypoint to use `maker-cli`:

```bash
# Show help
docker run -it --rm \
  --entrypoint /usr/local/bin/maker-cli \
  ghcr.io/zircote/maker-rs:latest --help

# Run a vote
docker run -it --rm \
  --entrypoint /usr/local/bin/maker-cli \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/zircote/maker-rs:latest \
  vote --prompt "What is 2+2?" --k 2

# Validate a response
docker run -it --rm \
  --entrypoint /usr/local/bin/maker-cli \
  ghcr.io/zircote/maker-rs:latest \
  validate --response "The answer is 4"

# Calibrate success probability
docker run -it --rm \
  --entrypoint /usr/local/bin/maker-cli \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/zircote/maker-rs:latest \
  calibrate --prompt "Simple math: 1+1=" --samples 10
```

## Claude Code Integration

To use the containerized MCP server with Claude Code, add to your MCP configuration:

```json
{
  "mcpServers": {
    "maker": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "OPENAI_API_KEY",
        "ghcr.io/zircote/maker-rs:latest"
      ],
      "env": {
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

## Building Locally

```bash
# Build for current architecture
docker build -t maker-rs:local .

# Build for specific platform
docker buildx build --platform linux/amd64 -t maker-rs:amd64 .
docker buildx build --platform linux/arm64 -t maker-rs:arm64 .

# Build multi-arch and push
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/zircote/maker-rs:dev \
  --push .
```

## Security

The container is built from `scratch` with:
- No shell or package manager
- No unnecessary system utilities
- Statically linked binaries
- Only CA certificates for HTTPS
- Minimal attack surface

For production deployments, consider:
- Using read-only root filesystem: `--read-only`
- Dropping all capabilities: `--cap-drop=ALL`
- Running as non-root (binaries support this)
- Network isolation as appropriate
