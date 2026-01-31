# Claude Code Setup Guide for MAKER

This guide explains how to integrate MAKER with Claude Code for zero-error LLM agent execution.

## Prerequisites

- Rust 1.70+ installed
- Claude Code (or another MCP-compatible client)
- An LLM provider configured (Ollama, OpenAI, or Anthropic)

## Installation

### Building from Source

```bash
git clone https://github.com/zircote/maker-rs.git
cd maker-rs
cargo build --release
```

The MCP server binary will be at `target/release/maker-mcp`.

### Using the Standalone CLI

If you prefer to use MAKER without MCP, you can use the CLI:

```bash
cargo build --release --bin maker-cli
./target/release/maker-cli --help
```

## Configuring Claude Code

Add the MAKER MCP server to your Claude Code configuration:

### macOS / Linux

Add to `~/.config/claude-code/config.json`:

```json
{
  "mcpServers": {
    "maker": {
      "command": "/path/to/maker-rs/target/release/maker-mcp",
      "args": []
    }
  }
}
```

### Windows

Add to `%APPDATA%\claude-code\config.json`:

```json
{
  "mcpServers": {
    "maker": {
      "command": "C:\\path\\to\\maker-rs\\target\\release\\maker-mcp.exe",
      "args": []
    }
  }
}
```

## Available MCP Tools

Once configured, the following tools are available in Claude Code:

### maker/vote

Execute SPRT voting on a prompt to get reliable, error-corrected responses.

```json
{
  "prompt": "What is 2+2?",
  "k_margin": 3,
  "max_samples": 50,
  "temperature_diversity": 0.1,
  "provider": "ollama"
}
```

**Parameters:**
- `prompt` (required): The prompt to vote on
- `k_margin` (required): Required vote margin for declaring winner (>= 1)
- `max_samples` (optional): Maximum samples before timeout
- `temperature_diversity` (optional): Temperature for diverse sampling
- `provider` (optional): LLM provider to use
- `adaptive` (optional): Enable adaptive k-margin
- `matcher` (optional): Matcher type ("exact", "embedding", "code")
- `ensemble` (optional): Enable multi-model ensemble voting

### maker/validate

Check if a response passes red-flag validation without voting.

```json
{
  "response": "The answer is 4",
  "token_limit": 700,
  "schema": {"type": "object"}
}
```

**Parameters:**
- `response` (required): The response content to validate
- `token_limit` (optional): Maximum token count
- `schema` (optional): Expected JSON schema

### maker/calibrate

Estimate per-step success rate (p) from calibration samples.

```json
{
  "samples": [
    {"prompt": "2+2?", "ground_truth": "4", "response": "4"},
    {"prompt": "3+3?", "ground_truth": "6", "response": "6"}
  ],
  "target_reliability": 0.95,
  "target_steps": 1000
}
```

**Parameters:**
- `samples` (required): Calibration samples with prompts and ground truth
- `target_reliability` (optional): Target reliability (default: 0.95)
- `target_steps` (optional): Target step count (default: 1000)

### maker/configure

Set default voting parameters for subsequent calls.

```json
{
  "k_default": 5,
  "temperature_diversity": 0.2,
  "token_limit": 1000,
  "provider": "openai",
  "adaptive_k": true
}
```

**Parameters:**
- `k_default` (optional): Default k-margin
- `temperature_diversity` (optional): Default temperature
- `token_limit` (optional): Default token limit
- `provider` (optional): Default LLM provider
- `adaptive_k` (optional): Enable/disable adaptive k-margin
- `matcher` (optional): Default matcher configuration
- `ensemble` (optional): Ensemble configuration

## LLM Provider Configuration

### Ollama (Local, Free)

1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull llama3.2`
3. MAKER uses Ollama by default

### OpenAI

Set the environment variable:
```bash
export OPENAI_API_KEY="your-api-key"
```

Then configure MAKER:
```json
{"provider": "openai"}
```

### Anthropic

Set the environment variable:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Then configure MAKER:
```json
{"provider": "anthropic"}
```

## Advanced Configuration

### Adaptive K-Margin

Enable adaptive k-margin to automatically adjust voting margins based on observed convergence:

```json
{
  "adaptive_k": true,
  "ema_alpha": 0.1,
  "k_bounds": [2, 10]
}
```

### Semantic Matching

Use embedding-based matching for semantic equivalence:

```json
{
  "matcher": {
    "type": "embedding",
    "threshold": 0.92,
    "provider": "ollama"
  }
}
```

### Code Matching

Use AST-based matching for code responses (requires `code-matcher` feature):

```bash
cargo build --release --features code-matcher
```

```json
{
  "matcher": {
    "type": "code",
    "language": "rust",
    "threshold": 0.95
  }
}
```

### Multi-Model Ensemble

Use multiple LLM providers for improved reliability:

```json
{
  "ensemble": {
    "models": [
      {"provider": "ollama", "name": "llama3.2", "weight": 1.0},
      {"provider": "openai", "name": "gpt-4", "weight": 1.5}
    ],
    "strategy": "reliability_weighted"
  }
}
```

## Health Check

Check server health status:

```bash
# Using the CLI
./target/release/maker-cli config --show

# The server also exports health status via the HealthChecker
```

## Troubleshooting

### Server Not Starting

1. Check the binary path in your config
2. Ensure the binary has execute permissions
3. Check logs: `RUST_LOG=debug maker-mcp`

### LLM Provider Errors

1. Verify API keys are set correctly
2. Check network connectivity
3. Ensure Ollama is running (if using local)

### Voting Not Converging

1. Increase `max_samples`
2. Reduce `k_margin` for faster convergence
3. Enable `adaptive_k` for automatic adjustment

### High Latency

1. Use a local provider (Ollama)
2. Reduce `k_margin`
3. Enable ensemble with `cost_aware` strategy

## Example Usage in Claude Code

Once configured, you can use MAKER tools in Claude Code:

```
User: Use MAKER to vote on "What is the capital of France?"

Claude: I'll use the maker/vote tool to get a reliable answer.
[Uses maker/vote with prompt="What is the capital of France?", k_margin=3]

Result: The voted answer is "Paris" with 5 votes and 100% confidence.
```

## Further Reading

- [MAKER Framework Documentation](../README.md)
- [System Design Specification](SystemDesignSpecification.txt)
- [API Reference](../target/doc/maker/index.html) (run `cargo doc --open`)
