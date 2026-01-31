# MCP Integration Guide

Complete guide for integrating MAKER with Claude Code, Claude Desktop, and other MCP-compatible clients.

## Overview

MAKER exposes its error-correction capabilities through the Model Context Protocol (MCP), allowing AI assistants like Claude to access voting, validation, and decomposition tools during conversations.

## Prerequisites

- Rust 1.70+ and Cargo
- An MCP-compatible client (Claude Code, Claude Desktop, etc.)
- An LLM provider (Ollama for local, or OpenAI/Anthropic API keys)

## Installation

### Build the MCP Server

```bash
git clone https://github.com/zircote/maker-rs.git
cd maker-rs
cargo build --release --bin maker-mcp
```

The binary will be at `target/release/maker-mcp`.

### Optional: Code Matching Feature

For AST-based code comparison (useful for coding tasks):

```bash
cargo build --release --bin maker-mcp --features code-matcher
```

## Configuration

### Claude Code

Add to `~/.config/claude-code/config.json`:

```json
{
  "mcpServers": {
    "maker": {
      "command": "/absolute/path/to/maker-rs/target/release/maker-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

### Claude Desktop

Add to Claude Desktop's settings.json:

```json
{
  "mcpServers": {
    "maker": {
      "command": "/absolute/path/to/maker-mcp"
    }
  }
}
```

### With LLM Provider API Keys

```json
{
  "mcpServers": {
    "maker": {
      "command": "/path/to/maker-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "RUST_LOG": "info"
      }
    }
  }
}
```

## Available MCP Tools

After configuration, Claude has access to these tools:

### maker/vote

Execute SPRT voting to get error-corrected responses.

**Request Schema:**
```json
{
  "prompt": "string (required) - The prompt to vote on",
  "k_margin": "integer (required) - Vote margin for winner (>= 1)",
  "max_samples": "integer (optional) - Max samples before timeout",
  "temperature_diversity": "number (optional) - Sampling temperature",
  "provider": "string (optional) - LLM provider (ollama, openai, anthropic)",
  "adaptive": "boolean (optional) - Enable adaptive k-margin",
  "matcher": "string (optional) - Matcher type (exact, embedding, code)",
  "ensemble": "boolean (optional) - Enable multi-model ensemble"
}
```

**Response:**
```json
{
  "winner": "The winning response",
  "vote_counts": {"response1": 5, "response2": 2},
  "total_samples": 7,
  "k_used": 3,
  "p_hat": 0.87,
  "matcher_type": "exact",
  "candidate_groups": 2
}
```

### maker/validate

Check if a response passes red-flag validation.

**Request Schema:**
```json
{
  "response": "string (required) - Response content to validate",
  "token_limit": "integer (optional) - Maximum token count",
  "schema": "object (optional) - Expected JSON schema"
}
```

**Response:**
```json
{
  "valid": true,
  "red_flags": []
}
```

### maker/calibrate

Estimate success rate from sample data.

**Request Schema:**
```json
{
  "samples": [
    {"prompt": "...", "ground_truth": "...", "response": "..."}
  ],
  "target_reliability": "number (optional, default: 0.95)",
  "target_steps": "integer (optional, default: 1000)"
}
```

**Response:**
```json
{
  "p_estimate": 0.85,
  "confidence_interval": [0.78, 0.91],
  "sample_count": 20,
  "recommended_k": 4
}
```

### maker/configure

Set default parameters for subsequent calls.

**Request Schema:**
```json
{
  "k_default": "integer (optional) - Default k-margin",
  "temperature_diversity": "number (optional) - Default temperature",
  "token_limit": "integer (optional) - Default token limit",
  "provider": "string (optional) - Default LLM provider",
  "adaptive_k": "boolean (optional) - Enable adaptive k-margin",
  "ema_alpha": "number (optional) - EMA smoothing factor",
  "k_bounds": "[min, max] (optional) - Adaptive k bounds",
  "matcher": "object (optional) - Matcher configuration",
  "ensemble": "object (optional) - Ensemble configuration"
}
```

## Usage Examples

### Basic Voting

```
User: What's the most reliable answer to "What is 15 * 17"?

Claude: I'll use MAKER's voting to get an error-corrected answer.

[Uses maker/vote with prompt="What is 15 * 17?", k_margin=3]

The voted answer is "255" with 5 consistent samples. This passed MAKER's
k-margin voting, meaning at least 5 samples agreed on this answer before
any alternative reached within 3 votes.
```

### Validating Generated Code

```
User: Generate a Python function and validate it meets our standards.

Claude: I'll generate the code and validate it with MAKER.

[Generates code, then uses maker/validate with token_limit=500]

Validation result:
- Valid: true
- No red flags detected
- Token count: 127 (within 500 limit)
```

### Multi-Step Task with Calibration

```
User: I need to process 100 data transformations. How reliable will this be?

Claude: Let me calibrate MAKER's parameters for your task.

[Uses maker/calibrate with sample data]

Based on calibration:
- Estimated accuracy: 87%
- Recommended k-margin: 4
- With k=4 voting on 100 steps, expected reliability: 97%

[Uses maker/configure to set these defaults]

Now MAKER is configured for your task. Each transformation will be
voted on with k=4 margin.
```

### Code Generation with AST Matching

```
User: Write a Rust function to calculate factorial. Make sure it's correct.

Claude: I'll use MAKER with code matching to ensure correctness.

[Uses maker/configure with matcher={"type": "code", "language": "rust"}]
[Uses maker/vote with the factorial prompt]

After 7 samples with AST-based comparison, the winning implementation is:

```rust
fn factorial(n: u64) -> u64 {
    (1..=n).product()
}
```

This was chosen because it's semantically equivalent to 5 other generated
versions (ignoring variable naming and formatting differences).
```

## Advanced Configuration

### Adaptive K-Margin

Automatically adjust k based on observed accuracy:

```json
{
  "adaptive_k": true,
  "ema_alpha": 0.1,
  "k_bounds": [2, 10]
}
```

- When accuracy is high, k decreases (faster convergence)
- When accuracy drops, k increases (more reliable)

### Semantic Matching

For non-deterministic responses:

```json
{
  "matcher": {
    "type": "embedding",
    "threshold": 0.92,
    "provider": "ollama"
  }
}
```

Groups semantically similar responses together.

### Multi-Model Ensemble

Use multiple LLM providers to decorrelate errors:

```json
{
  "ensemble": {
    "models": [
      {"provider": "ollama", "model": "llama3.2", "cost_tier": "cheap"},
      {"provider": "openai", "model": "gpt-4", "cost_tier": "expensive"}
    ],
    "strategy": "cost_aware"
  }
}
```

Strategies:
- `round_robin`: Even distribution across models
- `cost_aware`: Cheap models first, escalate on disagreement
- `reliability_weighted`: More samples from reliable models

## Health Monitoring

Check server health:

```bash
# Via CLI
./target/release/maker-cli config --show

# Debug logs
RUST_LOG=debug ./target/release/maker-mcp
```

## Troubleshooting

### "Tool not found: maker/vote"

1. Verify the server path in your config is absolute
2. Check the server has execute permissions: `chmod +x maker-mcp`
3. Restart Claude Code/Desktop after config changes

### "Provider error: connection refused"

For Ollama:
```bash
ollama serve  # Start the server
ollama pull llama3.2  # Download a model
```

For OpenAI/Anthropic:
- Verify API key is set in the MCP config's `env` section
- Check network connectivity

### Slow Response Times

1. Use Ollama (local) instead of cloud providers
2. Reduce k_margin
3. Enable adaptive_k
4. Use cost_aware ensemble strategy

### Voting Never Converges

1. Increase max_samples
2. Reduce k_margin
3. Use semantic matching for creative tasks
4. Check if the prompt is ambiguous

## Security Considerations

- API keys in MCP config are only passed to the MAKER server
- MAKER validates all inputs with schema enforcement
- Red-flag filtering discards malformed responses
- Token limits prevent resource exhaustion

## Next Steps

- [PROMPT-ENGINEERING.md](./PROMPT-ENGINEERING.md) - How to prompt Claude to use MAKER effectively
- [guides/CODING.md](./guides/CODING.md) - Using MAKER for code generation
- [guides/DATA-ANALYSIS.md](./guides/DATA-ANALYSIS.md) - Using MAKER for data tasks
