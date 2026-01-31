# MAKER Quick Start Guide

Get up and running with MAKER in 5 minutes.

## What is MAKER?

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a framework that makes LLM agents reliable. Even when your model is only 85% accurate per step, MAKER can achieve 95%+ success on 1,000+ step tasks through statistical voting and error correction.

## Installation

### Option 1: Build from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/zircote/maker-rs.git
cd maker-rs

# Build both binaries
cargo build --release

# Binaries are now at:
# - target/release/maker-cli    (standalone CLI)
# - target/release/maker-mcp    (MCP server for Claude Code)
```

### Option 2: Install via Cargo

```bash
cargo install maker
```

## Choose Your Integration Path

### Path A: Using with Claude Code / Claude Desktop

MAKER works as an MCP (Model Context Protocol) server that gives Claude access to error-correction tools.

1. **Add to your MCP configuration:**

   macOS/Linux: `~/.config/claude-code/config.json`

   ```json
   {
     "mcpServers": {
       "maker": {
         "command": "/path/to/maker-rs/target/release/maker-mcp"
       }
     }
   }
   ```

2. **Restart Claude Code**

3. **Use MAKER in your conversations:**
   ```
   You: Use MAKER to vote on the best solution for "implement a fibonacci function in Rust"

   Claude: I'll use the maker/vote tool to get a reliable, error-corrected response.
   [Executes voting with k_margin=3]
   Winner: "fn fib(n: u64) -> u64 { ... }" with 5 consistent votes
   ```

See [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) for complete setup instructions.

### Path B: Using the Standalone CLI

The CLI provides all MAKER functionality without requiring Claude Code.

```bash
# Basic voting on a prompt
echo "What is the capital of France?" | maker-cli vote -k 3

# Validate a response
echo '{"answer": "Paris"}' | maker-cli validate --schema '{"type": "object"}'

# Calibrate from test data
maker-cli calibrate -f samples.json

# Get help
maker-cli --help
```

See [CLI-REFERENCE.md](./CLI-REFERENCE.md) for complete command reference.

### Path C: Using as a Rust Library

```rust
use maker::core::{calculate_kmin, vote_with_margin, VoteConfig, MockLlmClient};

// 1. Calculate the k-margin needed for your task
let k = calculate_kmin(
    0.85,    // p: per-step accuracy (85%)
    0.95,    // t: target reliability (95%)
    100,     // s: number of steps
    1,       // m: must be 1 for MAKER
).unwrap();

// 2. Create an LLM client
let client = MockLlmClient::constant("example answer");

// 3. Run error-corrected voting
let config = VoteConfig::default();
let result = vote_with_margin("What is 2+2?", k, &client, config).unwrap();

println!("Winner: {} with {} votes", result.winner, result.total_samples);
```

## Core Concepts in 60 Seconds

### K-Margin Voting
Instead of trusting a single LLM response, MAKER samples multiple times and waits until one answer leads others by `k` votes. Higher `k` = more reliable but more samples needed.

### Red-Flag Validation
Before counting a vote, MAKER checks for obvious problems: token limits exceeded, invalid JSON, repetitive patterns. Bad responses are discarded, not repaired.

### m=1 Decomposition
Complex tasks are broken into atomic subtasks where each subtask is simple enough that voting makes sense. This is the "microagent" pattern.

### Adaptive K-Margin
MAKER can automatically adjust `k` based on observed accuracy. If the model is performing well, `k` decreases (faster). If accuracy drops, `k` increases (more reliable).

## Next Steps

| Goal | Document |
|------|----------|
| Set up Claude Code integration | [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) |
| Learn the CLI commands | [CLI-REFERENCE.md](./CLI-REFERENCE.md) |
| Use MAKER for coding tasks | [guides/CODING.md](./guides/CODING.md) |
| Use MAKER for data analysis | [guides/DATA-ANALYSIS.md](./guides/DATA-ANALYSIS.md) |
| Use MAKER for ML pipelines | [guides/ML-PIPELINES.md](./guides/ML-PIPELINES.md) |
| Prompt your assistant effectively | [PROMPT-ENGINEERING.md](./PROMPT-ENGINEERING.md) |

## Troubleshooting

### "maker-cli: command not found"
Add the binary to your PATH or use the full path:
```bash
export PATH="$PATH:/path/to/maker-rs/target/release"
```

### "connection refused" with Ollama
Ensure Ollama is running:
```bash
ollama serve
ollama pull llama3.2
```

### Voting not converging
- Reduce k_margin (try 2 instead of 5)
- Increase max_samples
- Enable adaptive_k for automatic adjustment

### Need more help?
- Check the [full documentation](./README.md)
- Open an issue: https://github.com/zircote/maker-rs/issues
