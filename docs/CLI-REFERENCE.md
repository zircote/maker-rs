# MAKER CLI Reference

Complete command-line interface documentation for `maker-cli`.

## Installation

```bash
cargo build --release --bin maker-cli
# Binary: target/release/maker-cli

# Optional: Add to PATH
export PATH="$PATH:$(pwd)/target/release"
```

## Global Options

```
maker-cli [OPTIONS] <COMMAND>

Options:
  -f, --format <FORMAT>    Output format [default: text] [possible values: text, json]
  -v, --verbose...         Increase verbosity (-v info, -vv debug, -vvv trace)
  -h, --help               Print help
  -V, --version            Print version
```

## Commands

### vote - Execute SPRT Voting

Run error-corrected voting on a prompt to get a reliable response.

```bash
maker-cli vote [OPTIONS]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-p, --prompt <PROMPT>` | string | stdin | The prompt to vote on (use `-` for stdin) |
| `-k, --k-margin <K>` | integer | 3 | Required vote margin for declaring winner |
| `-n, --max-samples <N>` | integer | 100 | Maximum samples before timeout |
| `-t, --temperature <T>` | float | 0.1 | Temperature diversity for sampling (0.0-1.0) |
| `--provider <PROVIDER>` | string | ollama | LLM provider (ollama, openai, anthropic) |
| `--adaptive` | flag | false | Enable adaptive k-margin |
| `--matcher <MATCHER>` | string | exact | Matcher type (exact, embedding, code) |

**Examples:**

```bash
# Basic voting with default settings
echo "What is 2+2?" | maker-cli vote

# Custom k-margin and provider
maker-cli vote -p "Explain quantum computing" -k 5 --provider openai

# JSON output for scripting
echo "Capital of France?" | maker-cli --format json vote -k 3

# Adaptive k-margin for variable accuracy
maker-cli vote -p "Complex math problem" --adaptive

# Code matching for programming tasks
maker-cli vote -p "Write a Rust function to reverse a string" --matcher code
```

**Output (JSON):**
```json
{
  "winner": "Paris",
  "votes": 5,
  "total_samples": 7,
  "k_margin": 3,
  "converged": true
}
```

---

### validate - Red-Flag Validation

Check if a response passes validation rules without voting.

```bash
maker-cli validate [OPTIONS]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-r, --response <RESPONSE>` | string | stdin | Response to validate (use `-` for stdin) |
| `-t, --token-limit <LIMIT>` | integer | none | Maximum token count |
| `-s, --schema <SCHEMA>` | string | none | Expected JSON schema |

**Examples:**

```bash
# Validate from stdin
echo '{"result": "success"}' | maker-cli validate

# With token limit
echo "Very long response..." | maker-cli validate -t 100

# With JSON schema validation
echo '{"name": "John", "age": 30}' | maker-cli validate \
  -s '{"type": "object", "required": ["name", "age"]}'

# JSON output
echo "test" | maker-cli --format json validate -t 1000
```

**Output (JSON):**
```json
{
  "valid": true,
  "red_flags": []
}
```

**Output with validation failures:**
```json
{
  "valid": false,
  "red_flags": [
    {
      "flag_type": "TokenLengthExceeded",
      "details": "Token count 150 exceeds limit 100"
    }
  ]
}
```

---

### calibrate - Estimate Success Rate

Estimate per-step success probability from calibration samples and recommend k-margin.

```bash
maker-cli calibrate [OPTIONS]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-f, --file <FILE>` | string | stdin | Path to JSON file with calibration samples |
| `-t, --target-reliability <T>` | float | 0.95 | Target task reliability (0.0-1.0) |
| `-t, --target-steps <S>` | integer | 1000 | Target step count |

**Input Format:**
```json
[
  {"prompt": "2+2?", "ground_truth": "4", "response": "4"},
  {"prompt": "3+3?", "ground_truth": "6", "response": "6"},
  {"prompt": "5*5?", "ground_truth": "25", "response": "24"}
]
```

**Examples:**

```bash
# From file
maker-cli --format json calibrate -f samples.json

# From stdin
cat samples.json | maker-cli calibrate

# Custom targets
maker-cli calibrate -f samples.json --target-reliability 0.99 --target-steps 500
```

**Output (JSON):**
```json
{
  "p_estimate": 0.67,
  "confidence_interval": [0.35, 0.88],
  "sample_count": 3,
  "recommended_k": 5
}
```

---

### config - View/Set Configuration

View or modify default MAKER parameters.

```bash
maker-cli config [OPTIONS]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--show` | flag | false | Show current configuration |
| `--k-margin <K>` | integer | - | Set default k-margin |
| `--provider <PROVIDER>` | string | - | Set default LLM provider |
| `--matcher <MATCHER>` | string | - | Set default matcher type |
| `--adaptive <BOOL>` | bool | - | Enable/disable adaptive k-margin |

**Examples:**

```bash
# Show current config
maker-cli config --show

# Set defaults
maker-cli config --k-margin 5 --provider openai --adaptive true

# JSON output
maker-cli --format json config --show
```

**Output (JSON):**
```json
{
  "k_margin": 3,
  "provider": "ollama",
  "matcher": "exact",
  "adaptive": false,
  "max_samples": 100
}
```

---

### decompose - Task Decomposition

Execute recursive task decomposition using MAKER's decomposition framework.

```bash
maker-cli decompose [OPTIONS]
```

**Options:**
| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-t, --task <TASK>` | string | stdin | Task description |
| `-d, --depth-limit <D>` | integer | 10 | Maximum recursion depth |
| `--timeout <SECONDS>` | integer | 60 | Timeout in seconds |

**Examples:**

```bash
# Simple decomposition
maker-cli decompose -t "Build a REST API with user authentication"

# With depth limit
maker-cli decompose -t "Refactor the payment module" -d 5

# JSON output for processing
maker-cli --format json decompose -t "Implement sorting algorithm"
```

**Output (JSON):**
```json
{
  "task_id": "task-1",
  "subtasks": [
    {
      "id": "task-1-1",
      "description": "Define function signature",
      "is_leaf": true
    },
    {
      "id": "task-1-2",
      "description": "Implement core logic",
      "is_leaf": true
    }
  ],
  "composition": "sequential",
  "depth": 2
}
```

---

### completions - Shell Completions

Generate shell completion scripts.

```bash
maker-cli completions <SHELL>
```

**Supported shells:** `bash`, `zsh`, `fish`, `powershell`

**Examples:**

```bash
# Bash
maker-cli completions bash > ~/.local/share/bash-completion/completions/maker-cli

# Zsh
maker-cli completions zsh > ~/.zfunc/_maker-cli

# Fish
maker-cli completions fish > ~/.config/fish/completions/maker-cli.fish

# PowerShell
maker-cli completions powershell > maker-cli.ps1
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid input, validation failure, etc.) |
| 2 | Invalid command-line arguments |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI provider |
| `ANTHROPIC_API_KEY` | API key for Anthropic provider |
| `RUST_LOG` | Log level (error, warn, info, debug, trace) |
| `OLLAMA_HOST` | Ollama server URL (default: http://localhost:11434) |

---

## Piping and Scripting

The CLI is designed for Unix-style piping:

```bash
# Pipeline: generate -> validate -> vote
echo "Write hello world" | \
  llm generate | \
  maker-cli validate -t 500 | \
  jq -r .valid

# Batch processing
for prompt in prompts/*.txt; do
  cat "$prompt" | maker-cli --format json vote -k 3 >> results.jsonl
done

# Integration with jq
maker-cli --format json vote -p "List 3 colors" | jq -r '.winner'
```

---

## See Also

- [QUICKSTART.md](./QUICKSTART.md) - Getting started guide
- [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) - Claude Code integration
- [PROMPT-ENGINEERING.md](./PROMPT-ENGINEERING.md) - Effective prompting strategies
