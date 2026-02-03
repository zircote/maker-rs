# MAKER Documentation

> ⚠️ **EXPERIMENTAL PROJECT** - This is a research experiment, not production software. See the [main README](../README.md) for details.

Welcome to the MAKER Framework documentation. MAKER explores zero-error long-horizon LLM execution through SPRT voting, red-flag validation, and microagent orchestration.

## Quick Navigation

### Getting Started

| Document | Description |
|----------|-------------|
| [QUICKSTART.md](./QUICKSTART.md) | 5-minute setup guide |
| [CLI-REFERENCE.md](./CLI-REFERENCE.md) | Complete CLI command reference |
| [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) | Claude Code / Claude Desktop setup |

### Using MAKER Effectively

| Document | Description |
|----------|-------------|
| [PROMPT-ENGINEERING.md](./PROMPT-ENGINEERING.md) | How to prompt your AI assistant to use MAKER |
| [CLAUDE-CODE-SETUP.md](./CLAUDE-CODE-SETUP.md) | Detailed Claude Code integration guide |

### Domain-Specific Guides

| Guide | Use Case |
|-------|----------|
| [guides/CODING.md](./guides/CODING.md) | Code generation, refactoring, multi-file changes |
| [guides/DATA-ANALYSIS.md](./guides/DATA-ANALYSIS.md) | ETL pipelines, SQL generation, data quality |
| [guides/ML-PIPELINES.md](./guides/ML-PIPELINES.md) | Model training, hyperparameter tuning, evaluation |

### Technical Reference

| Document | Description |
|----------|-------------|
| [project/BENCHMARKS.md](./project/BENCHMARKS.md) | Performance benchmarks and results |
| [project/SystemDesignSpecification.txt](./project/SystemDesignSpecification.txt) | Architecture deep-dive |

## Documentation by Persona

### I want to use MAKER with Claude Code

1. Start with [QUICKSTART.md](./QUICKSTART.md) (Path A)
2. Follow [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) for setup
3. Learn [PROMPT-ENGINEERING.md](./PROMPT-ENGINEERING.md) for effective usage

### I want to use the CLI directly

1. Start with [QUICKSTART.md](./QUICKSTART.md) (Path B)
2. Reference [CLI-REFERENCE.md](./CLI-REFERENCE.md) for commands
3. See domain guides for task-specific examples

### I want to integrate MAKER into my Rust project

1. Start with [QUICKSTART.md](./QUICKSTART.md) (Path C)
2. Read API docs: `cargo doc --open`
3. See [examples/](../examples/) for code samples

### I work on coding tasks

1. Configure with [MCP-INTEGRATION.md](./MCP-INTEGRATION.md)
2. Read [guides/CODING.md](./guides/CODING.md)
3. Use code matcher with tree-sitter

### I work on data analysis

1. Configure with [MCP-INTEGRATION.md](./MCP-INTEGRATION.md)
2. Read [guides/DATA-ANALYSIS.md](./guides/DATA-ANALYSIS.md)
3. Use ETL decomposition patterns

### I work on ML pipelines

1. Configure with [MCP-INTEGRATION.md](./MCP-INTEGRATION.md)
2. Read [guides/ML-PIPELINES.md](./guides/ML-PIPELINES.md)
3. Use pipeline decomposition and metric validation

## Core Concepts

### K-Margin Voting

MAKER doesn't trust single LLM responses. It samples multiple times and waits until one answer leads others by `k` votes. Higher `k` = more reliable but more samples needed.

### Red-Flag Validation

Before counting a vote, MAKER checks for problems:
- Token limit exceeded
- Invalid JSON/schema
- Repetitive patterns

Bad responses are discarded, never repaired.

### m=1 Decomposition

Complex tasks are broken into atomic subtasks where each subtask is simple enough for voting. This is the "microagent" pattern.

### Adaptive K-Margin

MAKER automatically adjusts `k` based on observed accuracy:
- High accuracy → lower k (faster)
- Low accuracy → higher k (more reliable)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User / AI Assistant                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 MCP Server (maker-mcp)                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │ maker/vote  │ │maker/validate│ │maker/calibrate│          │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Core MAKER Library                         │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐   │
│  │  Voting   │ │ RedFlags  │ │ Matchers  │ │Decomposition│   │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Providers                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │ Ollama  │ │ OpenAI  │ │Anthropic│                        │
│  └─────────┘ └─────────┘ └─────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Version History

- **v0.3.0** (2026-01-31): Recursive Decomposition, Domain Decomposers, CLI, Async Executor
- **v0.2.0** (2026-01-31): Ensemble Voting, Adaptive K, Semantic Matching, Benchmarks
- **v0.1.0** (2026-01-30): Core MAKER algorithms, MCP server, Red-flag validation

See [CHANGELOG.md](../CHANGELOG.md) for full details.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development setup and guidelines.

## License

MIT - see [LICENSE](../LICENSE)
