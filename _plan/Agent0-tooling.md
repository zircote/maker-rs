# Integration Plan: maker-rs as an Instrument for Agent Zero

**Created:** 2026-01-31
**Status:** Research Complete / Implementation Pending
**Version:** 1.0

---

## Executive Summary

After researching both projects, I've identified **two distinct Agent0 projects** and determined the most viable integration target:

| Project | Type | Best For |
|---------|------|----------|
| [aiming-lab/Agent0](https://github.com/aiming-lab/Agent0) | Research framework (self-evolving agents) | Training methodology, not runtime integration |
| [agent0ai/agent-zero](https://github.com/agent0ai/agent-zero) | Runtime AI assistant framework | **Primary integration target** |

**Recommendation:** Integrate maker-rs with **Agent Zero** (agent0ai) due to its:
- Mature tool/instrument architecture
- Bidirectional MCP support
- Docker-based execution environment
- Extension mechanisms for custom behavior

---

## Agent Zero Architecture Overview

Agent Zero operates on a hierarchical multi-agent model with these key components:

```
┌─────────────────────────────────────────────────────────────┐
│                         User                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                     Agent 0                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Tools     │ │ Instruments │ │  Extensions │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
│         │               │               │                    │
│         └───────────────┼───────────────┘                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Message Loop & Memory                   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│ Subordinate│  │    MCP    │  │  Docker   │
│   Agents  │  │  Servers  │  │ Container │
└───────────┘  └───────────┘  └───────────┘
```

### Key Components

1. **Agents**: Primary decision-makers that receive instructions, reason through problems, and coordinate tool usage
2. **Tools**: Built-in functionalities always present in system prompt (code execution, memory, search)
3. **Instruments**: Custom scripts stored in memory, recalled when needed, unlimited quantity
4. **Extensions**: Python modules that hook into the message loop for deep customization
5. **MCP Integration**: Bidirectional support for external MCP servers as tools

---

## Integration Approaches

### Approach 1: MCP Server Integration (Recommended)

**Effort:** Low | **Stability:** High | **Features:** Full

Agent Zero already supports MCP clients. maker-mcp can be registered as an external MCP server.

#### Configuration

Add to Agent Zero's MCP server configuration:

```json
{
  "maker": {
    "type": "local",
    "command": "/path/to/maker-mcp",
    "args": [],
    "env": {
      "OPENAI_API_KEY": "${OPENAI_API_KEY}",
      "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
    }
  }
}
```

#### Available Tools

| Tool | Agent Zero Name | Description |
|------|-----------------|-------------|
| `maker/vote` | `maker.vote` | SPRT voting for error-corrected responses |
| `maker/validate` | `maker.validate` | Red-flag validation |
| `maker/calibrate` | `maker.calibrate` | Success rate estimation |
| `maker/configure` | `maker.configure` | Runtime configuration |

#### Integration Flow

```
Agent 0 receives task
       │
       ▼
Identifies need for reliable output
       │
       ▼
Calls maker.vote via MCP
       │
       ▼
maker-mcp samples LLM, votes, returns winner
       │
       ▼
Agent 0 uses validated result
```

#### Advantages
- Zero code changes to either project
- Uses existing MCP infrastructure
- Full feature access
- Maintains clear separation of concerns

#### Disadvantages
- Requires maker-mcp binary in Docker container
- MCP overhead for each call

---

### Approach 2: Instrument Integration

**Effort:** Low | **Stability:** High | **Features:** CLI subset

Instruments are Agent Zero's mechanism for custom scripts stored in memory and recalled when needed.

#### Directory Structure

```
/instruments/custom/maker-vote/
├── maker-vote.md          # Description for agent
└── maker-vote.sh          # Execution script

/instruments/custom/maker-validate/
├── maker-validate.md
└── maker-validate.sh

/instruments/custom/maker-calibrate/
├── maker-calibrate.md
└── maker-calibrate.sh
```

#### Instrument Definition: maker-vote

**maker-vote.md:**
```markdown
# MAKER Vote Instrument

Use this instrument when you need a reliable, error-corrected answer.

## When to Use
- Critical calculations
- Multi-step task outputs
- Any output where accuracy matters

## Parameters
- prompt: The question or task (required)
- k_margin: Vote margin (default: 3, higher = more reliable)
- provider: LLM provider (ollama, openai, anthropic)

## Example
To get a reliable answer to a math problem:
~maker-vote "What is 15 * 17?" --k-margin 5

## Output
Returns JSON with winner, vote counts, and confidence.
```

**maker-vote.sh:**
```bash
#!/bin/bash
/opt/maker/maker-cli --format json vote -p "$1" -k "${2:-3}" --provider "${3:-ollama}"
```

#### Advantages
- Simple implementation
- No MCP complexity
- Stored in agent memory (infinite instruments)
- Auto-detected by agent

#### Disadvantages
- CLI invocation overhead
- Limited to shell script capabilities
- Less integrated than MCP approach

---

### Approach 3: Custom Python Tool

**Effort:** Medium | **Stability:** Medium | **Features:** Full with customization

Implement a Python tool class that wraps maker-rs.

#### Tool Implementation

**File:** `/python/tools/maker_tool.py`

```python
from python.helpers.tool import Tool, Response
import subprocess
import json

class MakerVote(Tool):
    """MAKER voting tool for error-corrected outputs."""

    async def execute(self, prompt: str, k_margin: int = 3,
                      provider: str = "ollama", **kwargs) -> Response:
        result = subprocess.run(
            ["maker-cli", "--format", "json", "vote",
             "-p", prompt, "-k", str(k_margin), "--provider", provider],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            return Response(
                message=f"MAKER vote failed: {result.stderr}",
                break_loop=False
            )

        output = json.loads(result.stdout)
        return Response(
            message=f"MAKER vote result: {output['winner']} "
                    f"(votes: {output['votes']}, samples: {output['total_samples']})",
            break_loop=False
        )
```

**Prompt file:** `/prompts/default/agent.system.tool.maker_vote.md`

```markdown
## Tool: maker_vote

Use this tool when you need a reliable, error-corrected answer.

### Parameters
- prompt (string, required): The question or task
- k_margin (int, default 3): Vote margin for winner
- provider (string, default "ollama"): LLM provider

### When to use
- Critical calculations where accuracy matters
- Multi-step tasks where errors compound
- Any output that will be used in subsequent steps

### Example
~~~json
{
    "prompt": "What is 15 * 17?",
    "k_margin": 5,
    "provider": "ollama"
}
~~~
```

#### Advantages
- Full Python flexibility
- Direct integration with Agent Zero's tool lifecycle
- Access to before_execution/after_execution hooks
- Can implement advanced features (caching, batching)

#### Disadvantages
- Requires Python wrapper maintenance
- Binary must be in container PATH
- More complex deployment

---

### Approach 4: Extension Integration

**Effort:** High | **Stability:** Medium | **Features:** Deep integration

Create Agent Zero extensions that automatically apply MAKER to agent outputs.

#### Extension Types

**1. Output Validation Extension**

Automatically validate all agent outputs before returning to user.

**File:** `/python/extensions/message_loop_prompts/50_maker_validation.py`

```python
async def validate_output(agent, output: str) -> str:
    """Validate agent output through MAKER red-flags."""
    result = await agent.call_tool("maker.validate", response=output)
    if not result.get("valid"):
        # Re-generate with voting
        return await agent.call_tool("maker.vote", prompt=output)
    return output
```

**2. Reliability-Aware Task Routing**

Route complex tasks through MAKER automatically.

**File:** `/python/extensions/message_loop_prompts/40_reliability_router.py`

```python
def should_use_maker(message: str) -> bool:
    """Determine if task requires MAKER reliability."""
    reliability_keywords = ["calculate", "ensure", "verify", "critical", "accurate"]
    return any(kw in message.lower() for kw in reliability_keywords)
```

#### Advantages
- Transparent to user/agent
- Automatic reliability enforcement
- Deep integration with message loop
- Can modify agent behavior dynamically

#### Disadvantages
- Complex implementation
- May add latency to all operations
- Requires thorough testing
- Harder to maintain

---

## Recommended Integration Strategy

### Phase 1: MCP Integration (Week 1)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 1.1 | Package maker-mcp for Docker | Dockerfile extension |
| 1.2 | Register MCP Server | Agent Zero config update |
| 1.3 | Test tool discovery | Verified maker.* tools available |
| 1.4 | Create prompt guidelines | System prompt additions |

**Acceptance Criteria:**
- [ ] maker-mcp binary accessible in Agent Zero Docker container
- [ ] All four maker.* tools discoverable via MCP
- [ ] Basic voting workflow functional end-to-end

### Phase 2: Instrument Library (Week 2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 2.1 | Create core instruments | maker-vote, maker-validate, maker-calibrate |
| 2.2 | Create domain instruments | maker-code, maker-data, maker-ml |
| 2.3 | Write usage documentation | Instrument guides |
| 2.4 | Test instrument recall | Verified agent finds instruments |

**Acceptance Criteria:**
- [ ] 6 instruments created and documented
- [ ] Agent correctly recalls instruments when relevant
- [ ] Domain instruments use appropriate MAKER settings

### Phase 3: Behavioral Integration (Week 3)

| Task | Description | Deliverable |
|------|-------------|-------------|
| 3.1 | System prompt modifications | MAKER-aware agent behavior |
| 3.2 | Memory integration | Calibration result storage |
| 3.3 | Accuracy tracking | Historical p-hat in memory |
| 3.4 | End-to-end testing | Full workflow validation |

**Acceptance Criteria:**
- [ ] Agent automatically uses MAKER for reliability keywords
- [ ] Calibration persists across sessions
- [ ] Performance benchmarks documented

---

## Technical Specifications

### Docker Container Requirements

```dockerfile
# Add to Agent Zero's Dockerfile
FROM agent0ai/agent-zero:latest

# Install MAKER binaries
RUN curl -L https://github.com/zircote/maker-rs/releases/download/v0.3.0/maker-cli-linux-amd64 \
    -o /usr/local/bin/maker-cli && chmod +x /usr/local/bin/maker-cli

RUN curl -L https://github.com/zircote/maker-rs/releases/download/v0.3.0/maker-mcp-linux-amd64 \
    -o /usr/local/bin/maker-mcp && chmod +x /usr/local/bin/maker-mcp

# Or build from source
# RUN apt-get update && apt-get install -y cargo
# RUN cargo install --git https://github.com/zircote/maker-rs maker
```

### MCP Server Configuration

**File:** Agent Zero MCP config (location varies by version)

```json
{
  "servers": {
    "maker": {
      "type": "local",
      "command": "maker-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "info",
        "OLLAMA_HOST": "http://host.docker.internal:11434"
      }
    }
  }
}
```

### LLM Provider Considerations

| Provider | Agent Zero | MAKER | Recommendation |
|----------|------------|-------|----------------|
| Ollama | Default | Default | Share instance, configure same model |
| OpenAI | Supported | Supported | Pass API key via environment |
| Anthropic | Supported | Supported | Pass API key via environment |

**Key Issue:** MAKER samples from LLMs independently. If Agent Zero and MAKER use the same provider, they may compete for rate limits.

**Solutions:**
1. Configure MAKER to use Ollama (local, no rate limits)
2. Use separate API keys with different rate limit pools
3. Implement request queuing/batching

### Performance Implications

| Operation | Latency Impact | Mitigation |
|-----------|---------------|------------|
| maker.vote (k=3) | +3-7 samples | Use adaptive k, start low |
| maker.vote (k=5) | +5-15 samples | Reserve for critical tasks |
| maker.validate | +1 validation | Minimal, always worthwhile |
| maker.calibrate | One-time setup | Cache in Agent Zero memory |

---

## Prompt Engineering for Agent Zero + MAKER

### System Prompt Addition

Add to `/prompts/default/agent.system.main.solving.md`:

```markdown
## Reliability with MAKER

When accuracy is critical, use MAKER tools:

1. **For reliable answers**: Use `maker.vote` with appropriate k_margin
   - Simple facts: k_margin=2
   - Calculations: k_margin=4
   - Critical decisions: k_margin=6

2. **For validation**: Use `maker.validate` before presenting:
   - Generated code
   - JSON/YAML outputs
   - Configuration files

3. **For multi-step tasks**: Use MAKER at each step where errors compound

Example workflow:
- User asks for accurate calculation
- You use maker.vote to get error-corrected result
- You use maker.validate to confirm no red flags
- You present the validated result
```

### Behavior Rule (Stored in Memory)

**File:** `/memory/behaviour.md` (auto-applied)

```markdown
## MAKER Reliability Rule

When the user says "accurate", "reliable", "verify", "ensure", or "critical":
- Always use maker.vote for generating answers
- Always use maker.validate before presenting outputs
- Report MAKER confidence in responses

When performing multi-step tasks (3+ steps):
- Use maker.vote for each step
- Track cumulative reliability
- Warn if reliability drops below 90%
```

### Domain-Specific Behaviors

**For Coding Tasks:**
```markdown
When generating code:
- Configure maker with code matcher: maker.configure(matcher="code")
- Use maker.vote for implementation
- Use maker.validate to check syntax
```

**For Data Tasks:**
```markdown
When transforming data:
- Use maker.vote for each transformation step
- Validate schema compliance with maker.validate
- Report data quality metrics
```

---

## Instrument Specifications

### Core Instruments

#### maker-vote

**Purpose:** Error-corrected answer generation

**Interface (`maker-vote.md`):**
```markdown
# MAKER Vote

Generates reliable, error-corrected answers through SPRT voting.

## Usage
~maker-vote "<prompt>" [k_margin] [provider]

## Parameters
- prompt: Question or task (required)
- k_margin: Vote margin, default 3 (optional)
- provider: ollama|openai|anthropic, default ollama (optional)

## Examples
~maker-vote "What is the square root of 144?"
~maker-vote "Calculate 15% of 847.50" 5
~maker-vote "Explain TCP handshake" 3 openai

## Output
JSON: {"winner": "...", "votes": N, "total_samples": N, "converged": true}
```

**Implementation (`maker-vote.sh`):**
```bash
#!/bin/bash
set -e
PROMPT="$1"
K_MARGIN="${2:-3}"
PROVIDER="${3:-ollama}"

maker-cli --format json vote -p "$PROMPT" -k "$K_MARGIN" --provider "$PROVIDER"
```

#### maker-validate

**Purpose:** Red-flag validation of outputs

**Interface (`maker-validate.md`):**
```markdown
# MAKER Validate

Validates output against red-flag rules.

## Usage
~maker-validate "<response>" [token_limit] [schema_file]

## Parameters
- response: Text to validate (required)
- token_limit: Maximum tokens, default none (optional)
- schema_file: JSON schema file path (optional)

## Examples
~maker-validate "The answer is 42"
~maker-validate '{"result": "success"}' 500 schema.json

## Output
JSON: {"valid": true|false, "red_flags": [...]}
```

#### maker-calibrate

**Purpose:** Estimate model accuracy for task planning

**Interface (`maker-calibrate.md`):**
```markdown
# MAKER Calibrate

Estimates per-step success probability from samples.

## Usage
~maker-calibrate <samples_file> [target_reliability] [target_steps]

## Parameters
- samples_file: JSON file with calibration samples (required)
- target_reliability: Target task reliability, default 0.95 (optional)
- target_steps: Number of steps in task, default 1000 (optional)

## Sample File Format
[
  {"prompt": "2+2?", "ground_truth": "4", "response": "4"},
  {"prompt": "3*3?", "ground_truth": "9", "response": "9"}
]

## Output
JSON: {"p_estimate": 0.87, "recommended_k": 4, "confidence_interval": [0.82, 0.92]}
```

### Domain Instruments

#### maker-code

**Purpose:** Code generation with AST matching

**Interface (`maker-code.md`):**
```markdown
# MAKER Code

Generates reliable code using AST-based semantic matching.

## Usage
~maker-code "<task>" <language> [k_margin]

## Parameters
- task: Code generation task (required)
- language: rust|python|javascript (required)
- k_margin: Vote margin, default 3 (optional)

## Examples
~maker-code "Implement binary search" rust
~maker-code "Write a decorator for retry logic" python 5

## Output
The winning code implementation with vote statistics.
```

#### maker-data

**Purpose:** Data transformation with validation

**Interface (`maker-data.md`):**
```markdown
# MAKER Data

Validates data transformations through MAKER pipeline.

## Usage
~maker-data "<transformation>" [schema_file]

## Parameters
- transformation: Data transformation description (required)
- schema_file: Expected output schema (optional)

## Examples
~maker-data "Convert CSV dates from MM/DD/YYYY to ISO format"
~maker-data "Calculate monthly revenue by region" output_schema.json
```

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| MCP connection failures | Low | Medium | Fallback to direct CLI via instruments |
| LLM rate limiting | Medium | Medium | Use local Ollama, separate API keys |
| Latency increase | High | Low | Adaptive k, document expected overhead |
| Docker image size | Low | Low | Use pre-built binaries (~10MB) |
| Version compatibility | Medium | Medium | Pin versions, document tested combinations |
| Agent confusion about when to use MAKER | Medium | Medium | Clear prompt guidelines, behavior rules |

---

## Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Integration time | < 1 week (Phase 1) | Development hours tracked |
| Error reduction | > 50% on multi-step tasks | A/B test with/without MAKER |
| Latency overhead | < 2x for typical operations | Benchmark suite |
| Instrument recall accuracy | > 90% | Test agent finds correct instrument |
| User satisfaction | Net positive feedback | User surveys/issues |

---

## Testing Plan

### Unit Tests

1. **MCP Connection**
   - Verify maker-mcp starts in Docker
   - Verify tool discovery returns 4 tools
   - Verify tool namespacing (maker.vote, etc.)

2. **Instrument Recall**
   - Verify agent finds maker-vote for "reliable answer" queries
   - Verify domain instruments match domain keywords

3. **End-to-End Workflows**
   - Math calculation with voting
   - Code generation with code matcher
   - Multi-step task with per-step validation

### Integration Tests

1. **Full Pipeline**
   ```
   User: "I need a reliable answer: what is 847 * 293?"
   Expected: Agent uses maker.vote, returns 248171
   ```

2. **Automatic Reliability**
   ```
   User: "Calculate the compound interest accurately"
   Expected: Agent detects "accurately", uses MAKER
   ```

3. **Multi-Step Task**
   ```
   User: "Build a data pipeline with 5 transformation steps"
   Expected: Agent uses MAKER for each step, tracks reliability
   ```

---

## Future Enhancements

### Phase 4: Advanced Integration (Future)

1. **Recursive Decomposition**
   - Expose maker/decompose via MCP
   - Agent Zero delegates complex tasks to MAKER decomposer
   - Subordinate agents execute decomposed subtasks

2. **Shared Memory**
   - Store MAKER calibration in Agent Zero knowledge base
   - Recall historical accuracy for similar tasks
   - Adaptive defaults based on task history

3. **Multi-Agent MAKER**
   - Each subordinate agent has MAKER access
   - Coordinate voting across agent hierarchy
   - Aggregate reliability across agent tree

4. **Visual Feedback**
   - MAKER voting progress in Agent Zero Web UI
   - Confidence indicators for responses
   - Historical reliability dashboard

---

## References

- [Agent Zero GitHub](https://github.com/agent0ai/agent-zero)
- [Agent Zero Architecture](https://www.agent-zero.ai/p/architecture/)
- [Agent Zero MCP Integration](https://deepwiki.com/frdel/agent-zero/4.2-mcp-integration-and-external-tools)
- [aiming-lab/Agent0 (Research)](https://github.com/aiming-lab/Agent0)
- [Agent0 Paper (arXiv:2511.16043)](https://arxiv.org/abs/2511.16043)
- [Model Context Protocol](https://github.com/modelcontextprotocol/servers)
- [MAKER Framework](https://github.com/zircote/maker-rs)

---

## Appendix: Agent Zero vs aiming-lab/Agent0 Comparison

| Aspect | Agent Zero (agent0ai) | Agent0 (aiming-lab) |
|--------|----------------------|---------------------|
| **Purpose** | Runtime AI assistant | Research on self-evolving agents |
| **Focus** | Task execution | Training methodology |
| **Tools** | Extensive (code, web, memory) | Python tool for reasoning |
| **MCP Support** | Yes (bidirectional) | No |
| **Extension Model** | Instruments, Extensions, Tools | Co-evolution framework |
| **Deployment** | Docker container | Research codebase |
| **Best For** | Production use | Academic research |
| **MAKER Integration** | High viability | Low viability |

**Decision:** Focus on Agent Zero (agent0ai) for practical integration.
