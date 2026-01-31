# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT open a public issue** for security vulnerabilities
2. Email the maintainers at: **security@maker-framework.dev** (or use GitHub Security Advisories)
3. Use [GitHub Security Advisories](https://github.com/zircote/maker-rs/security/advisories/new) to report privately

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days for critical issues

### Disclosure Policy

- We follow coordinated disclosure practices
- Public disclosure after fix is available (or 90 days, whichever comes first)
- Credit will be given to reporters (unless anonymity is requested)

## Security Model

### MCP Tool Security

MAKER operates as an MCP (Model Context Protocol) server. The security model assumes:

1. **User Responsibility**: Prompts provided to MAKER tools come from the calling agent (e.g., Claude Code). Users are responsible for the content of their prompts.

2. **Input Validation**: MAKER validates all inputs:
   - Maximum prompt length: 10,000 characters
   - Schema validation on all tool requests (`deny_unknown_fields`)
   - Red-flagging of malformed LLM outputs

3. **No Prompt Execution**: MAKER does not execute prompts as code. It passes them to LLM providers for text generation.

### Guardrails

MAKER implements several security guardrails:

#### Schema Enforcement
- All MCP tool inputs validated against strict JSON schemas
- Agent outputs validated with `StrictAgentOutput` schema
- Unknown fields rejected in strict mode

#### Red-Flagging
- Token length limits prevent runaway outputs
- Format violations detected and logged
- Malformed responses discarded (not repaired)

#### Microagent Isolation
- Each agent handles exactly one subtask (m=1)
- No history passed between agents
- State hash validation prevents corruption

#### Logging
- Schema violations logged at WARN level
- Content hashes logged (not full content) for privacy
- No sensitive data in logs

### Known Limitations

1. **Prompt Injection**: MAKER passes user prompts to LLM providers. If the underlying LLM is vulnerable to prompt injection, MAKER cannot fully mitigate this at the framework level. We recommend:
   - Using models with robust instruction-following
   - Implementing application-level prompt sanitization
   - Monitoring for unexpected outputs

2. **LLM Provider Security**: MAKER relies on external LLM APIs (OpenAI, Anthropic, Ollama). API key security and provider security practices are outside MAKER's scope.

3. **Cost Attacks**: Malicious users could attempt to inflate API costs through repeated voting requests. Rate limiting should be implemented at the application level.

## Security Checklist for Deployers

- [ ] Secure API keys in environment variables (not hardcoded)
- [ ] Implement rate limiting on MCP tool calls
- [ ] Monitor logs for unusual patterns
- [ ] Keep MAKER and dependencies updated
- [ ] Review LLM provider security practices
- [ ] Consider network isolation for sensitive deployments

## Dependencies

MAKER uses the following security-relevant dependencies:

- `reqwest`: HTTPS client for API calls
- `rmcp`: MCP protocol implementation
- `serde`: JSON serialization with schema validation
- `tokio`: Async runtime

We monitor dependencies for security advisories via `cargo audit`.

## Acknowledgments

We thank the security researchers who help keep MAKER secure through responsible disclosure.
