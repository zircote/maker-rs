---
# Documentation Review Configuration
# Updated by /doc-setup on 2026-01-30

doc_paths:
  - "*.md"
  - "src/**/*.rs"
  - "examples/**/*.rs"
  - "tests/**/*.rs"
  - "benches/**/*.rs"
  - ".github/ISSUE_TEMPLATE/*.md"
  - ".claude/CLAUDE.md"
  - "docs/**/*"
  - "_plan/*.md"

ignore:
  - "node_modules/"
  - "vendor/"
  - "target/"
  - "workspace/node_modules/"
  - "*.local.md"
  - ".claude/skills/"
  - ".claude/agents/"
  - ".claude/commands/"
  - ".claude/templates/"
  - ".claude/docs/"

standards:
  require_description: true
  max_heading_depth: 4
  require_code_examples: true
  check_links: true
  check_spelling: true
  check_rust_doc_comments: true
  require_public_api_docs: true

api_docs:
  openapi_path: null
  asyncapi_path: null
  generate_from_code: true
  rust_doc_path: "target/doc/maker/index.html"

site_generator:
  type: none
  config_path: null

output:
  verbosity: detailed
  format: markdown
---

# Project Documentation Notes

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a Rust library and MCP server for zero-error long-horizon LLM agent execution.

## Key Documentation Locations

- **README.md**: Primary user-facing documentation with quickstart, architecture, and MCP tool reference
- **CHANGELOG.md**: Keep a Changelog format, tracks releases
- **SECURITY.md**: Vulnerability reporting and security model
- **src/**: Rust doc comments (`///` and `//!`) on all public APIs
- **examples/**: Runnable examples with inline documentation
- **_plan/**: Project planning artifacts (RUNSHEET, EXECUTION-PLAN, CONTEXT-BRIEF)

## Rust Documentation Standards

- All public functions, structs, enums, and traits must have `///` doc comments
- Module-level `//!` comments required for each module file
- Doc examples (`/// # Examples`) required for key public APIs
- `cargo doc --no-deps` must build without warnings (`RUSTDOCFLAGS=-Dwarnings`)

## Quality Expectations

- Internal markdown links must resolve to actual files
- Code examples in README must be valid Rust (or clearly marked as pseudocode)
- Mermaid diagrams must render in GitHub markdown
- CHANGELOG follows Keep a Changelog format with ISO 8601 dates
- API documentation generated via `cargo doc` must be complete and warning-free
