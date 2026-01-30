---
name: changelog
description: Create or update the version history following Keep a Changelog 1.1.0 format. Use as the final step after all artifacts are generated.
allowed-tools: Read, Grep, Glob
---

Generate or update `CHANGELOG.md` at the repository root.

## Process

1. List all generated artifacts with creation timestamps
2. Format entries following Keep a Changelog 1.1.0
3. Link to all referenced files

## Required Format

```markdown
# Changelog

All notable changes to this project's planning artifacts are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- [artifact-name](link): Brief description

### Changed
- [artifact-name](link): What changed
```

## Quality Criteria

- [ ] Follows Keep a Changelog 1.1.0 format exactly
- [ ] All artifacts listed with file links
- [ ] Grouped by type (Added, Changed, Fixed)
- [ ] ISO 8601 date stamps
