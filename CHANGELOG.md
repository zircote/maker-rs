# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SVG-VALIDATION.md**: Shared SVG post-write validation reference at `.claude/docs/SVG-VALIDATION.md`, eliminating duplicated instructions across skills

### Changed
- **validate-svg.js**: Modernized to ES2015+ style (const/let, arrow functions, template literals), removed dead code and unused parameters, simplified result construction
- **generate-assets.js**: Extracted `writeSvgPng` and `svgWrap` helpers to eliminate repeated `sharp(Buffer.from(svg)).png().toFile()` boilerplate across shape functions
- **svg-gantt-generator, infographic-generator**: Replaced inline validation instructions with a link to the shared `SVG-VALIDATION.md` reference
- **context-elicitor agent**: Simplified to a thin orchestration wrapper that delegates to the `/project-context` skill instead of restating its full interview protocol
- **project-architect agent**: Removed duplicated epic/story scaling rules; now references the `jira-structure` skill's scaling table for those columns
- **validate-artifacts workflow**: Simplified link checker by using `grep -oP` with a lookbehind to extract link targets directly, removing the two-stage `sed` pipeline
- **timeline-architect, documentation-architect agents**: Replaced inline SVG validation instructions with a link to the shared `SVG-VALIDATION.md` reference
- **build-deck.js**: Removed unused `html` variable read from slide files

### Fixed
- **validate-artifacts workflow**: Fixed broken link checker conditional that assigned a literal string instead of evaluating the bash expression

## [0.1.0] - 2026-01-30

### Added
- **SVG XML validation**: `workspace/validate-svg.js` linter that parses SVG files through `@xmldom/xmldom`, checks for well-formed XML, unescaped entities, and valid SVG structure (`xmlns`, dimensions, root element)
- **`validate-svg` npm script**: Run `npm run validate-svg` to lint all `.github/*.svg` files or pass specific paths
- **Mandatory post-write validation**: `svg-gantt-generator` and `infographic-generator` skills now require running the XML linter after writing SVG files, with a fix-and-retry loop until validation passes
- **Agent validation steps**: `timeline-architect` and `documentation-architect` agents include explicit SVG validation gates that block progression on malformed XML
- **Template scaffolding**: Initial project planning template structure with workspace tooling, `generate-assets.js` and `build-deck.js` scripts
- **Claude agents**: Project orchestration agents (`project-architect`, `context-elicitor`, `research-specialist`, `timeline-architect`, `governance-architect`, `metrics-architect`, `documentation-architect`)
- **Claude skills**: Full skill suite for artifact generation (`project-plan`, `gantt-chart`, `jira-structure`, `raci-chart`, `risk-register`, `severity-classification`, `best-practices`, `success-metrics`, `runbook-template`, `readme-generator`, `changelog`, `svg-gantt-generator`, `infographic-generator`, `executive-briefing`, `dependency-analyzer`, `domain-research`, `project-context`)
- **Claude commands**: Convenience commands for batch artifact generation (`generate-all`, `docs`, `metrics`, `plan`, `research`, `timeline`, `governance`, `validate`)
- **GitHub workflows**: CI workflows (`validate-artifacts`, `build-visuals`) and issue templates (`epic`, `story`, `risk`)
- **GitHub assets**: Social preview SVG and README infographic SVG
- **README**: Project documentation with badges, usage instructions, and artifact reference table

### Changed
- **Template restructure**: Moved `PROJECT-CONTEXT.md` template to `.claude/templates/PROJECT-CONTEXT.md`; `/project-context` skill writes populated output to root `PROJECT-CONTEXT.md`
- **Documentation relocated**: `README.md` and `docs/` moved under `.claude/`; root `README.md` is now a placeholder for template users
- **Execution plan command**: Simplified `/create-execution-plan` command
- Updated writer files (skill, command, agent) to reference `.claude/templates/` as the template source
- Updated documentation (CLAUDE.md, README.md, COMMAND-GUIDE.md) with template vs populated file distinction
- **SVG quality criteria**: `gantt-chart` skill quality checklist now includes XML validation as first gate
- **Documentation audit fixes**: Reconciled artifact count to 16 across all documents, completed single-artifact command list, added language tags to code blocks, genericized issue template phase names, annotated internal skills as non-user-invocable, added `validate-svg.js` to build commands, noted private exemplar repository

### Removed
- `QUICKSTART.md` — consolidated into `.claude/README.md` quick start section
