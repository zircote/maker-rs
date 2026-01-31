# Using MAKER for Coding Tasks

A practical guide to achieving reliable code generation with MAKER's error-correction framework.

## Why MAKER for Coding?

Code generation has unique challenges:
- Multiple valid solutions (different variable names, approaches)
- Syntax must be exactly correct
- Errors compound in multi-file changes
- Testing alone isn't enough for correctness

MAKER addresses these through:
- **AST-based matching**: Groups semantically equivalent code
- **Red-flag validation**: Catches syntax errors before counting votes
- **m=1 decomposition**: Breaks complex changes into atomic operations

## Setup

### Enable Code Matching

Build MAKER with tree-sitter support:

```bash
cargo build --release --features code-matcher
```

### Configure for Your Language

```json
{
  "matcher": {
    "type": "code",
    "language": "rust",  // rust, python, or javascript
    "threshold": 0.95
  }
}
```

## Core Patterns

### Pattern 1: Single Function Generation

Generate a single function with voting:

```
Configure MAKER with:
- matcher: code (rust)
- k_margin: 3

Use maker/vote for:
"Implement a function that checks if a string is a valid IPv4 address"

Then validate:
- Compiles with rustc
- Handles edge cases (empty string, extra dots, values > 255)
```

**What MAKER does:**
1. Samples multiple implementations
2. Parses each with tree-sitter
3. Groups by AST structure (ignoring variable names, comments)
4. Returns the implementation that wins by k_margin votes

### Pattern 2: Refactoring Workflow

For refactoring, use decomposition to ensure atomicity:

```
This is a refactoring task. Use MAKER decomposition:

Task: "Extract the database connection logic into a separate module"

1. Use maker/decompose to create subtasks:
   - Create new module file
   - Move connection struct
   - Update imports in affected files
   - Add re-export from parent module

2. For each subtask, use maker/vote with k_margin=3

3. After each subtask, use maker/validate to ensure:
   - Valid syntax
   - No duplicate definitions
   - All imports resolve
```

### Pattern 3: Multi-File Changes

MAKER's filesystem state tracks cross-file dependencies:

```
I need to rename a function across the codebase. Use MAKER with:
- Multi-file orchestration enabled
- Atomic commits

Files affected:
- src/lib.rs (definition)
- src/utils.rs (usage)
- tests/integration.rs (usage)

For each file:
1. Use maker/vote to generate the change
2. Validate syntax
3. Track dependencies (lib.rs must change first)
4. Commit atomically only if all changes valid
```

## Language-Specific Tips

### Rust

```
Configure MAKER for Rust:
{
  "matcher": {
    "type": "code",
    "language": "rust",
    "threshold": 0.95
  }
}

Voting considerations:
- High threshold (0.95) because Rust is strict
- k_margin=4 for functions with generics
- Always validate with rustc before accepting

Example:
"Implement Iterator for a custom struct that yields pairs from two slices"
```

### Python

```
Configure MAKER for Python:
{
  "matcher": {
    "type": "code",
    "language": "python",
    "threshold": 0.90  // Lower due to dynamic typing variance
  }
}

Voting considerations:
- Lower threshold accepts more stylistic variance
- k_margin=3 is usually sufficient
- Validate with ast.parse() or pylint

Example:
"Write a decorator that retries a function up to 3 times on exception"
```

### JavaScript/TypeScript

```
Configure MAKER for JavaScript:
{
  "matcher": {
    "type": "code",
    "language": "javascript",
    "threshold": 0.90
  }
}

Voting considerations:
- Handle async/await vs Promise variance
- Arrow functions vs function declarations
- Consider ES module vs CommonJS

Example:
"Implement a debounce function that waits 300ms before executing"
```

## Handling Common Scenarios

### Ambiguous Requirements

When requirements are ambiguous, MAKER's voting reveals consensus:

```
Use maker/vote with k_margin=5 and max_samples=20 for:
"Implement error handling for this function"

<function code here>

After voting, explain:
- What error handling approach won
- What alternatives were considered
- Why the winner was most common
```

### Performance-Critical Code

For performance-sensitive code, add validation:

```
Generate optimized sorting implementation with MAKER:

1. Use maker/vote with k_margin=3 for:
   "Implement a cache-friendly sorting algorithm for large arrays"

2. Validate with maker/validate:
   - Token limit: 100 lines
   - No unnecessary allocations
   - Uses iterators where possible

3. After implementation, benchmark and report Big-O complexity
```

### Test Generation

Generate tests alongside implementation:

```
For this function, use MAKER to generate:

1. The implementation (maker/vote, code matcher)
2. Unit tests (maker/vote, code matcher)
3. Property-based test cases (maker/vote, exact matcher for test names)

Validate that tests:
- Cover edge cases
- Are independent
- Have clear assertions
```

## Decomposition Strategies

MAKER's CodingDecomposer provides three levels:

### Function-Level Decomposition

Best for: Adding new features, modules

```
Decompose at function level:
"Add user authentication to the API"

Results in subtasks:
- Implement User struct
- Implement password hashing
- Implement login endpoint
- Implement session management
- Add authentication middleware
```

### Block-Level Decomposition

Best for: Refactoring within functions

```
Decompose at block level:
"Refactor this 100-line function to be more readable"

Results in subtasks:
- Extract validation logic (lines 10-25)
- Extract database query (lines 30-50)
- Extract response formatting (lines 60-80)
- Compose in main function
```

### Line-Level Decomposition

Best for: Precise edits, bug fixes

```
Decompose at line level:
"Fix the off-by-one error in this loop"

Results in single atomic change:
- Change: line 42, `i < len` to `i <= len`
```

## Red Flags for Code

MAKER automatically rejects samples with:

| Red Flag | Detection | Example |
|----------|-----------|---------|
| Syntax error | Tree-sitter parse failure | Missing semicolon |
| Infinite loop pattern | Static analysis | `while true {}` without break |
| Token limit exceeded | Length check | Function > 500 tokens |
| Format violation | Schema check | Missing required fields |

## Workflow: Complete Feature Implementation

```
# Complete workflow for adding a feature

## Phase 1: Design
Use maker/vote to design the interface:
"Design the public API for a rate limiter module"

## Phase 2: Decompose
Use maker/decompose to create implementation plan:
"Implement the rate limiter with sliding window algorithm"

## Phase 3: Implement
For each subtask, use maker/vote with code matcher:
- Token bucket structure
- Add token method
- Check rate method
- Configuration builder

## Phase 4: Validate
For each implementation, use maker/validate:
- Syntax check (tree-sitter)
- Token limit (< 100 lines per function)
- Style check (clippy/pylint rules)

## Phase 5: Test
Use maker/vote to generate tests:
- Unit tests for each method
- Integration test for full flow
- Edge case tests

## Phase 6: Document
Use maker/vote with embedding matcher:
- Generate docstrings
- Write module documentation
- Add usage examples
```

## Troubleshooting

### Code Matcher Not Grouping Equivalents

**Problem:** Semantically equivalent code not being grouped

**Solutions:**
1. Lower the threshold: `0.90` instead of `0.95`
2. Ensure language is set correctly
3. Check that tree-sitter parsed successfully

### Voting Not Converging for Complex Code

**Problem:** max_samples reached without winner

**Solutions:**
1. Break into smaller subtasks (m=1 decomposition)
2. Add more specific requirements in prompt
3. Use semantic matcher for conceptual questions, code matcher for implementation

### Syntax Errors Passing Validation

**Problem:** Invalid code accepted

**Solutions:**
1. Enable the `code-matcher` feature flag
2. Add explicit compilation step in validation
3. Use project-specific linter in validation chain

## See Also

- [../PROMPT-ENGINEERING.md](../PROMPT-ENGINEERING.md) - General prompting strategies
- [DATA-ANALYSIS.md](./DATA-ANALYSIS.md) - MAKER for data tasks
- [ML-PIPELINES.md](./ML-PIPELINES.md) - MAKER for ML workflows
