# Contributing to MAKER Framework

Thank you for your interest in contributing to the MAKER Framework!

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/maker-rs.git`
3. **Create a branch**: `git checkout -b feature/your-feature-name`

## Development Setup

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run lints
cargo clippy
```

## Code Standards

### Test Coverage
- **95% minimum test coverage is mandatory** for all code
- Run coverage: `cargo llvm-cov --html`
- All new features must include comprehensive tests

### Code Style
- Follow Rust idioms and conventions
- Use `cargo fmt` before committing
- Address all `cargo clippy` warnings
- Document public APIs with doc comments

### Commit Messages
- Use conventional commit format: `type(scope): description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
- Keep first line under 72 characters

## Pull Request Process

1. **Ensure tests pass**: `cargo test`
2. **Check coverage**: Must maintain 95%+ coverage
3. **Update documentation** if adding new features
4. **Reference related issues** in PR description
5. **Request review** from maintainers

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Follow the issue templates provided
- Include reproduction steps for bugs
- Reference the severity classification in [SEVERITY-CLASSIFICATION.md](./SEVERITY-CLASSIFICATION.md)

## Code of Conduct

Be respectful, constructive, and inclusive. We're building reliability tooling for the AI community.

## Questions?

Open a GitHub Discussion or reach out to the maintainers.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
