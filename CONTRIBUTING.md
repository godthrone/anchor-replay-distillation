# Contributing to ARD

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/your-org/anchor-replay-distillation.git
cd anchor-replay-distillation
uv sync --extra dev
```

## Code Quality

```bash
# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/ard/

# Linting
uv run ruff check src/ tests/
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes and ensure all tests pass
3. Run type checking and linting
4. Add/update tests for new functionality
5. Submit a PR with a clear description

## Commit Convention

Use conventional commits: `type: description`

- `feat:` — new feature
- `fix:` — bug fix
- `docs:` — documentation
- `refactor:` — code restructuring
- `test:` — test additions/changes
- `chore:` — build/config changes

## License

By contributing, you agree that your contributions will be licensed
under the MIT License.