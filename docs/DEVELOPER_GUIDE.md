# Developer Guide

This guide covers setting up a development environment for chatty using uv.

## Prerequisites

- **Python 3.11+** — Required for chatty development
- **uv** — The only supported package manager for this project

### Installing uv

If you don't have uv installed:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv
```

For other installation methods, see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Setting Up the Development Environment

### 1. Clone the Repository

```bash
git clone <repo-url>
cd chatty
```

### 2. Create the Virtual Environment

Create a virtual environment named `.venv`:

```bash
uv venv --python 3.11 .venv
```

> **Note:** uv creates the environment in `.venv/` by default. The `--python 3.11` flag ensures you're using the correct Python version.

### 3. Activate the Environment

```bash
# macOS / Linux
source .venv/bin/activate

# Or use uv run to execute commands without activating
uv run <command>
```

### 4. Install Dependencies

Install all dependencies including development tools:

```bash
uv sync --dev
```

This installs:
- **Runtime dependencies:** textual, httpx, httpx-sse, pydantic, pydantic-settings, typer, rich
- **Optional:** tiktoken (token counting)
- **Development:** pytest, pytest-asyncio, respx, ruff, mypy, pre-commit

## Pre-commit Hooks

Install the pre-commit hooks to ensure code quality on every commit:

```bash
pre-commit install
```

The hooks run automatically on `git commit` and check:
- Code formatting (ruff format)
- Linting (ruff check)
- Type hints (mypy)

To run hooks manually on all files:

```bash
pre-commit run --all-files
```

## Running Tests

### Unit Tests

Run the full test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_config.py -v
```

### Coverage Reports

Generate a coverage report:

```bash
pytest --cov=chatty --cov-report=term-missing
```

### Integration Tests

Integration tests require a real LLM endpoint and are skipped by default:

```bash
# Set the test endpoint
export CHATTY_TEST_ENDPOINT="https://your-test-endpoint/v1"
export CHATTY_TEST_API_KEY="your-test-key"

# Run integration tests
pytest -m integration
```

## Code Quality Tools

### Linting and Formatting

Check for issues:

```bash
ruff check .
```

Auto-fix issues:

```bash
ruff check --fix .
```

Format code:

```bash
ruff format .
```

### Type Checking

Run mypy for type checking:

```bash
mypy src/chatty
```

### Textual Async Patterns

When working with Textual's async APIs, be aware of these patterns that mypy enforces:

#### Workers: Pass Method References, Not Coroutines

```python
# ❌ WRONG — mypy error: unused-coroutine
self.run_worker(self._send_message(), ...)

# ✅ CORRECT — pass method reference, Textual invokes it
self.run_worker(self._send_message, ...)
```

Textual's `run_worker()` accepts `Callable[[], Coroutine]` (a method reference) rather than a coroutine object. This lets Textual manage when to invoke and cancel the coroutine.

#### Actions: Use post_message Instead of Calling Async Methods

```python
# ❌ WRONG — action_submit() returns unused coroutine
input_widget.action_submit()

# ✅ CORRECT — post the event through Textual's message queue
self.post_message(Input.Submitted(input_widget, content))
```

Widget action methods like `action_submit()` are async. Calling them directly creates an unused coroutine. Instead, post the corresponding event through Textual's message queue.

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes

3. Run tests and checks:
   ```bash
   pytest
   ruff check .
   mypy src/chatty
   ```

4. Commit (pre-commit hooks will run automatically):
   ```bash
   git commit -m "Add my feature"
   ```

### Running chatty Locally

Run chatty commands using `uv run`:

```bash
# Start the chat UI
uv run chatty chat

# Run diagnostics
uv run chatty doctor

# Show configuration
uv run chatty print-config
```

Or activate the environment first:

```bash
source .venv/bin/activate
chatty chat
```

### Adding Dependencies

Add a runtime dependency:

```bash
uv add httpx
```

Add a development dependency:

```bash
uv add --dev pytest-cov
```

## Project Structure

```
chatty/
├── src/chatty/          # Main package
│   ├── __init__.py
│   ├── cli.py           # Typer CLI entrypoint
│   ├── config.py        # Configuration management
│   ├── client/          # HTTP client layer
│   ├── core/            # Business logic
│   ├── rag/             # RAG provider protocol
│   └── ui/              # Textual TUI
├── tests/               # Test suite
│   ├── conftest.py      # Shared fixtures
│   ├── test_*.py        # Unit tests
│   └── snapshots/       # UI snapshots
├── docs/                # Documentation
├── scripts/             # Build/install scripts
├── pyproject.toml       # Project configuration
└── README.md
```

## Offline Install (Air-Gapped HPC)

For air-gapped systems where `uv sync` cannot reach PyPI:

### On Connected Machine (Build Wheelhouse)

```bash
# Build wheelhouse with all dependencies
# (automatically generates requirements.lock if not present)
./scripts/build-wheelhouse.sh
```

This creates `wheelhouse/` containing all wheel files.

### Transfer to Air-Gapped Machine

Transfer the following to the target machine:
- `wheelhouse/` directory (all .whl files)
- `requirements.lock` file
- `scripts/` directory
- `src/` directory
- `pyproject.toml`

You can use any transfer method: USB drive, `scp`, shared filesystem, etc.

### On Air-Gapped Machine (Install)

```bash
# Install from local wheelhouse (no network required)
./scripts/install-offline.sh
```

This will:
1. Create a virtual environment if needed
2. Install all packages from the local wheelhouse
3. Install chatty itself

### Verify Installation

```bash
source .venv/bin/activate
chatty --help
chatty doctor  # Check configuration
```

### Notes

- The wheelhouse is platform-specific (macOS wheels won't work on Linux)
- If targeting a different platform, build the wheelhouse on that platform type
- `requirements.lock` is generated from `uv.lock` and includes exact hashes
- Some packages may download as source tarballs if no wheel exists for your platform

## Troubleshooting

### uv sync fails

Ensure you have the correct Python version:

```bash
python --version  # Should be 3.11+
uv venv --python 3.11 .venv
uv sync --dev
```

### Pre-commit hooks fail

Run the hooks manually to see detailed errors:

```bash
pre-commit run --all-files
```

### Tests fail with import errors

Ensure dependencies are installed:

```bash
uv sync --dev
```

### tiktoken installation fails

tiktoken is optional. If it fails to install (common on some HPC systems), chatty will still work with degraded token counting:

```bash
# Install without tiktoken
uv sync --dev --no-install tiktoken
```
