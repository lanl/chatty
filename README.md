# chatty

A terminal UI chatbot for OpenAI-compatible endpoints, built with [Textual](https://textual.textualize.io/).

Designed for laptop and HPC environments where browser-based UIs aren't practical.

## Installation

### Standard (connected network)

```bash
git clone <repo-url>
cd chatty
uv sync
```

### Offline (air-gapped HPC)

```bash
# On connected machine: build wheelhouse
scripts/build-wheelhouse.sh

# Transfer wheelhouse/ to air-gapped system

# On air-gapped machine: install from wheelhouse
scripts/install-offline.sh
```

## Configuration

### Basic (laptop)

Set environment variables:

```bash
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="your-api-key"
export CHATTY_MODEL="gpt-4.1"  # optional
```

> **Note:** Environment variables are fine for personal laptops. On shared systems (HPC, servers), use `api_key_file` instead—see HPC section below.

Or create `~/.config/chatty/config.toml`:

```toml
base_url = "https://your-endpoint/v1"
model = "gpt-4.1"
temperature = 0.2
stream = true
system_prompt = "You are a helpful assistant."
```

### HPC / Institutional Networks

On shared systems, **do not use environment variables for API keys** (they appear in shell history, job logs, `/proc`).

```toml
# ~/.config/chatty/config.toml

# Use api_key_file instead of api_key
api_key_file = "/home/user/.secrets/chatty-api-key"  # chmod 600

# TLS: custom CA bundle for institutional endpoints
ca_bundle = "/etc/pki/tls/certs/institutional-ca.pem"
verify_tls = true  # default; set false only for controlled dev

# Proxy (if required by network)
http_proxy = "http://proxy.internal:8080"
no_proxy = "localhost,127.0.0.1"
```

Create the key file:
```bash
echo "your-api-key" > ~/.secrets/chatty-api-key
chmod 600 ~/.secrets/chatty-api-key
```

## Usage

```bash
# Start the chat UI
chatty chat

# Start with a query file (for long committee queries)
chatty chat --query-file committee-query.txt

# Test connectivity and configuration
chatty doctor

# Show resolved configuration (secrets redacted)
chatty print-config
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+C` | Quit |
| `Ctrl+T` | Toggle streaming |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+O` | Load query from file |
| `Ctrl+N` | New session |
| `Ctrl+Y` | Copy message to clipboard |
| `Enter` | Send message |
| `Shift+Enter` | Insert newline |
| `Esc` | Cancel generation |

## Features

- **Markdown rendering** with syntax-highlighted code blocks
- **Context tracking** — status bar shows token usage (e.g., "12K / 128K")
- **File loading** — load long queries from files (`Ctrl+O` or `--query-file`)
- **Streaming** — real-time token display with cancellation support
- **Offline-friendly** — works without RAG (v0.1), graceful errors with RAG (v0.3+)

## Development

```bash
uv sync --dev
pre-commit install
pytest
```

## License

TBD
