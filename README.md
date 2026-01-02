# chatty

A terminal UI chatbot for OpenAI-compatible endpoints, built with [Textual](https://textual.textualize.io/).

Designed for laptop and HPC environments where browser-based UIs aren't practical.

## Installation

```bash
# Clone and install with uv
git clone <repo-url>
cd chatty
uv sync
```

## Configuration

Set environment variables:

```bash
export OPENAI_BASE_URL="https://your-endpoint/v1"
export OPENAI_API_KEY="your-api-key"
export CHATTY_MODEL="gpt-4.1"  # optional, uses endpoint default
```

Or create `~/.config/chatty/config.toml`:

```toml
base_url = "https://your-endpoint/v1"
model = "gpt-4.1"
temperature = 0.2
stream = true
system_prompt = "You are a helpful assistant."
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
| `Ctrl+K` | Compress context |
| `Ctrl+N` | New session |
| `Ctrl+Y` | Copy message to clipboard |
| `Esc` | Cancel generation |

## Features

- **Markdown rendering** with syntax-highlighted code blocks
- **Context tracking** — status bar shows token usage (e.g., "12K / 128K")
- **Context compression** — LLM summarizes history when context fills up
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
