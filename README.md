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
```

## Usage

```bash
# Start the chat UI
chatty chat

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
| `Esc` | Cancel generation |

## Development

```bash
uv sync --dev
pre-commit install
pytest
```

## License

TBD
