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
| `Ctrl+E` | Submit query |
| `Ctrl+Q` | Quit |
| `Ctrl+C` | Copy last response |
| `Ctrl+S` | Save session |
| `Ctrl+L` | Load session |
| `Ctrl+N` | New session |
| `Ctrl+O` | Load file |
| `Esc` | Interrupt generation |
| `Enter` | Insert newline |

**Power user shortcuts** (hidden from footer, accessible via `Ctrl+P` palette):
- `Ctrl+R` — Regenerate last response
- `Ctrl+T` — Toggle streaming mode

## Features

- **Markdown rendering** with syntax-highlighted code blocks
- **Context tracking** — status bar shows token usage (e.g., "12K / 128K")
- **File loading** — load long queries from files (`Ctrl+O` or `--query-file`)
- **Streaming** — real-time token display with cancellation support
- **Copy to clipboard** — `Ctrl+C` copies last response (file fallback for HPC)
- **Session management** — save and resume conversations across runs
- **Transcript logging** — save conversations to JSONL files for review
- **Offline-friendly** — works without RAG (v0.1), graceful errors with RAG (v0.3+)

## Session Management

Save conversations and resume them later. Sessions store the full conversation state including messages and system prompt.

**Keyboard shortcuts:**
- `Ctrl+S` — Save current session
- `Ctrl+L` — Browse and load saved sessions

**CLI option:**
```bash
chatty chat --session ./sessions/session-abc123.json
```

**Configuration:**
```toml
# In chatty.toml
session_path = "./sessions"  # default: repo-local
```

**Environment variable:**
```bash
export CHATTY_SESSION_PATH="~/.config/chatty/sessions"
```

**Supported locations:**
- `./sessions` — repo-local (default, good for development)
- `~/.config/chatty/sessions` — user config dir
- `~/.local/share/chatty/sessions` — XDG data dir

Session names are auto-generated from the first user message. Files are JSON format:
```json
{
  "version": "1.0",
  "metadata": {"name": "Tell me about Python", "message_count": 4, ...},
  "messages": [...]
}
```

---

## Transcript Logging

Save conversations to JSONL files for later review, auditing, or debugging.

```toml
# In chatty.toml or ~/.config/chatty/config.toml
transcript_enabled = true
transcript_path = "./transcripts"  # or any path
```

**Supported locations:**
- `./transcripts` — repo-local (good for development)
- `~/.config/chatty/transcripts` — user config dir (default)
- `~/.local/share/chatty/transcripts` — XDG data dir

**Output format:** One JSON object per line:
```json
{"timestamp": "2026-01-09T10:32:15.123", "role": "user", "content": "Hello"}
{"timestamp": "2026-01-09T10:32:18.456", "role": "assistant", "content": "Hi!", "model": "gpt-4.1", "response_time_s": 2.3}
```

When transcript logging is enabled, chatty shows the transcript file path on startup.

## Tips

### Adjusting Font Size

Font size is controlled by your terminal emulator, not chatty. To increase readability:

**macOS Terminal / iTerm2:**
- `Cmd +` to zoom in, `Cmd -` to zoom out
- Or: Preferences → Profiles → Text → Font size (14-16pt recommended)

**VS Code integrated terminal:**
- `Cmd +` to zoom in
- Or: Settings → Terminal › Integrated: Font Size

**Linux terminals:**
- `Ctrl + Shift +` typically
- Or: Preferences → Profile → Font

### Local Endpoints (LM Studio, Ollama)

When using local endpoints, make sure your `base_url` includes `/v1`:

**LM Studio:**
```toml
base_url = "http://localhost:1234/v1"
```

**Ollama:**
```toml
base_url = "http://localhost:11434/v1"
```

> **Note:** These are the default ports. If you've configured a different port, adjust accordingly.

## Complete Example Config

Create `chatty.toml` in your project directory or `~/.config/chatty/config.toml`:

<details>
<summary>📄 Click to expand full example</summary>

```toml
# chatty configuration
# Config file search order:
#   1. CHATTY_CONFIG env var (explicit override)
#   2. ./chatty.toml (repo-local)
#   3. ~/.config/chatty/config.toml (user default)

# =============================================================================
# REQUIRED: API Endpoint
# =============================================================================

# Base URL for OpenAI-compatible API
base_url = "https://your-endpoint/v1"

# API Key (use ONE of these methods):
# Option 1: Direct key (OK for personal laptop, not for shared systems)
api_key = "your-api-key-here"

# Option 2: Key file (recommended for HPC/shared systems)
# api_key_file = "/home/user/.secrets/chatty-api-key"

# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Model name (depends on your endpoint)
model = "gpt-4.1"

# Temperature (0.0 = deterministic, 1.0+ = creative)
temperature = 0.2

# =============================================================================
# BEHAVIOR
# =============================================================================

# Enable streaming responses (show tokens as they arrive)
stream = true

# System prompt (sets assistant behavior)
system_prompt = "You are a helpful assistant."

# Request timeout in seconds
timeout_s = 60

# =============================================================================
# TLS / NETWORK (for institutional endpoints)
# =============================================================================

# Custom CA bundle for institutional endpoints
# ca_bundle = "/etc/pki/tls/certs/institutional-ca.pem"

# Disable TLS verification (only for controlled dev environments!)
# verify_tls = false

# HTTP proxy (if required by your network)
# http_proxy = "http://proxy.internal:8080"

# Hosts that bypass the proxy
# no_proxy = "localhost,127.0.0.1"

# =============================================================================
# SESSION PERSISTENCE
# =============================================================================

# Session location options (Ctrl+S to save, Ctrl+L to load):
#   "./sessions"                        # repo-local (default)
#   "~/.config/chatty/sessions"         # user config dir
#   "~/.local/share/chatty/sessions"    # XDG data dir
# session_path = "./sessions"

# =============================================================================
# COPY FALLBACK (for headless HPC)
# =============================================================================

# Where to save copied content when clipboard is unavailable:
#   "./copies"                          # repo-local (default)
#   "~/.config/chatty/copies"           # user config dir
# copy_fallback_path = "./copies"

# =============================================================================
# TRANSCRIPT LOGGING
# =============================================================================

# Enable JSONL transcript logging (saves all conversations)
# transcript_enabled = false

# Transcript location options:
#   "./transcripts"                     # repo-local (good for development)
#   "~/.config/chatty/transcripts"      # user config dir (default)
#   "~/.local/share/chatty/transcripts" # XDG data dir (for large archives)
# transcript_path = "~/.config/chatty/transcripts"
```

</details>

## Development

```bash
uv sync --dev
pre-commit install
pytest
```

## License

TBD
