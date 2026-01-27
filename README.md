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
| `Ctrl+P` | Submit query |
| `Ctrl+O` | Load file |
| `Esc` | Interrupt / Close search |
| `Ctrl+F` | Find (search in chat) |
| `Ctrl+C` | Copy last response |
| `Ctrl+S` | Save session |
| `Ctrl+L` | Load session |
| `Ctrl+N` | New session |
| `Ctrl+Q` | Quit |
| `Enter` | Insert newline |
| `F1` | Help |

**Power user shortcuts** (hidden from footer):
- `Ctrl+E` — Export Markdown
- `Ctrl+R` — Regenerate last response
- `Ctrl+T` — Toggle streaming mode
- `Ctrl+G` — Switch model

**Search mode:** `Ctrl+F` opens search. Press again to jump to next match. `Esc` to close.

**Help:** Press `F1` to open the in-app help showing all shortcuts, current config, and tips.

## Features

- **Markdown rendering** with syntax-highlighted code blocks
- **Context tracking** — status bar shows token usage (e.g., "12K / 128K")
- **Search in chat** — `Ctrl+F` to find text in conversation history with word highlighting
- **File loading** — load long queries from files (`Ctrl+O` or `--query-file`)
- **Streaming** — real-time token display with cancellation support
- **Copy to clipboard** — `Ctrl+C` copies last response (file fallback for HPC)
- **Session management** — save and resume conversations across runs
- **Transcript logging** — save conversations to JSONL files for review
- **Context compression** — `Ctrl+J` compresses long conversations, `Ctrl+Y` undoes
- **RAG support** — query pre-built scientific literature corpora via litkit (v0.4+)
- **Offline-friendly** — works without RAG, graceful errors with RAG

## Session Management

Save conversations and resume them later. Sessions store the full conversation state including messages and system prompt.

**Keyboard shortcuts:**
- `Ctrl+S` — Save current session (prompts for name on first save)
- `Ctrl+L` — Browse and load saved sessions
- `r` — Rename selected session (in session browser)
- `d` — Delete selected session (in session browser, with confirmation)

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

---

## RAG Mode (Retrieval-Augmented Generation)

chatty can query a pre-built scientific literature corpus to ground LLM responses in source documents. This feature is **optional** and requires:

1. **litkit** — The retrieval engine (separate package)
2. **A pre-built vector store** — FAISS indices + SQLite database

> **Important:** chatty only **queries** existing indices. Building indices is done via the `litkit` CLI (see [Building a Vector Store](#building-a-vector-store)).

### Installing RAG Dependencies

```bash
# Install chatty with RAG dependencies (faiss-cpu, numpy)
uv add chatty[rag]

# Then install litkit (choose based on your access):

# Option A: Local development (editable install)
cd ~/Code/litkit && uv pip install -e .

# Option B: Public GitHub (coming soon)
# uv add litkit
```

### Configuring RAG

Add to your `chatty.toml`:

```toml
[rag]
provider = "litkit"              # Enable RAG (default: "none")
workspace = "~/litkit/workspace" # Path to pre-built indices (optional, auto-detects)
top_papers = 500                 # Stage 1: papers to shortlist
top_chunks = 30                  # Stage 2: chunks for LLM context
```

Or use environment variables:

```bash
export CHATTY_RAG_PROVIDER="litkit"
export CHATTY_RAG_WORKSPACE="~/litkit/workspace"
```

### Building a Vector Store

chatty requires a pre-built vector store. Build one using litkit:

```bash
# 1. Prepare your papers (JATS/NXML XML in tar archives)
mkdir -p ~/litkit/workspace/tar_shards
cp your-papers.tar ~/litkit/workspace/tar_shards/

# 2. Build the index (Mac/workstation)
cd ~/Code/litkit
source .venv/bin/activate
litkit --build-only --faiss-writer \
       --tar-dir workspace/tar_shards \
       --papers-index flat \
       --chunks-index flat

# 3. Verify the build
ls ~/litkit/workspace/indices/
# Should show: papers.faiss, chunks.faiss
ls ~/litkit/workspace/sqlite/
# Should show: litkit.sqlite3
```

For detailed litkit documentation, see the litkit repository:
- `LITKIT_MAC_GUIDE.md` — Mac/workstation setup
- `LITKIT_CLUSTER_GUIDE.md` — HPC cluster deployment

### Using RAG

Once configured, chatty automatically retrieves relevant documents for each query:

```bash
chatty chat
# Type: "What are the mechanisms of HIV infection?"
# chatty retrieves relevant papers → injects context → LLM responds with citations
```

### Technical Note: Subprocess Isolation

Chatty runs litkit retrieval in a completely isolated subprocess using `subprocess.run(close_fds=True)`. This is required for compatibility with Textual's terminal I/O, which creates file descriptors that conflict with Python's multiprocessing.

**What this means:**
- Each query spawns a fresh Python subprocess for retrieval
- The subprocess loads litkit, performs embedding + FAISS search, returns JSON results
- First query takes ~30 seconds (model loading); subsequent queries spawn fresh subprocesses

**Impact:** Slightly higher latency per query compared to in-process retrieval, but guarantees stability with Textual's asyncio-based UI. FAISS vector search remains the main bottleneck for large corpora.

**RAG Keyboard Shortcuts (v0.4+):**

| Key | Action |
|-----|--------|
| `Ctrl+I` | Toggle inspect mode (preview context before LLM) |

---

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

# Context window size (tokens)
context_window = 128000

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
# MARKDOWN EXPORT
# =============================================================================

# Where to save exported Markdown files (Ctrl+E):
#   "./exports"                         # repo-local (default)
#   "~/.config/chatty/exports"          # user config dir
# export_path = "./exports"

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

## Known Limitations

### Reasoning / Thinking Models

Models that expose chain-of-thought reasoning (DeepSeek R1, QwQ, Apriel Thinker, etc.) will output their internal reasoning directly in the response. Chatty does not filter or hide this content because:

- There is no standard format across providers (some use `<think>` tags, others use markers like `[BEGIN FINAL RESPONSE]`, etc.)
- Parsing free-form text for thinking markers is brittle and breaks when models update
- The only robust approach is using structured API responses, which most local/open models don't support

**Workaround:** Use non-reasoning variants of models (e.g., `gpt-4.1` instead of `o3-mini`) if you want cleaner output.

## Windows Support (Experimental)

Chatty should work on Windows, but **has not been tested**. Feedback and bug reports from Windows users are welcome.

### Requirements

- Windows 10 or 11
- **Windows Terminal** (required for proper rendering — legacy `cmd.exe` won't work)
- PowerShell 7+ (recommended) or Windows PowerShell 5.1
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```powershell
git clone <repo-url>
cd chatty
uv sync
chatty chat
```

### Configuration

**Environment variables** (PowerShell syntax):

```powershell
$env:OPENAI_BASE_URL = "https://your-endpoint/v1"
$env:OPENAI_API_KEY = "your-api-key"
$env:CHATTY_MODEL = "gpt-4.1"
```

**Config file location:**

```
%USERPROFILE%\.config\chatty\config.toml
```

Or use `chatty.toml` in the project directory.

### Known Caveats

1. **Not tested** — There may be edge cases with terminal I/O, keyboard shortcuts, or clipboard handling
2. **Offline install scripts are bash-only** — For air-gapped Windows systems, use WSL or manually install from the wheelhouse:
   ```powershell
   pip install --no-index --find-links=wheelhouse chatty
   ```
3. **Keyboard shortcuts** — Some shortcuts may conflict with PowerShell defaults (e.g., `Ctrl+C` for copy vs interrupt)
4. **Path separators** — Config file paths in `chatty.toml` should use forward slashes (`/`) or escaped backslashes (`\\`)

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Garbled display / missing colors | Use Windows Terminal, not `cmd.exe` |
| `faiss-cpu` fails to install | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |
| Clipboard not working | Chatty will fallback to file-based copy (see `copy_fallback_path` config) |
| `chatty` command not found | Ensure uv's bin directory is in `$env:PATH` |

## License

TBD
