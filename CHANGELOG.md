# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**RAG Provider Integration**
- `rag_provider` config option (`"none"` default, `"litkit"` in v0.3)
- `CHATTY_RAG_PROVIDER` environment variable
- `get_provider()` factory function in `chatty.rag`
- `UnknownProviderError` for invalid provider names
- `RAGProvider.augment()` now called in chat flow (NullProvider passthrough)
- `last_rag_metadata` stored for future citation display

**Session Save/Load**
- `Ctrl+S` saves current conversation to JSON file
- `--session` CLI option loads a saved session on startup
- `session_path` config option (default: `~/.config/chatty/sessions`)
- `CHATTY_SESSION_PATH` environment variable
- Auto-generated session names from first user message
- Session format: JSON with version field for future compatibility

**Session History Browser**
- `Ctrl+L` opens modal to browse saved sessions
- OptionList displays session name, date, message count
- Enter or double-click to load, Escape to cancel
- Empty state shows helpful hint to save with Ctrl+S

**Copy to Clipboard**
- `Ctrl+Y` copies last assistant response to system clipboard
- Toast notification confirms copy action
- `copy_fallback_path` config for headless HPC (default: `./copies`)
- `CHATTY_COPY_FALLBACK_PATH` environment variable
- Falls back to timestamped file when clipboard unavailable

### Changed

- Chat flow now uses `RAGProvider.augment()` instead of building messages directly
- User messages added to conversation after successful LLM response (enables query rewriting in v0.3)
- `action_regenerate()` updated to work with deferred message flow
- `Ctrl+N` (new session) now resets session tracking

### Developer Notes

- v0.2 acceptance criterion: `RAGProvider.augment()` called on every user message
- v0.2 acceptance criterion: `NullProvider` produces identical behavior to v0.1
- Session module: `SessionMetadata`, `Session`, `save_session()`, `load_session()`, `list_sessions()`

## [0.1.0] - 2026-01-09

Initial release of chatty — a terminal UI chatbot for OpenAI-compatible endpoints.

### Added

**Core Features**
- Full-screen TUI built with Textual
- OpenAI-compatible async client with SSE streaming
- Markdown rendering with syntax-highlighted code blocks
- Configuration via TOML file, environment variables, and CLI flags
- `chatty chat` — Launch interactive chat UI
- `chatty doctor` — Diagnose configuration and connectivity
- `chatty print-config` — Display resolved configuration (secrets redacted)

**HPC & Institutional Network Support**
- TLS/CA bundle configuration (`ca_bundle`, `verify_tls`)
- Secure API key storage via `api_key_file` (chmod 600)
- HTTP proxy support (`http_proxy`, `no_proxy`)
- Offline installation via wheelhouse for air-gapped systems
- Scripts: `build-wheelhouse.sh`, `install-offline.sh`

**Input & Interaction**
- Keyboard shortcuts (Ctrl+E submit, Ctrl+Q quit, Ctrl+N new session, etc.)
- Load queries from files (`Ctrl+O` or `--query-file`)
- Multi-line paste support
- Interrupt generation with `Esc`

**Context & Session Management**
- In-memory conversation history
- Context window tracking (server-reported or tiktoken estimate)
- New session (`Ctrl+N`) clears history

**Feedback & Polish**
- Color-coded messages (user/assistant/system)
- "Thinking..." spinner with elapsed time
- Response time shown after generation
- Inline error display with actionable hints
- Auto-retry with countdown on transient failures

**Transcript Logging**
- JSONL transcript files with timestamps
- Optional metadata: model, response_time_s, tokens
- Configurable via `transcript_enabled` and `transcript_path`

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+E` | Submit query |
| `Ctrl+Q` | Quit |
| `Ctrl+N` | New session |
| `Ctrl+O` | Load file |
| `Esc` | Interrupt generation |
| `Enter` | Insert newline |
| `Ctrl+R` | Regenerate last response |
| `Ctrl+T` | Toggle streaming mode |

### Configuration

Config file search order:
1. `CHATTY_CONFIG` environment variable
2. `./chatty.toml` (repo-local)
3. `~/.config/chatty/config.toml` (user default)

### Notes

- **UI test coverage deferred** — Requires `pytest-textual-snapshot` setup; will be addressed when UI stabilizes.
- **RAGProvider protocol exists** — `NullProvider` implemented as passthrough; wiring into chat flow deferred to v0.2.
- **tiktoken is optional** — Token counting degrades gracefully if unavailable.

[0.1.0]: https://github.com/your-org/chatty/releases/tag/v0.1.0
