# Retrieve your first passage

Use a laptop suite containing Chatty 0.4.6 or later. Check with `chatty --version` in a fresh terminal. Run these commands from any directory on an Apple Silicon Mac or Windows x64 laptop. Internet access is needed for the initial model download.

## Select a workspace

`chatty workspace` prints the selected path. The default is `Chatty/workspace` in your home directory, outside the replaceable runtime. To use an existing index, set the top-level `rag_workspace` in `~/.config/chatty/config.toml` to its workspace directory (the parent of `indices` and `sqlite`). Use an absolute path, for example `rag_workspace = 'D:\Research library'` on Windows. Chatty uses `CHATTY_RAG_WORKSPACE`, then `LITKIT_WORKSPACE`, then the configured path. The shell commands below give LitKit the same selection.

For the example, choose a new workspace. The following command creates `example/articles/rain-gauge.xml` from the installed package. **If you already have a built index**, omit `--example`, check the models, then go straight to the retrieval check and ask about your own corpus.

**macOS** (zsh or Bash)

```bash
workspace="$(chatty workspace --example)"
export LITKIT_WORKSPACE="$workspace"
export LITKIT_FORCE_DEVICE=cpu
```

**Windows** (PowerShell)

```powershell
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = 'utf-8'
$workspace = chatty workspace --example
$env:LITKIT_WORKSPACE = $workspace
$env:LITKIT_FORCE_DEVICE = 'cpu'
```

The remaining commands work in either shell. When opening a fresh terminal for another build, repeat the workspace selection without `--example`.

## Cache the embedding models

```sh
chatty models
chatty models --download
```

SPECTER2 embeds paper titles and abstracts (about 441 MB). All-MPNet-base-v2 embeds passages (about 439 MB). The status lists both models, their cache location, and whether their files are ready. Downloads go under `hf_cache/hub` in the workspace. To share an existing model cache, set `HF_HOME` to the absolute **parent** of its `hub` directory before running these commands. Rerunning the download reuses complete snapshots.

## Package and index the article

```sh
text-fetch tarball create --xml-dir "$workspace/example/articles" --out "$workspace/example/tars/sample.tar.gz"
litkit --build-only --faiss-writer --tar-dir "$workspace/example/tars" --papers-index flat --chunks-index flat
```

text-fetch validates the native JATS XML and writes the archive. LitKit reads that archive and writes two FAISS indexes and a SQLite database. For your own corpus, point `--xml-dir` at your JATS files. PDF-to-JATS conversion is a separate text-fetch workflow that needs a running GROBID service, commonly started with Docker. Native JATS goes directly into the archive.

## Verify retrieval

```sh
chatty doctor --retrieval
chatty retrieve "How much rain did the copper gauge collect?"
```

Look for the source **Rainfall at the cedar grove** and the passage “The copper rain gauge collected seventeen millimeters of rain during the overnight storm.” Both commands work without API credentials. If the local check names a missing index file, check the selected workspace and the LitKit build output.

## Connect an endpoint for chat

Edit the top-level settings in your user configuration (`~/.config/chatty/config.toml` on both platforms). Create the directory and file if needed. Use your provider's base URL and model ID.

```toml
rag_provider = "litkit"
base_url = "https://api.openai.com/v1"
model = "gpt-4.1"
api_key_file = "~/.config/chatty/api-key"
```

Store the key on its own line in that file. On macOS, run `chmod 600 ~/.config/chatty/api-key`. To supply the key through `OPENAI_API_KEY`, omit the `api_key_file` setting. If you selected a custom workspace with a shell variable, set `rag_workspace` in the same config to use it in future terminals. Run `chatty doctor` to check local retrieval and endpoint connectivity, then `chatty chat` and ask the same question.
