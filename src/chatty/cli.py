"""Typer CLI entrypoint for chatty."""

from pathlib import Path
from typing import Annotated

import typer

from chatty.config import format_config_with_sources, load_config
from chatty.diagnostics import format_doctor_result, run_doctor

app = typer.Typer(
    name="chatty",
    help="A terminal UI chatbot for OpenAI-compatible endpoints.",
    no_args_is_help=True,
)


@app.command()
def chat(
    query_file: Annotated[
        typer.FileText | None,
        typer.Option("--query-file", "-q", help="Load initial query from file"),
    ] = None,
    session: Annotated[
        Path | None,
        typer.Option("--session", help="Load saved session from file"),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", "-u", help="LLM endpoint URL", envvar="OPENAI_BASE_URL"),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name"),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", "-k", help="API key (prefer api_key_file on HPC)"),
    ] = None,
    api_key_file: Annotated[
        Path | None,
        typer.Option("--api-key-file", help="Path to API key file (chmod 600)"),
    ] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", "-t", help="Sampling temperature"),
    ] = None,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Disable streaming"),
    ] = False,
    system_prompt: Annotated[
        str | None,
        typer.Option("--system-prompt", "-s", help="System prompt"),
    ] = None,
) -> None:
    """Launch the chat UI."""
    # Build CLI overrides dict (only non-None values)
    cli_overrides: dict[str, object] = {}
    if base_url is not None:
        cli_overrides["base_url"] = base_url
    if model is not None:
        cli_overrides["model"] = model
    if api_key is not None:
        cli_overrides["api_key"] = api_key
    if api_key_file is not None:
        cli_overrides["api_key_file"] = api_key_file
    if temperature is not None:
        cli_overrides["temperature"] = temperature
    if no_stream:
        cli_overrides["stream"] = False
    if system_prompt is not None:
        cli_overrides["system_prompt"] = system_prompt

    config_with_sources = load_config(cli_overrides=cli_overrides)

    # Get query file path if provided
    query_file_path: str | None = None
    if query_file is not None:
        query_file_path = query_file.name

    # Get session file path if provided
    session_path: Path | None = None
    if session is not None:
        session_path = session.expanduser()
        if not session_path.exists():
            typer.echo(f"Error: Session file not found: {session_path}", err=True)
            raise typer.Exit(1)

    # Launch the UI
    from chatty.ui.app import main as run_chat_ui

    run_chat_ui(
        query_file=query_file_path,
        session_file=session_path,
        config_with_sources=config_with_sources,
    )


@app.command()
def doctor(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed output"),
    ] = False,
    skip_connectivity: Annotated[
        bool,
        typer.Option("--skip-connectivity", help="Skip network connectivity checks"),
    ] = False,
) -> None:
    """Run connectivity diagnostics."""
    config_with_sources = load_config()
    result = run_doctor(config_with_sources, verbose=verbose, skip_connectivity=skip_connectivity)
    typer.echo(format_doctor_result(result, verbose=verbose))
    raise SystemExit(result.exit_code)


@app.command(name="print-config")
def print_config(
    show_sources: Annotated[
        bool,
        typer.Option("--sources", "-s", help="Show where each value comes from"),
    ] = True,
) -> None:
    """Show resolved configuration (secrets redacted)."""
    config_with_sources = load_config()

    if show_sources:
        typer.echo("Configuration:")
        typer.echo(format_config_with_sources(config_with_sources))
    else:
        typer.echo("Configuration:")
        for key, value in config_with_sources.config.display().items():
            typer.echo(f"  {key}: {value}")


if __name__ == "__main__":
    app()
