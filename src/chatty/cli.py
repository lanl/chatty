"""Typer CLI entrypoint for chatty."""

import typer

app = typer.Typer(
    name="chatty",
    help="A terminal UI chatbot for OpenAI-compatible endpoints.",
    no_args_is_help=True,
)


@app.command()
def chat(
    query_file: typer.FileText | None = typer.Option(
        None, "--query-file", "-q", help="Load initial query from file"
    ),
) -> None:
    """Launch the chat UI."""
    # TODO: Implement chat UI launch
    _ = query_file  # Will be used when UI is implemented
    typer.echo("Chat UI not yet implemented")


@app.command()
def doctor() -> None:
    """Run connectivity diagnostics."""
    # TODO: Implement diagnostics
    typer.echo("Doctor not yet implemented")


@app.command(name="print-config")
def print_config() -> None:
    """Show resolved configuration (secrets redacted)."""
    # TODO: Implement config display
    typer.echo("Print config not yet implemented")


if __name__ == "__main__":
    app()
