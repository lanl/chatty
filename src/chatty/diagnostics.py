"""Connectivity diagnostics for chatty doctor command."""

from __future__ import annotations

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from chatty.client.http import build_sync_client

if TYPE_CHECKING:
    from chatty.config import ConfigWithSources


# Exit codes per ARCHITECTURE.md spec
EXIT_OK = 0
EXIT_MISSING_CONFIG = 1
EXIT_AUTH_FAILURE = 2
EXIT_NETWORK_FAILURE = 3
EXIT_API_INCOMPATIBLE = 4


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""

    passed: bool
    message: str
    detail: str | None = None


@dataclass
class DoctorResult:
    """Aggregate result of all doctor checks."""

    exit_code: int
    config_checks: list[DiagnosticResult]
    connectivity_checks: list[DiagnosticResult]


def check_config(config_with_sources: ConfigWithSources) -> list[DiagnosticResult]:
    """Check configuration validity."""
    results: list[DiagnosticResult] = []
    config = config_with_sources.config
    sources = config_with_sources.sources

    # Check base_url
    if config.base_url:
        results.append(
            DiagnosticResult(
                passed=True,
                message=f"base_url: {config.base_url}",
                detail=f"from: {sources.get('base_url', 'unknown')}",
            )
        )
    else:
        results.append(
            DiagnosticResult(
                passed=False,
                message="base_url: (not set)",
                detail="Set OPENAI_BASE_URL or add base_url to config.toml",
            )
        )

    # Check api_key or api_key_file
    if config.api_key_file:
        file_path = config.api_key_file.expanduser()
        if file_path.exists():
            # Check permissions
            mode = file_path.stat().st_mode & 0o777
            if mode & 0o077:
                results.append(
                    DiagnosticResult(
                        passed=False,
                        message=f"api_key_file: {file_path}",
                        detail=f"Insecure permissions ({oct(mode)}). Run: chmod 600 {file_path}",
                    )
                )
            else:
                results.append(
                    DiagnosticResult(
                        passed=True,
                        message="api_key: ********",
                        detail=f"from: api_key_file → {file_path}",
                    )
                )
        else:
            results.append(
                DiagnosticResult(
                    passed=False,
                    message=f"api_key_file: {file_path}",
                    detail="File not found",
                )
            )
    elif config.api_key:
        results.append(
            DiagnosticResult(
                passed=True,
                message="api_key: ********",
                detail=f"from: {sources.get('api_key', 'unknown')}",
            )
        )
    else:
        results.append(
            DiagnosticResult(
                passed=False,
                message="api_key: (not set)",
                detail="Set OPENAI_API_KEY or api_key_file",
            )
        )

    # Check CA bundle if specified
    if config.ca_bundle:
        ca_path = Path(config.ca_bundle)
        if ca_path.exists():
            results.append(
                DiagnosticResult(
                    passed=True,
                    message=f"ca_bundle: {config.ca_bundle}",
                    detail=f"from: {sources.get('ca_bundle', 'unknown')}",
                )
            )
        else:
            results.append(
                DiagnosticResult(
                    passed=False,
                    message=f"ca_bundle: {config.ca_bundle}",
                    detail="File not found",
                )
            )
    else:
        results.append(
            DiagnosticResult(
                passed=True,
                message="ca_bundle: system",
                detail="Using system CA certificates",
            )
        )

    # Show other config values
    results.append(
        DiagnosticResult(
            passed=True,
            message=f"model: {config.model}",
            detail=f"from: {sources.get('model', 'unknown')}",
        )
    )
    results.append(
        DiagnosticResult(
            passed=True,
            message=f"stream: {config.stream}",
            detail=f"from: {sources.get('stream', 'unknown')}",
        )
    )
    results.append(
        DiagnosticResult(
            passed=True,
            message=f"verify_tls: {config.verify_tls}",
            detail=f"from: {sources.get('verify_tls', 'unknown')}",
        )
    )

    return results


def check_connectivity(config_with_sources: ConfigWithSources) -> list[DiagnosticResult]:
    """Check network connectivity to the endpoint."""
    results: list[DiagnosticResult] = []
    config = config_with_sources.config

    # Skip connectivity checks if config is invalid
    if not config.base_url:
        results.append(
            DiagnosticResult(
                passed=False,
                message="TLS handshake: skipped",
                detail="base_url not configured",
            )
        )
        return results

    try:
        # Use shared HTTP client builder
        with build_sync_client(config, timeout=10.0) as client:
            # Step 1: TLS handshake (just connect)
            try:
                # Try to reach the models endpoint (doesn't require auth usually)
                client.get(f"{config.base_url}/models")
                results.append(
                    DiagnosticResult(
                        passed=True,
                        message="TLS handshake successful",
                    )
                )
            except ssl.SSLError as e:
                results.append(
                    DiagnosticResult(
                        passed=False,
                        message="TLS handshake failed",
                        detail=str(e),
                    )
                )
                return results
            except httpx.ConnectError as e:
                results.append(
                    DiagnosticResult(
                        passed=False,
                        message="Connection failed",
                        detail=str(e),
                    )
                )
                return results

            # Step 2: Authentication check
            try:
                api_key = config.get_api_key()
            except ValueError as e:
                results.append(
                    DiagnosticResult(
                        passed=False,
                        message="Authentication: could not load API key",
                        detail=str(e),
                    )
                )
                return results

            headers = {"Authorization": f"Bearer {api_key}"}

            # Step 3: Test /chat/completions endpoint
            try:
                chat_response = client.post(
                    f"{config.base_url}/chat/completions",
                    json={
                        "model": config.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                    },
                    headers=headers,
                )

                if chat_response.status_code == 401:
                    results.append(
                        DiagnosticResult(
                            passed=False,
                            message="Authentication failed",
                            detail="Invalid API key",
                        )
                    )
                    return results
                elif chat_response.status_code == 403:
                    results.append(
                        DiagnosticResult(
                            passed=False,
                            message="Authentication failed",
                            detail="API key does not have access to this endpoint",
                        )
                    )
                    return results

                results.append(
                    DiagnosticResult(
                        passed=True,
                        message="Authentication valid",
                    )
                )

                if chat_response.status_code == 404:
                    results.append(
                        DiagnosticResult(
                            passed=False,
                            message="Endpoint does not support /chat/completions",
                            detail="Got 404. Check that base_url points to an OpenAI-compatible API.",
                        )
                    )
                    return results

                if chat_response.status_code >= 400:
                    # Try to get error message from response
                    try:
                        error_data = chat_response.json()
                        error_msg = error_data.get("error", {}).get("message", chat_response.text)
                    except Exception:
                        error_msg = chat_response.text[:200]

                    results.append(
                        DiagnosticResult(
                            passed=False,
                            message=f"API error: {chat_response.status_code}",
                            detail=error_msg,
                        )
                    )
                    return results

                # Success!
                results.append(
                    DiagnosticResult(
                        passed=True,
                        message="Endpoint supports /chat/completions",
                    )
                )

            except httpx.TimeoutException:
                results.append(
                    DiagnosticResult(
                        passed=False,
                        message="Request timeout",
                        detail="Endpoint did not respond in time",
                    )
                )
                return results

    except Exception as e:
        results.append(
            DiagnosticResult(
                passed=False,
                message="Unexpected error",
                detail=str(e),
            )
        )

    return results


def run_doctor(
    config_with_sources: ConfigWithSources,
    *,
    verbose: bool = False,  # noqa: ARG001 - reserved for future use
    skip_connectivity: bool = False,
) -> DoctorResult:
    """Run all diagnostic checks and return aggregated result."""
    config_checks = check_config(config_with_sources)

    # Determine if we should skip connectivity
    config_passed = all(
        c.passed for c in config_checks if "base_url" in c.message or "api_key" in c.message
    )

    if skip_connectivity or not config_passed:
        connectivity_checks = []
        exit_code = EXIT_MISSING_CONFIG if not config_passed else EXIT_OK
    else:
        connectivity_checks = check_connectivity(config_with_sources)

        # Determine exit code from connectivity results
        if not connectivity_checks:
            exit_code = EXIT_OK
        elif any("TLS" in c.message and not c.passed for c in connectivity_checks) or any(
            "Connection" in c.message and not c.passed for c in connectivity_checks
        ):
            exit_code = EXIT_NETWORK_FAILURE
        elif any("Authentication" in c.message and not c.passed for c in connectivity_checks):
            exit_code = EXIT_AUTH_FAILURE
        elif any("does not support" in c.message for c in connectivity_checks) or any(
            "API error" in c.message for c in connectivity_checks
        ):
            exit_code = EXIT_API_INCOMPATIBLE
        elif all(c.passed for c in connectivity_checks):
            exit_code = EXIT_OK
        else:
            exit_code = EXIT_NETWORK_FAILURE

    return DoctorResult(
        exit_code=exit_code,
        config_checks=config_checks,
        connectivity_checks=connectivity_checks,
    )


def _should_show_detail(check: DiagnosticResult, verbose: bool) -> bool:
    """Determine if we should show the detail for a check."""
    if not check.detail:
        return False
    return verbose or not check.passed


def format_doctor_result(result: DoctorResult, *, verbose: bool = False) -> str:
    """Format doctor result for display."""
    lines = []

    lines.append("Configuration:")
    for check in result.config_checks:
        symbol = "✓" if check.passed else "✗"
        lines.append(f"  {symbol} {check.message}")
        if _should_show_detail(check, verbose):
            lines.append(f"    {check.detail}")

    if result.connectivity_checks:
        lines.append("")
        lines.append("Connectivity:")
        for check in result.connectivity_checks:
            symbol = "✓" if check.passed else "✗"
            lines.append(f"  {symbol} {check.message}")
            if _should_show_detail(check, verbose):
                lines.append(f"    {check.detail}")

    lines.append("")
    if result.exit_code == EXIT_OK:
        lines.append("All checks passed.")
    else:
        lines.append(f"Some checks failed (exit code {result.exit_code}).")

    return "\n".join(lines)
