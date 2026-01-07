"""Tests for diagnostics/doctor command."""

from pathlib import Path

import pytest
import respx
from pydantic import SecretStr

from chatty.config import Config, ConfigWithSources
from chatty.diagnostics import (
    EXIT_API_INCOMPATIBLE,
    EXIT_AUTH_FAILURE,
    EXIT_MISSING_CONFIG,
    EXIT_OK,
    DiagnosticResult,
    check_config,
    check_connectivity,
    format_doctor_result,
    run_doctor,
)


@pytest.fixture
def valid_config() -> ConfigWithSources:
    """Create a valid config for testing."""
    config = Config(
        base_url="https://api.test.com/v1",
        api_key=SecretStr("test-key"),
        model="gpt-4",
    )
    sources = {
        "base_url": "env:OPENAI_BASE_URL",
        "api_key": "env:OPENAI_API_KEY",
        "model": "default",
        "stream": "default",
        "verify_tls": "default",
        "ca_bundle": "default",
    }
    return ConfigWithSources(config=config, sources=sources)


@pytest.fixture
def missing_url_config() -> ConfigWithSources:
    """Config missing base_url."""
    config = Config(
        base_url="",
        api_key=SecretStr("test-key"),
    )
    sources = {"base_url": "default", "api_key": "env:OPENAI_API_KEY"}
    return ConfigWithSources(config=config, sources=sources)


@pytest.fixture
def missing_key_config() -> ConfigWithSources:
    """Config missing api_key."""
    config = Config(
        base_url="https://api.test.com/v1",
        api_key=None,
    )
    sources = {"base_url": "env:OPENAI_BASE_URL", "api_key": "default"}
    return ConfigWithSources(config=config, sources=sources)


def test_check_config_valid(valid_config: ConfigWithSources) -> None:
    """Test config check with valid config."""
    results = check_config(valid_config)

    base_url_check = next(r for r in results if "base_url" in r.message)
    assert base_url_check.passed is True

    api_key_check = next(r for r in results if "api_key" in r.message)
    assert api_key_check.passed is True


def test_check_config_missing_base_url(missing_url_config: ConfigWithSources) -> None:
    """Test config check with missing base_url."""
    results = check_config(missing_url_config)

    base_url_check = next(r for r in results if "base_url" in r.message)
    assert base_url_check.passed is False
    assert "Set OPENAI_BASE_URL" in (base_url_check.detail or "")


def test_check_config_missing_api_key(missing_key_config: ConfigWithSources) -> None:
    """Test config check with missing api_key."""
    results = check_config(missing_key_config)

    api_key_check = next(r for r in results if "api_key" in r.message)
    assert api_key_check.passed is False


def test_check_config_api_key_file(tmp_path: Path) -> None:
    """Test config check with api_key_file."""
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key")
    key_file.chmod(0o600)

    config = Config(base_url="https://api.test.com/v1", api_key_file=key_file)
    config_with_sources = ConfigWithSources(config=config, sources={"api_key_file": "config.toml"})

    results = check_config(config_with_sources)
    api_key_check = next(r for r in results if "api_key" in r.message)
    assert api_key_check.passed is True
    assert "api_key_file" in (api_key_check.detail or "")


def test_check_config_api_key_file_bad_permissions(tmp_path: Path) -> None:
    """Test config check with insecure api_key_file."""
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key")
    key_file.chmod(0o644)  # Insecure

    config = Config(base_url="https://api.test.com/v1", api_key_file=key_file)
    config_with_sources = ConfigWithSources(config=config, sources={})

    results = check_config(config_with_sources)
    api_key_check = next(r for r in results if "api_key_file" in r.message)
    assert api_key_check.passed is False
    assert "Insecure permissions" in (api_key_check.detail or "")


def test_check_config_api_key_file_not_found() -> None:
    """Test config check with missing api_key_file."""
    config = Config(base_url="https://api.test.com/v1", api_key_file=Path("/nonexistent/key"))
    config_with_sources = ConfigWithSources(config=config, sources={})

    results = check_config(config_with_sources)
    api_key_check = next(r for r in results if "api_key_file" in r.message)
    assert api_key_check.passed is False
    assert "File not found" in (api_key_check.detail or "")


def test_check_config_ca_bundle_exists(tmp_path: Path) -> None:
    """Test config check with existing CA bundle."""
    ca_file = tmp_path / "ca.pem"
    ca_file.write_text("CERTIFICATE")

    config = Config(
        base_url="https://api.test.com/v1",
        api_key=SecretStr("key"),
        ca_bundle=str(ca_file),
    )
    config_with_sources = ConfigWithSources(config=config, sources={"ca_bundle": "config.toml"})

    results = check_config(config_with_sources)
    ca_check = next(r for r in results if "ca_bundle" in r.message)
    assert ca_check.passed is True


def test_check_config_ca_bundle_not_found() -> None:
    """Test config check with missing CA bundle."""
    config = Config(
        base_url="https://api.test.com/v1",
        api_key=SecretStr("key"),
        ca_bundle="/nonexistent/ca.pem",
    )
    config_with_sources = ConfigWithSources(config=config, sources={})

    results = check_config(config_with_sources)
    ca_check = next(r for r in results if "ca_bundle" in r.message)
    assert ca_check.passed is False
    assert "File not found" in (ca_check.detail or "")


@respx.mock
def test_check_connectivity_success(valid_config: ConfigWithSources) -> None:
    """Test connectivity check success."""
    respx.get("https://api.test.com/v1/models").respond(200, json={"data": []})
    respx.post("https://api.test.com/v1/chat/completions").respond(
        200, json={"choices": [{"message": {"content": "ok"}}]}
    )

    results = check_connectivity(valid_config)

    assert any("TLS handshake successful" in r.message for r in results)
    assert any("Authentication valid" in r.message for r in results)
    assert any("supports /chat/completions" in r.message for r in results)


@respx.mock
def test_check_connectivity_auth_failure(valid_config: ConfigWithSources) -> None:
    """Test connectivity check with auth failure."""
    respx.get("https://api.test.com/v1/models").respond(200, json={"data": []})
    respx.post("https://api.test.com/v1/chat/completions").respond(
        401, json={"error": {"message": "Invalid API key"}}
    )

    results = check_connectivity(valid_config)

    assert any("Authentication failed" in r.message for r in results)


@respx.mock
def test_check_connectivity_404(valid_config: ConfigWithSources) -> None:
    """Test connectivity check with 404."""
    respx.get("https://api.test.com/v1/models").respond(200, json={"data": []})
    respx.post("https://api.test.com/v1/chat/completions").respond(404)

    results = check_connectivity(valid_config)

    assert any("does not support /chat/completions" in r.message for r in results)


def test_check_connectivity_no_base_url(missing_url_config: ConfigWithSources) -> None:
    """Test connectivity skipped when no base_url."""
    results = check_connectivity(missing_url_config)

    assert len(results) == 1
    assert "skipped" in results[0].message


def test_run_doctor_all_pass(valid_config: ConfigWithSources) -> None:
    """Test run_doctor with all checks passing."""
    with respx.mock:
        respx.get("https://api.test.com/v1/models").respond(200, json={"data": []})
        respx.post("https://api.test.com/v1/chat/completions").respond(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

        result = run_doctor(valid_config)

    assert result.exit_code == EXIT_OK


def test_run_doctor_missing_config(missing_url_config: ConfigWithSources) -> None:
    """Test run_doctor with missing config."""
    result = run_doctor(missing_url_config)

    assert result.exit_code == EXIT_MISSING_CONFIG
    assert len(result.connectivity_checks) == 0  # Skipped


def test_run_doctor_auth_failure(valid_config: ConfigWithSources) -> None:
    """Test run_doctor with auth failure."""
    with respx.mock:
        respx.get("https://api.test.com/v1/models").respond(200)
        respx.post("https://api.test.com/v1/chat/completions").respond(401)

        result = run_doctor(valid_config)

    assert result.exit_code == EXIT_AUTH_FAILURE


def test_run_doctor_api_incompatible(valid_config: ConfigWithSources) -> None:
    """Test run_doctor with API incompatibility."""
    with respx.mock:
        respx.get("https://api.test.com/v1/models").respond(200)
        respx.post("https://api.test.com/v1/chat/completions").respond(404)

        result = run_doctor(valid_config)

    assert result.exit_code == EXIT_API_INCOMPATIBLE


def test_run_doctor_skip_connectivity(valid_config: ConfigWithSources) -> None:
    """Test run_doctor with skip_connectivity flag."""
    result = run_doctor(valid_config, skip_connectivity=True)

    assert result.exit_code == EXIT_OK
    assert len(result.connectivity_checks) == 0


def test_format_doctor_result_success() -> None:
    """Test formatting successful doctor result."""
    from chatty.diagnostics import DoctorResult

    result = DoctorResult(
        exit_code=EXIT_OK,
        config_checks=[DiagnosticResult(passed=True, message="base_url: https://test")],
        connectivity_checks=[DiagnosticResult(passed=True, message="TLS handshake successful")],
    )

    output = format_doctor_result(result)

    assert "✓ base_url" in output
    assert "✓ TLS handshake" in output
    assert "All checks passed" in output


def test_format_doctor_result_failure() -> None:
    """Test formatting failed doctor result."""
    from chatty.diagnostics import DoctorResult

    result = DoctorResult(
        exit_code=EXIT_MISSING_CONFIG,
        config_checks=[
            DiagnosticResult(
                passed=False, message="base_url: (not set)", detail="Set OPENAI_BASE_URL"
            )
        ],
        connectivity_checks=[],
    )

    output = format_doctor_result(result)

    assert "✗ base_url" in output
    assert "Set OPENAI_BASE_URL" in output
    assert "Some checks failed" in output


def test_format_doctor_result_verbose() -> None:
    """Test verbose formatting shows all details."""
    from chatty.diagnostics import DoctorResult

    result = DoctorResult(
        exit_code=EXIT_OK,
        config_checks=[
            DiagnosticResult(
                passed=True, message="base_url: https://test", detail="from: env:OPENAI_BASE_URL"
            )
        ],
        connectivity_checks=[],
    )

    output = format_doctor_result(result, verbose=True)

    assert "from: env:OPENAI_BASE_URL" in output
