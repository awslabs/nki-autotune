"""Pytest configuration for caller-selected remote acceptance-test hosts."""

from __future__ import annotations

from typing import cast

import pytest

_HOST_OPTIONS = (("trn2_hosts", "--trn2-hosts"), ("cpu_hosts", "--cpu-hosts"))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register remote host options without repository-specific defaults."""
    group = parser.getgroup("remote hosts")
    group.addoption(
        "--trn2-hosts",
        nargs="+",
        default=None,
        dest="trn2_hosts",
        metavar="HOST",
        help="SSH destinations for Trn2 profiling tests",
    )
    group.addoption(
        "--cpu-hosts",
        nargs="+",
        default=None,
        dest="cpu_hosts",
        metavar="HOST",
        help="SSH destinations for CPU simulation tests",
    )


def _configured_hosts(config: pytest.Config, destination: str, option: str) -> tuple[str, ...]:
    """Return and validate hosts supplied through one repeatable option."""
    raw_hosts = cast(list[str] | None, config.getoption(destination))
    hosts = tuple(raw_hosts or ())
    invalid_host = next((host for host in hosts if not host.strip()), None)
    if invalid_host is not None:
        raise pytest.UsageError(f"{option} requires a non-empty SSH destination")
    return hosts


def _required_hosts(config: pytest.Config, destination: str, option: str) -> tuple[str, ...]:
    """Return configured hosts or fail with the option needed by the selected tests."""
    hosts = _configured_hosts(config, destination, option)
    if not hosts:
        raise pytest.UsageError(f"selected tests require {option} HOST")
    return hosts


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Reject selected remote tests whose corresponding host option is absent."""
    missing_options: list[str] = []
    for destination, option in _HOST_OPTIONS:
        needs_hosts = any(isinstance(item, pytest.Function) and destination in item.fixturenames for item in items)
        if needs_hosts and not _configured_hosts(config, destination, option):
            missing_options.append(f"{option} HOST")
    if missing_options:
        raise pytest.UsageError(f"selected tests require {', '.join(missing_options)}")


@pytest.fixture(scope="session")
def trn2_hosts(pytestconfig: pytest.Config) -> tuple[str, ...]:
    """Return caller-selected Trn2 profiling hosts."""
    return _required_hosts(pytestconfig, "trn2_hosts", "--trn2-hosts")


@pytest.fixture(scope="session")
def cpu_hosts(pytestconfig: pytest.Config) -> tuple[str, ...]:
    """Return caller-selected CPU simulation hosts."""
    return _required_hosts(pytestconfig, "cpu_hosts", "--cpu-hosts")
