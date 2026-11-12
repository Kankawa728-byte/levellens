"""Shared fixtures. Tests never write into the real results or cache folders."""

import pytest


@pytest.fixture(autouse=True)
def isolated_folders(tmp_path, monkeypatch):
    """Point the results and cache folders at a temporary folder for every test."""
    monkeypatch.setenv("LEVELLENS_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("LEVELLENS_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("LEVELLENS_DEVICE", raising=False)
    return tmp_path
