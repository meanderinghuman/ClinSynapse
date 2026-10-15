"""Isolate tests from the machine: no CS_*/OPEN_AI_* env vars and no .env file.

CI sets CS_REPLAY_ONLY for the smoke eval, and developers have real keys in .env; neither may
change what a unit test sees.
"""

import os

import pytest

from clinsynapse.config import Settings

_PREFIXES = ("CS_", "OPEN_AI_", "ARC_", "GROQ_", "LANGFUSE_")


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith(_PREFIXES):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
