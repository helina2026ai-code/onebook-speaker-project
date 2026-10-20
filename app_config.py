"""Resolve deployment settings without exposing or changing stored secrets."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from dotenv import dotenv_values

from qa_engine import ROOT, Settings

KEY_NAMES = ("SUPABASE_SERVICE_ROLE_KEY", "SUPABASE_ANON_KEY")
CONFIG_KEYS = ("SUPABASE_URL", *KEY_NAMES, "QA_MATCH_THRESHOLD", "QA_ALLOW_LOCAL_FALLBACK")


def _merge_layer(values: dict, layer: Mapping) -> None:
    # A credential supplied by a higher-priority source replaces lower-source
    # aliases too, so a local service key cannot override a cloud anon key.
    if any(name in layer for name in KEY_NAMES):
        for name in KEY_NAMES:
            values.pop(name, None)
    values.update({name: layer[name] for name in CONFIG_KEYS if name in layer})


def read_settings(*, secrets: Mapping | None = None, env: Mapping | None = None, dotenv_path: Path | None = None) -> Settings:
    """Priority: Streamlit Secrets > process environment > local .env > defaults."""
    values = {}
    _merge_layer(values, {key: value for key, value in dotenv_values(dotenv_path or ROOT / ".env").items() if key in CONFIG_KEYS and value is not None})
    process_env = os.environ if env is None else env
    _merge_layer(values, process_env)
    if secrets is not None:
        _merge_layer(values, secrets)
    return Settings.from_mapping(values)
