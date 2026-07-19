"""YAML config loading helpers used by every script in the project."""

from pathlib import Path

import yaml


def project_root() -> Path:
    """Return the repository root (two levels above this file)."""
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Load a YAML config file into a plain dict.

    Relative paths are resolved against the repository root so scripts work
    no matter which directory they are launched from.
    """
    path = Path(path)
    if not path.is_absolute():
        path = project_root() / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel: str | Path) -> Path:
    """Resolve a config-declared relative path against the repository root."""
    rel = Path(rel)
    return rel if rel.is_absolute() else project_root() / rel
