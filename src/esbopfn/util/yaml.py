"""YAML helpers with safe loading and stable dumping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path | str) -> Any:
    """Load one YAML document with the safe loader. An empty file yields an empty dict."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {} if data is None else data


def dump_yaml(data: Any, path: Path | str) -> None:
    """Write ``data`` as block-style YAML, keeping the key order, and create parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False, allow_unicode=True)


def dumps_yaml(data: Any) -> str:
    """Return ``data`` as a block-style YAML string."""
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
