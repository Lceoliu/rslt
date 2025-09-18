from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    try:
        import yaml  # type: ignore

        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "PyYAML is required to load YAML config. Install via: pip install pyyaml"
        ) from e

