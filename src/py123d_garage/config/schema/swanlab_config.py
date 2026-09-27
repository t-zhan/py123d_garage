from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwanLabConfig:
    enabled: bool = False
    project: str = "k-space"
    mode: str = "local"
    name: str | None = None
