from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    data: dict[str, Any]

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.data["project"]["output_dir"])

    @property
    def seed(self) -> int:
        return int(self.data["project"].get("random_seed", 42))

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def input_path(self, key: str) -> Path:
        return self.resolve(self.data["inputs"][key])

    def task_names(self, include_descriptive: bool = True) -> list[str]:
        tasks = self.data["tasks"]
        names = list(tasks.get("primary", [])) + list(tasks.get("secondary", []))
        if include_descriptive:
            names += list(tasks.get("descriptive_only", []))
        return names


def load_config(path: str | Path, root: str | Path | None = None) -> PipelineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    workspace = Path(root).resolve() if root else config_path.parent.parent.resolve()
    return PipelineConfig(root=workspace, data=data)

