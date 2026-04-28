"""Per-epoch metrics, dumpable to JSON."""
from __future__ import annotations

from pathlib import Path
import json


class TrainingHistory:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, **kwargs) -> None:
        self.records.append(kwargs)

    def to_dict(self) -> dict[str, list]:
        if not self.records:
            return {}
        keys: list[str] = []
        for r in self.records:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        return {k: [r.get(k) for r in self.records] for k in keys}

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def __len__(self) -> int:
        return len(self.records)
