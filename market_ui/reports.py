from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CURRENT_RESEARCH_REPORT_SCHEMA = 4


@dataclass(frozen=True, slots=True)
class ResearchReportInventory:
    current: tuple[Path, ...]
    stale: tuple[Path, ...]
    invalid: tuple[Path, ...]


def inventory_research_reports(artifacts_directory: Path) -> ResearchReportInventory:
    current: list[Path] = []
    stale: list[Path] = []
    invalid: list[Path] = []
    for path in artifacts_directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            valid_shape = (
                isinstance(payload, dict)
                and isinstance(payload.get("test_regression"), dict)
                and isinstance(payload.get("test_strategy"), dict)
            )
            schema = int(payload.get("schema_version", 0)) if valid_shape else 0
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            invalid.append(path)
            continue
        if not valid_shape:
            continue
        if schema == CURRENT_RESEARCH_REPORT_SCHEMA:
            current.append(path)
        elif 0 < schema < CURRENT_RESEARCH_REPORT_SCHEMA:
            stale.append(path)
        else:
            invalid.append(path)

    def modified_time(path: Path) -> float:
        return path.stat().st_mtime

    return ResearchReportInventory(
        current=tuple(sorted(current, key=modified_time, reverse=True)),
        stale=tuple(sorted(stale, key=modified_time, reverse=True)),
        invalid=tuple(sorted(invalid, key=modified_time, reverse=True)),
    )


def discover_research_reports(artifacts_directory: Path) -> list[Path]:
    return list(inventory_research_reports(artifacts_directory).current)
