from __future__ import annotations

import json
from pathlib import Path

from market_ui.reports import CURRENT_RESEARCH_REPORT_SCHEMA, inventory_research_reports


def write_report(path: Path, schema: int) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "test_regression": {},
                "test_strategy": {},
            }
        ),
        encoding="utf-8",
    )


def test_report_inventory_separates_current_and_stale_outputs(tmp_path: Path) -> None:
    write_report(tmp_path / "current.json", CURRENT_RESEARCH_REPORT_SCHEMA)
    write_report(tmp_path / "stale.json", CURRENT_RESEARCH_REPORT_SCHEMA - 1)
    (tmp_path / "invalid.json").write_text("{", encoding="utf-8")
    (tmp_path / "unrelated.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    inventory = inventory_research_reports(tmp_path)
    assert [path.name for path in inventory.current] == ["current.json"]
    assert [path.name for path in inventory.stale] == ["stale.json"]
    assert [path.name for path in inventory.invalid] == ["invalid.json"]
