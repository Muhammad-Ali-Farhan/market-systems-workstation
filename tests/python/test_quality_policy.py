"""Repository-level quality and public-branding guardrails."""

from __future__ import annotations

import json
import re
import tokenize
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def python_files() -> tuple[Path, ...]:
    """Return maintained Python source/test files covered by lint-style checks."""
    roots = tuple((ROOT / "market_engine").rglob("*.py")) + tuple(
        (ROOT / "tests" / "python").rglob("*.py")
    )
    return tuple(sorted(set(roots)))


def test_ruff_policy_is_explicit() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    assert payload["tool"]["ruff"]["lint"]["select"] == ["E4", "E7", "E9", "F"]


def test_public_branding_is_engineering_first() -> None:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        payload = tomllib.load(stream)
    assert payload["project"]["name"] == "market-data-execution-engine"
    assert (ROOT / "market_engine" / "cli" / "workstation.py").exists()
    assert (ROOT / "market_engine" / "ui").is_dir()
    app_source = (ROOT / "market_engine" / "ui" / "app.py").read_text(encoding="utf-8")
    assert "class MarketWorkstation(tk.Tk):" in app_source
    assert "QuantWorkstation" not in app_source
    assert not (ROOT / "PORTFOLIO_NOTES.md").exists()
    assert not (ROOT / "UPGRADE_SUMMARY.md").exists()


def test_no_portfolio_agent_instruction_files() -> None:
    assert not (ROOT / "AGENTS.md").exists()
    assert not (ROOT / ".cursorignore").exists()
    assert not any(ROOT.glob("INTERVIEW_GUIDE*.md"))


def test_no_placeholder_free_f_strings() -> None:
    issues: list[str] = []
    fstring_start = getattr(tokenize, "FSTRING_START", -1)
    fstring_end = getattr(tokenize, "FSTRING_END", -1)
    for path in python_files():
        with tokenize.open(path) as stream:
            tokens = tokenize.generate_tokens(stream.readline)
            stack: list[dict[str, object]] = []
            for token in tokens:
                if token.type == fstring_start:
                    stack.append({"line": token.start[0], "has_expression": False})
                elif stack and token.type == tokenize.OP and token.string == "{":
                    stack[-1]["has_expression"] = True
                elif token.type == fstring_end and stack:
                    current = stack.pop()
                    if not bool(current["has_expression"]):
                        relative = path.relative_to(ROOT)
                        issues.append(f"{relative}:{current['line']}")
    assert not issues, "No-placeholder f-strings: " + ", ".join(issues)


def test_windows_native_build_selects_msvc_and_ninja() -> None:
    script = (ROOT / "scripts" / "windows" / "build_native.ps1").read_text(encoding="utf-8")
    assert "vswhere.exe" in script
    assert "VsDevCmd.bat" in script
    assert "-G Ninja" in script
    assert "-DCMAKE_CXX_COMPILER=$Compiler" in script
    assert "-A x64" not in script


def test_architecture_documents_both_market_data_paths() -> None:
    architecture = (ROOT / "docs" / "architecture" / "overview.md").read_text(encoding="utf-8")
    assert "top-of-book path" in architecture
    assert "sequence-correct Level-2 path" in architecture
    assert "not an L2 reconstruction engine" not in architecture


def test_public_version_metadata_is_consistent() -> None:
    from market_engine.version import PROJECT_USER_AGENT, PROJECT_VERSION

    with (ROOT / "pyproject.toml").open("rb") as stream:
        python_project = tomllib.load(stream)
    vcpkg = json.loads((ROOT / "vcpkg.json").read_text(encoding="utf-8"))
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    match = re.search(
        r"project\(market_data_execution_engine VERSION ([0-9]+\.[0-9]+\.[0-9]+)",
        cmake,
    )

    assert match is not None
    assert python_project["project"]["version"] == PROJECT_VERSION
    assert vcpkg["version-semver"] == PROJECT_VERSION
    assert match.group(1) == PROJECT_VERSION
    assert PROJECT_USER_AGENT == f"market-data-execution-engine/{PROJECT_VERSION}"

    capture_source = (ROOT / "market_engine" / "market_data" / "l2_capture.py").read_text(
        encoding="utf-8"
    )
    assert 'headers={"User-Agent": PROJECT_USER_AGENT}' in capture_source
    assert "market-systems-workstation/4.0" not in capture_source


def test_public_documents_match_current_repository_state() -> None:
    setup = (ROOT / "docs" / "development" / "repository-setup.md").read_text(encoding="utf-8")
    walkthrough = (ROOT / "docs" / "architecture" / "system-design.md").read_text(encoding="utf-8")

    assert "This repository is already public and initialized." in setup
    assert "git init" in setup
    assert "Do not run `git init` again" in setup
    assert "signed 64-bit fixed-point integers" in walkthrough
    assert "quantities into unsigned 64-bit fixed-point integers" in walkthrough
