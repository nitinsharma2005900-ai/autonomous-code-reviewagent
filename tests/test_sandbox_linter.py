"""Lint Tool Isolation tests for the custom @tool run_linter_tool sandbox."""

from __future__ import annotations

from pathlib import Path

from app.tools.sandbox_linter import run_linter_tool, run_sandbox_analysis


def test_run_linter_tool_metadata() -> None:
    """Verify run_linter_tool is properly registered as a LangChain @tool."""
    assert run_linter_tool.name == "run_linter_tool"
    assert "flake8" in run_linter_tool.description.lower()


def test_lint_tool_identifies_syntax_errors() -> None:
    """Assert the linter tool successfully identifies syntax errors in bad Python code."""
    bad_syntax_code = "def broken_function(x, y\n    return x + y\n"

    report = run_linter_tool.invoke({"code_string": bad_syntax_code})

    assert "PASS:" not in report
    assert "E999" in report or "SyntaxError" in report


def test_lint_tool_identifies_style_and_formatting_issues() -> None:
    """Assert flake8 unused imports / spacing and black formatting issues are reported."""
    messy_code = "import sys\ndef add(a,b):\n  return a+b\n"

    result = run_sandbox_analysis(messy_code)

    assert result.passed is False
    assert any("F401" in issue for issue in result.flake8_issues)
    assert result.black_needs_formatting is True
    report = result.to_report()
    assert "FLAKE8 ISSUES:" in report
    assert "BLACK FORMATTING ISSUES:" in report


def test_lint_tool_passes_clean_code() -> None:
    """Assert PEP8 and Black compliant Python code passes cleanly."""
    clean_code = 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n'

    report = run_linter_tool.invoke({"code_string": clean_code})

    assert report.startswith("PASS:")


def test_sandbox_never_executes_untrusted_code(tmp_path: Path) -> None:
    """Ensure static analysis never executes arbitrary top-level Python statements."""
    marker = tmp_path / "should_not_exist.txt"
    malicious_code = (
        f"from pathlib import Path\n" f"Path({str(marker)!r}).write_text('pwned')\n"
    )

    run_linter_tool.invoke({"code_string": malicious_code})

    assert not marker.exists()
