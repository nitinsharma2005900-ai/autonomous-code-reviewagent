"""Custom LangChain sandbox execution tool for Python static analysis (flake8 & black)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import tool

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LintResult:
    """Structured output returned by the isolated linting sandbox."""

    passed: bool
    flake8_issues: list[str] = field(default_factory=list)
    black_needs_formatting: bool = False
    black_diff: str = ""
    errors: list[str] = field(default_factory=list)
    raw_output: str = ""

    def to_report(self) -> str:
        """Render a deterministic human- and LLM-readable diagnostic report."""
        if self.passed and not self.flake8_issues and not self.black_needs_formatting and not self.errors:
            return "PASS: No flake8 or black formatting issues detected."

        sections: list[str] = []
        if self.flake8_issues:
            sections.append("FLAKE8 ISSUES:\n" + "\n".join(self.flake8_issues))
        if self.black_needs_formatting:
            diff_text = self.black_diff.strip() or "File would be reformatted by black."
            sections.append(f"BLACK FORMATTING ISSUES:\n{diff_text}")
        if self.errors:
            sections.append("SANDBOX EXECUTION ERRORS:\n" + "\n".join(self.errors))
        return "\n\n".join(sections)


def _sanitized_env() -> dict[str, str]:
    """Build a minimal environment for subprocess execution without leaking secrets."""
    allowed_keys = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "PYTHONIOENCODING",
        "LANG",
        "LC_ALL",
    }
    env = {k: v for k, v in os.environ.items() if k.upper() in allowed_keys}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def run_sandbox_analysis(
    code_string: str,
    filename: str = "candidate.py",
    run_flake8: bool | None = None,
    run_black: bool | None = None,
    timeout_seconds: int | None = None,
) -> LintResult:
    """Write Python source code to an isolated temporary directory and execute linters.

    User code is never executed; only static analysis processes (`flake8` and
    `black --check`) are invoked with restricted environment variables and timeouts.
    """
    settings = get_settings()
    do_flake8 = settings.enable_flake8_check if run_flake8 is None else run_flake8
    do_black = settings.enable_black_check if run_black is None else run_black
    timeout = timeout_seconds or settings.linter_timeout_seconds

    safe_name = Path(filename).name or "candidate.py"
    if not safe_name.endswith(".py"):
        safe_name = f"{safe_name}.py"

    flake8_issues: list[str] = []
    black_needs_formatting = False
    black_diff = ""
    errors: list[str] = []
    raw_chunks: list[str] = []

    env = _sanitized_env()

    with tempfile.TemporaryDirectory(prefix="code_review_sandbox_") as temp_dir:
        temp_path = Path(temp_dir)
        target_file = temp_path / safe_name
        target_file.write_text(code_string, encoding="utf-8")

        if do_flake8:
            flake8_cmd = [sys.executable, "-m", "flake8"]
            if settings.flake8_config_path.exists():
                flake8_cmd.extend(["--config", str(settings.flake8_config_path)])
            flake8_cmd.append(str(target_file))

            try:
                proc = subprocess.run(
                    flake8_cmd,
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                combined = "\n".join(
                    part for part in (proc.stdout.strip(), proc.stderr.strip()) if part
                )
                if combined:
                    raw_chunks.append(f"[flake8 rc={proc.returncode}]\n{combined}")
                if proc.returncode != 0:
                    for line in combined.splitlines():
                        normalized = line.replace(str(target_file), safe_name).strip()
                        if normalized:
                            flake8_issues.append(normalized)
            except subprocess.TimeoutExpired:
                msg = f"flake8 timed out after {timeout}s"
                logger.error(msg)
                errors.append(msg)
            except OSError as exc:
                msg = f"flake8 execution error: {exc}"
                logger.error(msg)
                errors.append(msg)

        if do_black:
            black_cmd = [
                sys.executable,
                "-m",
                "black",
                "--check",
                "--diff",
                "--no-color",
                str(target_file),
            ]
            try:
                proc = subprocess.run(
                    black_cmd,
                    cwd=temp_dir,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                stdout = proc.stdout.replace(str(target_file), safe_name).strip()
                stderr = proc.stderr.replace(str(target_file), safe_name).strip()
                if stdout or stderr:
                    raw_chunks.append(
                        f"[black rc={proc.returncode}]\n{stdout}\n{stderr}".strip()
                    )
                if proc.returncode == 1:
                    black_needs_formatting = True
                    black_diff = stdout or stderr
                elif proc.returncode == 123:
                    # Black exit code 123 indicates an internal syntax/parse error
                    black_needs_formatting = True
                    errors.append(f"SyntaxError detected by black: {stderr or stdout}")
                elif proc.returncode != 0:
                    errors.append(
                        f"black exited with code {proc.returncode}: {stderr or stdout}"
                    )
            except subprocess.TimeoutExpired:
                msg = f"black timed out after {timeout}s"
                logger.error(msg)
                errors.append(msg)
            except OSError as exc:
                msg = f"black execution error: {exc}"
                logger.error(msg)
                errors.append(msg)

    passed = not flake8_issues and not black_needs_formatting and not errors
    return LintResult(
        passed=passed,
        flake8_issues=flake8_issues,
        black_needs_formatting=black_needs_formatting,
        black_diff=black_diff,
        errors=errors,
        raw_output="\n\n".join(raw_chunks),
    )


@tool
def run_linter_tool(code_string: str) -> str:
    """Writes python code to a temporary file and runs flake8 and black analysis."""
    result = run_sandbox_analysis(code_string=code_string)
    return result.to_report()
