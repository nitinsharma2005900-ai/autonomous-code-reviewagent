"""LLM-powered Python code refactoring engine with feedback-aware regeneration."""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)

SYSTEM_PROMPT = """You are a senior Python software engineer performing automated code review and refactoring.
Your task is to refactor the provided Python module to:
1. Resolve all flake8 and black linting/formatting issues.
2. Strictly incorporate any human reviewer feedback when provided.
3. Preserve the original functional behavior, public function/class signatures, and business logic.
4. Return ONLY valid Python source code without explanatory prose or markdown commentary."""


class LLMCallable(Protocol):
    """Protocol for any LangChain ChatModel or test double."""

    def invoke(self, messages: list[Any]) -> Any: ...


def _strip_markdown_fences(text: str) -> str:
    """Extract raw Python code if the LLM wraps its response in fenced code blocks."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip() + "\n"
    return text.strip() + "\n"


def _convert_simple_append_loops_to_comprehensions(code: str) -> str:
    """AST helper converting simple `res = []; for x in it: res.append(expr)` patterns."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    changed = False

    def transform_body(stmts: list[ast.stmt]) -> list[ast.stmt]:
        nonlocal changed
        new_stmts: list[ast.stmt] = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            # Recursively transform nested function/class bodies
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stmt.body = transform_body(stmt.body)

            if (
                i + 1 < len(stmts)
                and isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.List)
                and len(stmt.value.elts) == 0
                and isinstance(stmts[i + 1], ast.For)
            ):
                list_var = stmt.targets[0].id
                for_node = stmts[i + 1]
                comp = _match_append_for_loop(list_var, for_node)
                if comp is not None:
                    new_assign = ast.Assign(
                        targets=[ast.Name(id=list_var, ctx=ast.Store())],
                        value=comp,
                    )
                    ast.copy_location(new_assign, stmt)
                    new_stmts.append(new_assign)
                    changed = True
                    i += 2
                    continue

            new_stmts.append(stmt)
            i += 1
        return new_stmts

    tree.body = transform_body(tree.body)
    if not changed:
        return code

    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _match_append_for_loop(list_var: str, for_node: ast.For) -> ast.ListComp | None:
    """Match a simple `for target in iter: [if cond:] list_var.append(expr)` loop."""
    if for_node.orelse or len(for_node.body) != 1:
        return None

    body_stmt = for_node.body[0]
    ifs: list[ast.expr] = []

    if isinstance(body_stmt, ast.If) and not body_stmt.orelse and len(body_stmt.body) == 1:
        ifs.append(body_stmt.test)
        body_stmt = body_stmt.body[0]

    if (
        isinstance(body_stmt, ast.Expr)
        and isinstance(body_stmt.value, ast.Call)
        and isinstance(body_stmt.value.func, ast.Attribute)
        and isinstance(body_stmt.value.func.value, ast.Name)
        and body_stmt.value.func.value.id == list_var
        and body_stmt.value.func.attr == "append"
        and len(body_stmt.value.args) == 1
        and not body_stmt.value.keywords
    ):
        generator = ast.comprehension(
            target=for_node.target,
            iter=for_node.iter,
            ifs=ifs,
            is_async=0,
        )
        return ast.ListComp(elt=body_stmt.value.args[0], generators=[generator])

    return None


def _format_with_black(code: str) -> str:
    """Format Python code using black's library API if syntax is valid."""
    try:
        import black

        return black.format_str(code, mode=black.Mode())
    except Exception:
        return code


class RefactorEngine:
    """Generates refactored Python code via OpenAI LLM or deterministic offline fallback."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: LLMCallable | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm

    def _get_llm(self) -> LLMCallable | None:
        if self._llm is not None:
            return self._llm
        if not self.settings.has_openai_credentials:
            return None

        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(
            model=self.settings.openai_model,
            temperature=self.settings.llm_temperature,
            api_key=self.settings.openai_api_key,
        )
        return self._llm

    def refactor_file(
        self,
        filename: str,
        code: str,
        lint_report: str,
        reviewer_feedback: str | None = None,
        previous_draft: str | None = None,
    ) -> str:
        """Produce a refactored version of `code` addressing lint errors and feedback."""
        llm = self._get_llm()
        if llm is not None:
            return self._refactor_with_llm(
                llm=llm,
                filename=filename,
                code=previous_draft or code,
                lint_report=lint_report,
                reviewer_feedback=reviewer_feedback,
            )

        logger.info(
            "OPENAI_API_KEY not configured; using deterministic fallback refactor for %s",
            filename,
        )
        return self._fallback_refactor(
            code=previous_draft or code,
            reviewer_feedback=reviewer_feedback,
        )

    def _refactor_with_llm(
        self,
        llm: LLMCallable,
        filename: str,
        code: str,
        lint_report: str,
        reviewer_feedback: str | None,
    ) -> str:
        feedback_section = (
            f"\n\nHUMAN REVIEWER FEEDBACK (MUST ADDRESS):\n{reviewer_feedback}"
            if reviewer_feedback
            else ""
        )
        user_prompt = (
            f"File: {filename}\n\n"
            f"DETECTED LINTING / FORMATTING ISSUES:\n{lint_report}"
            f"{feedback_section}\n\n"
            f"SOURCE CODE:\n```python\n{code}\n```\n\n"
            "Return the complete refactored Python file."
        )
        response = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )
        content = response.content if hasattr(response, "content") else str(response)
        return _strip_markdown_fences(str(content))

    def _fallback_refactor(
        self,
        code: str,
        reviewer_feedback: str | None = None,
    ) -> str:
        """Apply safe deterministic PEP8 formatting and feedback-guided AST transforms."""
        updated = code
        if reviewer_feedback:
            fb_lower = reviewer_feedback.lower()
            if "list comprehension" in fb_lower or "comprehension" in fb_lower:
                updated = _convert_simple_append_loops_to_comprehensions(updated)
            else:
                # Record reviewer feedback cleanly if no specialized AST rule matches
                marker = f"# Reviewer feedback applied: {reviewer_feedback.strip()}"
                if marker not in updated:
                    updated = f"{marker}\n{updated}"

        formatted = _format_with_black(updated)
        if not formatted.endswith("\n"):
            formatted += "\n"
        return formatted
