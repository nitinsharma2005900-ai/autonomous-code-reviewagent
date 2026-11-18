"""LangGraph state schema definition for the autonomous code review pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


class ReviewState(TypedDict, total=False):
    """State carried across all nodes and persisted by the LangGraph checkpointer.

    Required core keys:
        - pr_url: Target GitHub Pull Request or repository URL.
        - code_contents: Mapping of file path -> original source code.
        - lint_errors: Mapping of file path -> sandbox linter output/diagnostics.
        - refactored_code: Mapping of file path -> proposed refactored source code.
        - is_approved: Human reviewer approval flag (None while pending).
        - reviewer_feedback: Optional human reviewer guidance for regeneration.
        - history: Append-only audit log of node executions and review transitions.
    """

    pr_url: str
    code_contents: dict[str, str]
    lint_errors: dict[str, str]
    refactored_code: dict[str, str]
    is_approved: Optional[bool]
    reviewer_feedback: Optional[str]
    history: Annotated[list[dict[str, Any]], operator.add]

    # Auxiliary PR & execution metadata
    repo_owner: str
    repo_name: str
    pr_number: Optional[int]
    branch: str
    revision_count: int
    commit_result: Optional[dict[str, Any]]
