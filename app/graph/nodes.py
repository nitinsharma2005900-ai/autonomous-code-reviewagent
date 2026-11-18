"""LangGraph pipeline node implementations for PR fetching, linting, refactoring, and applying."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.config import Settings, get_settings
from app.graph.state import ReviewState
from app.services.github_client import GitHubClient, PullRequestRef, parse_github_url
from app.services.refactor_engine import RefactorEngine
from app.tools.sandbox_linter import run_linter_tool

logger = logging.getLogger(__name__)

# Realistic sample module used only when simulating mock webhooks in dry-run mode
# without live GitHub credentials or inline files.
DEFAULT_SAMPLE_PR_FILES: dict[str, str] = {
    "src/data_processor.py": (
        "import os\n"
        "def filter_positive_squares(numbers):\n"
        "    result=[]\n"
        "    for n in numbers:\n"
        "        if n>0:\n"
        "            result.append(n*n)\n"
        "    return result\n"
    )
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineDependencies:
    """Injectable service dependencies for all graph nodes."""

    settings: Settings = field(default_factory=get_settings)
    github_client: GitHubClient = field(default_factory=GitHubClient)
    refactor_engine: RefactorEngine = field(default_factory=RefactorEngine)


def create_fetch_pr_node(deps: PipelineDependencies) -> Callable[[ReviewState], dict[str, Any]]:
    """Create `fetch_pr_node` that pulls modified files from GitHub into graph state."""

    def fetch_pr_node(state: ReviewState) -> dict[str, Any]:
        pr_url = state.get("pr_url", "")
        branch = state.get("branch", "main")
        pr_number = state.get("pr_number")
        pr_ref = parse_github_url(pr_url=pr_url, branch=branch, pr_number=pr_number)

        existing_files = state.get("code_contents") or {}
        source_mode = "payload"

        if existing_files:
            code_contents = dict(existing_files)
            resolved_branch = pr_ref.branch
        elif deps.settings.has_github_credentials or not deps.settings.dry_run_mode:
            source_mode = "github_api"
            code_contents, resolved_branch = deps.github_client.fetch_pr_files(pr_ref)
        else:
            source_mode = "dry_run_sample"
            logger.warning(
                "GITHUB_TOKEN not set and DRY_RUN_MODE=true; using sample PR files for %s",
                pr_ref.pr_url,
            )
            code_contents = dict(DEFAULT_SAMPLE_PR_FILES)
            resolved_branch = pr_ref.branch

        logger.info(
            "[fetch_pr_node] Loaded %d file(s) for %s (branch=%s, source=%s)",
            len(code_contents),
            pr_ref.pr_url,
            resolved_branch,
            source_mode,
        )

        return {
            "pr_url": pr_ref.pr_url,
            "repo_owner": pr_ref.owner,
            "repo_name": pr_ref.repo,
            "pr_number": pr_ref.pr_number,
            "branch": resolved_branch,
            "code_contents": code_contents,
            "history": [
                {
                    "node": "fetch_pr_node",
                    "timestamp": _utc_now(),
                    "source": source_mode,
                    "files_fetched": list(code_contents.keys()),
                    "branch": resolved_branch,
                }
            ],
        }

    return fetch_pr_node


def create_lint_code_node(deps: PipelineDependencies) -> Callable[[ReviewState], dict[str, Any]]:
    """Create `lint_code_node` that runs the sandboxed LangChain linter tool on each file."""

    def lint_code_node(state: ReviewState) -> dict[str, Any]:
        code_contents = state.get("code_contents") or {}
        lint_errors: dict[str, str] = {}

        for filename, source_code in code_contents.items():
            report = run_linter_tool.invoke({"code_string": source_code})
            lint_errors[filename] = str(report)
            logger.info(
                "[lint_code_node] Analyzed %s -> %s",
                filename,
                "PASS" if str(report).startswith("PASS:") else "ISSUES_FOUND",
            )

        return {
            "lint_errors": lint_errors,
            "history": [
                {
                    "node": "lint_code_node",
                    "timestamp": _utc_now(),
                    "files_linted": list(lint_errors.keys()),
                    "issues_detected": sum(
                        1 for r in lint_errors.values() if not r.startswith("PASS:")
                    ),
                }
            ],
        }

    return lint_code_node


def create_propose_refactor_node(
    deps: PipelineDependencies,
) -> Callable[[ReviewState], dict[str, Any]]:
    """Create `propose_refactor_node` that invokes the LLM to generate refactored code."""

    def propose_refactor_node(state: ReviewState) -> dict[str, Any]:
        code_contents = state.get("code_contents") or {}
        lint_errors = state.get("lint_errors") or {}
        previous_refactors = state.get("refactored_code") or {}
        reviewer_feedback = state.get("reviewer_feedback")
        revision = int(state.get("revision_count") or 0) + 1

        refactored_code: dict[str, str] = {}
        for filename, original_code in code_contents.items():
            report = lint_errors.get(filename, "PASS: No issues reported.")
            refactored_code[filename] = deps.refactor_engine.refactor_file(
                filename=filename,
                code=original_code,
                lint_report=report,
                reviewer_feedback=reviewer_feedback,
                previous_draft=previous_refactors.get(filename),
            )

        logger.info(
            "[propose_refactor_node] Generated revision #%d for %d file(s) (feedback=%s). Pausing before apply_changes_node.",
            revision,
            len(refactored_code),
            bool(reviewer_feedback),
        )

        return {
            "refactored_code": refactored_code,
            "is_approved": None,
            "revision_count": revision,
            "history": [
                {
                    "node": "propose_refactor_node",
                    "timestamp": _utc_now(),
                    "revision": revision,
                    "feedback_applied": reviewer_feedback,
                    "files_refactored": list(refactored_code.keys()),
                    "status": "awaiting_human_approval",
                }
            ],
        }

    return propose_refactor_node


def create_apply_changes_node(
    deps: PipelineDependencies,
) -> Callable[[ReviewState], dict[str, Any]]:
    """Create `apply_changes_node` that commits approved refactored code back to GitHub."""

    def apply_changes_node(state: ReviewState) -> dict[str, Any]:
        if not state.get("is_approved"):
            logger.warning(
                "[apply_changes_node] Blocked unapproved write; routing back to propose_refactor_node."
            )
            return {
                "history": [
                    {
                        "node": "apply_changes_node",
                        "timestamp": _utc_now(),
                        "status": "rejected_rerouting",
                        "reviewer_feedback": state.get("reviewer_feedback"),
                    }
                ]
            }

        pr_ref = PullRequestRef(
            owner=state.get("repo_owner", ""),
            repo=state.get("repo_name", ""),
            pr_number=state.get("pr_number"),
            branch=state.get("branch", "main"),
            pr_url=state.get("pr_url", ""),
        )
        refactored_code = state.get("refactored_code") or {}
        commit_result = deps.github_client.commit_files_to_branch(
            pr_ref=pr_ref,
            files=refactored_code,
            commit_message=(
                f"refactor: apply autonomous code review fixes "
                f"(revision {state.get('revision_count', 1)})"
            ),
        )

        logger.info(
            "[apply_changes_node] Applied %d refactored file(s) to %s@%s (status=%s)",
            len(refactored_code),
            pr_ref.full_name,
            pr_ref.branch,
            commit_result.get("status"),
        )

        return {
            "commit_result": commit_result,
            "history": [
                {
                    "node": "apply_changes_node",
                    "timestamp": _utc_now(),
                    "status": "committed",
                    "commit_result": commit_result,
                }
            ],
        }

    return apply_changes_node
