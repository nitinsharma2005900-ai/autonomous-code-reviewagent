"""Pydantic request and response models for FastAPI webhook and review endpoints."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.github_client import parse_github_url


class GitHubBranchRef(BaseModel):
    """GitHub Pull Request branch reference."""

    ref: str = Field(..., min_length=1, description="Git branch name")
    sha: Optional[str] = Field(default=None, description="Head commit SHA")


class GitHubPullRequestObject(BaseModel):
    """Subset of GitHub's standard `pull_request` webhook object."""

    html_url: Optional[str] = Field(default=None, description="Pull Request web URL")
    number: Optional[int] = Field(default=None, ge=1, description="Pull Request number")
    title: Optional[str] = Field(default=None, description="Pull Request title")
    head: Optional[GitHubBranchRef] = Field(default=None, description="Source branch info")
    base: Optional[GitHubBranchRef] = Field(default=None, description="Target branch info")


class GitHubRepositoryObject(BaseModel):
    """Subset of GitHub's standard `repository` webhook object."""

    full_name: Optional[str] = Field(default=None, description="Repository full name (owner/repo)")
    html_url: Optional[str] = Field(default=None, description="Repository HTML URL")
    clone_url: Optional[str] = Field(default=None, description="Repository clone URL")


class GitHubWebhookPayload(BaseModel):
    """Accepts both standard GitHub PR webhook payloads and simplified mock PR payloads."""

    model_config = ConfigDict(extra="allow")

    # Standard GitHub webhook fields
    action: Optional[str] = Field(default="opened", description="GitHub PR event action")
    number: Optional[int] = Field(default=None, ge=1, description="Pull Request number")
    pull_request: Optional[GitHubPullRequestObject] = Field(default=None)
    repository: Optional[GitHubRepositoryObject] = Field(default=None)

    # Simplified / mock payload fields (for local testing & verification)
    pr_url: Optional[str] = Field(default=None, description="Direct Pull Request URL")
    repository_url: Optional[str] = Field(default=None, description="GitHub repository URL")
    repo_url: Optional[str] = Field(default=None, description="Alias for repository_url")
    branch: Optional[str] = Field(default=None, description="Source branch name")
    branch_name: Optional[str] = Field(default=None, description="Alias for branch")
    pr_number: Optional[int] = Field(default=None, ge=1, description="Pull Request number")
    files: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional inline filename->code map for local/offline webhook simulations",
    )

    def resolve_parameters(self) -> tuple[str, str, Optional[int]]:
        """Extract normalized `(pr_url, branch, pr_number)` from the payload."""
        resolved_pr_number = (
            self.pr_number
            or self.number
            or (self.pull_request.number if self.pull_request else None)
        )
        resolved_branch = (
            self.branch
            or self.branch_name
            or (
                self.pull_request.head.ref
                if self.pull_request and self.pull_request.head
                else None
            )
            or "main"
        )
        resolved_repo_url = (
            self.repository_url
            or self.repo_url
            or (self.repository.html_url if self.repository else None)
            or (
                f"https://github.com/{self.repository.full_name}"
                if self.repository and self.repository.full_name
                else None
            )
        )
        resolved_pr_url = (
            self.pr_url
            or (self.pull_request.html_url if self.pull_request else None)
            or resolved_repo_url
        )

        ref = parse_github_url(
            pr_url=resolved_pr_url,
            repository_url=resolved_repo_url,
            branch=resolved_branch,
            pr_number=resolved_pr_number,
        )
        return ref.pr_url, ref.branch, ref.pr_number

    @model_validator(mode="after")
    def validate_github_coordinates(self) -> "GitHubWebhookPayload":
        """Ensure the payload contains a parseable GitHub PR or repository URL."""
        self.resolve_parameters()
        return self


class ApprovalRequest(BaseModel):
    """Human-in-the-loop review decision payload."""

    approved: bool = Field(
        ...,
        description="Set true to commit refactored code to GitHub, or false to reject and regenerate.",
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Optional reviewer instructions used to regenerate the refactoring draft when rejected.",
    )


class ReviewResponse(BaseModel):
    """Current state snapshot of an autonomous code review thread."""

    thread_id: str = Field(..., description="Unique LangGraph checkpointer thread identifier")
    status: str = Field(
        ...,
        description="Pipeline execution status: awaiting_approval, in_progress, or completed",
    )
    pr_url: str = Field(..., description="Target Pull Request or repository URL")
    branch: str = Field(..., description="Target Git branch")
    next_nodes: list[str] = Field(
        default_factory=list,
        description="Next scheduled LangGraph node(s) waiting after the interrupt gate",
    )
    revision_count: int = Field(default=0, description="Number of refactoring drafts generated")
    is_approved: Optional[bool] = Field(default=None, description="Human reviewer approval state")
    reviewer_feedback: Optional[str] = Field(
        default=None, description="Most recent reviewer feedback"
    )
    lint_errors: dict[str, str] = Field(
        default_factory=dict, description="Per-file sandbox flake8/black diagnostics"
    )
    refactored_code: dict[str, str] = Field(
        default_factory=dict, description="Per-file proposed refactored Python code"
    )
    commit_result: Optional[dict[str, Any]] = Field(
        default=None, description="GitHub commit metadata once changes are applied"
    )
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="Chronological execution and review audit log"
    )
    checkpoint_id: Optional[str] = Field(
        default=None, description="Current LangGraph checkpoint identifier"
    )


class CheckpointHistoryResponse(BaseModel):
    """Ordered checkpoint history for a LangGraph review thread."""

    thread_id: str
    checkpoints: list[dict[str, Any]]


class HealthResponse(BaseModel):
    """Service health metadata."""

    status: str
    version: str
    checkpointer: str
