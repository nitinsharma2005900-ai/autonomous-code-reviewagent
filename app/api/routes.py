"""FastAPI router exposing GitHub webhook ingestion and HITL review endpoints."""

from __future__ import annotations

import hashlib
import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status

from app.api.schemas import (
    ApprovalRequest,
    CheckpointHistoryResponse,
    GitHubWebhookPayload,
    HealthResponse,
    ReviewResponse,
)
from app.config import Settings, get_settings
from app.graph.workflow import (
    ReviewPipelineService,
    ReviewSnapshot,
    ThreadAlreadyCompletedError,
    ThreadNotFoundError,
)
from app.services.github_client import GitHubAPIError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_pipeline_service(request: Request) -> ReviewPipelineService:
    """Resolve the shared ReviewPipelineService from FastAPI app state."""
    return request.app.state.pipeline_service


def get_app_settings(request: Request) -> Settings:
    """Resolve the Settings instance bound to this FastAPI application."""
    return getattr(request.app.state, "settings", get_settings())


def _to_review_response(snapshot: ReviewSnapshot) -> ReviewResponse:
    state = snapshot.state
    return ReviewResponse(
        thread_id=snapshot.thread_id,
        status=snapshot.status,
        pr_url=state.get("pr_url", ""),
        branch=state.get("branch", "main"),
        next_nodes=snapshot.next_nodes,
        revision_count=int(state.get("revision_count") or 0),
        is_approved=state.get("is_approved"),
        reviewer_feedback=state.get("reviewer_feedback"),
        lint_errors=state.get("lint_errors") or {},
        refactored_code=state.get("refactored_code") or {},
        commit_result=state.get("commit_result"),
        history=state.get("history") or [],
        checkpoint_id=snapshot.checkpoint_id,
    )


def _verify_webhook_signature(
    raw_body: bytes,
    signature_header: str | None,
    secret: str,
) -> None:
    """Validate GitHub HMAC-SHA256 webhook signature when a secret is configured."""
    if not secret:
        return
    if not signature_header or not signature_header.startswith("sha256="):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed X-Hub-Signature-256 header.",
        )
    expected = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    )
    if not hmac.compare_digest(expected, signature_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid GitHub webhook signature.",
        )


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health_check(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    """Return service health and active checkpointer configuration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        checkpointer=settings.checkpointer_type,
    )


@router.post(
    "/api/webhooks/github",
    response_model=ReviewResponse,
    status_code=status.HTTP_200_OK,
    tags=["Webhooks"],
    summary="Ingest GitHub Pull Request webhook and start LangGraph review thread",
)
async def github_webhook(
    payload: GitHubWebhookPayload,
    request: Request,
    x_hub_signature_256: str | None = Header(default=None, alias="X-Hub-Signature-256"),
    pipeline: ReviewPipelineService = Depends(get_pipeline_service),
    settings: Settings = Depends(get_app_settings),
) -> ReviewResponse:
    """Receive a GitHub PR event payload, run sandbox linting & refactoring, and halt at HITL gate."""
    if settings.github_webhook_secret:
        raw_body = await request.body()
        _verify_webhook_signature(
            raw_body, x_hub_signature_256, settings.github_webhook_secret
        )

    pr_url, branch, pr_number = payload.resolve_parameters()

    try:
        snapshot = pipeline.start_review(
            pr_url=pr_url,
            branch=branch,
            pr_number=pr_number,
            code_contents=payload.files,
        )
    except GitHubAPIError as exc:
        logger.error("GitHub API error during webhook processing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return _to_review_response(snapshot)


@router.post(
    "/api/reviews/{thread_id}/approve",
    response_model=ReviewResponse,
    status_code=status.HTTP_200_OK,
    tags=["Reviews"],
    summary="Submit human approval or rejection feedback to resume a paused review thread",
)
def approve_review(
    thread_id: str,
    decision: ApprovalRequest,
    pipeline: ReviewPipelineService = Depends(get_pipeline_service),
) -> ReviewResponse:
    """Approve and commit refactored code, or reject with feedback to rollback and regenerate."""
    try:
        snapshot = pipeline.submit_review_decision(
            thread_id,
            approved=decision.approved,
            feedback=decision.feedback,
        )
    except ThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ThreadAlreadyCompletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except GitHubAPIError as exc:
        logger.error("GitHub API error during commit write-back: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return _to_review_response(snapshot)


@router.get(
    "/api/reviews/{thread_id}",
    response_model=ReviewResponse,
    tags=["Reviews"],
    summary="Retrieve current state of a review thread",
)
def get_review(
    thread_id: str,
    pipeline: ReviewPipelineService = Depends(get_pipeline_service),
) -> ReviewResponse:
    """Fetch the persisted LangGraph checkpoint state for a review thread."""
    try:
        snapshot = pipeline.get_thread(thread_id)
    except ThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return _to_review_response(snapshot)


@router.get(
    "/api/reviews/{thread_id}/history",
    response_model=CheckpointHistoryResponse,
    tags=["Reviews"],
    summary="Retrieve checkpoint state history for a review thread",
)
def get_review_history(
    thread_id: str,
    pipeline: ReviewPipelineService = Depends(get_pipeline_service),
) -> CheckpointHistoryResponse:
    """Return LangGraph checkpointer history snapshots for auditing and rollback inspection."""
    try:
        checkpoints = pipeline.get_checkpoint_history(thread_id)
    except ThreadNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    return CheckpointHistoryResponse(thread_id=thread_id, checkpoints=checkpoints)


@router.get(
    "/api/reviews",
    response_model=list[ReviewResponse],
    tags=["Reviews"],
    summary="List active review threads",
)
def list_reviews(
    pipeline: ReviewPipelineService = Depends(get_pipeline_service),
) -> list[ReviewResponse]:
    """List review threads tracked during the current server lifecycle."""
    return [_to_review_response(snap) for snap in pipeline.list_threads()]
