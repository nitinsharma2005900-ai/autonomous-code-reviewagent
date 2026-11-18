"""FastAPI endpoint integration tests covering Verification Test Cases 1-3 and validation errors."""

from __future__ import annotations

import hashlib
import hmac
import json

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings
from app.graph.workflow import ReviewPipelineService
from app.main import create_app


def test_health_endpoint(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "1.0.0"


def test_e2e_verification_cases_1_2_3(api_client: TestClient) -> None:
    """Execute Verification Test Cases 1, 2, and 3 sequentially through FastAPI endpoints."""
    # --- Test Case 1: Webhook PR Ingestion, Linting, & HITL Pause Gate ---
    webhook_resp = api_client.post(
        "/api/webhooks/github",
        json={
            "repository_url": "https://github.com/acme/review-agent-demo",
            "branch": "feature/refactor-loops",
            "pr_number": 12,
        },
    )
    assert webhook_resp.status_code == 200
    tc1 = webhook_resp.json()
    thread_id = tc1["thread_id"]
    assert thread_id
    assert tc1["status"] == "awaiting_approval"
    assert tc1["next_nodes"] == ["apply_changes_node"]
    assert tc1["branch"] == "feature/refactor-loops"
    assert tc1["lint_errors"]
    assert tc1["refactored_code"]
    assert tc1["commit_result"] is None

    # --- Test Case 2: Feedback Input & Refactoring Proposal Re-route Loop ---
    reject_resp = api_client.post(
        f"/api/reviews/{thread_id}/approve",
        json={
            "approved": False,
            "feedback": "Use list comprehensions instead of loops",
        },
    )
    assert reject_resp.status_code == 200
    tc2 = reject_resp.json()
    assert tc2["status"] == "awaiting_approval"
    assert tc2["next_nodes"] == ["apply_changes_node"]
    assert tc2["revision_count"] == 2
    assert tc2["reviewer_feedback"] == "Use list comprehensions instead of loops"
    refactored_values = "\n".join(tc2["refactored_code"].values())
    assert "[n * n for n in numbers if n > 0]" in refactored_values
    assert tc2["commit_result"] is None

    # --- Test Case 3: Approval and Git Commit Write-Back ---
    approve_resp = api_client.post(
        f"/api/reviews/{thread_id}/approve",
        json={"approved": True},
    )
    assert approve_resp.status_code == 200
    tc3 = approve_resp.json()
    assert tc3["status"] == "completed"
    assert tc3["next_nodes"] == []
    assert tc3["is_approved"] is True
    assert tc3["commit_result"] is not None
    assert tc3["commit_result"]["branch"] == "feature/refactor-loops"

    # Confirm GET /api/reviews/{thread_id} and history reflect completed state
    get_resp = api_client.get(f"/api/reviews/{thread_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["status"] == "completed"

    hist_resp = api_client.get(f"/api/reviews/{thread_id}/history")
    assert hist_resp.status_code == 200
    assert len(hist_resp.json()["checkpoints"]) >= 4


def test_standard_github_pr_webhook_payload(api_client: TestClient) -> None:
    """Verify standard GitHub webhook event structure is parsed accurately."""
    payload = {
        "action": "opened",
        "number": 88,
        "pull_request": {
            "html_url": "https://github.com/org-x/service-y/pull/88",
            "number": 88,
            "head": {"ref": "feat/clean-api", "sha": "deadbeef"},
            "base": {"ref": "main"},
        },
        "repository": {
            "full_name": "org-x/service-y",
            "html_url": "https://github.com/org-x/service-y",
        },
    }
    resp = api_client.post("/api/webhooks/github", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["pr_url"] == "https://github.com/org-x/service-y/pull/88"
    assert data["branch"] == "feat/clean-api"
    assert data["status"] == "awaiting_approval"


def test_validation_error_exceptions(api_client: TestClient) -> None:
    """Assert invalid payloads raise FastAPI 422 validation errors and unknown threads return 404."""
    # Empty webhook payload -> 422 Validation Error
    empty_resp = api_client.post("/api/webhooks/github", json={})
    assert empty_resp.status_code == 422

    # Malformed repository URL -> 422 Validation Error
    bad_url_resp = api_client.post(
        "/api/webhooks/github",
        json={"repository_url": "not_a_github_url", "branch": "main"},
    )
    assert bad_url_resp.status_code == 422

    # Missing required `approved` field -> 422 Validation Error
    missing_approved = api_client.post(
        "/api/reviews/some-thread/approve",
        json={"feedback": "Forgot approved boolean"},
    )
    assert missing_approved.status_code == 422

    # Invalid type for `approved` -> 422 Validation Error
    bad_type = api_client.post(
        "/api/reviews/some-thread/approve",
        json={"approved": "not-a-bool"},
    )
    assert bad_type.status_code == 422

    # Non-existent thread -> 404 Not Found
    not_found = api_client.post(
        "/api/reviews/nonexistent-thread-id/approve",
        json={"approved": True},
    )
    assert not_found.status_code == 404

    # Double approval on completed thread -> 409 Conflict
    created = api_client.post(
        "/api/webhooks/github",
        json={"repository_url": "https://github.com/acme/demo", "branch": "feat/once"},
    ).json()
    tid = created["thread_id"]
    assert (
        api_client.post(
            f"/api/reviews/{tid}/approve", json={"approved": True}
        ).status_code
        == 200
    )
    conflict = api_client.post(f"/api/reviews/{tid}/approve", json={"approved": True})
    assert conflict.status_code == 409


def test_webhook_signature_verification() -> None:
    """Verify HMAC-SHA256 signature validation when GITHUB_WEBHOOK_SECRET is configured."""
    secret = "super-secret-webhook-key"
    settings = Settings(
        GITHUB_TOKEN="",
        OPENAI_API_KEY="",
        GITHUB_WEBHOOK_SECRET=secret,
        CHECKPOINTER_TYPE="memory",
        DRY_RUN_MODE=True,
    )
    service = ReviewPipelineService(settings=settings, checkpointer=MemorySaver())
    app = create_app(settings=settings, pipeline_service=service)

    payload_bytes = json.dumps(
        {"repository_url": "https://github.com/acme/secure", "branch": "feat/hmac"}
    ).encode("utf-8")
    valid_sig = (
        "sha256="
        + hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
    )

    with TestClient(app) as client:
        # Missing signature -> 401
        resp_missing = client.post(
            "/api/webhooks/github",
            content=payload_bytes,
            headers={"Content-Type": "application/json"},
        )
        assert resp_missing.status_code == 401

        # Valid signature -> 200
        resp_ok = client.post(
            "/api/webhooks/github",
            content=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": valid_sig,
            },
        )
        assert resp_ok.status_code == 200
