"""LangGraph state machine, HITL interrupt gate, and checkpointer rollback assertions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from app.config import Settings
from app.graph.checkpointer import create_checkpointer
from app.graph.nodes import PipelineDependencies
from app.graph.workflow import ReviewPipelineService
from app.services.github_client import GitHubClient
from app.services.refactor_engine import RefactorEngine
from tests.conftest import SAMPLE_BUGGY_CODE, RecordingFakeLLM


def test_graph_pauses_at_apply_changes_interrupt_gate(
    pipeline_service: ReviewPipelineService,
) -> None:
    """Verify the pipeline runs fetch -> lint -> propose_refactor and halts before apply_changes_node."""
    snapshot = pipeline_service.start_review(
        pr_url="https://github.com/acme/demo/pull/10",
        branch="feature/refactor",
        code_contents={"src/calc.py": SAMPLE_BUGGY_CODE},
    )

    assert snapshot.status == "awaiting_approval"
    assert snapshot.next_nodes == ["apply_changes_node"]

    # Verify all required ReviewState keys are populated in state
    for key in (
        "pr_url",
        "code_contents",
        "lint_errors",
        "refactored_code",
        "is_approved",
        "reviewer_feedback",
        "history",
    ):
        assert key in snapshot.state

    assert "F401" in snapshot.state["lint_errors"]["src/calc.py"]
    assert snapshot.state["is_approved"] is None
    assert snapshot.state["commit_result"] is None
    assert snapshot.state["revision_count"] == 1


def test_checkpointer_rollback_on_rejection_and_resume_on_approval(
    test_settings: Settings,
    fake_llm: RecordingFakeLLM,
) -> None:
    """Assert rejection rolls back to propose_refactor_node without committing, and approval commits."""
    mock_github = MagicMock(spec=GitHubClient)
    mock_github.commit_files_to_branch.return_value = {
        "status": "committed",
        "branch": "feature/list-comp",
        "commit_sha": "abc123def456",
        "committed_files": ["src/calc.py"],
    }

    deps = PipelineDependencies(
        settings=test_settings,
        github_client=mock_github,
        refactor_engine=RefactorEngine(settings=test_settings, llm=fake_llm),
    )
    service = ReviewPipelineService(settings=test_settings, deps=deps)

    # Step 1: Initial review pauses before apply_changes_node
    initial = service.start_review(
        pr_url="https://github.com/acme/demo/pull/11",
        branch="feature/list-comp",
        code_contents={"src/calc.py": SAMPLE_BUGGY_CODE},
    )
    assert initial.status == "awaiting_approval"
    assert initial.next_nodes == ["apply_changes_node"]
    assert "for n in numbers:" in initial.state["refactored_code"]["src/calc.py"]
    mock_github.commit_files_to_branch.assert_not_called()

    # Step 2: Human rejects with feedback -> rollback to propose_refactor_node & re-pause
    rejected = service.submit_review_decision(
        initial.thread_id,
        approved=False,
        feedback="Use list comprehensions instead of loops",
    )
    assert rejected.status == "awaiting_approval"
    assert rejected.next_nodes == ["apply_changes_node"]
    assert rejected.state["revision_count"] == 2
    assert (
        rejected.state["reviewer_feedback"]
        == "Use list comprehensions instead of loops"
    )
    assert (
        "[n * n for n in numbers if n > 0]"
        in rejected.state["refactored_code"]["src/calc.py"]
    )
    mock_github.commit_files_to_branch.assert_not_called()

    # Verify rollback event and regenerated refactor appear in history & checkpoints
    nodes_visited = [entry["node"] for entry in rejected.state["history"]]
    assert nodes_visited.count("propose_refactor_node") == 2
    assert "human_review_gate" in nodes_visited
    assert len(service.get_checkpoint_history(initial.thread_id)) >= 4

    # Step 3: Human approves -> resumes apply_changes_node and completes thread
    approved = service.submit_review_decision(initial.thread_id, approved=True)
    assert approved.status == "completed"
    assert approved.next_nodes == []
    assert approved.state["is_approved"] is True
    assert approved.state["commit_result"]["commit_sha"] == "abc123def456"
    mock_github.commit_files_to_branch.assert_called_once()


def test_sqlite_checkpointer_persists_state_across_service_restarts(
    tmp_path: Path,
) -> None:
    """Verify SqliteSaver persists thread state and supports rollback across separate service instances."""
    db_file = tmp_path / "checkpoints.sqlite"
    settings = Settings(
        GITHUB_TOKEN="",
        OPENAI_API_KEY="",
        CHECKPOINTER_TYPE="sqlite",
        SQLITE_DB_PATH=str(db_file),
        DRY_RUN_MODE=True,
    )

    cp1, conn1 = create_checkpointer(settings=settings)
    service1 = ReviewPipelineService(
        settings=settings, checkpointer=cp1, sqlite_conn=conn1
    )
    started = service1.start_review(
        pr_url="https://github.com/acme/demo/pull/99",
        branch="feature/sqlite-persist",
        code_contents={"src/data.py": SAMPLE_BUGGY_CODE},
    )
    tid = started.thread_id
    service1.close()

    # Re-open a brand new service instance backed by the same SQLite database
    cp2, conn2 = create_checkpointer(settings=settings)
    service2 = ReviewPipelineService(
        settings=settings, checkpointer=cp2, sqlite_conn=conn2
    )
    try:
        restored = service2.get_thread(tid)
        assert restored.status == "awaiting_approval"
        assert restored.next_nodes == ["apply_changes_node"]

        # Reject with feedback using fallback AST comprehension transform
        revised = service2.submit_review_decision(
            tid,
            approved=False,
            feedback="Use list comprehensions instead of loops",
        )
        assert revised.status == "awaiting_approval"
        assert revised.state["revision_count"] == 2
        assert (
            "[n * n for n in numbers if n > 0]"
            in revised.state["refactored_code"]["src/data.py"]
        )

        completed = service2.submit_review_decision(tid, approved=True)
        assert completed.status == "completed"
        assert completed.state["commit_result"]["status"] == "committed_dry_run"
    finally:
        service2.close()


def test_direct_unapproved_resume_routes_back_without_committing(
    test_settings: Settings,
    fake_llm: RecordingFakeLLM,
) -> None:
    """Safety net: even if graph is resumed directly with is_approved=False, apply_changes_node refuses to write."""
    mock_github = MagicMock(spec=GitHubClient)
    deps = PipelineDependencies(
        settings=test_settings,
        github_client=mock_github,
        refactor_engine=RefactorEngine(settings=test_settings, llm=fake_llm),
    )
    service = ReviewPipelineService(settings=test_settings, deps=deps)
    snap = service.start_review(
        pr_url="https://github.com/acme/demo/pull/55",
        branch="feature/safety",
        code_contents={"src/calc.py": SAMPLE_BUGGY_CODE},
    )

    config = {"configurable": {"thread_id": snap.thread_id}}
    service.graph.update_state(
        config,
        {
            "is_approved": False,
            "reviewer_feedback": "Use list comprehensions instead of loops",
        },
    )
    service.graph.invoke(None, config=config)

    after = service.get_thread(snap.thread_id)
    assert after.status == "awaiting_approval"
    assert after.next_nodes == ["apply_changes_node"]
    assert after.state["revision_count"] == 2
    mock_github.commit_files_to_branch.assert_not_called()
