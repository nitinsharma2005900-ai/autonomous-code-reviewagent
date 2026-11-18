"""LangGraph StateGraph construction, HITL interrupt gates, and checkpointer rollback service."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.config import Settings, get_settings
from app.graph.checkpointer import create_checkpointer
from app.graph.nodes import (
    PipelineDependencies,
    create_apply_changes_node,
    create_fetch_pr_node,
    create_lint_code_node,
    create_propose_refactor_node,
)
from app.graph.state import ReviewState

logger = logging.getLogger(__name__)


class ThreadNotFoundError(LookupError):
    """Raised when a review thread_id does not exist in the checkpointer."""


class ThreadAlreadyCompletedError(RuntimeError):
    """Raised when attempting to submit a decision to a finished review thread."""


@dataclass
class ReviewSnapshot:
    """Normalized view of a LangGraph review thread state."""

    thread_id: str
    status: str
    next_nodes: list[str]
    state: dict[str, Any]
    checkpoint_id: str | None = None


def route_after_apply(state: ReviewState) -> str:
    """Safety conditional edge: re-route to refactoring if apply was entered unapproved."""
    if not state.get("is_approved"):
        return "propose_refactor_node"
    return END


def build_review_graph(
    checkpointer: BaseCheckpointSaver,
    deps: PipelineDependencies | None = None,
) -> Any:
    """Compile the stateful LangGraph review pipeline with HITL interrupt before writes."""
    resolved_deps = deps or PipelineDependencies()

    builder = StateGraph(ReviewState)
    builder.add_node("fetch_pr_node", create_fetch_pr_node(resolved_deps))
    builder.add_node("lint_code_node", create_lint_code_node(resolved_deps))
    builder.add_node("propose_refactor_node", create_propose_refactor_node(resolved_deps))
    builder.add_node("apply_changes_node", create_apply_changes_node(resolved_deps))

    builder.add_edge(START, "fetch_pr_node")
    builder.add_edge("fetch_pr_node", "lint_code_node")
    builder.add_edge("lint_code_node", "propose_refactor_node")
    builder.add_edge("propose_refactor_node", "apply_changes_node")
    builder.add_conditional_edges(
        "apply_changes_node",
        route_after_apply,
        {
            "propose_refactor_node": "propose_refactor_node",
            END: END,
        },
    )

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_changes_node"],
    )


class ReviewPipelineService:
    """High-level orchestrator managing LangGraph threads, HITL gates, and rollbacks."""

    def __init__(
        self,
        settings: Settings | None = None,
        deps: PipelineDependencies | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        sqlite_conn: sqlite3.Connection | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.deps = deps or PipelineDependencies(settings=self.settings)
        if checkpointer is None:
            self.checkpointer, self._sqlite_conn = create_checkpointer(settings=self.settings)
        else:
            self.checkpointer = checkpointer
            self._sqlite_conn = sqlite_conn

        self.graph = build_review_graph(checkpointer=self.checkpointer, deps=self.deps)
        self._known_threads: list[str] = []

    def close(self) -> None:
        """Close underlying SQLite connection if owned by this service."""
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None

    @staticmethod
    def _thread_config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _derive_status(next_nodes: tuple[str, ...]) -> str:
        if "apply_changes_node" in next_nodes:
            return "awaiting_approval"
        if next_nodes:
            return "in_progress"
        return "completed"

    def _snapshot_from_graph(self, thread_id: str) -> ReviewSnapshot:
        config = self._thread_config(thread_id)
        snapshot = self.graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise ThreadNotFoundError(f"Review thread '{thread_id}' was not found.")

        next_nodes = list(snapshot.next)
        checkpoint_id = None
        if snapshot.config and "configurable" in snapshot.config:
            checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")

        return ReviewSnapshot(
            thread_id=thread_id,
            status=self._derive_status(snapshot.next),
            next_nodes=next_nodes,
            state=dict(snapshot.values),
            checkpoint_id=checkpoint_id,
        )

    def start_review(
        self,
        *,
        pr_url: str,
        branch: str = "main",
        pr_number: int | None = None,
        code_contents: dict[str, str] | None = None,
        thread_id: str | None = None,
    ) -> ReviewSnapshot:
        """Trigger a new LangGraph review thread and halt before `apply_changes_node`."""
        tid = thread_id or str(uuid.uuid4())
        config = self._thread_config(tid)

        initial_state: ReviewState = {
            "pr_url": pr_url,
            "branch": branch,
            "pr_number": pr_number,
            "code_contents": dict(code_contents) if code_contents else {},
            "lint_errors": {},
            "refactored_code": {},
            "is_approved": None,
            "reviewer_feedback": None,
            "revision_count": 0,
            "commit_result": None,
            "history": [],
        }

        logger.info("Starting review thread %s for PR %s (branch=%s)", tid, pr_url, branch)
        self.graph.invoke(initial_state, config=config)

        if tid not in self._known_threads:
            self._known_threads.append(tid)

        result = self._snapshot_from_graph(tid)
        logger.info(
            "Thread %s paused at checkpoint %s (status=%s, next=%s)",
            tid,
            result.checkpoint_id,
            result.status,
            result.next_nodes,
        )
        return result

    def submit_review_decision(
        self,
        thread_id: str,
        *,
        approved: bool,
        feedback: str | None = None,
    ) -> ReviewSnapshot:
        """Process a human reviewer decision: either commit changes or rollback and re-refactor."""
        current = self._snapshot_from_graph(thread_id)
        if current.status == "completed":
            raise ThreadAlreadyCompletedError(
                f"Review thread '{thread_id}' is already completed."
            )

        config = self._thread_config(thread_id)
        now = datetime.now(timezone.utc).isoformat()

        if approved:
            logger.info("Thread %s approved by reviewer; resuming apply_changes_node.", thread_id)
            self.graph.update_state(
                config,
                {
                    "is_approved": True,
                    "reviewer_feedback": feedback,
                    "history": [
                        {
                            "node": "human_review_gate",
                            "timestamp": now,
                            "decision": "approved",
                            "feedback": feedback,
                        }
                    ],
                },
            )
            self.graph.invoke(None, config=config)
            return self._snapshot_from_graph(thread_id)

        # Rejection path: persist reviewer feedback and rollback checkpoint to lint_code_node
        # so the graph re-routes directly to propose_refactor_node and pauses again at the gate.
        logger.info(
            "Thread %s rejected with feedback=%r; rolling back to propose_refactor_node.",
            thread_id,
            feedback,
        )
        self.graph.update_state(
            config,
            {
                "is_approved": False,
                "reviewer_feedback": feedback or "Please improve code quality and readability.",
                "history": [
                    {
                        "node": "human_review_gate",
                        "timestamp": now,
                        "decision": "rejected",
                        "feedback": feedback,
                        "action": "rollback_to_propose_refactor_node",
                    }
                ],
            },
            as_node="lint_code_node",
        )
        self.graph.invoke(None, config=config)
        updated = self._snapshot_from_graph(thread_id)
        logger.info(
            "Thread %s regenerated refactor revision #%s and paused again (next=%s)",
            thread_id,
            updated.state.get("revision_count"),
            updated.next_nodes,
        )
        return updated

    def get_thread(self, thread_id: str) -> ReviewSnapshot:
        """Return the latest persisted snapshot for a thread."""
        return self._snapshot_from_graph(thread_id)

    def list_threads(self) -> list[ReviewSnapshot]:
        """Return snapshots for all threads started in this session."""
        snapshots: list[ReviewSnapshot] = []
        for tid in self._known_threads:
            try:
                snapshots.append(self._snapshot_from_graph(tid))
            except ThreadNotFoundError:
                continue
        return snapshots

    def get_checkpoint_history(self, thread_id: str) -> list[dict[str, Any]]:
        """Return ordered checkpoint metadata entries for auditing/time-travel."""
        # Validate thread exists first
        self._snapshot_from_graph(thread_id)
        config = self._thread_config(thread_id)
        entries: list[dict[str, Any]] = []
        for state_snap in self.graph.get_state_history(config):
            cp_id = (
                state_snap.config.get("configurable", {}).get("checkpoint_id")
                if state_snap.config
                else None
            )
            entries.append(
                {
                    "checkpoint_id": cp_id,
                    "next": list(state_snap.next),
                    "revision_count": state_snap.values.get("revision_count", 0),
                    "is_approved": state_snap.values.get("is_approved"),
                    "reviewer_feedback": state_snap.values.get("reviewer_feedback"),
                }
            )
        return entries
