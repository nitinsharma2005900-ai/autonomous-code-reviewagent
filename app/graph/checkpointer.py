"""Persistent checkpointer factory supporting SqliteSaver and MemorySaver."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def create_checkpointer(
    settings: Settings | None = None,
    checkpointer_type: Literal["sqlite", "memory"] | None = None,
    sqlite_path: str | Path | None = None,
) -> tuple[BaseCheckpointSaver, sqlite3.Connection | None]:
    """Instantiate the configured LangGraph checkpointer.

    Returns ``(checkpointer, sqlite_connection)`` so callers can cleanly close
    the SQLite connection on application shutdown.
    """
    cfg = settings or get_settings()
    selected = checkpointer_type or cfg.checkpointer_type

    if selected == "memory":
        logger.info("Initialized in-memory LangGraph checkpointer (MemorySaver)")
        return MemorySaver(), None

    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(sqlite_path) if sqlite_path is not None else cfg.resolved_sqlite_path
    if str(db_path) != ":memory:":
        db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    logger.info("Initialized SQLite LangGraph checkpointer at %s", db_path)
    return saver, conn
