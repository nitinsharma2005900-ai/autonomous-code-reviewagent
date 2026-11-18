"""FastAPI application factory for the Autonomous Code Review & Refactoring Agent."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.graph.workflow import ReviewPipelineService
from app.logging_config import configure_logging


def create_app(
    settings: Settings | None = None,
    pipeline_service: ReviewPipelineService | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owns_service = pipeline_service is None
        service = pipeline_service or ReviewPipelineService(settings=resolved_settings)
        application.state.pipeline_service = service
        try:
            yield
        finally:
            if owns_service:
                service.close()

    application = FastAPI(
        title="Autonomous Code Review & Refactoring Agent",
        description=(
            "LangGraph-powered stateful code review pipeline with sandboxed flake8/black "
            "static analysis, LLM-assisted refactoring, persistent checkpoints, "
            "human-in-the-loop (HITL) approval gates with rollback loops, and GitHub API integration."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.state.settings = resolved_settings
    if pipeline_service is not None:
        application.state.pipeline_service = pipeline_service

    application.include_router(router)
    return application


app = create_app()
