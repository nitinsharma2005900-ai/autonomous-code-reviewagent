"""Shared pytest fixtures for sandbox, LangGraph, GitHub API, and FastAPI tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings
from app.graph.nodes import PipelineDependencies
from app.graph.workflow import ReviewPipelineService
from app.main import create_app
from app.services.github_client import GitHubClient
from app.services.refactor_engine import RefactorEngine

SAMPLE_BUGGY_CODE = (
    "import os\n"
    "def filter_positive_squares(numbers):\n"
    "    result=[]\n"
    "    for n in numbers:\n"
    "        if n>0:\n"
    "            result.append(n*n)\n"
    "    return result\n"
)


@dataclass
class FakeLLMResponse:
    content: str


@dataclass
class RecordingFakeLLM:
    """Deterministic fake LLM that records prompts and reacts to reviewer feedback."""

    calls: list[list[Any]] = field(default_factory=list)

    def invoke(self, messages: list[Any]) -> FakeLLMResponse:
        self.calls.append(messages)
        prompt_text = "\n".join(str(getattr(m, "content", m)) for m in messages)
        if "list comprehension" in prompt_text.lower():
            return FakeLLMResponse(
                content=(
                    "```python\n"
                    "def filter_positive_squares(numbers):\n"
                    "    return [n * n for n in numbers if n > 0]\n"
                    "```\n"
                )
            )
        return FakeLLMResponse(
            content=(
                "```python\n"
                "def filter_positive_squares(numbers):\n"
                "    result = []\n"
                "    for n in numbers:\n"
                "        if n > 0:\n"
                "            result.append(n * n)\n"
                "    return result\n"
                "```\n"
            )
        )


@pytest.fixture
def test_settings() -> Settings:
    """Isolated settings using in-memory checkpointer and dry-run mode."""
    return Settings(
        GITHUB_TOKEN="",
        OPENAI_API_KEY="",
        CHECKPOINTER_TYPE="memory",
        DRY_RUN_MODE=True,
        LOG_LEVEL="WARNING",
    )


@pytest.fixture
def fake_llm() -> RecordingFakeLLM:
    return RecordingFakeLLM()


@pytest.fixture
def pipeline_service(
    test_settings: Settings,
    fake_llm: RecordingFakeLLM,
) -> ReviewPipelineService:
    """ReviewPipelineService wired with MemorySaver and deterministic fake LLM."""
    deps = PipelineDependencies(
        settings=test_settings,
        github_client=GitHubClient(settings=test_settings),
        refactor_engine=RefactorEngine(settings=test_settings, llm=fake_llm),
    )
    return ReviewPipelineService(
        settings=test_settings,
        deps=deps,
        checkpointer=MemorySaver(),
    )


@pytest.fixture
def api_client(
    test_settings: Settings,
    pipeline_service: ReviewPipelineService,
) -> TestClient:
    """FastAPI TestClient backed by the isolated pipeline_service."""
    application = create_app(settings=test_settings, pipeline_service=pipeline_service)
    with TestClient(application) as client:
        yield client
