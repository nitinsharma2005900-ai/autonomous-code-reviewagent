"""Unit tests for GitHub REST API client PR file retrieval and commit write-back."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from app.config import Settings
from app.services.github_client import (
    GitHubAPIError,
    GitHubClient,
    PullRequestRef,
    parse_github_url,
)


def test_parse_github_url_pr_and_repo() -> None:
    pr_ref = parse_github_url(
        pr_url="https://github.com/acme/review-demo/pull/42", branch="feat/lint"
    )
    assert pr_ref.owner == "acme"
    assert pr_ref.repo == "review-demo"
    assert pr_ref.pr_number == 42
    assert pr_ref.branch == "feat/lint"

    repo_ref = parse_github_url(
        repository_url="https://github.com/acme/review-demo.git", branch="dev"
    )
    assert repo_ref.owner == "acme"
    assert repo_ref.repo == "review-demo"
    assert repo_ref.pr_number is None
    assert repo_ref.branch == "dev"

    with pytest.raises(ValueError):
        parse_github_url(pr_url="not-a-valid-url")


def test_fetch_pr_files_from_github_api() -> None:
    """Verify PR metadata, file listing, base64 decoding, and filtering of non-Python/removed files."""
    sample_py = "def ok():\n    return 1\n"
    encoded_py = base64.b64encode(sample_py.encode("utf-8")).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/acme/demo/pulls/7":
            return httpx.Response(200, json={"head": {"ref": "feature/cleanup"}})
        if path == "/repos/acme/demo/pulls/7/files":
            return httpx.Response(
                200,
                json=[
                    {"filename": "app/util.py", "status": "modified"},
                    {"filename": "app/deleted.py", "status": "removed"},
                    {"filename": "README.md", "status": "modified"},
                ],
            )
        if path == "/repos/acme/demo/contents/app/util.py":
            assert request.url.params.get("ref") == "feature/cleanup"
            return httpx.Response(
                200, json={"encoding": "base64", "content": encoded_py}
            )
        return httpx.Response(404, json={"message": "Not Found"})

    settings = Settings(GITHUB_TOKEN="ghp_real_test_token_123", DRY_RUN_MODE=False)
    http_client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(handler),
    )
    client = GitHubClient(settings=settings, http_client=http_client)

    pr_ref = PullRequestRef(owner="acme", repo="demo", pr_number=7, branch="main")
    files, head_branch = client.fetch_pr_files(pr_ref)

    assert head_branch == "feature/cleanup"
    assert list(files.keys()) == ["app/util.py"]
    assert files["app/util.py"] == sample_py
    http_client.close()


def test_commit_files_to_branch_calls_github_contents_put() -> None:
    """Verify commit_files_to_branch reads existing blob SHA and PUTs base64 updated content."""
    recorded_puts: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if (
            request.method == "GET"
            and request.url.path == "/repos/acme/demo/contents/src/mod.py"
        ):
            return httpx.Response(200, json={"sha": "oldblobsha123"})
        if (
            request.method == "PUT"
            and request.url.path == "/repos/acme/demo/contents/src/mod.py"
        ):
            body = json.loads(request.content.decode("utf-8"))
            recorded_puts.append(body)
            return httpx.Response(200, json={"commit": {"sha": "newcommitsha999"}})
        return httpx.Response(404, json={"message": "Not Found"})

    settings = Settings(GITHUB_TOKEN="ghp_real_test_token_123", DRY_RUN_MODE=False)
    http_client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(handler),
    )
    client = GitHubClient(settings=settings, http_client=http_client)

    pr_ref = PullRequestRef(
        owner="acme", repo="demo", pr_number=7, branch="feature/cleanup"
    )
    result = client.commit_files_to_branch(
        pr_ref,
        files={"src/mod.py": "x = 1\n"},
        commit_message="refactor: clean code",
    )

    assert result["status"] == "committed"
    assert result["commit_sha"] == "newcommitsha999"
    assert len(recorded_puts) == 1
    assert recorded_puts[0]["sha"] == "oldblobsha123"
    assert recorded_puts[0]["branch"] == "feature/cleanup"
    assert base64.b64decode(recorded_puts[0]["content"]).decode("utf-8") == "x = 1\n"
    http_client.close()


def test_github_client_raises_on_api_error() -> None:
    """Ensure HTTP error status codes surface as GitHubAPIError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    settings = Settings(GITHUB_TOKEN="ghp_invalid", DRY_RUN_MODE=False)
    http_client = httpx.Client(
        base_url="https://api.github.com",
        transport=httpx.MockTransport(handler),
    )
    client = GitHubClient(settings=settings, http_client=http_client)

    with pytest.raises(GitHubAPIError) as exc_info:
        client.fetch_file_content("acme", "demo", "main.py", ref="main")

    assert exc_info.value.status_code == 401
    http_client.close()
