"""GitHub REST API client for Pull Request inspection and branch commit write-back."""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

PR_URL_PATTERN = re.compile(
    r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<pr_number>\d+)(?:/.*)?$"
)
REPO_URL_PATTERN = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub REST API request fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class PullRequestRef:
    """Parsed repository and Pull Request coordinates."""

    owner: str
    repo: str
    pr_number: int | None = None
    branch: str = "main"
    pr_url: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_github_url(
    pr_url: str | None = None,
    repository_url: str | None = None,
    branch: str | None = None,
    pr_number: int | None = None,
) -> PullRequestRef:
    """Parse owner, repo, and optional PR number from GitHub URLs."""
    target_url = (pr_url or repository_url or "").strip()
    if not target_url:
        raise ValueError("Either pr_url or repository_url must be provided.")

    parsed = urlparse(target_url)
    path = parsed.path.rstrip("/")

    pr_match = PR_URL_PATTERN.match(path)
    if pr_match:
        owner = pr_match.group("owner")
        repo = pr_match.group("repo")
        resolved_pr = pr_number or int(pr_match.group("pr_number"))
        canonical_pr_url = pr_url or f"https://github.com/{owner}/{repo}/pull/{resolved_pr}"
        return PullRequestRef(
            owner=owner,
            repo=repo,
            pr_number=resolved_pr,
            branch=branch or "main",
            pr_url=canonical_pr_url,
        )

    repo_match = REPO_URL_PATTERN.match(path)
    if repo_match:
        owner = repo_match.group("owner")
        repo = repo_match.group("repo")
        canonical_pr_url = (
            pr_url
            if pr_url
            else (
                f"https://github.com/{owner}/{repo}/pull/{pr_number}"
                if pr_number
                else f"https://github.com/{owner}/{repo}"
            )
        )
        return PullRequestRef(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            branch=branch or "main",
            pr_url=canonical_pr_url,
        )

    raise ValueError(f"Unable to parse GitHub owner/repository from URL: {target_url}")


class GitHubClient:
    """Client for reading Pull Request files and committing refactored code via GitHub API."""

    def __init__(
        self,
        settings: Settings | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._external_client = http_client

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "autonomous-code-review-agent/1.0",
        }
        if self.settings.has_github_credentials:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        return headers

    def _client(self) -> httpx.Client:
        if self._external_client is not None:
            return self._external_client
        return httpx.Client(
            base_url=self.settings.github_api_base_url,
            headers=self._build_headers(),
            timeout=self.settings.github_timeout_seconds,
        )

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[Any]:
        own_client = self._external_client is None
        client = self._client()
        try:
            response = client.request(
                method,
                endpoint,
                params=params,
                json=json_body,
                headers=self._build_headers(),
            )
            if response.status_code >= 400:
                raise GitHubAPIError(
                    f"GitHub API {method} {endpoint} failed with {response.status_code}: {response.text}",
                    status_code=response.status_code,
                )
            return response.json()
        except httpx.HTTPError as exc:
            raise GitHubAPIError(f"GitHub API transport error for {endpoint}: {exc}") from exc
        finally:
            if own_client:
                client.close()

    def fetch_pr_files(
        self,
        pr_ref: PullRequestRef,
        extensions: tuple[str, ...] = (".py",),
    ) -> tuple[dict[str, str], str]:
        """Fetch modified source files from a GitHub Pull Request or branch.

        Returns a tuple of ``(code_contents, resolved_head_branch)``.
        """
        head_branch = pr_ref.branch

        if pr_ref.pr_number is not None:
            pr_meta = self._request(
                "GET",
                f"/repos/{pr_ref.owner}/{pr_ref.repo}/pulls/{pr_ref.pr_number}",
            )
            if isinstance(pr_meta, dict):
                head_info = pr_meta.get("head") or {}
                if head_info.get("ref"):
                    head_branch = head_info["ref"]

            files_payload = self._request(
                "GET",
                f"/repos/{pr_ref.owner}/{pr_ref.repo}/pulls/{pr_ref.pr_number}/files",
            )
            if not isinstance(files_payload, list):
                raise GitHubAPIError("Unexpected response format when listing PR files.")

            code_contents: dict[str, str] = {}
            for file_entry in files_payload:
                filename = file_entry.get("filename", "")
                status = file_entry.get("status", "")
                if status == "removed" or not filename.endswith(extensions):
                    continue
                content = self.fetch_file_content(
                    pr_ref.owner, pr_ref.repo, filename, ref=head_branch
                )
                code_contents[filename] = content

            logger.info(
                "Fetched %d Python file(s) from PR %s#%s (branch=%s)",
                len(code_contents),
                pr_ref.full_name,
                pr_ref.pr_number,
                head_branch,
            )
            return code_contents, head_branch

        # Fallback when only repo + branch is given: list files from root tree
        tree_payload = self._request(
            "GET",
            f"/repos/{pr_ref.owner}/{pr_ref.repo}/git/trees/{head_branch}",
            params={"recursive": "1"},
        )
        if not isinstance(tree_payload, dict):
            raise GitHubAPIError("Unexpected response format when fetching repository tree.")

        code_contents = {}
        for item in tree_payload.get("tree", []):
            path = item.get("path", "")
            if item.get("type") == "blob" and path.endswith(extensions):
                code_contents[path] = self.fetch_file_content(
                    pr_ref.owner, pr_ref.repo, path, ref=head_branch
                )
        return code_contents, head_branch

    def fetch_file_content(
        self,
        owner: str,
        repo: str,
        file_path: str,
        ref: str,
    ) -> str:
        """Retrieve and decode a single file from the GitHub Contents API."""
        payload = self._request(
            "GET",
            f"/repos/{owner}/{repo}/contents/{file_path}",
            params={"ref": ref},
        )
        if not isinstance(payload, dict) or "content" not in payload:
            raise GitHubAPIError(f"File content missing in GitHub response for {file_path}")

        encoding = payload.get("encoding", "base64")
        raw_content = payload["content"]
        if encoding == "base64":
            return base64.b64decode(raw_content).decode("utf-8")
        return str(raw_content)

    def commit_files_to_branch(
        self,
        pr_ref: PullRequestRef,
        files: dict[str, str],
        commit_message: str = "refactor: apply autonomous code review fixes",
    ) -> dict[str, Any]:
        """Commit approved refactored files back to the target GitHub branch."""
        if not files:
            return {
                "status": "no_changes",
                "branch": pr_ref.branch,
                "committed_files": [],
                "commit_shas": [],
            }

        if self.settings.dry_run_mode and not self.settings.has_github_credentials:
            digest_input = f"{pr_ref.full_name}:{pr_ref.branch}:{sorted(files.items())}"
            simulated_sha = hashlib.sha1(digest_input.encode("utf-8")).hexdigest()
            logger.info(
                "[DRY-RUN] Simulated commit %s to %s@%s for %d file(s)",
                simulated_sha[:8],
                pr_ref.full_name,
                pr_ref.branch,
                len(files),
            )
            return {
                "status": "committed_dry_run",
                "repository": pr_ref.full_name,
                "branch": pr_ref.branch,
                "commit_sha": simulated_sha,
                "committed_files": list(files.keys()),
                "message": commit_message,
            }

        committed_files: list[str] = []
        commit_shas: list[str] = []

        for file_path, new_content in files.items():
            existing_sha: str | None = None
            try:
                existing = self._request(
                    "GET",
                    f"/repos/{pr_ref.owner}/{pr_ref.repo}/contents/{file_path}",
                    params={"ref": pr_ref.branch},
                )
                if isinstance(existing, dict):
                    existing_sha = existing.get("sha")
            except GitHubAPIError as exc:
                if exc.status_code != 404:
                    raise

            encoded_content = base64.b64encode(new_content.encode("utf-8")).decode("ascii")
            body: dict[str, Any] = {
                "message": f"{commit_message} ({file_path})",
                "content": encoded_content,
                "branch": pr_ref.branch,
            }
            if existing_sha:
                body["sha"] = existing_sha

            result = self._request(
                "PUT",
                f"/repos/{pr_ref.owner}/{pr_ref.repo}/contents/{file_path}",
                json_body=body,
            )
            committed_files.append(file_path)
            if isinstance(result, dict):
                commit_info = result.get("commit") or {}
                if commit_info.get("sha"):
                    commit_shas.append(commit_info["sha"])

        logger.info(
            "Committed %d refactored file(s) to %s@%s",
            len(committed_files),
            pr_ref.full_name,
            pr_ref.branch,
        )
        return {
            "status": "committed",
            "repository": pr_ref.full_name,
            "branch": pr_ref.branch,
            "commit_sha": commit_shas[-1] if commit_shas else None,
            "commit_shas": commit_shas,
            "committed_files": committed_files,
            "message": commit_message,
        }
