"""
Thin wrapper around calling the FastAPI backend.

API_BASE_URL is read from the environment (see .env / .env.example) --
never hardcoded, per the assignment's requirements.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 60  # local LLM generation can take a while


class ApiError(Exception):
    """Raised when the backend is unreachable or returns an error."""


def check_health() -> dict:
    """GET /health -- used to show a status indicator in the sidebar."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach backend at {API_BASE_URL}: {exc}") from exc


def ask_question(question: str) -> dict:
    """POST /query -- returns {'answer': str, 'sources': list[str]}."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.ConnectionError as exc:
        raise ApiError(
            f"Could not connect to the backend at {API_BASE_URL}. "
            "Is it running? (uvicorn app.main:app --reload)"
        ) from exc
    except requests.Timeout as exc:
        raise ApiError(
            f"The backend took too long to respond (>{REQUEST_TIMEOUT_SECONDS}s). "
            "The local model may be slow on this hardware -- try again."
        ) from exc

    if resp.status_code == 422:
        raise ApiError("The question was rejected as invalid by the backend.")
    if resp.status_code == 503:
        raise ApiError("The backend's vector store isn't loaded yet. Wait a moment and try again.")
    if not resp.ok:
        raise ApiError(f"Backend returned an error ({resp.status_code}): {resp.text}")

    return resp.json()
