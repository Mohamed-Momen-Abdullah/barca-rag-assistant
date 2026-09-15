"""
Tests for the /query and /health endpoints.

These tests mock the retrieval/generation services rather than requiring a
live Ollama instance or a real vector store, so `pytest` can run in CI or on
a fresh clone without any model downloads.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.retrieval import retrieval_service

client = TestClient(app)


def test_query_happy_path():
    """POST /query with a valid question returns a grounded answer + sources."""
    fake_hits = [
        {
            "text": "FC Barcelona was founded in 1899 by Joan Gamper.",
            "metadata": {"source_group": "wikipedia", "filename": "history_of_fc_barcelona.txt"},
            "distance": 0.1,
        }
    ]
    # is_loaded is a read-only property derived from _collection, so we set
    # the underlying attribute directly rather than patching the property.
    with patch.object(
        retrieval_service, "_collection", object()
    ), patch.object(retrieval_service, "retrieve", return_value=fake_hits), patch(
        "app.services.generation.ollama.generate"
    ) as mock_generate:
        mock_generate.return_value = {
            "response": "FC Barcelona was founded in 1899 by Joan Gamper. [Source 1]"
        }
        response = client.post("/query", json={"question": "When was FC Barcelona founded?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert "history_of_fc_barcelona.txt" in body["sources"]


def test_query_invalid_input_returns_422():
    """POST /query missing the required 'question' field returns 422."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_health_endpoint():
    """GET /health reports whether the vector store is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "vector_store_loaded" in body