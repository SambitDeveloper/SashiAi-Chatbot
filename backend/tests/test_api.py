import logging

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import FakeTranslator


def client_for(settings, repo, translator=None, **overrides) -> TestClient:
    for k, v in overrides.items():
        setattr(settings, k, v)
    return TestClient(create_app(settings=settings, repo=repo, translator=translator, use_default_translator=False))


def test_health_ready_meta(settings, repo):
    with client_for(settings, repo) as c:
        assert c.get("/api/health").json() == {"status": "ok"}
        assert c.get("/api/ready").json()["llmConfigured"] is False
        assert c.get("/api/meta").json()["totalResidents"] == 88


def test_query_endpoint(settings, repo):
    fake = FakeTranslator(
        {"How many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}}}
    )
    with client_for(settings, repo, fake) as c:
        r = c.post("/api/query", json={"query": "How many residents are active?", "session_id": "abc-123"})
        assert r.status_code == 200
        assert r.json()["response"]["count"] == 68


def test_input_validation(settings, repo):
    with client_for(settings, repo) as c:
        assert c.post("/api/query", json={"query": "   "}).status_code == 400
        assert c.post("/api/query", json={"query": "x" * 501}).status_code == 400
        assert c.post("/api/query", json={"query": "hi", "session_id": "bad id!"}).status_code == 400
        assert c.post("/api/query", json={"query": "hi", "unexpected": 1}).status_code == 400


def test_security_headers(settings, repo):
    with client_for(settings, repo) as c:
        h = c.get("/api/health").headers
        assert h["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in h["content-security-policy"]


def test_data_browser_disabled_by_default(settings, repo):
    with client_for(settings, repo) as c:
        assert c.get("/api/residents").status_code == 403


def test_data_browser_hides_contacts(settings, repo):
    with client_for(settings, repo, enable_data_browser=True) as c:
        rows = c.get("/api/residents").json()["residents"]
        assert len(rows) == 88 and all(r["emergencyContact"]["phone"] == "" for r in rows)


def test_api_token(settings, repo):
    with client_for(settings, repo, api_token="s" * 32) as c:
        assert c.get("/api/meta").status_code == 401
        # changed 2026-11-27: Authorization carries the user's own access token; the service secret is x-api-key
        assert c.get("/api/meta", headers={"x-api-key": "s" * 32}).status_code == 200
        assert c.get("/api/health").status_code == 200  # health stays public for load balancers


@pytest.mark.parametrize("enabled", [False, True])
def test_query_responses_are_logged_exactly_as_returned_only_when_enabled(settings, repo, caplog, enabled):
    fake = FakeTranslator({"How many residents are active?": {"action": "count", "filters": {"status": {"include": ["active"]}}}})
    route_log = logging.getLogger("app.api.routes")
    with client_for(settings, repo, fake, log_api_responses=enabled) as c:
        route_log.addHandler(caplog.handler)  # after create_app(), which reconfigures logging
        try:
            body = c.post("/api/query", json={"query": "How many residents are active?"}).json()
        finally:
            route_log.removeHandler(caplog.handler)
    logged = [r for r in caplog.records if r.getMessage() == "api response"]
    assert len(logged) == int(enabled)
    if enabled:
        assert logged[0].response == body  # the same object the UI receives


def test_response_logging_is_refused_in_production():
    with pytest.raises(ValueError, match="not allowed in production"):
        Settings(_env_file=None, environment="production", llm_enabled=False, log_api_responses=True)
