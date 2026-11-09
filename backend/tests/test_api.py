from fastapi.testclient import TestClient

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
        assert c.get("/api/meta", headers={"Authorization": "Bearer " + "s" * 32}).status_code == 200
        assert c.get("/api/health").status_code == 200  # health stays public for load balancers
