"""Intent scoring with Aurelio semantic-router (scoring and logging only; it does not change routing).

Routes (kinds of question + example utterances) live in app/data/intent_routes.yaml, in semantic-router's own
file format, so intents are added by editing that file — no code changes. The encoder is the same local
bge-small model as the topic check (already on this server / in the Docker image): no API calls, no tokens.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

DEFAULT_ROUTES_FILE = Path(__file__).resolve().parents[1] / "data" / "intent_routes.yaml"


@dataclass
class IntentScore:
    route: str  # best-matching route
    confidence: float  # its similarity score
    margin: float  # lead over the second-best route (small = unsure)
    scores: dict[str, float]  # every route's score


class SemanticIntentRouter:
    def __init__(self, routes_file: Path | str = DEFAULT_ROUTES_FILE, cache_dir: str | None = None, router=None):
        self.routes_file = Path(routes_file)
        self._cache_dir = cache_dir
        self._router = router  # injectable for tests
        # read once: routes, their direct-answer metadata and the `direct:` rules (shared with DirectQueryBuilder)
        self.config: dict = (
            yaml.safe_load(self.routes_file.read_text(encoding="utf-8")) if self.routes_file.exists() else {}
        )
        self.route_names: list[str] = [r["name"] for r in self.config.get("routes", [])]

    @property
    def ready(self) -> bool:
        return self._router is not None

    def load(self) -> None:
        """Build the router from the YAML file (blocking; call via start())."""
        from semantic_router import IndexedRouter, Route
        from semantic_router.encoders.fastembed import FastEmbedEncoder

        config = self.config
        routes = [
            Route(
                name=r["name"],
                utterances=r["utterances"],
                description=r.get("description"),
                metadata=r.get("metadata") or {},
            )
            for r in config.get("routes", [])
        ]
        encoder = FastEmbedEncoder(name=config["encoder"]["name"], cache_dir=self._cache_dir)
        self._router = IndexedRouter(
            encoder=encoder,
            routes=routes,
            top_k=config.get("top_k", 5),
            aggregation=config.get("aggregation", "mean"),
        )
        self.route_names = [r.name for r in routes]

    async def start(self) -> None:
        try:
            await asyncio.to_thread(self.load)
            log.info("semantic router ready", extra={"routes": self.route_names, "file": str(self.routes_file)})
        except Exception:
            log.exception("semantic router could not load; questions are not scored")

    def score(self, text: str) -> IntentScore | None:
        if not self.ready:
            return None
        choices = self._router(text, limit=None)
        choices = choices if isinstance(choices, list) else [choices]
        scores = {c.name: round(float(c.confidence or 0.0), 4) for c in choices if c.name}
        if not scores:
            return None
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        return IntentScore(ranked[0][0], ranked[0][1], round(ranked[0][1] - second, 4), scores)
