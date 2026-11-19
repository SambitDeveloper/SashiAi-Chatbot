"""Local topic check: stops clearly off-topic questions before they reach the language model.

Uses a small open-source embedding model (BAAI/bge-small-en-v1.5, ~65 MB) that runs on this server:
no API, no tokens, and the question never leaves the server. It turns sentences into vectors and compares
the question's meaning with reference examples (topic_examples.py).

Conservative by design:
  • A question is blocked only when it is clearly closer to the off-topic examples than to the resident
    examples (by at least `margin`). Anything borderline goes to the language model as before.
  • If the model cannot load, the gate disables itself and every question goes to the language model.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.nlu.topic_examples import OFF_TOPIC, RESIDENT

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: str | None):
        from fastembed import TextEmbedding  # imported lazily: only needed when the gate is enabled

        self._model = TextEmbedding(model_name, cache_dir=cache_dir)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array(list(self._model.embed(texts)), dtype=np.float32)


@dataclass
class GateResult:
    blocked: bool
    resident_score: float
    off_topic_score: float


def _normalise(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


class TopicGate:
    def __init__(self, margin: float, embedder_factory=None, model_name: str = "", cache_dir: str | None = None):
        self.margin = margin
        self._factory = embedder_factory or (lambda: FastEmbedEmbedder(model_name, cache_dir))
        self._embedder: Embedder | None = None
        self._resident: np.ndarray | None = None
        self._off_topic: np.ndarray | None = None
        self.model_name = model_name
        self.blocked_count = 0

    @property
    def ready(self) -> bool:
        return self._embedder is not None

    @property
    def embedder(self) -> Embedder | None:
        """The loaded BGE embedder (None until start() finished), shared with the semantic plan cache so the same
        model, preprocessing and vectors are used for storing and searching templates."""
        return self._embedder

    def load(self) -> None:
        """Load the model and embed the reference examples (blocking; call via start())."""
        embedder = self._factory()
        self._resident = _normalise(embedder.embed(RESIDENT))
        self._off_topic = _normalise(embedder.embed(OFF_TOPIC))
        self._embedder = embedder

    async def start(self) -> None:
        try:
            await asyncio.to_thread(self.load)
            log.info("topic check ready", extra={"model": self.model_name, "margin": self.margin})
        except Exception:
            log.exception("topic check could not load; all questions will go to the language model")

    def check(self, text: str) -> GateResult:
        if not self.ready:
            return GateResult(False, 0.0, 0.0)
        q = _normalise(self._embedder.embed([text]))[0]
        resident = float((self._resident @ q).max())
        off_topic = float((self._off_topic @ q).max())
        blocked = off_topic - resident >= self.margin
        if blocked:
            self.blocked_count += 1
        return GateResult(blocked, resident, off_topic)
