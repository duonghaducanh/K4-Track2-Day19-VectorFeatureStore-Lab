"""HybridMemoryAgent — episodic memory (vector) + stable profile (feature store).

This is the minimal POC behind bonus/ARCHITECTURE.md. It deliberately REUSES
the lab's own building blocks instead of reinventing them:

  * `app.embeddings.Embedder`   -> the same embedding backend as NB1/NB2
  * Qdrant (in-memory)          -> per-user filtered vector memory, as in NB5
  * `feast.FeatureStore`        -> online profile lookup, as in NB4

Design rule (from the brief): optimize *clarity*, not speed. Every method is
short and reads like the architecture diagram.

Privacy: every read and write is filtered by `user_id`. This is HARD
isolation, not the soft metadata filter that NB7 shows leaking. There is no
`namespaced=False` switch here on purpose -- a memory leak between users is a
legal incident (Nghi dinh 13), not a caching bug.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

from qdrant_client import QdrantClient, models

from app.embeddings import Embedder

# ── constants ─────────────────────────────────────────────────────────────
MEMORY_COLLECTION = "lab19_bonus_memory"
CHUNK_TOKENS = 256          # QD1: block ~256 token, semantic break
CHUNK_OVERLAP = 32          # keeps an idea from being cut in half
TOP_K = 5

DEFAULT_FEATURES = [
    "user_profile_features:topic_affinity",
    "user_profile_features:preferred_language",
    "user_profile_features:reading_speed_wpm",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]

# Fallback profile so the POC runs even before `feast apply` (NB4). The demo
# must exit 0 on a clean checkout; a hard dependency on Feast would break that.
_FALLBACK_PROFILE = {
    "topic_affinity": ["cloud"],
    "preferred_language": ["vi"],
    "reading_speed_wpm": [220],
    "queries_last_hour": [7],
    "distinct_topics_24h": [4],
}


def _chunk(text: str, max_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into ~max_tokens blocks with a small overlap.

    Whitespace tokens are a deliberate simplification (see ARCHITECTURE.md
    section 4): fast, dependency-free, and good enough for the POC. A
    production system would use a real VN sentence splitter for the *break*
    points and keep whole sentences for the embedding.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    step = max_tokens - overlap
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), step)]


class HybridMemoryAgent:
    """One agent per process; holds a Qdrant client + the shared embedder."""

    def __init__(self, feast_repo: Path | None = None) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=models.VectorParams(
                size=self.embedder.dim, distance=models.Distance.COSINE
            ),
        )
        self._next_id = 0
        self.feature_store = self._load_feast(feast_repo)
        # In-memory "recent activity" buffer: the streaming half of the
        # architecture, kept tiny so the POC needs no Redis.
        self.recent_queries: dict[str, list[str]] = {}

    # ── feature store ─────────────────────────────────────────────────────
    @staticmethod
    def _load_feast(feast_repo: Path | None):
        """Return a FeatureStore if the registry exists, else None (graceful)."""
        repo = feast_repo or (Path(__file__).resolve().parent.parent
                              / "app" / "feast_repo")
        try:
            from feast import FeatureStore
            if (Path(repo) / "registry.db").exists():
                return FeatureStore(repo_path=str(repo))
        except Exception:                                   # noqa: BLE001
            return None
        return None

    def _profile(self, user_id: str) -> dict:
        """Online lookup; falls back to a default profile if Feast is cold."""
        if self.feature_store is not None:
            try:
                raw = self.feature_store.get_online_features(
                    features=DEFAULT_FEATURES,
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                # to_dict() returns lists per feature; flatten to scalars
                return {k: (v[0] if isinstance(v, list) else v) for k, v in raw.items()}
            except Exception:                               # noqa: BLE001
                pass
        return {k: (v[0] if isinstance(v, list) else v)
                for k, v in _FALLBACK_PROFILE.items()}

    # ── write path ────────────────────────────────────────────────────────
    def remember(self, text: str, user_id: str = "u_001",
                 topic: str | None = None) -> None:
        """Chunk -> embed -> upsert, tagged with user_id (hard isolation)."""
        chunks = _chunk(text)
        vectors = list(self.embedder.embed(chunks))
        points = []
        for chunk, vec in zip(chunks, vectors):
            points.append(models.PointStruct(
                id=self._next_id,
                vector=vec.tolist(),
                payload={
                    "user_id": user_id,
                    "topic": topic,
                    "text": chunk,
                    "ts": time.time(),
                    "source": "episodic",
                },
            ))
            self._next_id += 1
        self.client.upsert(collection_name=MEMORY_COLLECTION, points=points)

    # ── read path ─────────────────────────────────────────────────────────
    def _search_memory(self, query: str, user_id: str, top_k: int = TOP_K) -> list[dict]:
        """Filtered vector search -- user_id filter is mandatory, never optional."""
        qv = next(self.embedder.embed([query])).tolist()
        hits = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=qv,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id))]),
            limit=top_k,
        ).points
        return [{"text": h.payload["text"], "score": float(h.score),
                 "topic": h.payload.get("topic")} for h in hits]

    def recall(self, query: str, user_id: str = "u_001") -> str:
        """Assemble context: profile (who) + memory (what) + activity (now)."""
        self.recent_queries.setdefault(user_id, []).append(query)

        profile = self._profile(user_id)
        memories = self._search_memory(query, user_id)

        # Personalisation re-rank: nudge memories whose topic matches the
        # user's affinity. A cheap stand-in for a learned re-ranker.
        affinity = profile.get("topic_affinity")
        for m in memories:
            if affinity and m.get("topic") == affinity:
                m["score"] += 0.05
        memories.sort(key=lambda m: -m["score"])

        recent = self.recent_queries[user_id][-3:]
        top = memories[:3]
        mem_lines = "\n".join(f"  - [{m['topic'] or 'general'}] {m['text'][:120]}"
                              for m in top) or "  - (chưa có memory)"
        return (
            f"USER PROFILE\n"
            f"  language={profile.get('preferred_language')}  "
            f"speed={profile.get('reading_speed_wpm')}wpm  "
            f"affinity={affinity}\n"
            f"RECENT ACTIVITY\n"
            f"  queries_last_hour={profile.get('queries_last_hour')}  "
            f"distinct_topics_24h={profile.get('distinct_topics_24h')}  "
            f"recent={recent}\n"
            f"TOP MEMORIES (query={query!r})\n"
            f"{mem_lines}"
        )
