import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from datetime import datetime, timedelta
import hashlib


CACHE_MAX_AGE_DAYS = 3
VENUE_MAX_AGE_DAYS = 30
SIMILARITY_THRESHOLD = 0.85  # 0-1, higher = more strict matching


class RAGManager:
    """
    Manages vector-based memory for the Melbourne Event Agent.

    Three collections:
    - search_cache     : recent DuckDuckGo results, expires after 3 days
    - user_preferences : event types the user has searched/liked
    - venue_knowledge  : venue details extracted from past searches
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.search_cache = self.client.get_or_create_collection(
            name="search_cache",
            embedding_function=self.ef
        )
        self.user_preferences = self.client.get_or_create_collection(
            name="user_preferences",
            embedding_function=self.ef
        )
        self.venue_knowledge = self.client.get_or_create_collection(
            name="venue_knowledge",
            embedding_function=self.ef
        )

    # ------------------------------------------------------------------
    # Search Cache
    # ------------------------------------------------------------------

    def store_search_result(self, query: str, result_text: str, params: dict):
        """Cache a live search result with its params and timestamp."""
        timestamp = datetime.now().isoformat()
        doc_id = self._make_id(query + timestamp)

        self.search_cache.add(
            documents=[result_text],
            metadatas=[{
                "query": query,
                "event_type": params.get("event_type", "events"),
                "timeframe": params.get("timeframe", ""),
                "budget": str(params.get("budget", "")),
                "timestamp": timestamp
            }],
            ids=[doc_id]
        )

    def retrieve_cached_search(self, query: str, params: dict):
        """
        Look for a semantically similar cached search.
        Returns result text if found and fresh, otherwise None.
        """
        if self.search_cache.count() == 0:
            return None

        results = self.search_cache.query(
            query_texts=[query],
            n_results=min(3, self.search_cache.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            return None

        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = 1 - distance  # ChromaDB returns cosine distance
            if similarity < SIMILARITY_THRESHOLD:
                continue

            # Check freshness
            if not self._is_fresh(meta["timestamp"], CACHE_MAX_AGE_DAYS):
                continue

            # Check event type matches (loose match)
            cached_type = meta.get("event_type", "")
            requested_type = params.get("event_type", "")
            if requested_type and cached_type and cached_type != requested_type:
                continue

            return doc  # Cache hit

        return None

    # ------------------------------------------------------------------
    # User Preferences
    # ------------------------------------------------------------------

    def store_user_preference(self, event_type: str):
        """
        Record that the user searched for this event type.
        Uses upsert: increments count if already exists.
        """
        pref_id = self._make_id(event_type)
        existing = self.user_preferences.get(ids=[pref_id])

        if existing["ids"]:
            count = int(existing["metadatas"][0].get("count", 1)) + 1
            self.user_preferences.update(
                ids=[pref_id],
                metadatas=[{
                    "event_type": event_type,
                    "count": str(count),
                    "last_searched": datetime.now().isoformat()
                }]
            )
        else:
            self.user_preferences.add(
                documents=[f"User is interested in {event_type} events in Melbourne"],
                metadatas=[{
                    "event_type": event_type,
                    "count": "1",
                    "last_searched": datetime.now().isoformat()
                }],
                ids=[pref_id]
            )

    def get_top_preferences(self, n: int = 3) -> list[str]:
        """Return the user's most searched event types."""
        if self.user_preferences.count() == 0:
            return []

        all_prefs = self.user_preferences.get(include=["metadatas"])
        sorted_prefs = sorted(
            all_prefs["metadatas"],
            key=lambda x: int(x.get("count", 0)),
            reverse=True
        )
        return [p["event_type"] for p in sorted_prefs[:n]]

    # ------------------------------------------------------------------
    # Venue Knowledge
    # ------------------------------------------------------------------

    def store_venue(self, venue_name: str, details: str):
        """Store or update venue details."""
        venue_id = self._make_id(venue_name.lower())
        timestamp = datetime.now().isoformat()

        existing = self.venue_knowledge.get(ids=[venue_id])
        if existing["ids"]:
            self.venue_knowledge.update(
                ids=[venue_id],
                documents=[details],
                metadatas=[{"venue_name": venue_name, "timestamp": timestamp}]
            )
        else:
            self.venue_knowledge.add(
                documents=[details],
                metadatas=[{"venue_name": venue_name, "timestamp": timestamp}],
                ids=[venue_id]
            )

    def retrieve_venue(self, query: str) -> str | None:
        """Find venue info relevant to the query."""
        if self.venue_knowledge.count() == 0:
            return None

        results = self.venue_knowledge.query(
            query_texts=[query],
            n_results=min(2, self.venue_knowledge.count()),
            include=["documents", "metadatas", "distances"]
        )

        if not results["documents"] or not results["documents"][0]:
            return None

        best_distance = results["distances"][0][0]
        if (1 - best_distance) < SIMILARITY_THRESHOLD:
            return None

        meta = results["metadatas"][0][0]
        if not self._is_fresh(meta["timestamp"], VENUE_MAX_AGE_DAYS):
            return None

        return results["documents"][0][0]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_fresh(self, timestamp_str: str, max_age_days: int) -> bool:
        try:
            stored = datetime.fromisoformat(timestamp_str)
            return datetime.now() - stored < timedelta(days=max_age_days)
        except (ValueError, TypeError):
            return False

    def _make_id(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
