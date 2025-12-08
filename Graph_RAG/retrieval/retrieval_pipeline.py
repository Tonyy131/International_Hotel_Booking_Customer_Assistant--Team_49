# Graph_RAG/retrieval/retrieval_pipeline.py
from typing import Dict, Any, List
from retrieval.baseline_retriever import BaselineRetriever
from retrieval.embedding_retriever import EmbeddingRetriever
from preprocessing.entity_extractor import EntityExtractor
from preprocessing.preprocess_intent import classify_user_intent
from neo4j_connector import Neo4jConnector

class RetrievalPipeline:
    """
    Orchestrates baseline + embedding retrieval and merges results into a single context
    structure suitable for feeding to the LLM prompt builder.
    """
    def __init__(self, neo4j_connector: Neo4jConnector = None):
        connector = neo4j_connector or Neo4jConnector()
        self.baseline = BaselineRetriever(connector)
        self.embed = EmbeddingRetriever(connector)

    def retrieve(self, intent: str, entities: Dict[str, Any], user_query: str, user_embeddings: bool = True, limit: int = 10) -> Dict[str, Any]:
        baseline_results = self.baseline.retrieve(intent, entities, limit=limit)

        embedding_results = []
        if user_embeddings:
            # For hotel-oriented intents we search hotels + reviews
            rating_filter = entities.get("rating_filter") if entities else None
            embedding_results = self.embed.sem_search_hotels(user_query, entities, top_k=limit, rating_filter=rating_filter)

        combined = self._merge_results(baseline_results, embedding_results)

        # Build a textual context summary (simple)
        context_text = self._build_context_text(combined)

        return {
            "baseline": baseline_results,
            "embeddings": embedding_results,
            "combined": combined,
            "context_text": context_text
        }

    def _merge_results(self, baseline: List[Dict], embedding: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Merge by hotel_id/name de-duplication.
        Returns a dict with keys 'hotels' and 'reviews' (reviews empty here unless you add review embed search).
        """
        hotels = []
        seen_ids = set()
        # baseline may return records with hotel_id or name
        for r in baseline or []:
            hid = r.get("hotel_id") or r.get("hotel", {}).get("hotel_id") if isinstance(r.get("hotel"), dict) else None
            name = r.get("name") or (r.get("h") and r.get("h").get("name"))
            key = hid or name
            if not key:
                continue
            if key in seen_ids:
                continue
            seen_ids.add(key)
            hotels.append({**r, "source": "baseline"})

        for r in embedding or []:
            hid = r.get("hotel_id") or r.get("id") or r.get("node") and r.get("node").get("hotel_id")
            name = r.get("name")
            key = hid or name
            if not key:
                continue
            if key in seen_ids:
                continue
            seen_ids.add(key)
            hotels.append({**r, "source": "embedding"})

        return {"hotels": hotels, "reviews": []}

    def _build_context_text(self, combined: Dict[str, Any]) -> str:
        """
        Create a human-readable context string summarizing retrieved KG info.
        The LLM prompt will use this to ground responses.
        """
        parts = []
        hotels = combined.get("hotels", [])
        if not hotels:
            return "No relevant hotels or reviews found in the knowledge graph."

        parts.append("Retrieved hotels:")
        for h in hotels:
            name = h.get("name") or (h.get("h") and h.get("h").get("name"))
            avg = h.get("avg_score") or h.get("average_reviews_score") or h.get("score")
            src = h.get("source", "unknown")
            line = f"- {name}"
            if avg is not None:
                line += f" (avg_score={avg})"
            line += f"  [via: {src}]"
            parts.append(line)

        return "\n".join(parts)
    


    def safe_retrieve(
        self,
        query: str,
        limit: int = 10,
        user_embeddings: bool = True,
    ) -> Dict[str, Any]:
        """
        Production-safe retrieval wrapper for the final chatbot.

        - Uses extracted intent
        - Never crashes
        - Normalizes return format
        - Fallbacks if something goes wrong
        - Suitable for chatbot + experiments with real pipeline behavior
        """

        # 1) Extract entities
        extractor = EntityExtractor()
        entities = extractor.extract(query)

        # 2) Classify intent
        try:
            intent_info = classify_user_intent(query)
            intent = intent_info.get("intent")
            if not intent:
                intent = "hotel_search"     # fallback
        except Exception:
            intent = "hotel_search"         # fallback

        # 3) Run the actual retrieval
        try:
            results = self.retrieve(
                intent=intent,
                entities=entities,
                user_query=query,
                user_embeddings=user_embeddings,
                limit=limit
            )
        except Exception as e:
            # Critical failure → return safe empty structure
            return {
                "intent": intent,
                "entities": entities,
                "baseline": [],
                "embeddings": [],
                "combined": {"hotels": [], "reviews": []},
                "context_text": f"[Retrieval Error: {str(e)}]"
            }

        # 4) Normalize structure and return clean results
        return {
            "intent": intent,
            "entities": entities,
            "baseline": results.get("baseline", []),
            "embeddings": results.get("embeddings", []),
            "combined": results.get("combined", {}),
            "context_text": results.get("context_text", "")
        }
