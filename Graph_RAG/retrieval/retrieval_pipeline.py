# Graph_RAG/retrieval/retrieval_pipeline.py
from typing import Dict, Any, List
from Graph_RAG.retrieval.baseline_retriever import BaselineRetriever
from Graph_RAG.retrieval.embedding_retriever import EmbeddingRetriever
from Graph_RAG.neo4j_connector import Neo4jConnector

class RetrievalPipeline:
    """
    Orchestrates baseline + embedding retrieval and merges results into a single context
    structure suitable for feeding to the LLM prompt builder.
    """
    def __init__(self, neo4j_connector: Neo4jConnector = None):
        connector = neo4j_connector or Neo4jConnector()
        self.baseline = BaselineRetriever(connector)
        self.embed = EmbeddingRetriever(connector)

    def retrieve(self, intent: str, entities: Dict[str, Any], user_query: str, use_embeddings: bool = True, limit: int = 10) -> Dict[str, Any]:
        baseline_results = self.baseline.retrieve(intent, entities, limit=limit)

        embedding_results = []
        if use_embeddings:
            # For hotel-oriented intents we search hotels + reviews
            embedding_results = self.embed.sem_search_hotels(user_query, top_k=limit)

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
