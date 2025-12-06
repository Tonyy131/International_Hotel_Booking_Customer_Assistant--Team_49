# Graph_RAG/retrieval/baseline_retriever.py
from typing import Dict, Any, Optional
from Graph_RAG.retrieval.query_templates import QUERY_TEMPLATES
from Graph_RAG.neo4j_connector import Neo4jConnector

class BaselineRetriever:
    """
    Select and execute Cypher templates based on intent + extracted entities.
    Returns raw lists of dict records from Neo4j.
    """
    def __init__(self, neo4j_connector: Optional[Neo4jConnector] = None):
        self.db = neo4j_connector or Neo4jConnector()

    def retrieve(self, intent: str, entities: Dict[str, Any], limit: int = 10):
        """
        intent: label from intent classification ('hotel_search','review_query','visa_query', etc.)
        entities: dict produced by your EntityExtractor (see preprocessing/entity_extractor.py). :contentReference[oaicite:3]{index=3}
        """
        intent = intent or "generic_qa"
        e = entities or {}

        # Route to appropriate template:
        if intent == "hotel_search":
            # priority: city -> country -> rating -> free text hotel name
            if e.get("cities"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_search_by_city"], {"city": e["cities"][0], "limit": limit})
            if e.get("countries"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_search_by_country"], {"country": e["countries"][0], "limit": limit})
            if e.get("rating"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_search_min_rating"], {"rating": e["rating"], "limit": limit})
            # fallback free text substring
            if e.get("hotels"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_by_name_substring"], {"q": e["hotels"][0], "limit": limit})

            return self.db.run_query(QUERY_TEMPLATES["top_hotels"], {"limit": limit})

        if intent == "review_query":
            if e.get("hotels"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_reviews_by_name"], {"hotel": e["hotels"][0], "limit": limit})
            return []

        if intent == "recommendation":
            traveller_type = e.get("traveller_type")
            if traveller_type:
                return self.db.run_query(QUERY_TEMPLATES["recommend_hotels_by_traveller_type"], {"traveller_type": traveller_type, "limit": limit})
            # fallback to top hotels
            return self.db.run_query(QUERY_TEMPLATES["top_hotels"], {"limit": limit})

        if intent == "visa_query":
            # origin/destination are lists in your extractor -> choose first if present
            origins = e.get("origin_country") or []
            dests = e.get("destination_country") or []
            if origins and dests:
                return self.db.run_query(QUERY_TEMPLATES["visa_requirements"], {"from": origins[0], "to": dests[0]})
            return []

        # Generic fallback: top hotels
        return self.db.run_query(QUERY_TEMPLATES["top_hotels"], {"limit": limit})
