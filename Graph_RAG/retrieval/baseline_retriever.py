# Graph_RAG/retrieval/baseline_retriever.py
from typing import Dict, Any, Optional
from retrieval.query_templates import QUERY_TEMPLATES
from neo4j_connector import Neo4jConnector

class BaselineRetriever:
    """
    Select and execute Cypher templates based on intent + extracted entities.
    Returns raw lists of dict records from Neo4j.
    """
    def __init__(self, neo4j_connector: Optional[Neo4jConnector] = None):
        self.db = neo4j_connector or Neo4jConnector()

    def retrieve(self, intent: str, entities: Dict[str, Any], limit: int = 10):
        intent = intent or "generic_qa"
        e = entities or {}

        def _exec_and_extract(cypher_key, params):
            """Run template, extract hotel dicts when available."""
            print(QUERY_TEMPLATES[cypher_key],"mizo")
            records = self.db.run_query(QUERY_TEMPLATES[cypher_key], params)
            cleaned = []
            for rec in records or []:
                # If record carries a hotel map:
                if "hotel" in rec and rec["hotel"] is not None:
                    h = rec["hotel"]
                    if isinstance(h, dict):
                        h["source"] = "baseline"
                        cleaned.append(h)
                    else:
                        # sometimes driver returns a Node-like object; try to coerce
                        try:
                            hmap = dict(h)
                            hmap["source"] = "baseline"
                            cleaned.append(hmap)
                        except Exception:
                            continue
                else:
                    # fallback: return the raw record (for queries like reviews or visa)
                    cleaned.append(rec)
            return cleaned

        # --- hotel_search intent ---
        if intent == "hotel_search":
            rf = e.get("rating_filter") or {"type": "none", "operator": None}
            cities = e.get("cities") or None
            countries = e.get("countries") or None
            if rf and rf.get("type") != "none" and rf.get("type") == "stars":
                op = rf.get("operator")
                if op == "gte" and rf.get("value") is not None:
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_stars", params)

                if op == "lte" and rf.get("value") is not None:
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_stars", params)
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_stars_range", params)

                if op == "eq" and rf.get("value") is not None:
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_stars", params)
            elif rf and rf.get("type") != "none":
                op = rf.get("operator")
                if op == "gte" and rf.get("value") is not None:
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_rating", params)

                if op == "lte" and rf.get("value") is not None:
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_rating", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_rating_range", params)

                if op == "eq" and rf.get("value") is not None:
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_rating", params)

            # Combined city + country search
            if e.get("cities") or e.get("countries"):
                return _exec_and_extract("hotel_search_by_city_or_country",
                                        {"cities": e.get("cities", []), "countries": e.get("countries", []), "limit": limit})

            # fallback free text substring
            if e.get("hotels"):
                return _exec_and_extract("hotel_by_name_substring", {"q": e["hotels"][0], "limit": limit})

            return _exec_and_extract("top_hotels", {"limit": limit})

        # --- review_query ---
        if intent == "review_query":
            if e.get("hotels"):
                return self.db.run_query(QUERY_TEMPLATES["hotel_reviews_by_name"], {"hotel": e["hotels"][0], "limit": limit})
            return []

        # --- recommendation ---
        if intent == "recommendation":
            traveller_type = e.get("traveller_type")
            if traveller_type:
                # keep same params shape (template returns hotel + freq, _exec_and_extract will pick hotel)
                return _exec_and_extract("recommend_hotels_by_traveller_type", {"traveller_type": traveller_type, "limit": limit})
            return _exec_and_extract("top_hotels", {"limit": limit})

        # --- visa_query ---
        if intent == "visa_query":
            origins = e.get("origin_country") or []
            dests = e.get("destination_country") or []
            if origins and dests:
                return self.db.run_query(QUERY_TEMPLATES["visa_requirements"], {"from": origins[0], "to": dests[0]})
            return []

        # Generic fallback
        return _exec_and_extract("top_hotels", {"limit": limit})
