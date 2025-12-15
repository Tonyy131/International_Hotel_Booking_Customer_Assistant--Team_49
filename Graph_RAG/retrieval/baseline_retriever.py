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
        intent = intent or "hotel_search"
        e = entities or {}

        def _exec_and_extract(cypher_key, params):
            """Run template, extract hotel dicts when available."""
            print(QUERY_TEMPLATES[cypher_key],cypher_key)
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

            return cleaned, QUERY_TEMPLATES[cypher_key]

        print("BaselineRetriever: intent =", intent, "entities =", e)
        # --- hotel_search intent ---
        if intent == "hotel_search":
            rf = e.get("rating_filter") or {"type": "none", "operator": None}
            cities = e.get("cities") or None
            countries = e.get("countries") or None
            if rf and rf.get("type") != "none" and rf.get("type") == "stars":
                op = rf.get("operator")
                if op == "gte" and rf.get("value") is not None:
                    params = {"stars": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                if op == "gte" and rf.get("value") is not None:
                    params = {"stars": rf["value"], "cities": cities, "countries": countries, "limit": limit}
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
            elif rf and rf.get("type") == "cleanliness":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_cleanliness", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_cleanliness", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_cleanliness_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_cleanliness", params)
                
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotel_cleanliness", params)
                
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotel_cleanliness", params)
            elif rf and rf.get("type") == "comfort":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_comfort", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_comfort", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_comfort_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_comfort", params)  
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotel_comfort", params)
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotel_comfort", params) 
            elif rf and rf.get("type") == "facilities":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_facilities", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_facilities", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_facilities_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_facilities", params)  
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotel_facilities", params)
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotel_facilities", params)
            elif rf and rf.get("type") == "staff":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_staff", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_staff", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_staff_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_staff", params)  
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotel_staff", params)
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotel_staff", params)
            elif rf and rf.get("type") == "money":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_value_for_money", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_value_for_money", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_value_for_money_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_value_for_money", params)  
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotel_value_for_money", params)
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotel_value_for_money", params)
            elif rf and rf.get("type") != "none":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_rating", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_rating", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_rating_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_rating", params)
                
                if op == "gte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("top_hotels", params)
                
                if op == "lte" :
                    params = {"cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("worst_hotels", params)

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
            rf = e.get("rating_filter") or {"type": "none", "operator": None}
            cities = e.get("cities") or None
            countries = e.get("countries") or None
            if rf and rf.get("type") != "none" and rf.get("type") == "stars":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_stars", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_stars", params)
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_stars_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_stars", params)
            elif rf and rf.get("type") == "cleanliness":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_cleanliness", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_cleanliness", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_cleanliness_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_cleanliness", params)
            elif rf and rf.get("type") == "comfort":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_comfort", params)
                
                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_comfort", params)
                
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_comfort_range", params)
                
                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_comfort", params)
            elif rf and rf.get("type") == "facilities":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_facilities", params)
                
                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_facilities", params)
                
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_facilities_range", params)
                
                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_facilities", params)
            elif rf and rf.get("type") == "staff":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_staff", params)
                
                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_staff", params)
                
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_staff_range", params)
                
                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_staff", params)
            elif rf and rf.get("type") == "money":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_value_formoney", params)
                
                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_value_for_money", params)
                
                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_value_for_money_range", params)
                
                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_value_for_money", params)
            elif rf and rf.get("type") != "none":
                op = rf.get("operator")
                if op == "gte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"rating": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_min_rating", params)

                if op == "lte" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"max": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_max_rating", params)

                if op == "between" and rf.get("min") is not None and rf.get("max") is not None:
                    params = {"min": rf["min"], "max": rf["max"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_rating_range", params)

                if op == "eq" and (rf.get("value") is not None and rf.get("value") != 0):
                    params = {"value": rf["value"], "cities": cities, "countries": countries, "limit": limit}
                    return _exec_and_extract("hotel_search_exact_rating", params)

            # Combined city + country search
            if e.get("cities") or e.get("countries"):
                return _exec_and_extract("hotel_search_by_city_or_country",
                                        {"cities": e.get("cities", []), "countries": e.get("countries", []), "limit": limit})

            if e.get("hotels"):
                return _exec_and_extract("hotel_reviews_by_name", {"hotel": e["hotels"][0], "limit": limit})
            return []

        # --- recommendation ---
        if intent == "recommendation":
            traveller_type = e.get("traveller_type")
            if traveller_type:
                # keep same params shape (template returns hotel + freq, _exec_and_extract will pick hotel)
                return _exec_and_extract("recommend_hotels_by_traveller_type", {"traveller_type": traveller_type, "limit": limit})
            if e.get("cities") or e.get("countries"):
                return _exec_and_extract("hotel_search_by_city_or_country",
                                        {"cities": e.get("cities", []), "countries": e.get("countries", []), "limit": limit})
            

            return _exec_and_extract("top_hotels", {"limit": limit})

        # --- visa_query ---
        if intent == "visa_query":
            origins = e.get("origin_country") or []
            dests = e.get("destination_country") or []
            if origins and dests:
                params = {
                    "from": origins[0], 
                    "to": dests[0]
                }
                return _exec_and_extract("visa_requirements", params)
            if origins:
                params = {"from": origins[0]}
                return _exec_and_extract("visa_requirements_by_origin", params)
            return [],""
        
        if intent == "hotel_visa":
            origins = e.get("origin_country") or []
            if origins:
                params = {"origin": origins[0], "limit": limit}
                return _exec_and_extract("hotel_search_visa_free", params)
            
            # If origin is missing, fallback to generic top hotels
            return _exec_and_extract("top_hotels", {"limit": limit})

        # Generic fallback
        return _exec_and_extract("top_hotels", {"limit": limit})
