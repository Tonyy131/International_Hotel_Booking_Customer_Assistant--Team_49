from typing import List, Dict, Any
from neo4j_connector import Neo4jConnector
from preprocessing.embedding_encoder import EmbeddingEncoder
# from preprocessing.entity_extractor import extract_entities


class EmbeddingRetriever:
    """
    Embedding-based retrieval for Hotels using Neo4j vector index.
    Supports:
      - Global semantic search
      - City-filtered semantic search (single or multiple)
      - Country-filtered semantic search (single or multiple)
    """

    def __init__(self, neo4j_connector: Neo4jConnector = None, model_name: str = "minilm"):
        self.db = neo4j_connector or Neo4jConnector()
        self.encoder = EmbeddingEncoder(model_name=model_name)
        if model_name == "bge":
            self.property_name = "embedding_bge"
            self.index_name = "hotel_embedding_bge_idx"
            self.dimensions = 768
        else:
            self.property_name = "embedding_minilm"
            self.index_name = "hotel_embedding_minilm_idx"
            self.dimensions = 384


    # GLOBAL SEARCH (no filters)
    def sem_search_hotels_global(self, embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {
            "index_name": self.index_name,
            "embedding": embedding,
            "top_k": top_k
        }
        if rating_filter and rating_filter.get("type") != "none" and rating_filter.get("type") == "stars":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.star_rating >= $rating_min AND h.star_rating <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "cleanliness":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min AND h.score_cleanliness <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "comfort":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min AND h.score_comfort <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]

        elif rating_filter and rating_filter.get("type") != "none":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min AND h.average_reviews_score <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        


        cypher = f"""
        MATCH (c:City)

        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.%s IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        cypher = cypher % self.property_name

        return self.db.run_query(cypher, params)


    # SINGLE CITY
    def sem_search_hotels_in_city(self, city: str, embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"city": city, "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none" and rating_filter.get("type") == "stars":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.star_rating >= $rating_min AND h.star_rating <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "cleanliness":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min AND h.score_cleanliness <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "comfort":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min AND h.score_comfort <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") != "none":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min AND h.average_reviews_score <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        


        cypher = f"""
        MATCH (c:City)
        WHERE toLower(c.name) = toLower($city)

        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.%s IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        cypher = cypher % self.property_name

        return self.db.run_query(cypher, params)

    # MULTIPLE CITIES
    def sem_search_hotels_in_cities(self, cities: List[str], embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"cities": [c.lower() for c in cities], "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none" and rating_filter.get("type") == "stars":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.star_rating >= $rating_min AND h.star_rating <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "cleanliness":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min AND h.score_cleanliness <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "comfort":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min AND h.score_comfort <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") != "none":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min AND h.average_reviews_score <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        


        cypher = f"""
        MATCH (c:City)
        WHERE toLower(c.name) IN $cities

        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.%s IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        cypher = cypher % self.property_name

        return self.db.run_query(cypher, params)

    # SINGLE COUNTRY
    def sem_search_hotels_in_country(self, country: str, embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"country": country, "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none" and rating_filter.get("type") == "stars":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.star_rating >= $rating_min AND h.star_rating <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "cleanliness":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min AND h.score_cleanliness <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "comfort":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min AND h.score_comfort <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") != "none":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min AND h.average_reviews_score <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        


        cypher = f"""
        MATCH (co:Country)
        WHERE toLower(co.name) = toLower($country)

        MATCH (c:City)-[:LOCATED_IN]->(co)
        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.%s IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        cypher = cypher % self.property_name

        return self.db.run_query(cypher, params)

    # MULTIPLE COUNTRIES
    def sem_search_hotels_in_countries(self, countries: List[str], embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"countries": [co.lower() for co in countries], "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none" and rating_filter.get("type") == "stars":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.star_rating <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.star_rating >= $rating_min AND h.star_rating <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "cleanliness":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_cleanliness <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_cleanliness >= $rating_min AND h.score_cleanliness <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") == "comfort":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.score_comfort <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.score_comfort >= $rating_min AND h.score_comfort <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        elif rating_filter and rating_filter.get("type") != "none":
            op = rating_filter.get("operator")
            if op == "gte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min"
                params["rating_min"] = rating_filter["value"]
            elif op == "lte" and rating_filter.get("value") is not None:
                rating_clause = "AND h.average_reviews_score <= $rating_max"
                params["rating_max"] = rating_filter["value"]
            elif op == "between" and rating_filter.get("min") is not None and rating_filter.get("max") is not None:
                rating_clause = "AND h.average_reviews_score >= $rating_min AND h.average_reviews_score <= $rating_max"
                params["rating_min"] = rating_filter["min"]
                params["rating_max"] = rating_filter["max"]
        


        cypher = f"""
        MATCH (co:Country)
        WHERE toLower(co.name) IN $countries

        MATCH (c:City)-[:LOCATED_IN]->(co)
        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.%s IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """

        cypher = cypher % self.property_name

        return self.db.run_query(cypher, params)
    
    def search_visa(self, origin_country:str , destination_country:str, embedding: List[float], top_k:int=10):
        if not origin_country or not destination_country:
            return []

        # We use toLower() for case-insensitive matching to be robust against user input variations
        cypher = """
        MATCH (from:Country {name: 'Japan'})
        MATCH (to:Country)
        WHERE from <> to
        OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to)
        RETURN
            from.name AS origin_country,
            to.name AS destination_country,
            COALESCE(v.visa_type, 'Visa Free') AS visa_type
        ORDER BY destination_country

        """

        params = {
            "origin": origin_country,
            "destination": destination_country
        }

        return self.db.run_query(cypher, params)
    
    def _build_rating_clause(self, rating_filter: dict, params: dict) -> str:
        if not rating_filter or rating_filter.get("type") == "none":
            return ""

        r_type = rating_filter.get("type")
        op = rating_filter.get("operator")
        val = rating_filter.get("value")
        min_val = rating_filter.get("min")
        max_val = rating_filter.get("max")
        
        # Map filter type to DB property
        field_map = {
            "stars": "h.star_rating",
            "cleanliness": "h.score_cleanliness",
            "comfort": "h.score_comfort",
            "reviews": "h.average_reviews_score" # Default fallback
        }
        db_field = field_map.get(r_type, "h.average_reviews_score")

        clause = ""
        if op == "gte" and val is not None:
            clause = f"AND {db_field} >= $rating_min"
            params["rating_min"] = val
        elif op == "lte" and val is not None:
            clause = f"AND {db_field} <= $rating_max"
            params["rating_max"] = val
        elif op == "between" and min_val is not None and max_val is not None:
            clause = f"AND {db_field} >= $rating_min AND {db_field} <= $rating_max"
            params["rating_min"] = min_val
            params["rating_max"] = max_val
            
        return clause
    

    def _build_pre_filter(self, rating_filter: dict, params: dict) -> str:
        """
        Builds WHERE clause for properties that exist ON THE HOTEL NODE.
        (Stars, Global Score).
        """
        if not rating_filter or rating_filter.get("type") not in ["stars", "reviews"]:
            return ""

        r_type = rating_filter.get("type")
        op = rating_filter.get("operator")
        val = rating_filter.get("value")
        min_val = rating_filter.get("min")
        max_val = rating_filter.get("max")
        
        # Map to static node properties
        field = "h.star_rating" if r_type == "stars" else "h.average_reviews_score"

        clause = ""
        if op == "gte" and val is not None:
            clause = f"AND {field} >= $pre_min"
            params["pre_min"] = float(val)
        elif op == "lte" and val is not None:
            clause = f"AND {field} <= $pre_max"
            params["pre_max"] = float(val)
        elif op == "between":
            clause = f"AND {field} >= $pre_min AND {field} <= $pre_max"
            params["pre_min"] = float(min_val)
            params["pre_max"] = float(max_val)
        elif op == "eq" and val is not None:
            clause = f"AND {field} = $pre_eq"
            params["pre_eq"] = float(val)
            
        return clause

    def _build_post_filter(self, rating_filter: dict, params: dict) -> str:
        """
        Builds WHERE clause for calculated aggregates FROM REVIEWS.
        (Cleanliness, Comfort, Facilities, Staff, Value).
        """
        dynamic_types = ["cleanliness", "comfort", "facilities", "staff", "money"]
        
        if not rating_filter or rating_filter.get("type") not in dynamic_types:
            return ""

        r_type = rating_filter.get("type")
        op = rating_filter.get("operator")
        val = rating_filter.get("value")
        min_val = rating_filter.get("min")
        max_val = rating_filter.get("max")
        
        # Map to the calculated variable names in Cypher
        field_map = {
            "cleanliness": "avg_cleanliness",
            "comfort": "avg_comfort",
            "facilities": "avg_facilities",
            "staff": "avg_staff",
            "money": "avg_money"
        }
        variable = field_map.get(r_type)

        clause = ""
        if op == "gte" and val is not None:
            clause = f"AND {variable} >= $post_min"
            params["post_min"] = float(val)
        elif op == "lte" and val is not None:
            clause = f"AND {variable} <= $post_max"
            params["post_max"] = float(val)
        elif op == "between":
            clause = f"AND {variable} >= $post_min AND {variable} <= $post_max"
            params["post_min"] = float(min_val)
            params["post_max"] = float(max_val)
            
        return clause
    
    def _search_hotels_generic(self, embedding: List[float], cities: List[str] = None, countries: List[str] = None, top_k: int = 25, rating_filter: dict = None):
        params = {
            "index_name": self.index_name,
            "embedding": embedding,
            "top_k": top_k,
            # Fetch more candidates initially to allow for post-filtering
            "fetch_k": top_k * 5 
        }

        # 1. Location Clause
        location_match = "MATCH (c:City)"
        if cities:
            location_match = "MATCH (c:City) WHERE toLower(c.name) IN $cities"
            params["cities"] = [c.lower() for c in cities]
        elif countries:
            location_match = """
            MATCH (co:Country) WHERE toLower(co.name) IN $countries
            MATCH (c:City)-[:LOCATED_IN]->(co)
            """
            params["countries"] = [c.lower() for c in countries]

        # 2. Build Filters
        # Pre-filter: Node properties (Stars)
        pre_filter = self._build_pre_filter(rating_filter, params)
        # Post-filter: Calculated averages (Cleanliness, etc.)
        post_filter = self._build_post_filter(rating_filter, params)

        # 3. Cypher Query
        cypher = f"""
        {location_match}
        MATCH (h:Hotel)-[:LOCATED_IN]->(c)
        WHERE h.{self.property_name} IS NOT NULL
        {pre_filter}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes('{self.index_name}', $fetch_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        MATCH (node)-[:LOCATED_IN]->(c_res:City)-[:LOCATED_IN]->(co_res:Country)
        OPTIONAL MATCH (node)<-[:REVIEWED]-(r:Review)

        WITH node, score, c_res, co_res, 
             collect(r.text)[0..3] AS review_texts,
             avg(r.score_cleanliness) AS avg_cleanliness,
             avg(r.score_comfort) AS avg_comfort,
             avg(r.score_facilities) AS avg_facilities,
             avg(r.score_staff) AS avg_staff,
             avg(r.score_value_for_money) AS avg_money
        
        WHERE true {post_filter}

        RETURN 
            node {{
                .*, 
                avg_score_cleanliness: avg_cleanliness,
                avg_score_comfort: avg_comfort,
                avg_score_facilities: avg_facilities,
                avg_score_staff: avg_staff,
                avg_score_value_for_money: avg_money
            }} AS h, 
            c_res.name AS city_name, 
            co_res.name AS country_name, 
            review_texts, 
            score
        ORDER BY score DESC
        LIMIT $top_k
        """

        results = self.db.run_query(cypher, params)

        # 4. Clean Results
        cleaned_results = []
        for row in results:
            if 'h' in row:
                hotel_props = dict(row['h'])
                hotel_props.pop('embedding_minilm', None)
                hotel_props.pop('embedding_bge', None)
                row['h'] = hotel_props
            cleaned_results.append(row)

        return cleaned_results
 
    def get_visa_free_countries(self, origin_country: str) -> List[str]:
        """
        Finds 'Visa Free' countries by looking for the ABSENCE of a 
        restrictive [:NEEDS_VISA] relationship.
        """
        cypher = """
        MATCH (origin:Country)
        WHERE toLower(origin.name) = toLower($origin)
        
        MATCH (dest:Country)
        WHERE dest <> origin 
        
        AND NOT (origin)-[:NEEDS_VISA]->(dest)
        
        RETURN dest.name AS country
        """
        
        params = {"origin": origin_country}
        results = self.db.run_query(cypher, params)
        
        found_countries = [r["country"].lower() for r in results]
        
        if not found_countries:
            print(f"No visa-free countries found for {origin_country}. (Check if Country nodes exist?)")
        
        return found_countries
    
    def search_countries_visa(self, origin_country, embedding):
        cypher = """
        MATCH (from:Country {name: $from})
        MATCH (to:Country)
        WHERE from <> to
        OPTIONAL MATCH (from)-[v:NEEDS_VISA]->(to)
        RETURN
            from.name AS origin_country,
            to.name AS destination_country,
            COALESCE(v.visa_type, 'Visa Free') AS visa_type
        ORDER BY destination_country
        """
        
        params = {"from": origin_country}
        return self.db.run_query(cypher, params)
        

    # MAIN ENTRY POINT
    def sem_search_hotels(self, query: str, entities, top_k: int = 10, rating_filter: dict = None, intent: str = "hotel_search"):
    
        embedding = self.encoder.encode(query)
        top_k = entities.get("limit")

        if intent == "visa_query":
            # Safely get the lists
            origins = entities.get("origin_country", [])
            dests = entities.get("destination_country", [])

            # Check if lists are not empty before accessing [0]
            origin_country = origins[0] if origins else None
            destination_country = dests[0] if dests else None
            
            # The search_visa method likely needs both. If one is missing, return empty.
            if origin_country and destination_country:
                return self.search_visa(origin_country, destination_country, embedding, top_k)
            if origin_country:
                return self.search_countries_visa(origin_country, embedding)
            else:
                return []
        

        cities = entities.get("cities", [])
        countries = entities.get("countries", [])

        visa_info_to_add = []
        if intent == "hotel_visa":
            origin_country = entities.get("origin_country", [None])[0]
            
            if origin_country:
                print(f"DEBUG: Processing 'hotel_visa' for origin: {origin_country}")
                
                allowed_countries = self.get_visa_free_countries(origin_country)
                
                if allowed_countries:
                    print(f"DEBUG: Found {len(allowed_countries)} visa-free destinations.")
                    countries = allowed_countries

                    for dest in allowed_countries:
                        visa_info_to_add.append({
                            "origin_country": origin_country,
                            "destination_country": dest.title(), # Formatting
                            "visa_type": "Visa Free" # Explicit status
                        })
                else:
                    print("DEBUG: No visa-free countries found.")
                    return [] 
            else:
                print("DEBUG: Intent is 'hotel_visa' but no 'origin_country' extracted.")


        hotel_results = self._search_hotels_generic(
            embedding=embedding, 
            cities=cities, 
            countries=countries, 
            top_k=top_k, 
            rating_filter=rating_filter
        )
        # The single generic method handles empty lists (Global), single items, or multiple items automatically.
        return hotel_results + visa_info_to_add
