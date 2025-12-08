from typing import List, Dict, Any
from neo4j_connector import Neo4jConnector
from preprocessing.embedding_encoder import EmbeddingEncoder
from preprocessing.entity_extractor import extract_entities


class EmbeddingRetriever:
    """
    Embedding-based retrieval for Hotels using Neo4j vector index.
    Supports:
      - Global semantic search
      - City-filtered semantic search (single or multiple)
      - Country-filtered semantic search (single or multiple)
    """

    def __init__(self, neo4j_connector: Neo4jConnector = None, index_name: str = "hotel_embedding_minilm_idx", model_name: str = "minilm"):
        self.db = neo4j_connector or Neo4jConnector()
        self.encoder = EmbeddingEncoder(model_name=model_name)
        self.index_name = index_name

    # GLOBAL SEARCH (no filters)
    def sem_search_hotels_global(self, embedding: List[float], top_k: int = 10):
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        """

        params = {
            "index_name": self.index_name,
            "embedding": embedding,
            "top_k": top_k
        }

        return self.db.run_query(cypher, params)

    # SINGLE CITY
    def sem_search_hotels_in_city(self, city: str, embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"city": city, "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none":
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
        WHERE h.embedding_minilm IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.db.run_query(cypher, params)

    # MULTIPLE CITIES
    def sem_search_hotels_in_cities(self, cities: List[str], embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"cities": [c.lower() for c in cities], "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none":
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
        WHERE h.embedding_minilm IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.db.run_query(cypher, params)

    # SINGLE COUNTRY
    def sem_search_hotels_in_country(self, country: str, embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"country": country, "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none":
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
        WHERE h.embedding_minilm IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.db.run_query(cypher, params)

    # MULTIPLE COUNTRIES
    def sem_search_hotels_in_countries(self, countries: List[str], embedding: List[float], top_k: int = 10, rating_filter: dict = None):
        rating_clause = ""
        params = {"countries": [co.lower() for co in countries], "index_name": self.index_name, "embedding": embedding, "top_k": top_k}
        if rating_filter and rating_filter.get("type") != "none":
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
        WHERE h.embedding_minilm IS NOT NULL
        {rating_clause}

        WITH collect(h) AS hotels

        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
        YIELD node, score
        WHERE node IN hotels

        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        return self.db.run_query(cypher, params)

    # MAIN ENTRY POINT
    def sem_search_hotels(self, query: str, entities, top_k: int = 10, rating_filter: dict = None):
        embedding = self.encoder.encode(query)

        cities = entities.get("cities", [])
        countries = entities.get("countries", [])

        # MULTI-CITY
        if len(cities) > 1:
            return self.sem_search_hotels_in_cities(cities, embedding, top_k, rating_filter)

        # SINGLE CITY
        if len(cities) == 1:
            results = self.sem_search_hotels_in_city(cities[0], embedding, top_k, rating_filter)
            if results:
                return results

        # MULTI-COUNTRY
        if len(countries) > 1:
            return self.sem_search_hotels_in_countries(countries, embedding, top_k, rating_filter)

        # SINGLE COUNTRY
        if len(countries) == 1:
            results = self.sem_search_hotels_in_country(countries[0], embedding, top_k, rating_filter)
            if results:
                return results

        # GLOBAL FALLBACK
        return self.sem_search_hotels_global(embedding, top_k)
