# Graph_RAG/retrieval/embedding_retriever.py
from typing import List, Dict, Any, Optional
from neo4j_connector import Neo4jConnector
# Use your encoder located in preprocessing
from preprocessing.embedding_encoder import EmbeddingEncoder  # stays in preprocessing as agreed. :contentReference[oaicite:4]{index=4}
import numpy as np
import math

class EmbeddingRetriever:
    """
    Perform semantic search using a precomputed vector index in Neo4j.
    Assumes you created appropriate vector indices in Neo4j (see notes below).
    """
    def __init__(self, neo4j_connector: Optional[Neo4jConnector] = None, encoder: Optional[EmbeddingEncoder] = None):
        self.db = neo4j_connector or Neo4jConnector()
        self.encoder = encoder or EmbeddingEncoder()

    # Example: semantic search for hotels using Neo4j vector index named 'hotel_vector_index'
    def sem_search_hotels(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        embedding = self.encoder.encode(query)
        if not embedding:
            return []

        # If your Neo4j has the vector index available (Neo4j GDS / vector plugin),
        # you can use the vector query procedure. Adjust name to your index name.
        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) YIELD node, score
        RETURN node.name AS name, node.hotel_id AS hotel_id, score
        """

        params = {
            "index_name": "hotel_vector_index",  # change if your index is different
            "top_k": top_k,
            "embedding": embedding
        }

        try:
            return self.db.run_query(cypher, params)
        except Exception as e:
            print("Vector query failed, falling back to substring match. Error:", e)
            # Fallback: text substring search
            return self.db.run_query(
                "MATCH (h:Hotel) WHERE toLower(h.name) CONTAINS toLower($q) RETURN h.name AS name, h.hotel_id AS hotel_id LIMIT $limit",
                {"q": query, "limit": top_k}
            )

    def sem_search_reviews(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        embedding = self.encoder.encode(query)
        if not embedding:
            return []

        cypher = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) YIELD node, score
        RETURN node.review_id AS review_id, node.text AS text, score
        """
        params = {"index_name": "review_vector_index", "top_k": top_k, "embedding": embedding}
        try:
            return self.db.run_query(cypher, params)
        except Exception as e:
            print("Review vector query failed:", e)
            return []
