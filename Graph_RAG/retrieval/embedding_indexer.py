from typing import Any, List, Dict
from neo4j_connector import Neo4jConnector
from retrieval.feature_builder import build_feature_text
from preprocessing.embedding_encoder import EmbeddingEncoder

class EmbeddingIndexer:
    """
    Generates vector embeddings for all Hotel nodes and stores them in Neo4j.

    Output properties added to each (:Hotel):
        - h.embedding_minilm   (List[float])
    """

    def __init__(self, neo4j_connector: Neo4jConnector):
        self.db = neo4j_connector or Neo4jConnector()
        self.encoder = EmbeddingEncoder()

    def fetch_hotels(self) -> List[Dict[str, Any]]:
        """
        Fetches all hotel nodes from the database.
        """
        cypher = """MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country) 
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WITH h, c.name AS city_name, co.name AS country_name, collect(r.text)[0...3] AS review_texts
        RETURN h, city_name, country_name, review_texts"""
        return self.db.run_query(cypher)
    
    def store_embedding(self, node_id: int, embedding: List[float]):
        """
        Stores the embedding vector in the specified node.
        """
        cypher = """
        MATCH (h:Hotel) WHERE id(h) = $node_id
        SET h.embedding_minilm = $embedding
        """
        params = {"node_id": node_id, "embedding": embedding}
        self.db.run_query(cypher, params)

    def index_all_hotels(self):
        """
        Fetches all hotels, generates embeddings, and stores them in the database.
        """
        records = self.fetch_hotels()
        for record in records:
            hotel_node = record["h"]
            node_id = hotel_node.id
            feature_text = build_feature_text(record)
            embedding = self.encoder.encode(feature_text)
            if embedding:
                self.store_embedding(node_id, embedding)
                print(f"Stored embedding for Hotel node ID {node_id}")
            else:
                print(f"Failed to generate embedding for Hotel node ID {node_id}")