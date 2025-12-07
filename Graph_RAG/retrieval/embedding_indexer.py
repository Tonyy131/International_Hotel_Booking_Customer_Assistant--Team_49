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

    def __init__(self, neo4j_connector: Neo4jConnector = None, model_name="minilm"):
        self.db = neo4j_connector or Neo4jConnector()
        self.encoder = EmbeddingEncoder(model_name=model_name)
        if model_name == "bge":
            self.property_name = "embedding_bge"
        else:
            self.property_name = "embedding_minilm"

    def ensure_vector_index(self):
        if self.property_name == "embedding_minilm":
            dims = 384
            index_name = "hotel_embedding_minilm_idx"
        else:
            dims = 768
            index_name = "hotel_embedding_bge_idx"

        cypher = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (h:Hotel) ON (h.{self.property_name})
        OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dims},
            `vector.similarity_function`: 'cosine'
        }}
        }};
        """

        self.db.run_query(cypher)


    def fetch_hotels(self) -> List[Dict[str, Any]]:
        """
        Fetches all hotel nodes from the database.
        """
        cypher = """MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country) 
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WITH h, c.name AS city_name, co.name AS country_name, collect(r.text)[0..3] AS review_texts
        RETURN h, city_name, country_name, review_texts"""
        return self.db.run_query(cypher)
    
    def store_embedding(self, node_id: int, embedding: List[float]):
        """
        Stores the embedding vector in the specified node.
        """
        cypher = f"""
        MATCH (h:Hotel) WHERE elementId(h) = $node_id
        SET h.{self.property_name} = $embedding
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
            node_id = hotel_node.element_id
            feature_text = build_feature_text(record)
            embedding = self.encoder.encode(feature_text)
            if embedding:
                self.store_embedding(node_id, embedding)
                print(f"Stored embedding for Hotel node ID {node_id}")
            else:
                print(f"Failed to generate embedding for Hotel node ID {node_id}")


if __name__ == "__main__":
    EmbeddingIndexer().ensure_vector_index()
    EmbeddingIndexer(model_name="bge").ensure_vector_index()
    EmbeddingIndexer().index_all_hotels()
    EmbeddingIndexer(model_name="bge").index_all_hotels()