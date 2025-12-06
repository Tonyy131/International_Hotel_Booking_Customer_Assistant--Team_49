# Graph_RAG/neo4j_connector.py
from neo4j import GraphDatabase
from typing import Any, Dict, List, Optional
import os

class Neo4jConnector:
    """
    Small wrapper for basic Neo4j operations used by the retrieval layer.
    Expects environment variables (or a config file) to supply connection info.
    """
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "test")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def run_query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None, fetch_one: bool = False) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return a list of dicts (records).
        If fetch_one=True, return a single record or empty list.
        """
        params = parameters or {}
        with self.driver.session() as session:
            try:
                result = session.run(cypher, params)
                records = []
                for r in result:
                    # Convert Neo4j Record to plain dict
                    rec = {}
                    for key in r.keys():
                        rec[key] = r.get(key)
                    records.append(rec)
                if fetch_one:
                    return records[:1]
                return records
            except Exception as e:
                print("Neo4j query error:", e)
                return []

    def close(self):
        self.driver.close()
