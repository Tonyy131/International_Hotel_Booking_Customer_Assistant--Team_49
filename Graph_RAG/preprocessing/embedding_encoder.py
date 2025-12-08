"""
EmbeddingEncoder: Encodes user queries into dense vectors
using the same model that is used for KG node embeddings.

Model: sentence-transformers/all-MiniLM-L6-v2
or BAAI/bge-small-en-v1.5
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingEncoder:
    """
    This class loads the embedding model and provides methods to encode text into vectors.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_2_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name="minilm"):
        if model_name == "bge":
            self.model = SentenceTransformer(self.MODEL_2_NAME)
        else:
            self.model = SentenceTransformer(self.MODEL_NAME)

    def encode(self, text: str) -> List[float]:
        """
        Encodes a single text string into a dense vector.
        """
        if not text or not text.strip():
            return []

        embedding = self.model.encode(text)
        return embedding.tolist()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Encodes a batch of text strings into dense vectors.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()