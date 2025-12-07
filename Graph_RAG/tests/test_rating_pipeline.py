from retrieval.embedding_retriever import EmbeddingRetriever
from preprocessing.entity_extractor import extract_entities
from preprocessing.embedding_encoder import EmbeddingEncoder

retriever = EmbeddingRetriever()
encoder = EmbeddingEncoder()

query = "Find me hotels in Cairo above 8"
entities = extract_entities(query)
emb = encoder.encode(query)

print("Global Embedding Search (Top 5):")
results = retriever.sem_search_hotels_global(emb, top_k=5)
for r in results:
    print(r)
