from retrieval.embedding_retriever import EmbeddingRetriever
from preprocessing.embedding_encoder import EmbeddingEncoder

query = "Find me hotels in Cairo above 8"

retriever_minilm = EmbeddingRetriever(model_name="minilm")
emb_minilm = EmbeddingEncoder("minilm").encode(query)
results_minilm = retriever_minilm.sem_search_hotels_global(emb_minilm, top_k=5)
print("MINILM RESULTS:")
for r in results_minilm:
    print(r)

retriever_bge = EmbeddingRetriever(model_name="bge")
emb_bge = EmbeddingEncoder("bge").encode(query)
results_bge = retriever_bge.sem_search_hotels_global(emb_bge, top_k=5)
print("\nBGE RESULTS:")
for r in results_bge:
    print(r)
