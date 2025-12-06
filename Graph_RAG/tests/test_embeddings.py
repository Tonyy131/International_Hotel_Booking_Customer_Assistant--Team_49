from preprocessing.embedding_encoder import EmbeddingEncoder

def run():
    encoder = EmbeddingEncoder()

    queries = [
        "Find me a hotel in Cairo near the Nile",
        "I want a cheap place in Berlin",
        "Luxury resort with spa in Dubai"
    ]

    for q in queries:
        print("=" * 50)
        print("Query:", q)
        vec = encoder.encode(q)
        print("Vector length:", len(vec))
        print("First 5 values:", vec[:5])

if __name__ == "__main__":
    run()
