# Graph_RAG/tests/test_baseline.py
# Run from project root:
#   python -m Graph_RAG.tests.test_baseline

from preprocessing.preprocess_intent import classify_user_intent
from preprocessing.entity_extractor import EntityExtractor
from retrieval.retrieval_pipeline import RetrievalPipeline


def run_baseline_test(query: str, use_embeddings: bool = False, limit: int = 10):
    """
    Runs a full baseline retrieval test for a given query.
    Returns a dictionary with intent, entities, baseline results and context.
    """

    print("\n==============================")
    print("TESTING QUERY:", query)
    print("==============================\n")

    # 1) Intent classification
    print("=== INTENT CLASSIFICATION ===")
    intent_info = classify_user_intent(query)
    print(intent_info)
    intent = intent_info["intent"]

    # 2) Entity extraction
    print("\n=== ENTITY EXTRACTION ===")
    extractor = EntityExtractor()
    entities = extractor.extract(query)
    print(entities)

    # 3) Retrieval pipeline
    print("\n=== BASELINE RETRIEVAL ===")
    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(intent, entities, query, use_embeddings=use_embeddings, limit=limit)

    # Output
    print("\n--- BASELINE RESULTS ---")
    for r in results["baseline"]:
        print(r)

    print("\n--- CONTEXT TEXT ---")
    print(results["context_text"])

    return {
        "intent": intent,
        "entities": entities,
        "baseline": results["baseline"],
        "context_text": results["context_text"],
    }


# Allow running as script
if __name__ == "__main__":
    test_query = "I want a hotel in ,Germany,and Italy for business class"
    run_baseline_test(test_query, use_embeddings=False)




# Use this when need to use in another file
# from Graph_RAG.tests.test_baseline import run_baseline_test

# results = run_baseline_test("Find hotels in Paris above rating 8")
