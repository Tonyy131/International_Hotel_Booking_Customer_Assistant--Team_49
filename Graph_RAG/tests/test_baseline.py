# test_baseline.py

from preprocessing.preprocess_intent import classify_user_intent
from preprocessing.entity_extractor import EntityExtractor
from retrieval.retrieval_pipeline import RetrievalPipeline


query = "I want a hotel in Cairo "

print("\n=== TEST: Intent Classification ===")
intent_info = classify_user_intent(query)
print(intent_info)
intent = intent_info["intent"]

print("\n=== TEST: Entity Extraction ===")
extractor = EntityExtractor()
entities = extractor.extract(query)
print(entities)

print("\n=== TEST: Baseline Retrieval ===")
pipeline = RetrievalPipeline()
results = pipeline.retrieve(intent, entities, query, use_embeddings=False)

print("\nBaseline Results:")
for r in results["baseline"]:
    print(r)

print("\nContext Text:")
print(results["context_text"])
