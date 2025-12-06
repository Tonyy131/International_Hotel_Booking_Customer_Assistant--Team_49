from preprocessing.preprocess_intent import classify_user_intent

tests = [
    "Could you maybe help me out?",
    "I want something nice around there",
    "Going to Turkey soon, anything I should know?",
    "What do people think about this place?",
    "I want to sort something out for my stay"
]

for q in tests:
    out = classify_user_intent(q, use_llm_fallback=True)
    print("="*60)
    print("Query:", q)
    print("Intent:", out["intent"])
    print("Source:", out["intent_source"])
    print("Scores:", out["intent_scores"])
    print("Top Score:", out["top_score"])
    print("Fallback Used:", out["fallback_used"])
