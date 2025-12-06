from preprocessing.preprocess_intent import classify_user_intent

# Simple test cases for intent classification
def run_tests():
    cases = [
        ("Suggest for me an hotel to stay in", "recommendation"),
        ("Can you help me in booking a hotel to stay in", "booking"),
        ("Do I need a visa to travel from Egypt to Turkey?", "visa_query"),
        ("Show me reviews for Hotel Nile Plaza", "review_query"),
        ("Find hotels in Cairo with rating above 8", "hotel_search"),
    ]
    for q, expected in cases:
        out = classify_user_intent(q, use_llm_fallback=False)
        ok = out["intent"] == expected
        print(f"Q: {q}\n→ intent: {out['intent']} (expected: {expected}) — {'OK' if ok else 'FAIL'}")
        print(" scores:", out["intent_scores"])
        print(" top:", out["top_score"], "fallback_used:", out["fallback_used"], "source:", out["intent_source"])
        print("-"*60)

if __name__ == "__main__":
    run_tests()
