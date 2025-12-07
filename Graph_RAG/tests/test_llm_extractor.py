from preprocessing.llm_entity_extractor import extract_with_llm
from preprocessing.entity_extractor import EntityExtractor   # your existing extractor

spacy_ex = EntityExtractor()

TEST_QUERIES = [
    "I want a hotel in Cairo with great reviews",
    "Book me something in Paris, coming from Egypt",
    "I live in New York and need a hotel in Stuttgart",
    "Looking for a 4 star hotel in Dubai for my wife and I",
    "Need a cheap hotel near Munich airport",
    "A family trip to Berlin with kids, rating above 8",
    "Going to Tokyo from Canada, need a business hotel",
    "Hotel for elderly couple in Madrid with breakfast",
    "Find me the Hilton in London",
    "Traveling from Brazil to Rome, need something central"
]


def pretty(d):
    import json
    return json.dumps(d, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    for q in TEST_QUERIES:
        print("=" * 80)
        print("QUERY:", q)

        print("\n--- Rule-based (spaCy) ---")
        spacy_out = spacy_ex.extract(q, use_llm=False)
        print(pretty(spacy_out))

        print("\n--- LLM Extractor ---")
        llm_out = extract_with_llm(q)
        print(pretty(llm_out))

        print("\n")
