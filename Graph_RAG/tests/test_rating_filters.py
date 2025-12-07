# tests/test_rating_queries.py

from preprocessing.entity_extractor import EntityExtractor

def run():

    extractor = EntityExtractor()

    queries = [

        # --- Minimum rating (score) ---
        "Find me hotels above 8 in Cairo.",
        "Show me hotels rated at least 8.5 in Berlin.",
        "Hotels with rating 9 or higher in Tokyo.",

        # --- Maximum rating (score) ---
        "Find hotels in Paris with rating below 9.",
        "Show me hotels with a rating less than 8.8 in Dubai.",

        # --- Rating ranges (score) ---
        "Find hotels in Rome with rating between 8.7 and 9.",
        "I want a hotel in Madrid with rating from 8.8 to 9.1.",

        # --- Star exact ---
        "Show me 5 star hotels in Dubai.",
        "Find 4 star hotels in Cairo.",

        # --- Star upper bound ---
        "I want hotels with less than 5 stars in London.",
        "Find hotels with under 4 stars in Berlin.",

        # --- Star ranges ---
        "Find hotels in Barcelona with 4 to 5 stars.",
        "Show me hotels in Munich between 3 and 5 stars.",

        # --- Combined with origin/destination ---
        "I live in Egypt and want a 4 star hotel in Paris.",
        "I'm traveling from New York to Tokyo and want a 5 star hotel.",

        # --- Combined with traveller type ---
        "We are a couple looking for 4 star hotels in Istanbul.",
        "A family of four wants a hotel rated between 8.8 and 9.2 in Dubai.",

        # --- Edge cases ---
        "Any cheap hotel with rating less than 6 in Rome?",
        "Show me terrible hotels with rating below 5 in Madrid.",
        "I want a 2 star hotel in Munich.",
        "Give me hotels between 1 and 3 stars in Paris.",

        # --- Make sure extraction still works normally ---
        "I live in new york and want a hotel in Stuttgart"
    ]

    for q in queries:
        print("=" * 60)
        print("Query:", q)
        entities = extractor.extract(q)
        print("Entities:", entities)


if __name__ == "__main__":
    run()
