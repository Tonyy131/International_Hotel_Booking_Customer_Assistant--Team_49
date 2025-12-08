from preprocessing.entity_extractor import EntityExtractor
from preprocessing.hotel_loader import load_hotels

def run():

    extractor = EntityExtractor()

    queries = [
        "Find me hotels above 8 in Cairo.",
        "We are a boy and a girl that want to go to a trip to Germany.",
        "I'm travelling alone to Japan.",
        "Show me 5 star hotels in Dubai.",
        "I am 27 years old and I'm female.",
        "I want to stay in the Colosseum Gardens hotel.",
        "I'm going from Egypt to Turkey.",
        "We are a family of 4 visiting Paris.",
        "I want excellent hotels in Berlin.",
        "I live in Canada and want to visit Panama.",
        "I live in Cairo and want a hotel in Manhattan.",
        "I live in new york and want a hotel in Stuttgart"
    ]

    for q in queries:
        print("=" * 60)
        print("Query:", q)
        entities = extractor.extract(q)
        print("Entities:", entities)


if __name__ == "__main__":
    run()
