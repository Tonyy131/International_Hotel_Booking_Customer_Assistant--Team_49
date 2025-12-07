"""
EntityExtractor: orchestrates ALL NLP extractors
- spaCy NER (cities, countries)
- hotel name matching
- country/city classification
- origin/destination detection
- traveller type extraction
- demographic extraction (age + gender)
- rating extraction
"""

from typing import List, Dict
from .spacy_extractor import (
    SpacyExtractor, 
    Country_City_Classifier,
    OriginDestinationDetector,
    TravellerTypeExtractor,
    DemographicsExtractor,
    RatingExtractor
)
from .hotel_matcher import HotelMatcher
from .hotel_loader import load_hotels
from .llm_entity_extractor import extract_with_llm


class EntityExtractor:
    def __init__(self):
        self.spacy_ex = SpacyExtractor()
        self.classifier = Country_City_Classifier()
        self.od_detector = OriginDestinationDetector(
            classifier=self.classifier,
            nlp=self.spacy_ex.nlp
        )
        self.traveller_type_ex = TravellerTypeExtractor()
        self.demographics_ex = DemographicsExtractor()
        self.rating_ex = RatingExtractor()
        hotel_names = load_hotels()
        self.hotel_matcher = HotelMatcher(hotel_names)

    def extract(self, text: str, use_llm:bool = True) -> Dict:
        text_low = text.lower()

        if use_llm:
            try:
                llm_result = extract_with_llm(text)
                if llm_result is None:
                    raise ValueError("LLM returned None")
                # Fix age_group (model sometimes outputs string "null")
                if isinstance(llm_result["age_group"], str) and llm_result["age_group"].lower() == "null":
                    llm_result["age_group"] = None

                return llm_result
            except Exception as e:
                print(f"LLM extraction failed: {e}. Falling back to rule-based extractor.")

        # 1) Extract GPE entities via spaCy (cities/countries)
        gpes = self.spacy_ex.extract_gpe_entities(text)

        # 2) Country/City Classification
        locs = self.classifier.classify(gpes)
        cities = locs["cities"]
        countries = locs["countries"]

        # 3) Origin / Destination Detection
        od = self.od_detector.extract(text, gpes)
        origin = od["origin_country"]
        destination = od["destination_country"]

        # 4) Hotel Name Detection
        hotel_exact = self.hotel_matcher.match_exact(text)
        hotel_fuzzy = self.hotel_matcher.match_fuzzy(text)

        hotels = []
        if hotel_exact:
            hotels.append(hotel_exact)
        elif hotel_fuzzy:
            hotels.append(hotel_fuzzy)

        # 5) Traveller Type Detection
        traveller_type = self.traveller_type_ex.extract(text)

        # 6) Demographic Extraction (age + gender)
        demographics = self.demographics_ex.extract(text)
        age_group = demographics["age_group"]
        gender = demographics["gender"]  # now a list if multiple genders detected

        # If multiple genders → infer group travel
        if gender and isinstance(gender, list) and len(gender) >= 2:
            traveller_type = "group"

        # 7) Rating Extraction
        rating = self.rating_ex.extract_rating(text)

        # 8) Return clean, unified entity dictionary
        return {
            "cities": cities,
            "countries": countries,
            "hotels": hotels,
            "origin_country": origin,
            "destination_country": destination,
            "traveller_type": traveller_type,
            "age_group": age_group,
            "gender": gender,
            "rating": rating
        }
    

_global_extractor = EntityExtractor()

def extract_entities(text: str) -> Dict:
    """Public helper to extract all entities using the global extractor."""
    return _global_extractor.extract(text)