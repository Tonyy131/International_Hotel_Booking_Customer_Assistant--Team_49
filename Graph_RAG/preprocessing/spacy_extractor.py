"""
spaCyExtractor: Responsible for detecting named entities using spaCy.
- GPE (cities & countries)
- LOC (locations)
"""

from typing import List
import spacy
import pycountry
import geonamescache
import re
import difflib
"""
we use both pycountry and geonamescache to classify GPEs into cities and countries.
- pycountry has comprehensive country data, including alternative names (and fuzzy search).
- geonamescache provides a large list of cities worldwide but not that much countries.
"""
class SpacyExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def extract_gpe_entities(self, text: str) -> List[str]:
        """
        Extract GPE and LOC entities from the input text.
        Returns a list of unique entity strings.
        """
        doc = self.nlp(text)
        entities = set()

        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                entities.add(ent.text)

        return list(entities)
    

#-------- city and country classification---------

class Country_City_Classifier:
    def __init__(self):
        self.gc = geonamescache.GeonamesCache()

        all_cities = self.gc.get_cities()
        self.city_names = {c["name"].lower() for c in all_cities.values()}
        self.country_names_gc = {c["name"].lower() for c in self.gc.get_countries().values()}
        self.country_names_pc = {country.name.lower(): country.name for country in pycountry.countries}


    def canonical_country_name(self, name: str) -> str:
        """
        Normalize any country input into its canonical ISO country name.
        """
        name_clean = name.strip().title()

        # Try pycountry fuzzy search (best method)
        try:
            match = pycountry.countries.search_fuzzy(name_clean)
            return match[0].name
        except LookupError:
            pass

        # Fallback if exact match exists in pycountry
        if name_clean.lower() in self.country_names_pc:
            return self.country_names_pc[name_clean.lower()]

        # Fallback to geonamescache countries
        for c in self.gc.get_countries().values():
            if c["name"].lower() == name_clean.lower():
                return c["name"]

        return name_clean  # last fallback


    def is_country(self, name: str) -> bool:
        name_low = name.lower().strip()

        # Reject if not title cased
        if name == name_low:
            return False

        #Exact name using pucountry
        if name_low in self.country_names_pc:
            return True
        
        #fuzzy match using pucountry
        try:
            matches = pycountry.countries.search_fuzzy(name_low)
            if matches:
                return True
        except LookupError:
            pass

        #Exact name using geonamescache
        if name_low in self.country_names_gc:
            return True

        return False
    

    def is_city(self, name: str) -> bool:
        name_low = name.lower().strip()
        
        # Exact match (Existing logic)
        if name_low in self.city_names:
            return True
        
        # Fuzzy match against the entire set of city names
        matches = difflib.get_close_matches(name_low, self.city_names, n=1, cutoff=0.9)
        if matches:
            return True

        return False
    
    def classify(self, gpe_list):
        cities = []
        countries = []

        for name in gpe_list:
            if self.is_city(name):
                cities.append(name)
            elif self.is_country(name):
                countries.append(self.canonical_country_name(name))

        return {
            "cities": cities,
            "countries": countries
        }
    

#------------Origin and Destination Detection------------

class OriginDestinationDetector:
    def __init__(self, classifier: Country_City_Classifier, nlp):
        self.classifier = classifier
        self.nlp = nlp
        self.gc = geonamescache.GeonamesCache()


    #-------- Helper to get prepositional object
    def get_pobj(self, prep_token):
        for child in prep_token.children:
            if child.dep_ == "pobj":
                tokens = [child.text]

                # include compound words: "South", "United", "New"
                for grandchild in child.children:
                    if grandchild.dep_ in ("compound", "amod"):
                        tokens.append(grandchild.text)


                return " ".join(tokens)
        return None

    #-------- Country classification helper
    def classify_country(self, name: str):
        name_clean = name.strip()

        if self.classifier.is_country(name_clean):
            try:
                match = pycountry.countries.search_fuzzy(name_clean)
                return match[0].name  
            except Exception:
                return name_clean
            
        name_low = name_clean.lower()
        all_cities = self.gc.get_cities()
        for city_data in all_cities.values():
            if city_data["name"].lower() == name_low:
                country_code = city_data["countrycode"] 
                country = pycountry.countries.get(alpha_2=country_code)
                if country:
                    return country.name
            
        return None


    def extract(self, text: str, gpe_list):
        doc = self.nlp(text)

        origin_city_sources = []

        origin = []
        destination = []

        # look for "from X"
        for token in doc:
            if token.text.lower() == "from" and token.dep_ == "prep":
                probj = self.get_pobj(token)
                if probj:
                    country = self.classify_country(probj)
                    if country and country not in origin:
                        origin.append(country)
                        origin_city_sources.append(probj)
                        
        # look for "live in X"
        for token in doc:
            if token.lemma_ == "live":
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "in":
                        probj = self.get_pobj(child)
                        if probj:
                            country = self.classify_country(probj)
                            if country and country not in origin:
                                origin.append(country)
                                origin_city_sources.append(probj)

        # look for "to Y"
        for token in doc:
            if token.text.lower() == "to" and token.dep_ == "prep":
                probj = self.get_pobj(token)
                if probj:
                    country = self.classify_country(probj)
                    if country and country not in destination:
                        destination.append(country)

        # look for verbs
        travel_verbs = {"go", "going", "travel", "travelling", "visit", "visiting", "fly", "heading"}
        for token in doc:
            if token.lemma_.lower() in travel_verbs and token.pos_ == "VERB":
                # look for direct objects (destinations)
                for child in token.children:
                    if child.dep_ == "dobj":
                        country = self.classify_country(child.text)
                        if country and country not in destination:
                            destination.append(country)

                # look for prepositional objects with "from" (origins)
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "from":
                        probj = self.get_pobj(child)
                        if probj:
                            country = self.classify_country(probj)
                            if country and country not in origin:
                                origin.append(country)

        # fallback to spacy GPEs if nothing found
        classified = self.classifier.classify(gpe_list)
        if not origin and "from" in text.lower():
            for country in classified["countries"]:
                origin.append(country)
                break

        if not destination and len (classified["countries"]) == 1:
            destination.append(classified["countries"][0])

        #if we have cities but no destination, infer country from city
        if not destination and classified["cities"]:
            for city in classified["cities"]:
                if city.lower() in {c.lower() for c in origin_city_sources}:
                    continue
                country = self.classify_country(city)
                if country and country not in destination:
                    destination.append(country)


        return {
            "origin_country": origin,
            "destination_country": destination
        }
    
class RatingExtractor:
    STAR_TO_RATING = {
        5: 9.0,
        4: 8.0,
        3: 7.0,
        2: 6.0,
        1: 5.0
    }

    HEURISTICS = {
        "excellent": 9.0,
        "very good": 8.0,
        "good": 7.0,
        "average": 6.0,
        "decent": 5.0,
        "average": 5.0,
        "terrible": 4.0,
    }

    def _normalize_score(self, numerator, denominator):
        """Converts score on any scale (e.g., X/5, X/100) to a 10.0 scale."""
        try:
            # Avoid division by zero
            if denominator == 0:
                return None
                
            return (numerator / denominator) * 10.0
        except TypeError:
            return None

    def extract_rating(self, text: str):
        text_low = text.lower()
        possible_ratings = []

        # 1. Extract and Normalize Numerical Ratings
        
        # Pattern 1: Look for X/Y scales (e.g., 4.5/5, 85 out of 100)
        scale_match = re.search(
            r"(\d+(?:\.\d+)?)\s*(?:out of|\/)\s*(\d+)", 
            text_low
        )
        if scale_match:
            numerator = float(scale_match.group(1))
            denominator = float(scale_match.group(2))
            normalized_score = self._normalize_score(numerator, denominator)
            if normalized_score is not None:
                possible_ratings.append(normalized_score)
        
        # Pattern 2: Look for standalone numbers explicitly linked to a minimum rating
        # Check for explicit 'rating' or 'score' phrase followed by a number
        m = re.search(r"(rating|score|minimum)\s+.*(\d+(\.\d+)?)", text_low)
        if m:
            possible_ratings.append(float(m.group(2)))

        # Check for comparison phrases followed by a number
        m = re.search(r"(above|over|greater than|higher than|at least|min(?:imum)?)\s+(\d+(\.\d+)?)", text_low)
        if m:
            possible_ratings.append(float(m.group(2)))

        # 2. Star → rating conversion
        
        # Pattern: (\d) star or (\d)-star, near "hotel" or alone
        star_match = re.search(r"(\d)\s*-?star", text_low)
        if star_match:
            stars = int(star_match.group(1))
            if stars in self.STAR_TO_RATING:
                possible_ratings.append(self.STAR_TO_RATING[stars])

        # 3. Heuristic descriptors
        
        for phrase, rating in self.HEURISTICS.items():
            # Ensure we only match whole words by using word boundaries (\b)
            if re.search(rf"\b{phrase}\b", text_low):
                possible_ratings.append(rating)

        # 4. Final Selection
        
        # Since the goal is to find the *minimum* required rating, 
        # the highest value found among all matches satisfies all constraints.
        if possible_ratings:
            return max(possible_ratings)

        # No rating found
        return None


class TravellerTypeExtractor:
    """
    Detects traveller type: solo, family, business, couple, group.
    """

    SOLO_PATTERNS = [
        "i am travelling alone",
        "i'm travelling alone",
        "traveling alone",
        "travelling alone",
        "alone",
        "solo",
        "by myself",
        "just me"
    ]

    FAMILY_PATTERNS = [
        "we are a family",
        "family trip",
        "family vacation",
        "family of",
        "kids",
        "children",
        "with my family"
    ]

    COUPLE_PATTERNS = [
        "with my wife",
        "with my husband",
        "with my girlfriend",
        "with my boyfriend",
        "couple",
        "honeymoon",
        "me and my wife",
        "me and my husband"
    ]

    BUSINESS_PATTERNS = [
        "business trip",
        "for business",
        "work trip",
        "corporate",
        "conference",
        "work travel"
    ]

    GROUP_PATTERNS = [
        "we are a group",
        "group of",
        "with my friends",
        "students",
        "school trip",
        "group travel"
    ]

    def infer_group(self, text: str, genders: list):
        t = text.lower()

        # If multiple genders → group
        if genders and len(genders) >= 2:
            return "group"

        # If plural pronouns → likely group
        if any(p in t for p in ["we ", "us ", "our "]):
            return "group"

        # If multiple persons joined with "and"
        if " and " in t and any(word in t for word in ["boy", "girl", "man", "woman", "friend", "friends"]):
            return "group"

        return None



    def extract(self, text: str):
        t = text.lower()

        for p in self.SOLO_PATTERNS:
            if p in t:
                return "solo"

        for p in self.FAMILY_PATTERNS:
            if p in t:
                return "family"

        for p in self.COUPLE_PATTERNS:
            if p in t:
                return "couple"

        for p in self.BUSINESS_PATTERNS:
            if p in t:
                return "business"

        for p in self.GROUP_PATTERNS:
            if p in t:
                return "group"

        return None


class DemographicsExtractor:
    """
    Extracts age and gender.
    """
    
    def extract_age(self, text: str):
        t = text.lower()

        # Pattern: "I am 25" / "I'm 32"
        m = re.search(r"\b(i am|i'm|age|aged)\s*(\d{1,2})\b", t)
        if m:
            age = int(m.group(2))
            return self._map_age_to_group(age)

        # Pattern: "between 20 and 30"
        m = re.search(r"between\s*(\d{1,2})\s*and\s*(\d{1,2})", t)
        if m:
            avg_age = (int(m.group(1)) + int(m.group(2))) // 2
            return self._map_age_to_group(avg_age)

        return None


    def _map_age_to_group(self, age: int):
        if age < 18: return None
        if 18 <= age <= 24: return "18-24"
        if 25 <= age <= 34: return "25-34"
        if 35 <= age <= 44: return "35-44"
        if 45 <= age <= 54: return "45-54"
        return "55+"

    def extract_gender(self, text: str):
        t = text.lower()

        genders = []

        if re.search(r"\b(male|man|boy)\b", t):
            genders.append("male")

        if re.search(r"\b(female|woman|girl)\b", t):
            genders.append("female")


        if genders:
            return genders
        return None

    
    def extract(self, text: str):
        return {
            "age_group": self.extract_age(text),
            "gender": self.extract_gender(text)
        }
