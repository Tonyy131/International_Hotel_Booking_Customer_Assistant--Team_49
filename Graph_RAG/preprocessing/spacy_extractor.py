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
        self.nlp = spacy.load("en_core_web_sm")

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
        
        self.city_map = {}
        
        all_cities = self.gc.get_cities()
        for c in all_cities.values():
            real_name = c["name"]
            low_name = real_name.lower()
            
            # Map exact name
            self.city_map[low_name] = real_name
            
            # AUTOMATIC GENERALIZATION:
            # If DB has "New York City", map "new york" -> "New York City"
            if low_name.endswith(" city"):
                short_name = low_name.replace(" city", "").strip()
                # Only add if it doesn't conflict with an existing city
                if short_name and short_name not in self.city_map:
                    self.city_map[short_name] = real_name

    

        # Country sets
        self.country_names_gc = {c["name"].lower() for c in self.gc.get_countries().values()}
        self.country_names_pc = {country.name.lower(): country.name for country in pycountry.countries}


    def canonical_country_name(self, name: str) -> str:
        """Normalize country name to ISO standard."""
        name_clean = name.strip().title()

        # Fuzzy match via pycountry
        try:
            match = pycountry.countries.search_fuzzy(name_clean)
            return match[0].name
        except LookupError:
            pass

        if name_clean.lower() in self.country_names_pc:
            return self.country_names_pc[name_clean.lower()]

        for c_name in self.country_names_gc:
             if c_name == name_clean.lower():
                 return c_name.title()

        return name_clean


    def is_country(self, name: str) -> bool:
        name_low = name.lower().strip()

        if name_low in self.country_names_pc: return True
        if name_low in self.country_names_gc: return True

        if name_low in self.city_map:
            return False
            
        try:
            if pycountry.countries.search_fuzzy(name_low):
                return True
        except LookupError:
            pass

        return False
    

    def is_city(self, name: str) -> bool:
        name_low = name.lower().strip()
        # lookup using our smart map
        if name_low in self.city_map:
            return True
        
        # Fuzzy fallback (slower)
        # We search against keys of city_map to catch partial typos
        matches = difflib.get_close_matches(name_low, self.city_map.keys(), n=1, cutoff=0.9)
        if matches:
            return True

        return False
    
    def normalize_city_name(self, name: str) -> str:
        """Returns the canonical name (e.g. 'new york' -> 'New York City')"""
        name_low = name.lower().strip()
        
        # 1. Direct Map Lookup
        if name_low in self.city_map:
            return self.city_map[name_low]
            
        # 2. Fuzzy Map Lookup (matches logic in is_city)
        matches = difflib.get_close_matches(name_low, self.city_map.keys(), n=1, cutoff=0.9)
        if matches:
            return self.city_map[matches[0]]
            
        return name

    def classify(self, gpe_list):
        cities = []
        countries = []

        for name in gpe_list:
            # 1. Country Check FIRST
            if self.is_country(name):
                countries.append(self.canonical_country_name(name))
            
            # 2. City Check SECOND 
            elif self.is_city(name):
                cities.append(self.normalize_city_name(name))

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
        
        # Pre-load cities for population-based lookup
        self.city_lookup = {}
        all_cities = self.gc.get_cities()
        
        for c in all_cities.values():
            name_low = c["name"].lower()
            pop = c.get("population", 0)
            code = c["countrycode"]
            
            # Resolve country name immediately
            country_obj = pycountry.countries.get(alpha_2=code)
            if country_obj:
                c_name = country_obj.name
                
                if name_low not in self.city_lookup:
                    self.city_lookup[name_low] = []
                self.city_lookup[name_low].append((pop, c_name))
        
        # Sort all entries by population descending
        for k in self.city_lookup:
            self.city_lookup[k].sort(key=lambda x: x[0], reverse=True)


    def classify_country(self, name: str):
        """
        Determines country from a name.
        Prioritizes: 1. Explicit Country 2. City with Highest Population
        """
        name_clean = name.strip()
        # Normalize (New York -> New York City)
        name_clean = self.classifier.normalize_city_name(name_clean)

        # 1. Check if it IS a country
        if self.classifier.is_country(name_clean):
            try:
                match = pycountry.countries.search_fuzzy(name_clean)
                return match[0].name  
            except Exception:
                return name_clean
            
        # 2. Check if it is a City (Lookup by Population)
        name_low = name_clean.lower()
        if name_low in self.city_lookup:
            # Return country of the most populous city match
            return self.city_lookup[name_low][0][1]
            
        return None


    def extract(self, text: str, gpe_list):
        doc = self.nlp(text)
        
        origin_countries = set()
        destination_countries = set()
        origin_sources = set()  # Tracks normalized city names used as origin

        # Triggers
        # "be" is handled specifically for "I am in..." vs "I will be in..."
        ORIGIN_VERBS = {"live", "reside", "stay", "born", "come", "hail"}
        ORIGIN_PREPS = {"from"}
        
        DEST_VERBS = {"visit", "travel", "go", "fly", "head", "vacation", "book", "want", "need", "plan"}
        DEST_NOUNS = {"trip", "holiday", "vacation", "hotel", "hostel", "apartment", "flight"}

        for ent in doc.ents:
            if ent.label_ not in {"GPE", "LOC"}: continue
            
            country_name = self.classify_country(ent.text)
            if not country_name: continue

            # Normalize city name (e.g., "New York" -> "new york city")
            city_norm = self.classifier.normalize_city_name(ent.text).lower()

            # --- ANCESTOR SEARCH (Scan up the tree) ---
            # We look up to 3 levels up for a trigger
            # e.g. "I [live] in the beautiful [city] of [New York]"
            
            is_origin = False
            is_dest = False
            
            # 1. Check immediate head for Prepositions ("from Paris", "to London")
            head = ent.root.head
            if head.text.lower() == "from":
                is_origin = True
            elif head.text.lower() == "to":
                is_dest = True
            
            # 2. Walk up the tree for Verbs/Nouns
            if not is_origin and not is_dest:
                # Ancestors stream: [in, city, of, live]...
                for token in ent.root.ancestors:
                    lemma = token.lemma_.lower()
                    
                    # ORIGIN CHECKS
                    if lemma in ORIGIN_VERBS:
                        is_origin = True
                        break
                    
                    # DESTINATION CHECKS
                    if lemma in DEST_VERBS:
                        is_dest = True
                        break
                    if token.pos_ == "NOUN" and lemma in DEST_NOUNS:
                        is_dest = True
                        break
                    
                    # Special Handling for "BE" (am, is, are)
                    # "I am in Egypt" (Origin) vs "I will be in Egypt" (Dest)
                    if lemma == "be":
                        # Check children of 'be' for future markers
                        is_future = any(child.lemma_ in {"will", "go", "plan", "want"} for child in token.children)
                        if is_future:
                            is_dest = True
                        else:
                            is_origin = True
                        break

            # --- ASSIGNMENT ---
            if is_origin:
                origin_countries.add(country_name)
                origin_sources.add(city_norm)
            elif is_dest:
                destination_countries.add(country_name)

        # --- FALLBACK LOGIC ---
        # If a city was mentioned but not caught by dependency parsing above,
        # we infer its role based on what we already know.
        
        classified = self.classifier.classify(gpe_list)
        
        for city in classified["cities"]:
            city_norm = self.classifier.normalize_city_name(city).lower()
            
            # Skip if this city was already identified as an origin source
            if city_norm in origin_sources:
                continue

            country = self.classify_country(city_norm)
            if not country: continue

            # Heuristic: If we already have an Origin, assume unused cities are Destinations
            if origin_countries:
                if country not in origin_countries:
                    destination_countries.add(country)
            
            # Heuristic: If NO origin found yet, and we have "from" in text, 
            # the first entity might be origin 
            elif "from" in text.lower() and not destination_countries:
                 origin_countries.add(country)
                 origin_sources.add(city_norm)
            
            # Default: If not origin, add to destination
            else:
                destination_countries.add(country)

        # Clean up: Ensure Countries found via entities are added
        for country in classified["countries"]:
             if country not in origin_countries and country not in destination_countries:
                 destination_countries.add(country)

        return {
            "origin_country": list(origin_countries),
            "destination_country": list(destination_countries)
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
