from typing import Any, List, Dict

def build_feature_text(record: Dict[str, Any]) -> str:
    """
    Builds descriptive text for a single hotel node to be used for vector embedding.

    Expected record keys (from Cypher):
      - "h"            : Hotel node (contains properties like name, star_rating, avg_score_*, etc.)
      - "city_name"    : str
      - "country_name" : str
      - "review_texts" : List[str]
    """

    hotel = record["h"]
    
    # 1. Basic Metadata
    name = hotel.get("name")
    city = record.get("city_name")
    country = record.get("country_name")
    review_texts: List[str] = record.get("review_texts", [])
    
    parts: List[str] = []

    # --- Section 1: Identity & Location ---
    if name:
        parts.append(f"{name}.")
    
    if city and country:
        parts.append(f"Located in {city}, {country}.")
    elif city:
        parts.append(f"Located in {city}.")
    elif country:
        parts.append(f"Located in {country}.")

    # --- Section 2: Ratings & Quality ---
    # Star Rating
    star_rating = hotel.get("star_rating")
    if star_rating is not None:
        parts.append(f"Star rating: {star_rating} stars.")

    # Global Average Score (from compute_average_review_scores)
    avg_score = hotel.get("average_reviews_score")
    if avg_score is not None:
        parts.append(f"Global Review Score: {round(avg_score, 1)}/10.")

    # Traveller Type Scores (Dynamic Extraction)
    # Matches properties like 'avg_score_solo_traveller' created in create_kg.py
    traveller_scores = []
    for key, value in hotel.items():
        if key.startswith("avg_score_") and value is not None:
            # key format: "avg_score_solo_traveller" -> "Solo Traveller"
            ttype = key.replace("avg_score_", "").replace("_", " ").title()
            traveller_scores.append(f"{ttype}: {value:.1f}")
    
    if traveller_scores:
        parts.append("Traveller Ratings: " + ", ".join(traveller_scores) + ".")

    # Base Category Scores (Cleanliness, Comfort, Facilities)
    subs = []
    for key in ["cleanliness_base", "comfort_base", "facilities_base"]:
        val = hotel.get(key)
        if val is not None:
            # "cleanliness_base" -> "Cleanliness"
            label = key.replace("_base", "").capitalize()
            subs.append(f"{label} {val}")

    if subs:
        parts.append("Category Scores: " + ", ".join(subs) + ".")

    # --- Section 3: Qualitative Data (Reviews) ---
    if review_texts:
        cleaned = []
        # We take up to 3 reviews to keep the context size manageable
        for txt in review_texts[:3]:
            # Clean newlines to prevent fragmentation
            snippet = str(txt).strip().replace("\n", " ")
            # Truncate very long reviews
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            cleaned.append(snippet)
        
        if cleaned:
            parts.append("Sample reviews: " + " | ".join(cleaned))

    # Fallback if no data exists
    if not parts and name:
        return name

    return " ".join(parts)