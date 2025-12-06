from typing import Any, List, Dict

def build_feature_text(record: Dict[str, Any]) -> str:
    """
    Builds descriptive text for a single hotel node.

    Expected record keys (from Cypher):
      - "h"            : Hotel node
      - "city_name"    : str
      - "country_name" : str
      - "review_texts" : List[str]
    """

    hotel = record["h"]

    name = hotel.get("name")
    star_rating = hotel.get("star_rating")
    cleanliness = hotel.get("cleanliness_base")
    comfort = hotel.get("comfort_base")
    facilities = hotel.get("facilities_base")
    avg_review_score = hotel.get("average_reviews_score")

    city = record.get("city_name")
    country = record.get("country_name")
    review_texts: List[str] = record.get("review_texts", [])

    parts: List[str] = []

    if name:
        parts.append(f"{name}.")

    if city and country:
        parts.append(f"Located in {city}, {country}.")
    elif city:
        parts.append(f"Located in {city}.")
    elif country:
        parts.append(f"Located in {country}.")

    if star_rating is not None:
        parts.append(f"Star rating: {star_rating} stars.")

    if avg_review_score is not None:
        parts.append(f"Average review score: {round(avg_review_score, 1)}/10.")

    subs = []
    if cleanliness is not None:
        subs.append(f"cleanliness {cleanliness}")
    if comfort is not None:
        subs.append(f"comfort {comfort}")
    if facilities is not None:
        subs.append(f"facilities {facilities}")

    if subs:
        parts.append("Base scores: " + ", ".join(subs) + ".")

    if review_texts:
        cleaned = []
        for txt in review_texts[:3]:
            snippet = txt.strip()
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            cleaned.append(snippet)
        parts.append("Sample reviews: " + " ".join(cleaned))

    if not parts and name:
        return name

    return " ".join(parts)
