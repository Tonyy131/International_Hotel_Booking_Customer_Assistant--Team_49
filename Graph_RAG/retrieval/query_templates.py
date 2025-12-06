# Graph_RAG/retrieval/query_templates.py

"""
A small library of Cypher templates for the hotel theme.
Add/extend these templates as your KG evolves.
Parameters are provided as dicts when executing queries.
"""

QUERY_TEMPLATES = {
    # Basic hotel search by city
    "hotel_search_by_city": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City {name: $city})
        RETURN h.name AS name, h.hotel_id AS hotel_id, h.star_rating AS stars, h.average_reviews_score AS avg_score
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Search hotels by country
    "hotel_search_by_country": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(co:Country {name: $country})
        RETURN h.name AS name, h.hotel_id AS hotel_id, h.star_rating AS stars, h.average_reviews_score AS avg_score
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Search hotels with minimum average rating
    "hotel_search_min_rating": """
        MATCH (h:Hotel)
        WHERE h.average_reviews_score >= $rating
        RETURN h.name AS name, h.hotel_id AS hotel_id, h.star_rating AS stars, h.average_reviews_score AS avg_score
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Get reviews for a specific hotel (by exact name or id)
    "hotel_reviews_by_name": """
        MATCH (r:Review)-[:REVIEWED]->(h:Hotel {name: $hotel})
        RETURN r.review_id AS review_id, r.text AS text, r.score_overall AS score, r.date AS date
        ORDER BY r.date DESC
        LIMIT $limit
    """,

    "hotel_reviews_by_id": """
        MATCH (r:Review)-[:REVIEWED]->(h:Hotel {hotel_id: $hotel_id})
        RETURN r.review_id AS review_id, r.text AS text, r.score_overall AS score, r.date AS date
        ORDER BY r.date DESC
        LIMIT $limit
    """,

    # Recommend hotels based on traveller type (simple co-occurrence)
    "recommend_hotels_by_traveller_type": """
        MATCH (t:Traveller {type: $traveller_type})-[:STAYED_AT]->(h:Hotel)
        RETURN h.name AS name, h.hotel_id AS hotel_id, count(*) AS freq, h.average_reviews_score AS avg_score
        ORDER BY freq DESC, h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Visa requirement between two countries
    "visa_requirements": """
        MATCH (from:Country {name:$from})-[v:NEEDS_VISA]->(to:Country {name:$to})
        RETURN v.visa_type AS visa_type
    """,

    # Hotels that match a textual search (exact name substring)
    "hotel_by_name_substring": """
        MATCH (h:Hotel)
        WHERE toLower(h.name) CONTAINS toLower($q)
        RETURN h.name AS name, h.hotel_id AS hotel_id, h.average_reviews_score AS avg_score
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Top N hotels overall
    "top_hotels": """
        MATCH (h:Hotel)
        RETURN h.name AS name, h.hotel_id AS hotel_id, h.average_reviews_score AS avg_score
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Hotel details by id
    "hotel_details_by_id": """
        MATCH (h:Hotel {hotel_id: $hotel_id})-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(co:Country)
        RETURN h, city.name AS city, co.name AS country
        LIMIT 1
    """
}
