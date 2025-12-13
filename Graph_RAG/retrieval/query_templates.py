# Graph_RAG/retrieval/query_templates.py

"""
A small library of Cypher templates for the hotel theme.
Add/extend these templates as your KG evolves.
Parameters are provided as dicts when executing queries.
"""

QUERY_TEMPLATES = {
    # Basic hotel search by city
    "hotel_search_by_city": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)
        WHERE c.name IN $cities
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY c.name, h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Search hotels by country
    "hotel_search_by_country": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(co:Country)
        WHERE co.name IN $countries
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, country: co.name } AS hotel        
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Search hotels with minimum average rating
    "hotel_search_min_rating": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.average_reviews_score >= $rating
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,
    # Search hotels with minimum star rating
    "hotel_search_min_stars": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating >= $stars
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.star_rating DESC
        LIMIT $limit
    """,

    # Search hotels with minimum average score
    "hotel_search_min_avg_score": """
        MATCH (h:Hotel)
        WHERE h.average_reviews_score >= $min_score
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score } AS hotel
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
        WITH h, count(*) AS freq, h.average_reviews_score AS avgScore
        ORDER BY freq DESC, avgScore DESC
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, average_reviews_score: avgScore } AS hotel,
            freq
        LIMIT $limit
    """,


    # Visa requirement between two countries
    "visa_requirements": """
        MATCH (from:Country {name:$from})-[v:NEEDS_VISA]->(to:Country {name:$to})
        RETURN from.name AS origin_country, 
               to.name AS destination_country, 
               v.visa_type AS visa_type
    """,
    # Visa requirements by origin country
        "visa_requirements_by_origin": """
            MATCH (from:Country {name: $from})-[v:NEEDS_VISA]->(to:Country)
            RETURN 
                from.name AS origin_country,
                to.name AS destination_country,
                v.visa_type AS visa_type
            ORDER BY destination_country
        """,
    # Countries that do NOT require a visa from the origin
        "visa_free_countries_by_origin": """
        MATCH (origin:Country {name: $from})
        MATCH (dest:Country)
        WHERE dest <> origin
        AND NOT (origin)-[:NEEDS_VISA]->(dest)
        RETURN
            origin.name AS origin_country,
            dest.name AS destination_country,
            "Visa Free" AS visa_type
        ORDER BY destination_country
    """,


    # Hotels that match a textual search (exact name substring)
    "hotel_by_name_substring": """
        MATCH (h:Hotel)
        WHERE toLower(h.name) CONTAINS toLower($q)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Top N hotels overall
    "top_hotels": """
        MATCH (h:Hotel)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Hotel details by id
    "hotel_details_by_id": """
        MATCH (h:Hotel {hotel_id: $hotel_id})-[:LOCATED_IN]->(city:City)-[:LOCATED_IN]->(co:Country)
        RETURN h, city.name AS city, co.name AS country
        LIMIT 1
    """,

    # Best hotel overall based on average rating
    "best_hotel_overall": """
        MATCH (h:Hotel)
        WHERE h.average_reviews_score IS NOT NULL
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # Search hotels by city or country
    "hotel_search_by_city_or_country": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE c.name IN $cities
        OR co.name IN $countries
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, city: c.name, country: co.name, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,

    # rating range (min AND max)
    "hotel_search_rating_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.average_reviews_score >= $min AND h.average_reviews_score <= $max
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,
        
    "hotel_search_stars_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating >= $min AND h.star_rating <= $max
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.star_rating DESC
        LIMIT $limit
    """,


    # rating less-or-equal (upper bound)
    "hotel_search_max_rating": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.average_reviews_score <= $max
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,
    # star rating less-or-equal (upper bound)
    "hotel_search_max_stars": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating <= $max
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.star_rating DESC
        LIMIT $limit
    """,


    # exact rating
    "hotel_search_exact_rating": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE floor(h.average_reviews_score * 10) / 10 = $value AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities) AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,
    # exact star rating
    "hotel_search_exact_stars": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        WHERE h.star_rating = $value
        AND ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating, average_reviews_score: h.average_reviews_score, city: c.name } AS hotel
        ORDER BY h.star_rating DESC
        LIMIT $limit
    """,
        # Search hotels with minimum average cleanliness score (ratings are on Rating nodes)
    "hotel_search_min_cleanliness": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE avg_cleanliness >= $rating
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_cleanliness: avg_cleanliness, city: c.name } AS hotel
        ORDER BY avg_cleanliness DESC
        LIMIT $limit
    """,

    # Search hotels with maximum average cleanliness score (upper bound)
    "hotel_search_max_cleanliness": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE avg_cleanliness <= $max
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_cleanliness: avg_cleanliness, city: c.name } AS hotel
        ORDER BY avg_cleanliness DESC
        LIMIT $limit
    """,

    # Search hotels with cleanliness avg in a range
    "hotel_search_cleanliness_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE avg_cleanliness >= $min AND avg_cleanliness <= $max
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_cleanliness: avg_cleanliness, city: c.name } AS hotel
        ORDER BY avg_cleanliness DESC
        LIMIT $limit
    """,

    # Search hotels with exact average cleanliness value
    "hotel_search_exact_cleanliness": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE floor(avg_cleanliness * 10) / 10 = $value
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_cleanliness: avg_cleanliness, city: c.name } AS hotel
        ORDER BY avg_cleanliness DESC
        LIMIT $limit
    """,

    # hotels with highest cleanliness ratings

    "top_hotel_cleanliness": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE avg_cleanliness IS NOT NULL
        RETURN h {
            .*, 
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_cleanliness: avg_cleanliness,
            city: c.name
        } AS hotel
        ORDER BY avg_cleanliness DESC
        LIMIT $limit
    """,

    # hotels with lowest cleanliness ratings
    "worst_hotel_cleanliness": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_cleanliness) AS avg_cleanliness
        WHERE avg_cleanliness IS NOT NULL
        RETURN h {
            .*, 
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_cleanliness: avg_cleanliness,
            city: c.name
        } AS hotel
        ORDER BY avg_cleanliness ASC
        LIMIT $limit
    """,

    # hotels with lowest ratings
    "worst_hotels": """
        MATCH (h:Hotel)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, average_reviews_score: h.average_reviews_score } AS hotel
        ORDER BY h.average_reviews_score ASC
        LIMIT $limit
    """,

        # Search hotels with minimum average comfort score
    "hotel_search_min_comfort": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_comfort) AS avg_comfort
        WHERE avg_comfort >= $rating
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_comfort: avg_comfort, city: c.name } AS hotel
        ORDER BY avg_comfort DESC
        LIMIT $limit
    """,

    # upper bound comfort
    "hotel_search_max_comfort": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_comfort) AS avg_comfort
        WHERE avg_comfort <= $max
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_comfort: avg_comfort, city: c.name } AS hotel
        ORDER BY avg_comfort DESC
        LIMIT $limit
    """,

    # comfort range
    "hotel_search_comfort_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_comfort) AS avg_comfort
        WHERE avg_comfort >= $min AND avg_comfort <= $max
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_comfort: avg_comfort, city: c.name } AS hotel
        ORDER BY avg_comfort DESC
        LIMIT $limit
    """,

    # exact comfort match
    "hotel_search_exact_comfort": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
          AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_comfort) AS avg_comfort
        WHERE floor(avg_comfort * 10) / 10 = $value
        RETURN h { .*, hotel_id: h.hotel_id, name: h.name, star_rating: h.star_rating,
                   average_reviews_score: h.average_reviews_score, avg_score_comfort: avg_comfort, city: c.name } AS hotel
        ORDER BY avg_comfort DESC
        LIMIT $limit
    """,
    # hotels with highest comfort ratings
    "top_hotel_comfort": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_comfort) AS avg_comfort
        WHERE avg_comfort IS NOT NULL
        RETURN h {
            .*, 
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_comfort: avg_comfort,
            city: c.name
        } AS hotel
        ORDER BY avg_comfort DESC
        LIMIT $limit
    """,
    # hotels with lowest comfort ratings
    "worst_hotel_comfort": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_comfort) AS avg_comfort
        WHERE avg_comfort IS NOT NULL
        RETURN h {
            .*, 
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_comfort: avg_comfort,
            city: c.name
        } AS hotel
        ORDER BY avg_comfort ASC
        LIMIT $limit
    """,


    # Find hotels in countries that do NOT require a visa from the origin
    "hotel_search_visa_free": """
        MATCH (origin:Country)
        WHERE toLower(origin.name) = toLower($origin)
        
        MATCH (dest:Country)
        WHERE dest <> origin
        AND NOT (origin)-[:NEEDS_VISA]->(dest)
        
        MATCH (h:Hotel)-[:LOCATED_IN]->(:City)-[:LOCATED_IN]->(dest)
        
        RETURN h { 
            .*, 
            hotel_id: h.hotel_id, 
            name: h.name, 
            average_reviews_score: h.average_reviews_score,
            country: dest.name,
            visa_status: "Visa Free" 
        } AS hotel
        ORDER BY h.average_reviews_score DESC
        LIMIT $limit
    """,
    # Search hotels with minimum average facilities score
    "hotel_search_min_facilities": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_facilities) AS avg_facilities
        WHERE avg_facilities >= $rating
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities DESC
        LIMIT $limit
    """,

    # Search hotels with maximum average facilities score
    "hotel_search_max_facilities": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_facilities) AS avg_facilities
        WHERE avg_facilities <= $max
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities DESC
        LIMIT $limit
    """,

    # Search hotels with facilities avg in a range
    "hotel_search_facilities_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_facilities) AS avg_facilities
        WHERE avg_facilities >= $min AND avg_facilities <= $max
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities DESC
        LIMIT $limit
    """,

    # Search hotels with exact average facilities value
    "hotel_search_exact_facilities": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_facilities) AS avg_facilities
        WHERE floor(avg_facilities * 10) / 10 = $value
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities DESC
        LIMIT $limit
    """,

    # Hotels with highest facilities ratings
    "top_hotel_facilities": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_facilities) AS avg_facilities
        WHERE avg_facilities IS NOT NULL
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities DESC
        LIMIT $limit
    """,

    # Hotels with lowest facilities ratings
    "worst_hotel_facilities": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_facilities) AS avg_facilities
        WHERE avg_facilities IS NOT NULL
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_facilities: avg_facilities,
            city: c.name
        } AS hotel
        ORDER BY avg_facilities ASC
        LIMIT $limit
    """,
    # Search hotels with minimum average staff score
    "hotel_search_min_staff": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_staff) AS avg_staff
        WHERE avg_staff >= $rating
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff DESC
        LIMIT $limit
    """,
    # Search hotels with maximum average staff score
    "hotel_search_max_staff": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_staff) AS avg_staff
        WHERE avg_staff <= $max
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff DESC
        LIMIT $limit
    """,
    # Search hotels with staff avg in a range
    "hotel_search_staff_range": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_staff) AS avg_staff
        WHERE avg_staff >= $min AND avg_staff <= $max
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff DESC
        LIMIT $limit
    """,
    # Search hotels with exact average staff value
    "hotel_search_exact_staff": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, co, avg(r.score_staff) AS avg_staff
        WHERE floor(avg_staff * 10) / 10 = $value
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff DESC
        LIMIT $limit
    """,
    # Hotels with highest staff ratings
    "top_hotel_staff": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_staff) AS avg_staff
        WHERE avg_staff IS NOT NULL
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff DESC
        LIMIT $limit
    """,
    # Hotels with lowest staff ratings
    "worst_hotel_staff": """
        MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
        OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
        WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
        AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
        WITH h, c, avg(r.score_staff) AS avg_staff
        WHERE avg_staff IS NOT NULL
        RETURN h {
            .*,
            hotel_id: h.hotel_id,
            name: h.name,
            star_rating: h.star_rating,
            average_reviews_score: h.average_reviews_score,
            avg_score_staff: avg_staff,
            city: c.name
        } AS hotel
        ORDER BY avg_staff ASC
        LIMIT $limit
    """,
    # Search hotels with minimum average value for money score
    "hotel_search_min_value_for_money": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, co, avg(r.score_value_for_money) AS avg_value
    WHERE avg_value >= $rating
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value DESC
    LIMIT $limit
    """,
    # Search hotels with maximum average value for money score
    "hotel_search_max_value_for_money": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, co, avg(r.score_value_for_money) AS avg_value
    WHERE avg_value <= $max
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value DESC
    LIMIT $limit
    """,
    # Search hotels with value for money avg in a range
    "hotel_search_value_for_money_range": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, co, avg(r.score_value_for_money) AS avg_value
    WHERE avg_value >= $min AND avg_value <= $max
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value DESC
    LIMIT $limit
    """,
    # Search hotels with exact average value for money value
    "hotel_search_exact_value_for_money": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, co, avg(r.score_value_for_money) AS avg_value
    WHERE floor(avg_value * 10) / 10 = $value
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value DESC
    LIMIT $limit
    """,
    # Hotels with highest value for money ratings
    "top_hotel_value_for_money": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, avg(r.score_value_for_money) AS avg_value
    WHERE avg_value IS NOT NULL
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value DESC
    LIMIT $limit
    """,
    # Hotels with lowest value for money ratings
    "worst_hotel_value_for_money": """
    MATCH (h:Hotel)-[:LOCATED_IN]->(c:City)-[:LOCATED_IN]->(co:Country)
    OPTIONAL MATCH (h)<-[:REVIEWED]-(r:Review)
    WHERE ($cities IS NULL OR size($cities) = 0 OR c.name IN $cities)
      AND ($countries IS NULL OR size($countries) = 0 OR co.name IN $countries)
    WITH h, c, avg(r.score_value_for_money) AS avg_value
    WHERE avg_value IS NOT NULL
    RETURN h {
        .*,
        hotel_id: h.hotel_id,
        name: h.name,
        star_rating: h.star_rating,
        average_reviews_score: h.average_reviews_score,
        avg_score_value_for_money: avg_value,
        city: c.name
    } AS hotel
    ORDER BY avg_value ASC
    LIMIT $limit
    """,

}
