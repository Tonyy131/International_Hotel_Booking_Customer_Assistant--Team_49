from neo4j import GraphDatabase
import pandas as pd

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key] = value
    return config

def create_identifiers(driver):
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Traveller) REQUIRE t.user_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (h:Hotel) REQUIRE h.hotel_id IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE")

def load_travellers(driver):
    travellers = pd.read_csv('users.csv')
    with driver.session() as session:
        for _, row in travellers.iterrows():
            session.run("""
                MERGE (t: Traveller {user_id: $user_id})
                SET t.age = $age,
                t.type = $type,
                t.gender = $gender
                        
                MERGE (c: Country {name: $country_name})
                        
                MERGE (t)-[:FROM_COUNTRY]->(c)
            """, parameters={
                "user_id": row['user_id'],
                "age": row['age_group'],
                "type": row['traveller_type'],
                "gender": row['user_gender'],
                "country_name": row['country']
            })


def load_hotels(driver):
    hotels = pd.read_csv('hotels.csv')
    with driver.session() as session:
        for _, row in hotels.iterrows():
            session.run("""
                MERGE (h: Hotel {hotel_id: $hotel_id})
                SET h.name = $name,
                h.star_rating = $star_rating,
                h.cleanliness_base = $cleanliness_base,
                h.comfort_base = $comfort_base,
                h.facilities_base = $facilities_base
                                                
                MERGE (c: City {name: $city_name})
                MERGE (h)-[:LOCATED_IN]->(c)
                        
                MERGE (b: Country {name: $country_name})
                MERGE (c)-[:LOCATED_IN]->(b)
            """, parameters={
                "hotel_id": row['hotel_id'],
                "name": row['hotel_name'],
                "star_rating": row['star_rating'],
                "cleanliness_base": row['cleanliness_base'],
                "comfort_base": row['comfort_base'],
                "facilities_base": row['facilities_base'],
                "city_name": row['city'],
                "country_name": row['country']
            })

def load_reviews(driver):
    reviews = pd.read_csv('reviews.csv')
    with driver.session() as session:
        for _, row in reviews.iterrows():
            session.run("""
                MERGE (r: Review {review_id: $review_id})
                SET r.text = $text,
                r.date = $date,
                r.score_overall = $score_overall,
                r.score_cleanliness = $score_cleanliness,
                r.score_comfort = $score_comfort,
                r.score_facilities = $score_facilities,
                r.score_location = $score_location,
                r.score_staff = $score_staff,
                r.score_value_for_money = $score_value_for_money
                        
                WITH r

                MATCH (t: Traveller {user_id: $user_id})
                MATCH (h: Hotel {hotel_id: $hotel_id})
                        
                MERGE (t)-[:WROTE]->(r)
                MERGE (r)-[:REVIEWED]->(h)
                MERGE (t)-[:STAYED_AT]->(h)
            """, parameters={
                "review_id": row['review_id'],
                "text": row['review_text'],
                "date": row['review_date'],
                "score_overall": row['score_overall'],
                "score_cleanliness": row['score_cleanliness'],
                "score_comfort": row['score_comfort'],
                "score_facilities": row['score_facilities'],
                "score_location": row['score_location'],
                "score_staff": row['score_staff'],
                "score_value_for_money": row['score_value_for_money'],
                "user_id": row['user_id'],
                "hotel_id": row['hotel_id']
            })

def load_visa(driver):
    visas = pd.read_csv("visa.csv")

    with driver.session() as session:
        for _, row in visas.iterrows():

            session.run("""
                MERGE (fromC:Country {name: $from})
                MERGE (toC:Country {name: $to})
            """, parameters={
                "from": row["from"],
                "to": row["to"]
            })

            if row["requires_visa"] == "Yes":
                session.run("""
                    MATCH (fromC:Country {name: $from})
                    MATCH (toC:Country {name: $to})
                    MERGE (fromC)-[v:NEEDS_VISA]->(toC)
                    SET v.visa_type = $visa_type
                """, parameters={
                    "from": row["from"],
                    "to": row["to"],
                    "visa_type": row["visa_type"]
                })


def main():
    config = read_config("config.txt")
    driver = GraphDatabase.driver(
        config["URI"],
        auth=(config["USERNAME"], config["PASSWORD"])
    )

    create_identifiers(driver)

    load_travellers(driver)
    load_hotels(driver)
    load_reviews(driver)
    load_visa(driver)

    driver.close()
    print("Knowledge Graph creation complete!")

if __name__ == "__main__":
    main()