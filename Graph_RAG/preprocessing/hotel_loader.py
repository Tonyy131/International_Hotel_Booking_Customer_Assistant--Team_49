import csv
from typing import List

HOTEL_CSV_PATH = "../Knowledge_Graph_DB/hotels.csv"

def load_hotels() -> List[str]:
    hotels = []
    with open(HOTEL_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hotels.append(row["hotel_name"].strip())
    return hotels