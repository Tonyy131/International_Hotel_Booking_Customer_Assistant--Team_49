import csv
import os
from typing import List

# Get absolute path relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
HOTEL_CSV_PATH = os.path.join(CURRENT_DIR, "..", "..", "Knowledge_Graph_DB", "hotels.csv")
HOTEL_CSV_PATH = os.path.abspath(HOTEL_CSV_PATH)

def load_hotels() -> List[str]:
    hotels = []
    if not os.path.exists(HOTEL_CSV_PATH):
        print(f"Warning: Hotel CSV not found at {HOTEL_CSV_PATH}")
        return []
    
    try:
        with open(HOTEL_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hotels.append(row["hotel_name"].strip())
    except Exception as e:
        print(f"Error loading hotels: {e}")
        return []
    
    return hotels