"""
HotelMatcher: Responsible only for hotel name recognition.
- Exact substring match
- Fuzzy match (handles typos / partial names)
- Canonical storage of hotel names

This keeps hotel matching isolated and reusable by other components.
"""

from typing import List, Optional
import difflib


class HotelMatcher:
    def __init__(self, hotel_names: List[str]):
        """
        hotel_names: list of hotel names loaded from hotels.csv
        """
        self.hotels_original = hotel_names
        self.hotels_lower = [h.lower() for h in hotel_names]

    # EXACT SUBSTRING MATCH
    def match_exact(self, text: str) -> Optional[str]:
        """
        If the user text contains any full hotel name as a substring,
        return the longest (most specific) match.
        """
        text_low = text.lower()
        matches = []

        for original, lowered in zip(self.hotels_original, self.hotels_lower):
            if lowered in text_low:
                matches.append((len(lowered), original))

        if matches:
            # longer hotel names are more specific
            matches.sort(reverse=True)
            return matches[0][1]

        return None


    # FUZZY MATCHING
    def match_fuzzy(self, text: str, cutoff: float = 0.7) -> Optional[str]:
        """
        Fuzzy match based on single words extracted from text.
        Example: "niel plazz" -> "Nile Plaza"
        """
        tokens = [t.strip() for t in text.replace(",", " ").split() if len(t) >= 3]

        best_match = None
        best_ratio = 0

        for token in tokens:
            matches = difflib.get_close_matches(token.lower(), self.hotels_lower, n=1, cutoff=cutoff)
            if matches:
                idx = self.hotels_lower.index(matches[0])
                candidate = self.hotels_original[idx]
                ratio = difflib.SequenceMatcher(None, token.lower(), matches[0]).ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = candidate

        return best_match
