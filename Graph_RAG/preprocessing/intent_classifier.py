from typing import Tuple, Dict

# ========== CONFIG ==========
INTENT_KEYWORDS = {
    "recommendation": ["recommend", "suggest", "best", "top", "suggestion", "any suggestions", "could you suggest"],
    "booking": ["book", "booking", "reserve", "reservation", "help me book", "help me in booking", "can you book", "i want to book"],
    "visa_query": ["visa", "visa requirements", "visa info", "passport", "entry", "immigration", "do i need a visa"],
    "review_query": ["review", "reviews", "rating", "ratings", "score", "scores", "feedback"],
    "hotel_search": ["hotel", "hotels", "stay", "staying", "accommodation", "find hotels", "find a hotel"],
    "generic_qa": ["what", "how", "when", "where", "who", "why"]
}

INTENT_WEIGHTS = {
    "recommendation": 2.0,
    "booking": 2.0,
    "visa_query": 1.5,
    "review_query": 1.2,
    "hotel_search": 1.0,
    "generic_qa": 0.5
}

# Only used if a true tie remains after confidence checks:
INTENT_PRIORITY = ["booking", "recommendation", "visa_query", "review_query", "hotel_search", "generic_qa"]

# High-confidence multi-token phrases (dominance overrides)
DOMINANCE_PHRASES = {
    "booking": ["help me book", "help me in booking", "can you book", "i want to book", "i'd like to book", "i want to reserve"],
    "visa_query": ["do i need a visa", "visa required", "need a visa", "visa information"],
    "recommendation": ["recommend", "suggest", "any suggestions", "could you suggest"]
}

# ========== HELPERS ==========
def _normalize(text: str) -> str:
    return text.lower().strip()

def _compute_weighted_scores(text_norm: str) -> Dict[str, float]:
    scores: Dict[str, float] = {intent: 0.0 for intent in INTENT_KEYWORDS}
    # First, account for dominance multi-token phrases by giving extra score if present
    for intent, phrases in DOMINANCE_PHRASES.items():
        for p in phrases:
            if p in text_norm:
                # Add a large boost so dominance phrases are clearly visible in scores
                scores[intent] += 5.0
    # Then keyword counts * weights
    for intent, kws in INTENT_KEYWORDS.items():
        count = 0
        for kw in kws:
            if kw in text_norm:
                count += 1
        scores[intent] += count * INTENT_WEIGHTS.get(intent, 1.0)
    return scores

# ========== MAIN INTERFACE ==========
def classify_intent_rule_with_confidence(
    text: str,
    min_score_to_accept: float = 1.0,
    margin_ratio: float = 1.25
) -> Tuple[str, Dict[str, float], float, bool]:
    """
    Returns:
      - intent_label (or "unknown" if uncertain)
      - weighted_scores dict
      - top_score (float)
      - fallback_needed (bool): True if caller should use LLM fallback
    Logic:
      1) compute weighted scores (dominance phrases add strong boost)
      2) if top_score < min_score_to_accept -> unknown
      3) if top/second < margin_ratio -> unknown
      4) otherwise return chosen intent (tie-break by INTENT_PRIORITY)
    """
    t = _normalize(text)
    scores = _compute_weighted_scores(t)
    vals = sorted(scores.values(), reverse=True)
    top = vals[0] if vals else 0.0
    second = vals[1] if len(vals) > 1 else 0.0

    # Absolute threshold check
    if top < min_score_to_accept:
        return "unknown", scores, top, True

    # Margin ratio check (avoid ambiguous close scores)
    eps = 1e-9 # to avoid div by zero
    ratio = top / (second + eps)
    if ratio < margin_ratio:
        return "unknown", scores, top, True

    # Choose winner(s)
    winners = [k for k, v in scores.items() if v == top]
    if len(winners) == 1:
        return winners[0], scores, top, False

    # Tie-break by priority
    for p in INTENT_PRIORITY:
        if p in winners:
            return p, scores, top, False

    return winners[0], scores, top, False
