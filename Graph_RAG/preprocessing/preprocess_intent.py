from typing import Dict, Any
from preprocessing.intent_classifier import classify_intent_rule_with_confidence
from preprocessing.llm_intent_classifier import classify_intent_llm_hf

def classify_user_intent(text: str,
                         use_llm_fallback: bool = True,
                         min_score: float = 0.1,
                         margin_ratio: float = 100,
                         llm_timeout: float = 1.5) -> Dict[str, Any]:
    """
    Orchestrates rule-based classification and optional LLM fallback.
    Returns an audit-friendly dict:
      {
        query, intent, intent_source, intent_scores, top_score, fallback_used
      }
    """
    label, scores, top_score, fallback_needed = classify_intent_rule_with_confidence(
        text,
        min_score_to_accept=min_score,
        margin_ratio=margin_ratio
    )
    intent_source = "rule-based"
    fallback_used = False
    final_intent = label

    if label == "unknown" and fallback_needed and use_llm_fallback:
        return {
            "query": text,
            "intent": classify_intent_llm_hf(text),
            "intent_source": "llm-fallback",
            "intent_scores": scores,
            "top_score": top_score,
            "fallback_used": True
        }

    return {
        "query": text,
        "intent": final_intent,
        "intent_source": intent_source,
        "intent_scores": scores,
        "top_score": top_score,
        "fallback_used": fallback_used
    }