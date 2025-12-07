import os
import json
from typing import Dict, Any
from huggingface_hub import InferenceClient


SYSTEM_PROMPT = """
You are a strict entity extractor for a hotel booking assistant.
You MUST output ONLY a valid JSON object following the schema below.
No text, no explanations, no backticks.

SCHEMA:
{
  "cities": ["City", ...],
  "countries": ["Country", ...],
  "hotels": ["Hotel Name", ...],
  "origin_country": ["Country", ...],
  "destination_country": ["Country", ...],
  "traveller_type": "solo|family|couple|business|group|null",
  "age_group": "18-24|25-34|35-44|45-54|55+|null",
  "gender": ["male","female"] or [],
  "rating": number or null,
  "confidence": {
      "cities": float,
      "countries": float,
      "hotels": float,
      "origin_country": float,
      "destination_country": float,
      "traveller_type": float,
      "age_group": float,
      "gender": float,
      "rating": float
  }
}

STRICT RULES:
1. DO NOT guess optional fields.
   - traveller_type, age_group, gender, rating MUST be null/empty unless explicitly stated.

2. ALWAYS infer countries from cities:
   Examples:
     - Cairo → Egypt
     - London → United Kingdom
     - Stuttgart → Germany

3. Infer origin vs destination logically:
   - Phrases like "from X", "I live in X", "coming from X" → origin_country = X.
   - Phrases like "in X", "to X", "going to X", "for X" → destination_country = X.

4. Ratings:
   - If user says "4 star" → convert to 8.0 (multiply by 2).
   - If user says "above 8", "at least 8", "8/10" → rating = 8.0.
   - Only extract rating if explicitly provided.

5. Hotels:
   - Only extract hotel names mentioned directly.
   - Do NOT invent or guess hotel names.

6. Output must ALWAYS be valid JSON matching the schema.
"""


def extract_with_llm(
    query: str,
    model: str = "deepseek-ai/DeepSeek-V3.2",
    max_tokens: int = 300
) -> Dict[str, Any]:

    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise RuntimeError("HF_API_KEY environment variable not set.")

    client = InferenceClient(model=model, token=api_key)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'User query: "{query}"'}
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
        content = response.choices[0].message["content"].strip()

    except Exception as e:
        # Hard failure → return safe empty structure
        return _empty_result()

    # Remove accidental code fences if any
    if content.startswith("```"):
        try:
            content = content.split("```")[1]
        except:
            return _empty_result()

    # Parse JSON safely
    try:
        parsed = json.loads(content)
        return _validate_and_fix(parsed)
    except Exception:
        return _empty_result()


# Helper Functions
def _empty_result() -> Dict[str, Any]:
    """Return a safe empty extraction result."""
    return {
        "cities": [],
        "countries": [],
        "hotels": [],
        "origin_country": [],
        "destination_country": [],
        "traveller_type": None,
        "age_group": None,
        "gender": [],
        "rating": None,
        "confidence": {
            "cities": 0.0,
            "countries": 0.0,
            "hotels": 0.0,
            "origin_country": 0.0,
            "destination_country": 0.0,
            "traveller_type": 0.0,
            "age_group": 0.0,
            "gender": 0.0,
            "rating": 0.0,
        }
    }


def _validate_and_fix(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensures missing fields won't break the pipeline.
    Also ensures all required keys exist.
    """
    template = _empty_result()
    for key in template:
        if key not in data:
            data[key] = template[key]
    return data
