import os
from huggingface_hub import InferenceClient

INTENT_LABELS = [
    "recommendation",
    "booking",
    "visa_query",
    "review_query",
    "hotel_search",
    "hotel_visa"
]

SYSTEM_PROMPT = f"""
You are an intent classification model.

Your task:
Given a user query, classify it into EXACTLY ONE of these intent labels:
{", ".join(INTENT_LABELS)}

### 1. INTENT HIERARCHY (Read this first):
- **CRITICAL RULE**: If the query mentions "visa", "passport", or "entry requirements" AND "hotels" (or accommodation), the intent is ALWAYS **'hotel_visa'**. This overrides 'recommendation' and 'hotel_search'.
- **CRITICAL RULE**: The label **'recommendation'** is RESERVED ONLY for specific **Traveller Types** (families, couples, business). If the user just says "recommend me a hotel in Cairo", that is **'hotel_search'**, NOT 'recommendation'.

### 2. DEFINITIONS:
- **hotel_visa**: User wants to find hotels based on visa requirements (e.g., "where can I go without a visa?", "hotels in visa-free countries") OR checks visa rules for a specific stay.
- **recommendation**: User asks for suggestions specific to a **TRAVELLER TYPE** (e.g., "best hotels for families", "romantic hotels for couples", "business hotels").
- **visa_query**: User asks PURELY about visa rules for a country (e.g., "Do I need a visa for France?"), with no mention of hotels.
- **hotel_search**: User wants to find/locate hotels based on location, price, amenities, or general "best" queries (e.g., "Find me a hotel", "Best hotels in London").
- **booking**: User wants to reserve or book.
- **review_query**: User asks for reviews or ratings.

### 3. EXAMPLES (Use these to guide your decision):
User: "Recommend a hotel for **families** in London"
Intent: recommendation

User: "Recommend a hotel in London"
Intent: hotel_search  <-- (No traveller type)

User: "Recommend the best hotels to go to **without needing a visa**"
Intent: hotel_visa    <-- (Visa keyword overrides 'recommend')

User: "Best hotels for **Egyptians** without a visa"
Intent: hotel_visa

User: "Do I need a visa for Paris?"
Intent: visa_query

User: "I want a hotel with a pool"
Intent: hotel_search

### RESPONSE FORMAT:
Respond with ONLY the label name. No punctuation.
"""


def classify_intent_llm_hf(text: str) -> str:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("HF_API_KEY is missing")
        return "hotel_search"

    client = InferenceClient(
        model="deepseek-ai/DeepSeek-V3.2", # or "meta-llama/Llama-3.1-8B-Instruct" if available
        token=api_key,
    )

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]

        response = client.chat_completion(
            messages=messages,
            max_tokens=10,
            temperature=0.0 # Keep this 0.0 for deterministic classification
        )

        label_raw = response.choices[0].message["content"].strip()

        # Clean up any accidental punctuation
        label_clean = label_raw.replace(".", "").replace('"', "").replace("'", "").lower()

        if label_clean in INTENT_LABELS:
            return label_clean

        # Fallback soft match
        for label in INTENT_LABELS:
            if label in label_clean:
                return label

        return "hotel_search"

    except Exception as e:
        print("Intent Classification Error:", e)
        return "hotel_search"