import os
from huggingface_hub import InferenceClient

INTENT_LABELS = [
    "recommendation",
    "booking",
    "visa_query",
    "review_query",
    "hotel_search",
    "generic_qa"
]

SYSTEM_PROMPT = f"""
You are an intent classification model.

Your task:
Given a user query, classify it into EXACTLY ONE of these intent labels:
{", ".join(INTENT_LABELS)}

INTENT DEFINITIONS:
- recommendation: User asks for suggestions, best options, or recommendations.
- booking: User wants to book, reserve, or arrange a hotel stay.
- visa_query: User asks about traveling to another country, requirements before travel, visa rules, travel documents, passport validity, entry restrictions.
- review_query: User asks for reviews, ratings, feedback, or opinions about a hotel.
- hotel_search: User wants to find hotels, locate places to stay, or search for accommodation options.
- generic_qa: Any general question not related to the above categories.

REQUIREMENTS:
- Respond with ONLY the label.
- No explanation.
- No additional text.
- No punctuation.
- No quotes.
- No sentences.
"""


def classify_intent_llm_hf(text: str) -> str:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        print("HF_API_KEY is missing")
        return "generic_qa"

    client = InferenceClient(
        model="deepseek-ai/DeepSeek-V3.2",
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
            temperature=0.0
        )

        label_raw = response.choices[0].message["content"].strip()

        if label_raw in INTENT_LABELS:
            return label_raw

        # soft match - in case model returns: "The intent is: booking"
        low = label_raw.lower()
        for label in INTENT_LABELS:
            if label in low:
                return label

        return "generic_qa"

    except Exception as e:
        print("DeepSeek HF Error:", e)
        return "generic_qa"
