"""
LLM Answer Generator using HuggingFace InferenceClient.

This module:
- Builds the final RAG prompt using context_text and user_query.
- Calls the HFLLMClient for any model name.
- Returns a structured dictionary with model output, latency, etc.
"""

from typing import Optional, Dict, Any
from llm.hf_client import HFClient


DEFAULT_PERSONA = (
    "You are a helpful, precise hotel recommendation assistant. "
    "You always follow the provided CONTEXT strictly. "
    "You never hallucinate facts not found in the context. "
    "If the context does not contain the answer, say so politely."
)

DEFAULT_TASK = (
    "Your task is to answer the user's hotel-related query using ONLY the given CONTEXT. \n"
    "Summaries should be short, factual, and helpful. \n"
    "If multiple hotels are relevant, list them in a bullet list with a short justification.\n"
    "You should NEVER user external resources.\n"
)

def build_prompt(
    context_text: str,
    user_query: str,
    persona: Optional[str] = None,
    task: Optional[str] = None
) -> str:
    persona = persona or DEFAULT_PERSONA
    task = task or DEFAULT_TASK

    prompt = (
        "===== CONTEXT =====\n"
        f"{context_text}\n\n"
        "===== INSTRUCTIONS =====\n"
        f"{persona}\n"
        f"{task}\n\n"
        "===== USER QUERY =====\n"
        f"{user_query}\n\n"
        "===== ANSWER (use only the CONTEXT above) =====\n"
    )
    return prompt


# MAIN LLM ANSWER FUNCTION
def answer_with_model(
    model_name: str,
    user_query: str,
    context_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    **generation_kwargs
) -> Dict[str, Any]:
    """
    Builds prompt, sends it to HFLLMClient(model_name), and returns structured info.

    Return format:
    {
        "model": model_name,
        "prompt": "<full prompt>",
        "generation": {
            "text": "...",
            "latency_s": float,
            "approx_input_tokens": int,
            "approx_output_tokens": int
        },
        "end_to_end_latency_s": float
    }
    """

    # Build the prompt
    prompt = build_prompt(
        context_text=context_text,
        user_query=user_query
    )

    # Initialize the HF client
    client = HFClient(model_name)

    # Generate response
    import time
    start = time.perf_counter()
    gen = client.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        **generation_kwargs
    )
    end = time.perf_counter()

    return {
        "model": model_name,
        "prompt": prompt,
        "generation": gen,
        "end_to_end_latency_s": end - start
    }