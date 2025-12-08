"""
LLM Answer Generator using HuggingFace InferenceClient.

This module:
- Builds the final RAG prompt using context_text and user_query.
- Calls the HFLLMClient for any model name.
- Returns a structured dictionary with model output, latency, etc.
"""