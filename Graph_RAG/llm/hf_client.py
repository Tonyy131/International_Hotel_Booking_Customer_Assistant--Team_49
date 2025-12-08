"""
HuggingFace Inference API client.

Provides:
  - HFClient(model_name)
  - client.generate(prompt, max_new_tokens=256, temperature=0.0)

Returns a dict:
  {
    "model": model_name,
    "text": "<generated text>",
    "latency_s": float,
    "approx_input_tokens": int,
    "approx_output_tokens": int,
    "raw_response": <original json response>
  }
"""
import os 
import time
from huggingface_hub import InferenceClient

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise RuntimeError("HF_API_KEY environment variable not set.")

def approx_token_count(text: str) -> int:
    """Approximate token count by splitting on whitespace."""
    return len(text.split())

class HFClient:
    """
    Wrapper around HuggingFace InferenceClient to standardize model generation calls.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = InferenceClient(model=model_name, token=HF_API_KEY)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.2, top_p: float = 0.95) -> dict:
        """        
        :param self: Description
        :param prompt: Description
        :type prompt: str
        :param max_new_tokens: Description
        :type max_new_tokens: int
        :param temperature: Description
        :type temperature: float
        :param top_p: Description
        :type top_p: float
        :return: Description
        :rtype: dict
        """

        start = time.perf_counter()
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=["===== END ====="],
            )
        except Exception as e:
            latency_s = time.perf_counter() - start
            return {
                "model": self.model_name,
                "text": "",
                "error": str(e),
                "latency_s": latency_s,
                "approx_input_tokens": approx_token_count(prompt),
                "approx_output_tokens": 0,
            }
        latency_s = time.perf_counter() - start

        return {
            "model": self.model_name,
            "text": response,
            "latency_s": latency_s,
            "approx_input_tokens": approx_token_count(prompt),
            "approx_output_tokens": approx_token_count(response),
            "raw_response": response
        }