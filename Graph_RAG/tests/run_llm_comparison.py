# experiments/run_llm_comparison.py
"""
This script:
- Runs multiple hotel-related queries
- Retrieves context via RetrievalPipeline
- Sends context + query into 4 LLMs (Llama3, Qwen2.5, Mistral7B)
- Collects:
    - output text
    - latency
    - token estimates
    - errors (if any)
- Saves:
    - experiments/results/results.json
    - experiments/results/summary.csv
"""

import json
import time
from pathlib import Path
import pandas as pd

from retrieval.retrieval_pipeline import RetrievalPipeline
from llm.llm_answerer import answer_with_model



# MODEL LIST
MODEL_CANDIDATES = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "deepseek-ai/DeepSeek-R1"
]



# TEST QUERIES
TEST_QUERIES = [
    "Find me hotels in Cairo above 8.",
    "Recommend 3 family-friendly hotels in Berlin.",
    "Which hotels in Tokyo have excellent cleanliness?",
    "Is a visa needed to travel from Egypt to Germany?",
    "Find boutique hotels in Paris.",
    "What are the best hotels in Istanbul for business travellers?",
]



# RESULTS DIRECTORY
RESULTS_DIR = Path("tests/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)




# MAIN EXPERIMENT FUNCTION
def run_experiment(models, queries, top_k=5):

    rp = RetrievalPipeline()
    results = []

    print("\n====================================")
    print("   Running Multi-Model Evaluation")
    print("====================================")

    for q_idx, query in enumerate(queries):
        print(f"\n=== Query {q_idx+1}/{len(queries)}: {query}")

        
        retrieval = rp.safe_retrieve(
            query=query,
            limit=top_k,
            user_embeddings=True
        )

        # Get context text
        context_text = retrieval.get("context_text", "")
        if not context_text:
            print("[WARNING] No context found. Using empty context.")

        
        # Query each model
        for model in models:
            print(f"  → Running model: {model}")

            start = time.perf_counter()

            try:
                out = answer_with_model(
                    model_name=model,
                    user_query=query,
                    context_text=context_text,
                    max_new_tokens=256,
                    temperature=0.2,
                    top_p=0.95
                )

                latency = out["generation"].get("latency_s", None)
                text = out["generation"].get("text", "")
                error = out["generation"].get("error", None)

            except Exception as e:
                latency = None
                text = ""
                error = str(e)

            end = time.perf_counter()

            # Store row
            row = {
                "query_index": q_idx,
                "query": query,
                "model": model,
                "latency_s": latency,
                "end_to_end_latency_s": end - start,
                "response_text": text,
                "error": error,
                "approx_input_tokens": out["generation"].get("approx_input_tokens", None) if not error else None,
                "approx_output_tokens": out["generation"].get("approx_output_tokens", None) if not error else None,
            }

            results.append(row)


    
    # Save results
    json_path = RESULTS_DIR / "results.json"
    csv_path = RESULTS_DIR / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    print(f"\nResults saved to:")
    print(" -", json_path.resolve())
    print(" -", csv_path.resolve())




# ENTRY POINT
if __name__ == "__main__":
    print("Starting model comparison...\n")
    run_experiment(
        models=MODEL_CANDIDATES,
        queries=TEST_QUERIES,
        top_k=5
    )
