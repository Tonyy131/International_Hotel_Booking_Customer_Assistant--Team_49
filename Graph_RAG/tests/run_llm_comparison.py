# experiments/run_llm_comparison.py
"""
This script:
- Runs multiple hotel-related queries
- Retrieves context via RetrievalPipeline
- Sends context + query into multiple LLMs
- Collects quantitative metrics:
    - latency
    - token usage
    - error rate
    - (optional) cost placeholder
- Saves:
    - results.json (raw results)
    - summary.csv (per-query results)
    - summary_metrics.csv (aggregated per-model metrics)
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
    "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek-ai/DeepSeek-V3.2",
    "openai/gpt-oss-20b"
]

JUDGE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# TEST QUERIES
TEST_QUERIES = [
    # "Find me hotels in Cairo above 8.",
    # "Recommend 3 family-friendly hotels in Berlin.",
    # "Which hotels in Tokyo have excellent cleanliness?",
    # "Is a visa needed to travel from Egypt to Germany?",
    # "Find boutique hotels in Paris.",
    "What are the best hotels for business travellers?",
]



# RESULTS DIRECTORY
RESULTS_DIR = Path("tests/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def judge_accuracy(
    question: str,
    context: str,
    answer: str
) -> int:
    """
    Uses an LLM judge to evaluate correctness and faithfulness.
    Returns 1 (correct) or 0 (incorrect).
    """

    if not answer or not context:
        return 0

    judge_prompt = f"""
You are an evaluation assistant.

You will be given:
1) A user question
2) Retrieved knowledge graph context
3) A model-generated answer

Evaluate the answer using ONLY the provided context.

Return ONLY one number:
- 1 if the answer is correct, relevant, and faithful to the context
- 0 otherwise

Do not explain your decision.
Do not use external knowledge.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""

    try:
        judge_out = answer_with_model(
            model_name=JUDGE_MODEL,
            user_query=judge_prompt,
            context_text="",      # Judge sees everything in the prompt
            max_new_tokens=5,
            temperature=0.0,      # 🔒 Deterministic
            top_p=1.0
        )

        judge_text = judge_out["generation"]["text"].strip()
        return 1 if judge_text.startswith("1") else 0

    except Exception:
        return 0



# MAIN EXPERIMENT FUNCTION
def run_experiment(models, queries, top_k=5):

    rp = RetrievalPipeline()
    results = []

    print("\n====================================")
    print("   Running Multi-Model Evaluation")
    print("====================================")

    for q_idx, query in enumerate(queries):
        print(f"\n=== Query {q_idx + 1}/{len(queries)}: {query}")

        retrieval = rp.safe_retrieve(
            query=query,
            limit=top_k,
            user_embeddings=True
        )

        context_text = retrieval.get("context_text", "")
        if not context_text:
            print("[WARNING] No context found. Using empty context.")

        for model in models:
            print(f"  → Running model: {model}")

            start = time.perf_counter()

            out = {}
            error = None

            try:
                out = answer_with_model(
                    model_name=model,
                    user_query=query,
                    context_text=context_text,
                    max_new_tokens=256,
                    temperature=0.2,
                    top_p=0.95
                )

                latency = out["generation"].get("latency_s")
                text = out["generation"].get("text", "")
                error = out["generation"].get("error")

            except Exception as e:
                latency = None
                text = ""
                error = str(e)

            accuracy_flag = judge_accuracy(
            question=query,
            context=context_text,
            answer=text
            )


            end = time.perf_counter()

            input_tokens = out.get("generation", {}).get("approx_input_tokens")
            output_tokens = out.get("generation", {}).get("approx_output_tokens")

            row = {
                "query_index": q_idx,
                "query": query,
                "model": model,

                # Performance
                "latency_s": latency,
                "end_to_end_latency_s": end - start,

                # Tokens
                "approx_input_tokens": input_tokens,
                "approx_output_tokens": output_tokens,
                "total_tokens": (
                    input_tokens + output_tokens
                    if input_tokens is not None and output_tokens is not None
                    else None
                ),

                # Cost (placeholder)
                "estimated_cost_usd": None,

                # Error tracking
                "error_flag": 1 if error else 0,
                "error": error,

                # Judge-based quantitative accuracy
                "accuracy_flag": accuracy_flag,

                # Output
                "response_text": text,
            }

            results.append(row)

    # Save raw results
    json_path = RESULTS_DIR / "results.json"
    csv_path = RESULTS_DIR / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)

    # ===== Aggregated Quantitative Metrics (REQUIRED) =====
    metrics_df = (
        df.groupby("model")
        .agg(
            avg_latency_s=("latency_s", "mean"),
            avg_end_to_end_latency_s=("end_to_end_latency_s", "mean"),
            avg_input_tokens=("approx_input_tokens", "mean"),
            avg_output_tokens=("approx_output_tokens", "mean"),
            avg_total_tokens=("total_tokens", "mean"),
            error_rate=("error_flag", "mean"),
            accuracy=("accuracy_flag", "mean"),
        )
        .reset_index()
    )

    metrics_path = RESULTS_DIR / "summary_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\nResults saved to:")
    print(" -", json_path.resolve())
    print(" -", csv_path.resolve())
    print(" -", metrics_path.resolve())


# ENTRY POINT
if __name__ == "__main__":
    print("Starting model comparison...\n")
    run_experiment(
        models=MODEL_CANDIDATES,
        queries=TEST_QUERIES,
        top_k=5
    )
