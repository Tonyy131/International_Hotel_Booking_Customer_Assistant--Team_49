# simple_qwen_test.py
import time
from retrieval.retrieval_pipeline import RetrievalPipeline
from llm.llm_answerer import answer_with_model

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TEST_QUERIES = [
    "Find me hotels with best cleanliness ratings",
]

def run_simple_qwen_test():
    rp = RetrievalPipeline()

    print("\n====================================")
    print("   Running Qwen Evaluation")
    print("====================================\n")

    for i, query in enumerate(TEST_QUERIES):
        print(f"\n=== Query {i+1}/{len(TEST_QUERIES)} ===")
        print("User:", query)

        # Retrieve context
        retrieval = rp.safe_retrieve(query=query, limit=5, user_embeddings=True)
        context_text = retrieval.get("context_text", "")

        if not context_text:
            print("[WARNING] No context found.")

        # Run Qwen
        print("\n→ Running Qwen...")
        start = time.perf_counter()

        try:
            out = answer_with_model(
                model_name=MODEL,
                user_query=query,
                context_text=context_text,
                max_new_tokens=256,
                temperature=0.2,
                top_p=0.95
            )

            latency = out["generation"].get("latency_s")
            response = out["generation"].get("text", "")
            error = out["generation"].get("error")

        except Exception as e:
            response = ""
            error = str(e)
            latency = None

        end = time.perf_counter()

        # Print results
        print("\n--- RESULT ---")
        if error:
            print("Error:", error)
        else:
            print(f"Model: {MODEL}")
            print(f"Latency: {latency:.4f} sec (model) | {end-start:.4f} sec (total)")
            print("\nResponse:\n", response)

        print("\n---------------------------")

if __name__ == "__main__":
    run_simple_qwen_test()
