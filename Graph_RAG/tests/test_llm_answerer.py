from llm.llm_answerer import answer_with_model

def test_llm():
    context = (
        "Hotel X: score 8.5, located in Cairo.\n"
        "Hotel Y: score 9.0, located in Cairo."
    )
    query = "Find me hotels in Cairo above 8."

    out = answer_with_model(
        "Qwen/Qwen2.5-1.5B-Instruct",
        user_query=query,
        context_text=context
    )

    print("\n=== MODEL OUTPUT ===")
    if "error" in out["generation"]:
        print("ERROR:", out["generation"]["error"])
    else:
        print(out["generation"]["text"])

    print("\n=== LATENCY ===")
    print(out["generation"].get("latency_s"))

if __name__ == "__main__":
    test_llm()
