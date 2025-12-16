import time
from typing import List, Dict
from retrieval.retrieval_pipeline import RetrievalPipeline
from rich.console import Console
from rich.table import Table

console = Console()

def extract_hotel_names(result: Dict) -> List[str]:
    """Helper to extract clean hotel names from results."""
    names = []
    hotels = result.get("combined", {}).get("hotels", [])
    for item in hotels:
        # Handle nested vs flat structure
        h_node = item.get("h", {}) if isinstance(item.get("h"), dict) else item
        name = h_node.get("name") or h_node.get("hotel_name") or "Unknown"
        names.append(name)
    return names

def run_experiment(query: str, limit: int = 5):
    console.rule(f"[bold blue]Running Experiment: '{query}'")

    # --- 1. Run MiniLM (384d) ---
    console.print("\n[bold yellow]1. Loading & Querying MiniLM...[/]")
    try:
        pipeline_mini = RetrievalPipeline(model_name="minilm")
        
        start_time = time.perf_counter()
        # We disable baseline to isolate the embedding model's performance
        res_mini = pipeline_mini.safe_retrieve(query, limit=limit, user_embeddings=True, user_baseline=False)
        duration_mini = time.perf_counter() - start_time
        
        mini_hotels = extract_hotel_names(res_mini)
        console.print(f"   ✅ Done in {duration_mini:.4f}s | Found {len(mini_hotels)} hotels")
    except Exception as e:
        console.print(f"   ❌ Failed: {e}")
        return

    # --- 2. Run BGE (768d) ---
    console.print("\n[bold cyan]2. Loading & Querying BGE...[/]")
    try:
        pipeline_bge = RetrievalPipeline(model_name="bge")
        
        start_time = time.perf_counter()
        res_bge = pipeline_bge.safe_retrieve(query, limit=limit, user_embeddings=True, user_baseline=False)
        duration_bge = time.perf_counter() - start_time
        
        bge_hotels = extract_hotel_names(res_bge)
        console.print(f"   ✅ Done in {duration_bge:.4f}s | Found {len(bge_hotels)} hotels")
    except Exception as e:
        console.print(f"   ❌ Failed: {e}")
        return

    # --- 3. Analysis & Comparison ---
    console.print("\n[bold green]--- Comparative Results ---[/]")

    # Create Comparison Table
    table = Table(title=f"Top {limit} Results for '{query}'")
    table.add_column("Rank", justify="center", style="dim")
    table.add_column("MiniLM Results", style="yellow")
    table.add_column("BGE Results", style="cyan")

    # Pad lists to ensure they are the same length for the table
    max_len = max(len(mini_hotels), len(bge_hotels))
    for i in range(max_len):
        h_mini = mini_hotels[i] if i < len(mini_hotels) else "-"
        h_bge = bge_hotels[i] if i < len(bge_hotels) else "-"
        
        # Highlight overlap
        if h_mini == h_bge and h_mini != "-":
            h_mini = f"[bold green]{h_mini} (Match)[/]"
            h_bge = f"[bold green]{h_bge} (Match)[/]"
        
        table.add_row(str(i+1), h_mini, h_bge)

    console.print(table)

    # Intersection Analysis
    set_mini = set(mini_hotels)
    set_bge = set(bge_hotels)
    overlap = set_mini.intersection(set_bge)
    unique_mini = set_mini - set_bge
    unique_bge = set_bge - set_mini

    console.print(f"\n[bold]Overlap ({len(overlap)}):[/] {', '.join(overlap) if overlap else 'None'}")
    console.print(f"[dim]Unique to MiniLM:[/dim] {', '.join(unique_mini)}")
    console.print(f"[dim]Unique to BGE:[/dim]    {', '.join(unique_bge)}")
    
    # Latency Comparison
    diff = duration_bge - duration_mini
    faster = "MiniLM" if diff > 0 else "BGE"
    console.print(f"\n⚡ [bold]{faster}[/] was faster by [bold]{abs(diff):.4f}s[/]")

if __name__ == "__main__":
    TEST_QUERY = "Hotels with high average rating score, high value for money score, in Egypt"
    
    run_experiment(TEST_QUERY, limit=5)