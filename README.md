# International Hotel Booking Customer Assistant (Team 49)

A research / demo project that builds a retrieval-augmented travel assistant for international hotel booking. The system combines a Neo4j knowledge graph (KG) of hotels, reviews, travellers, and visa rules with embedding-based retrieval and LLM answer generation (via HuggingFace Inference API). The project includes a Streamlit demo, KG ingestion scripts, embedding indexing, entity extraction, and an orchestrated retrieval pipeline suitable for RAG prompts.

Contents
- Features
- Architecture overview
- Quickstart (install, configure, run)
- Build the Knowledge Graph (KG)
- Index hotel embeddings (vector index)
- Run the Streamlit assistant
- Programmatic usage examples
- Tests & evaluation
- Troubleshooting & tips
- Key files & where to look

Features
- Neo4j-based knowledge graph for hotels, travellers, reviews, visa info.
- Entity extraction (spaCy + rule-based + optional LLM fallback).
- Intent classification to route queries.
- Embedding generation with sentence-transformers (MiniLM) and optional BGE model.
- Vector indexing and semantic search stored in Neo4j.
- LLM answer generation using HuggingFace Inference API with structured RAG prompt.
- Streamlit UI for interactive demos.

Tech stack
- Python
- Neo4j (graph database)
- sentence-transformers (all-MiniLM-L6-v2) and optional BGE
- spaCy (en_core_web_md)
- huggingface_hub (InferenceClient)
- Streamlit for UI
- NetworkX + Plotly for visualization (used by app)

Prerequisites
- Python 3.8+
- Neo4j 5.x (or compatible with VECTOR index support)
- A HuggingFace Inference API key (if you want to call an HF model)
- Optional: GPU for faster embedding/model inference (sentence-transformers/BGE)

1) Clone the repo
Run from a shell:
```
git clone https://github.com/Tonyy131/International_Hotel_Booking_Customer_Assistant--Team_49.git
cd International_Hotel_Booking_Customer_Assistant--Team_49
```

2) Create a Python virtual environment and install dependencies
```
python -m venv .venv
# Activate the venv (examples)
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r Graph_RAG/requirements.txt
```

Notes on spaCy model install
- The repository pins the en_core_web_md wheel in requirements. If the model is not installed after the above, run:
```
python -m spacy download en_core_web_md
```
(or follow the wheel link in Graph_RAG/requirements.txt).

Configuration (Neo4j credentials and HF API)
You can either create a config file used by the KG script or set environment variables.

Option A — config file (recommended for local KG creation)
Create file `Knowledge_Graph_DB/config.txt` with:
```
URI=neo4j://127.0.0.1:7687
USERNAME=neo4j
PASSWORD=your_neo4j_password
```
(There is a template at Knowledge_Graph_DB/config_file_template.txt.)

Option B — Environment variables (recommended for running code)
Set these environment vars in your environment or in a .env file (if you use something to load envs):
- NEO4J_URI (e.g. neo4j://127.0.0.1:7687)
- NEO4J_USER
- NEO4J_PASSWORD
- HF_API_KEY (your HuggingFace Inference API key) — required to instantiate HFClient in Graph_RAG/llm/hf_client.py

Examples (Unix/macOS):
```
export NEO4J_URI=neo4j://127.0.0.1:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export HF_API_KEY=hf_xxx....
```

3) Build the Knowledge Graph (KG)
The repo includes CSV files in Knowledge_Graph_DB/ and a script to create nodes/relations.

- Ensure the config file exists at Knowledge_Graph_DB/config.txt (see template) OR ensure NEO4J_* env vars are set.
- Run:
```
python Knowledge_Graph_DB/create_kg.py
```
What it does:
- Creates constraints for unique identifiers.
- Loads travellers/users, hotels, reviews and visa relations from CSV files under Knowledge_Graph_DB/.
- Computes aggregated scores and per-traveller-type averages.

Notes:
- create_kg.py expects a Neo4j connection reachable from where you run it.
- If you prefer to run Cypher manually, example queries are in Knowledge_Graph_DB/queries.txt.

4) Index hotel embeddings in Neo4j (vector index)
The embedding indexer computes and stores vector properties on Hotel nodes. There are two model options: "minilm" (default, 384d) and "bge" (768d).

Run from repo root with package mode:
```
# default Minilm index + embedding storage
python -m Graph_RAG.retrieval.embedding_indexer

# or create and index using BGE (if you want)
python -c "from Graph_RAG.retrieval.embedding_indexer import EmbeddingIndexer; EmbeddingIndexer(model_name='bge').ensure_vector_index(); EmbeddingIndexer(model_name='bge').index_all_hotels()"
```

Important:
- The indexer uses Graph_RAG/preprocessing/embedding_encoder.py which uses sentence-transformers for embeddings by default (all-MiniLM-L6-v2). The first run will download models and may take time.
- Ensure Neo4j version supports VECTOR indexes (Neo4j 5.x). The index creation DDL is in the indexer.

5) Run the Streamlit demo app
From the repository root:
```
streamlit run Graph_RAG/app.py
```
- The Streamlit UI provides an interactive assistant powered by the retrieval pipeline + LLM answerer.
- Ensure Neo4j is running and vectors are available for semantic search. If HF_API_KEY is not set, some LLM features will be disabled or instantiation will fail.

Programmatic usage examples
Below are lightweight examples to integrate the core components in your own code.

A) Retrieval-only usage (retrieve context text)
```
from Graph_RAG.retrieval.retrieval_pipeline import RetrievalPipeline
from Graph_RAG.neo4j_connector import Neo4jConnector

# Optionally supply connector (uses env vars if not)
connector = Neo4jConnector()
pipeline = RetrievalPipeline(neo4j_connector=connector, model_name="minilm")

query = "I'm traveling from Spain to Germany next month and want hotels in Berlin for a family of four; any good options?"
results = pipeline.safe_retrieve(query, limit=5, user_embeddings=True, use_llm=False)

print("Intent:", results["intent"])
print("Entities:", results["entities"])
print("Context text:
", results["context_text"])
```

B) Full RAG answer generation with HF model
```
from Graph_RAG.retrieval.retrieval_pipeline import RetrievalPipeline
from Graph_RAG.llm.llm_answerer import answer_with_model
from Graph_RAG.neo4j_connector import Neo4jConnector

# 1) Retrieve
connector = Neo4jConnector()
pipeline = RetrievalPipeline(neo4j_connector=connector)
query = "Recommend family-friendly hotels in Berlin near the city center with good cleanliness scores."

retrieval = pipeline.safe_retrieve(query, limit=5, user_embeddings=True, use_llm=False)
context = retrieval["context_text"]

# 2) Ask the LLM — model_name must be a valid HuggingFace Inference model identifier
response = answer_with_model(
    model_name="gpt2",  # replace with a supported chat/inference model you have access to
    user_query=query,
    context_text=context,
    max_new_tokens=256,
    temperature=0.2
)

print("Prompt sent to the model:
", response["prompt"][:1000])
print("Generated text:
", response["generation"].get("text"))
print("Latencies:", response["generation"].get("latency_s"), response.get("end_to_end_latency_s"))
```
Notes:
- Replace the model_name above with an appropriate HF model that supports the chat/completions API used by HFClient.
- HFClient requires HF_API_KEY set in env.

Running unit tests
- The repo contains tests under Graph_RAG/tests/. To run:
```
pip install pytest
pytest -q
```
Some tests may require Neo4j or external models; you can run selected unit tests or mock dependencies as needed.

Troubleshooting
- HF_API_KEY missing: HFClient raises a RuntimeError when created without HF_API_KEY. Set the environment variable before running code that uses HFClient.
- Neo4j connection errors: Verify NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD or the config file. Ensure the Neo4j server is reachable and that Bolt/Neo4j URI is correct.
- spaCy model errors: If spaCy cannot load en_core_web_md, run `python -m spacy download en_core_web_md` or reinstall the wheel from the requirements.
- Embedding indexing slow or out-of-memory: embedding model downloads and encoding can be memory / CPU intensive. Consider batch sizes or running on a machine with more RAM/GPU.
- Vector index errors: Neo4j must support vector indexes (Neo4j 5+). If creating the index fails, confirm Neo4j version and the DDL syntax in Graph_RAG/retrieval/embedding_indexer.py.

Security & privacy
- The app may send prompts and context to an external LLM provider (HuggingFace Inference). Avoid sending personally identifiable information (PII) in prompts in production unless consented.
- Store HF API keys and DB credentials securely (do not commit them to the repository).

Files of interest (quick pointers)
- Graph_RAG/app.py — Streamlit application / UI
- Graph_RAG/retrieval/retrieval_pipeline.py — Orchestration of baseline + embedding retrieval and context building
- Graph_RAG/llm/hf_client.py — HuggingFace Inference API wrapper (HFClient)
- Graph_RAG/llm/llm_answerer.py — Final prompt builder + answer_with_model()
- Graph_RAG/neo4j_connector.py — Small Neo4j wrapper for queries
- Graph_RAG/retrieval/embedding_indexer.py — Compute & store node embeddings and create vector indexes
- Graph_RAG/preprocessing/embedding_encoder.py — Embedding model wrapper (MiniLM / BGE)
- Graph_RAG/preprocessing/entity_extractor.py — Entity extraction orchestration (spaCy, hotel matching, etc.)
- Knowledge_Graph_DB/create_kg.py — Build the KG from CSVs
- Knowledge_Graph_DB/queries.txt — Example Cypher queries
- Graph_RAG/requirements.txt — Python dependencies

Research / evaluation artifacts
- The repository contains evaluation notebooks and CSVs (e.g., Team_49_Hotel_Project.ipynb, tests/*, and various PDFs) used by the team for quantitative and qualitative analysis.

Contributing
- This repository contains research code and demos. For contributions:
  - Open an issue describing the feature or bug.
  - Create a branch, run tests locally, follow repository conventions.
  - Consider adding tests for new behavior and updating the README if you add new scripts.

License
- No license file is included in the repository. If you plan to reuse or publish, add an appropriate LICENSE file (e.g., MIT, Apache-2.0).

Acknowledgements
- The project uses open-source components: Neo4j, HuggingFace, SentenceTransformers, spaCy, Streamlit, and others. See Graph_RAG/requirements.txt for details.
