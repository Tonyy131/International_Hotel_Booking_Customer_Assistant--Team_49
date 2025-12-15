import streamlit as st
import time
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List

# --- Backend Imports ---
try:
    from retrieval.retrieval_pipeline import RetrievalPipeline
    from llm.llm_answerer import answer_with_model
except ImportError:
    st.error("Backend modules not found. Please ensure your project structure is correct.")

# --- Page Config ---
st.set_page_config(
    page_title="Graph-RAG Travel Assistant",
    page_icon="🌍",
    layout="wide"
)

# --- Sidebar Configuration (Moved up for Initialization) ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Requirement: Compare at least 3 models
    model_name = st.selectbox(
        "Select LLM Model",
        options=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.2",
            "deepseek-ai/DeepSeek-V3.2",
            "openai/gpt-oss-20b"
        ],
        index=0
    )

    # NEW: Embedding Model Selection
    # Maps user-friendly names to internal keys expected by RetrievalPipeline
    embedding_option = st.radio(
        "Embedding Model",
        options=["MiniLM (ll-MiniLM-L6-v2)", "BGE (bge-small-en-v1.5)"],
        index=0
    )
    
    # Map selection to backend keys
    if "MiniLM" in embedding_option:
        selected_embedding_model = "minilm"
    else:
        selected_embedding_model = "bge"

    # Requirement: Retrieval Method Selection
    retrieval_method = st.radio(
        "Retrieval Strategy",
        options=["Hybrid (Baseline + Embeddings)", "Baseline Only", "Embeddings Only"],
        index=0
    )
    
    # Logic mapping for the pipeline flags
    use_embeddings = True
    use_baseline = True
    if retrieval_method == "Baseline Only":
        use_embeddings = False
    elif retrieval_method == "Embeddings Only":
        use_baseline = False

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 1. Robust Initialization ---
# Update cache to depend on the selected embedding model
@st.cache_resource(show_spinner=f"Loading Knowledge Graph with {selected_embedding_model}...")
def get_pipeline(embedding_model_name: str):
    """
    Initialize the RetrievalPipeline. 
    Arguments are hashed; changing the model_name will reload the pipeline.
    """
    return RetrievalPipeline(model_name=embedding_model_name)

try:
    # Pass the selected model from sidebar to the pipeline
    pipeline = get_pipeline(selected_embedding_model)
    backend_ready = True
except Exception as e:
    st.error(f"Failed to connect to Knowledge Graph: {e}")
    backend_ready = False

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. Visualization Helper ---
def visualize_subgraph(combined_results: Dict):
    hotels = combined_results.get("hotels", [])
    visa_info = combined_results.get("visa_info", [])
    others = combined_results.get("others", [])
    
    if not hotels and not visa_info and not others:
        return None

    G = nx.Graph()
    
    # --- DEDUPLICATION TRACKERS ---
    # We track both IDs and Names. If a new node matches EITHER, we skip it.
    seen_ids = set()
    seen_names = set()

    def add_hotel_to_graph(item):
        # 1. Unpack safely (Handle Nested vs Flat)
        if isinstance(item.get("h"), dict):
            h_node = item["h"]
            city = item.get("city_name") or h_node.get("city")
        else:
            h_node = item
            city = item.get("city_name") or item.get("city")

        # 2. Extract Identifiers
        raw_id = h_node.get("hotel_id")
        raw_name = h_node.get("name") or h_node.get("hotel_name")
        
        # Normalize identifiers for comparison
        canonical_id = str(raw_id).strip() if raw_id is not None else None
        canonical_name = raw_name.strip().lower() if raw_name else None
        
        # 3. CRITICAL: Aggressive Deduplication
        if canonical_id and canonical_id in seen_ids:
            return
        if canonical_name and canonical_name in seen_names:
            return

        # 4. Register as Seen
        if canonical_id:
            seen_ids.add(canonical_id)
        if canonical_name:
            seen_names.add(canonical_name)

        # 5. Define Node ID
        # Use canonical_id if available (it's unique), otherwise hash the name
        node_key = canonical_id if canonical_id else f"name_{canonical_name}"
        display_label = raw_name or f"Hotel {node_key}"
        
        # 6. Add Node to Graph
        G.add_node(node_key, label=display_label, color='#FF6B6B', size=20, type='Hotel') 
        
        # 7. Add Location
        if city:
            city_clean = city.strip()
            city_id = f"City_{city_clean}"
            if not G.has_node(city_id):
                G.add_node(city_id, label=city_clean, color='#4ECDC4', size=15, type='City')
            
            if not G.has_edge(node_key, city_id):
                G.add_edge(node_key, city_id, label="LOCATED_IN")
            
        # 8. Add Rating
        score = (h_node.get("average_reviews_score") or 
                 h_node.get("total_avg_score") or 
                 h_node.get("star_rating"))
                 
        if score and isinstance(score, (int, float)) and float(score) > 8.0:
            rating_label = f"⭐ {float(score):.1f}"
            r_id = f"Rate_{node_key}"
            
            G.add_node(r_id, label=rating_label, color='#FFE66D', size=10, type='Rating') 
            G.add_edge(node_key, r_id, label="HAS_RATING")

    # --- PROCESS ORDER FLIPPED ---
    
    # 1. Process 'others' (Baseline) FIRST
    # These are usually exact string matches, so we prioritize them.
    for o in others:
        if "hotel_name" in o or "name" in o:
            add_hotel_to_graph(o)

    # 2. Process 'hotels' (Embeddings) SECOND
    # If the hotel was already added by Baseline, 'seen_names' will block it here.
    for h in hotels:
        add_hotel_to_graph(h)
        
    # --- Process Visa ---
    for v in visa_info:
        origin = v.get("origin_country")
        dest = v.get("destination_country")
        req = v.get("visa_type", "Unknown")
        if origin and dest:
            G.add_node(origin, label=origin, color='#9D50BB', size=25, type='Country')
            G.add_node(dest, label=dest, color='#9D50BB', size=25, type='Country')
            G.add_edge(origin, dest, label=req)

    # --- Render ---
    if G.number_of_nodes() == 0:
        return None
        
    pos = nx.spring_layout(G, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    elabel_x, elabel_y, elabel_text = [], [], []
    for edge in G.edges(data=True):
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            elabel_x.append((x0 + x1) / 2)
            elabel_y.append((y0 + y1) / 2)
            elabel_text.append(edge[2].get('label', ''))

    edge_label_trace = go.Scatter(x=elabel_x, y=elabel_y, mode='text', text=elabel_text, textposition='middle center', hoverinfo='none', textfont=dict(size=10, color="#15ff00"))

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_color.append(G.nodes[node]['color'])
            node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", hoverinfo='text', marker=dict(color=node_color, size=node_size, line_width=2))

    return go.Figure(data=[edge_trace, edge_label_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=0,l=0,r=0,t=0), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=300))

# --- 4. Main Chat Interface ---
st.title("🌍 Graph-RAG Travel Assistant")
st.markdown(f"Ask about hotels, visas, or reviews. \n\n*Current Embedding Model: `{selected_embedding_model}`*")

# Render Chat History
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Requirement: View KG-retrieved context, Cypher queries, & Visualization
        if "data" in msg:
            with st.expander(f"🔍 View Graph Reasoning (Found {len(msg['data'].get('combined',{}).get('hotels',[]))} nodes)"):
                
                # Tabbed view for cleaner UX
                tab1, tab2, tab3 = st.tabs(["📄 KG Context", "🕸️ Graph Visualization", "⚡ Cypher Query"])
                
                with tab1:
                    st.markdown("**Raw Retrieval Context:**")
                    st.text(msg["data"].get("context_text", "No context retrieved."))
                
                with tab2:
                    # Get the whole combined dictionary
                    combined_data = msg["data"].get("combined", {})
                    
                    # Check if we have ANY data (hotels or visa)
                    has_data = combined_data.get("hotels") or combined_data.get("visa_info")
                    
                    if has_data:
                        # Pass the whole dictionary
                        fig = visualize_subgraph(combined_data)
                        st.plotly_chart(fig, width='stretch', key=f"graph_{i}")
                    else:
                        st.info("No graph nodes (hotels or visa info) found to visualize.")
                
                with tab3:
                    intent = msg["data"].get("intent", "Unknown")
                    st.markdown(f"**Classified Intent:** `{intent}`")
                    
                    # NEW: Retrieve the ACTUAL query string from the backend response
                    actual_cypher = msg["data"].get("cypher_query", "")
                    
                    st.markdown("**Executed Cypher Query:**")
                    if actual_cypher:
                        st.code(actual_cypher, language="cypher")
                    else:
                        st.info("No Cypher query was executed for this request (or Embeddings Only mode used).")

# Input Handler
if prompt := st.chat_input("Ex: Find high-rated hotels in Cairo"):
    if not backend_ready:
        st.error("Backend is unavailable. Please check Neo4j connection.")
    else:
        # 1. Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process with Assistant
        with st.chat_message("assistant"):
            status_box = st.status("🔍 Processing...", expanded=True)
            
            try:
                start_time = time.perf_counter()
                
                # A. Retrieval Step
                status_box.write(f"Querying Graph (Embeddings: {selected_embedding_model})...")
                
                # Calling the cached pipeline - safe_retrieve handles errors gracefully
                retrieval_result = pipeline.safe_retrieve(
                    query=prompt,
                    limit=5,
                    user_embeddings=use_embeddings,
                    user_baseline=use_baseline,
                    use_llm=True
                )
                
                context_text = retrieval_result.get("context_text", "")
                
                # B. LLM Generation Step
                status_box.write(f"Generating answer with {model_name}...")
                
                out = answer_with_model(
                    model_name=model_name,
                    user_query=prompt,
                    context_text=context_text,
                    max_new_tokens=512,
                    temperature=0.2,
                    top_p=0.95
                )
                
                response_text = out["generation"].get("text", "I couldn't generate a response.")
                latency = out["generation"].get("latency_s", 0)
                
                total_time = time.perf_counter() - start_time
                status_box.update(label="✅ Complete", state="complete", expanded=False)
                
                # Display Result
                st.markdown(response_text)
                st.caption(f"⏱️ Total: {total_time:.2f}s | LLM: {latency:.2f}s | Nodes Retrieved: {len(retrieval_result.get('combined',{}).get('hotels',[]))}")

                # 3. Append Assistant Message (with data for visualization)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "data": retrieval_result # Store full result for the expander view
                })
                
                # Rerun to render the new message with the expander correctly
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Error", state="error")
                st.error(f"An error occurred: {str(e)}")