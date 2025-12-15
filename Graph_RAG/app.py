import streamlit as st
import time
import networkx as nx
import plotly.graph_objects as go
from typing import Dict, List
import os
import json
from datetime import datetime

# --- Backend Imports ---
try:
    from retrieval.retrieval_pipeline import RetrievalPipeline
    from llm.llm_answerer import answer_with_model
except ImportError:
    st.error("Backend modules not found. Please ensure your project structure is correct.")

# --- Page Config ---
st.set_page_config(
    page_title="Graph-RAG Travel Assistant",
    layout="wide"
)

# --- Animated Background (CSS only, no emojis) ---
st.markdown(
        """
        <style>
        :root {
            --bg1: #0f2027;
            --bg2: #203a43;
            --bg3: #2c5364;
            --sb1: #0b1720;
            --sb2: #122634;
            --sb-border: rgba(255,255,255,0.08);
            --text: #e8f1f5;
            --muted: #a9bdc8;
            --accent: #34b3ff;
            --accent-weak: rgba(52,179,255,0.15);
        }

        /* Smooth animated gradient base */
        .stApp {
            background: linear-gradient(120deg, var(--bg1), var(--bg2), var(--bg3));
            background-size: 400% 400%;
            animation: gradientShift 28s ease-in-out infinite;
        }

        /* Base responsive typography */
        html, body { font-size: 16px; }
        @media (max-width: 1200px) { html, body { font-size: 15px; } }
        @media (max-width: 992px)  { html, body { font-size: 14px; } }
        @media (max-width: 768px)  { html, body { font-size: 13px; } }
        @media (max-width: 576px)  { html, body { font-size: 12px; } }

        /* Soft floating color blobs overlay */
        .stApp::before, .stApp::after {
            content: "";
            position: fixed;
            top: -20vh; left: -20vw; right: -20vw; bottom: -20vh;
            pointer-events: none;
            z-index: 0;
            background:
                radial-gradient(closest-side at 25% 35%, rgba(255, 0, 120, 0.08), transparent 60%),
                radial-gradient(closest-side at 75% 65%, rgba(0, 200, 255, 0.07), transparent 60%),
                radial-gradient(closest-side at 60% 25%, rgba(255, 200, 0, 0.05), transparent 60%);
            filter: blur(64px);
            transform: translate3d(0,0,0);
        }

        .stApp::before {
            animation: floatBlob1 26s ease-in-out infinite alternate;
        }
        .stApp::after {
            animation: floatBlob2 34s ease-in-out infinite alternate;
            opacity: 0.85;
        }

        /* Ensure app content stays above the background layers */
        .stApp > div {
            position: relative;
            z-index: 1;
        }

        /* Sidebar styling to match background palette */
        [data-testid="stSidebar"] > div {
            background: linear-gradient(160deg, var(--sb1) 0%, var(--sb2) 100%);
            border-right: 1px solid var(--sb-border);
            box-shadow: 8px 0 24px rgba(0,0,0,0.25);
            color: var(--text);
            height: 100vh;
            overflow-y: auto;
            overscroll-behavior-y: contain;
            scrollbar-width: thin;
        }
        /* Responsive sidebar width adjustments */
        [data-testid="stSidebar"] { width: 22rem; }
        @media (max-width: 1200px) { [data-testid="stSidebar"] { width: 20rem; } }
        @media (max-width: 992px)  { [data-testid="stSidebar"] { width: 18rem; } }
        @media (max-width: 768px)  { [data-testid="stSidebar"] { width: 16rem; } }
        @media (max-width: 576px)  { [data-testid="stSidebar"] { width: 14rem; } }

        /* WebKit scrollbar styling for sidebar */
        [data-testid="stSidebar"] > div::-webkit-scrollbar {
            width: 10px;
        }
        [data-testid="stSidebar"] > div::-webkit-scrollbar-track {
            background: rgba(255,255,255,0.05);
        }
        [data-testid="stSidebar"] > div::-webkit-scrollbar-thumb {
            background: rgba(52,179,255,0.35);
            border-radius: 8px;
            border: 2px solid rgba(0,0,0,0.2);
        }
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span { color: var(--text) !important; }
        [data-testid="stSidebar"] .markdown-text-container p { color: var(--muted) !important; }

        /* Inputs in sidebar */
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stTextInput > div > div,
        [data-testid="stSidebar"] .stMultiSelect > div > div,
        [data-testid="stSidebar"] .stNumberInput > div > div,
        [data-testid="stSidebar"] .stDateInput > div > div,
        [data-testid="stSidebar"] .stTimeInput > div > div {
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--sb-border);
            color: var(--text);
        }
        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: var(--accent-weak);
            border-radius: 6px;
        }
        [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div {
            color: var(--text);
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(180deg, rgba(52,179,255,0.18), rgba(52,179,255,0.10));
            color: var(--text);
            border: 1px solid var(--accent);
            border-radius: 8px;
            transition: all .15s ease-in-out;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            box-shadow: 0 0 0 3px var(--accent-weak);
            transform: translateY(-1px);
        }

        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes floatBlob1 {
            0%   { transform: translate(-2%, -1%) scale(1.00) rotate(0deg); }
            50%  { transform: translate(3%, 2%) scale(1.05) rotate(10deg); }
            100% { transform: translate(-1%, 1%) scale(1.00) rotate(0deg); }
        }

        @keyframes floatBlob2 {
            0%   { transform: translate(1%, -2%) scale(1.00) rotate(0deg); }
            50%  { transform: translate(-3%, 1%) scale(1.07) rotate(-8deg); }
            100% { transform: translate(2%, -1%) scale(1.00) rotate(0deg); }
        }

        /* Responsive content containers */
        .block-container { padding: 1.5rem 2rem 2.5rem; }
        @media (max-width: 992px)  { .block-container { padding: 1.25rem 1.5rem 2rem; } }
        @media (max-width: 768px)  { .block-container { padding: 1rem 1rem 1.5rem; } }
        @media (max-width: 576px)  { .block-container { padding: 0.75rem 0.75rem 1rem; } }

        /* Make Plotly charts responsive */
        .js-plotly-plot, .plotly { width: 100% !important; }
        .stPlotlyChart { width: 100% !important; }
        @media (max-width: 576px) { .stPlotlyChart { min-height: 240px; } }
        </style>
        """,
        unsafe_allow_html=True,
)

# --- Sidebar Configuration (Moved up for Initialization) ---
with st.sidebar:
    st.title("Settings")
    
    # Requirement: Compare at least 3 models
    model_name = st.selectbox(
        "Model",
        options=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.2",
            "deepseek-ai/DeepSeek-V3.2",
            "openai/gpt-oss-20b"
        ],
        index=0
    )

    # NEW: Embedding Model Selection (Shown only when using embeddings)
    selected_embedding_model = None

    # Requirement: Retrieval Method Selection
    retrieval_method = st.radio(
        "Retrieval",
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

    # Conditionally show Embedding picker only when embeddings are used
    if use_embeddings:
        embedding_option = st.radio(
            "Embedding",
            options=["MiniLM (ll-MiniLM-L6-v2)", "BGE (bge-small-en-v1.5)"],
            index=0,
            horizontal=False
        )
        if "MiniLM" in embedding_option:
            selected_embedding_model = "minilm"
        else:
            selected_embedding_model = "bge"

    st.markdown("---")
    # Local persistence disabled permanently
    NO_LOCAL_SAVE = True
    # --- Chat Session Management ---
    # Persist chats to disk so users can reopen old chats across models
    CHAT_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
    # Do not create chat directory when local saving is disabled

    def list_chat_sessions():
        sessions = []
        # When local save is disabled, list from in-memory archive
        if NO_LOCAL_SAVE:
            archived = st.session_state.get("archived_sessions", [])
            for data in archived:
                sessions.append({
                    "id": data.get("id"),
                    "model": data.get("model_name", "unknown"),
                    "created_at": data.get("created_at", ""),
                    "title": data.get("title", data.get("id"))
                })
            sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
            return sessions
        # Otherwise, read from disk
        if not os.path.isdir(CHAT_DIR):
            return sessions
        for name in os.listdir(CHAT_DIR):
            if name.endswith(".json"):
                try:
                    with open(os.path.join(CHAT_DIR, name), "r", encoding="utf-8") as f:
                        data = json.load(f)
                    sessions.append({
                        "id": data.get("id", name[:-5]),
                        "model": data.get("model_name", "unknown"),
                        "created_at": data.get("created_at", ""),
                        "title": data.get("title", data.get("id", name[:-5]))
                    })
                except Exception:
                    continue
        # Sort newest first
        sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return sessions

    def _remove_emojis(text: str) -> str:
        try:
            # Basic emoji removal via unicode ranges
            import re
            emoji_pattern = re.compile(
                "[\U0001F600-\U0001F64F]"  # emoticons
                "|[\U0001F300-\U0001F5FF]"  # symbols & pictographs
                "|[\U0001F680-\U0001F6FF]"  # transport & map symbols
                "|[\U0001F1E0-\U0001F1FF]"  # flags
                "|[\U00002700-\U000027BF]"  # dingbats
                "|[\U0001F900-\U0001F9FF]"  # supplemental symbols
                "|[\U0001FA70-\U0001FAFF]"  # symbols & pictographs ext-A
                ",",
                flags=re.UNICODE,
            )
            return emoji_pattern.sub("", text)
        except Exception:
            return text

    def save_current_chat(session_id: str, model: str, messages: List[Dict]):
        payload = {
            "id": session_id,
            "model_name": model,
            "created_at": st.session_state.get("session_created_at", datetime.utcnow().isoformat()),
            "title": _remove_emojis((
                # Use first words of the first user message as title when available
                (next((m.get("content", "") for m in messages if m.get("role") == "user"), None) or st.session_state.get("session_title", session_id)).split("\n")[0][:60]
            )),
            "messages": [
                {
                    **m,
                    "content": _remove_emojis(m.get("content", ""))
                }
                for m in messages
            ],
        }
        if not NO_LOCAL_SAVE:
            path = os.path.join(CHAT_DIR, f"{session_id}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            # Store in-memory archive
            archived = st.session_state.get("archived_sessions", [])
            # replace if same id exists
            archived = [s for s in archived if s.get("id") != payload["id"]]
            archived.append(payload)
            st.session_state.archived_sessions = archived

    def load_chat(session_id: str):
        if NO_LOCAL_SAVE:
            data = next((s for s in st.session_state.get("archived_sessions", []) if s.get("id") == session_id), None)
            if not data:
                return
        else:
            path = os.path.join(CHAT_DIR, f"{session_id}.json")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        st.session_state.messages = data.get("messages", [])
        st.session_state.current_session_id = data.get("id", session_id)
        st.session_state.session_created_at = data.get("created_at", datetime.utcnow().isoformat())
        st.session_state.session_title = data.get("title", session_id)

    # Initialize session id
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = f"chat-{int(time.time())}"
        st.session_state.session_created_at = datetime.utcnow().isoformat()
        st.session_state.session_title = "New Chat"

    # Controls
    st.subheader("Chat Sessions")
    if st.button("New Chat"):
            # Save existing chat before starting a new one
            if st.session_state.get("messages"):
                save_current_chat(st.session_state.current_session_id, model_name, st.session_state.messages)
                # Also append structured entries to tests/results/results.json for analysis
                try:
                    if not NO_LOCAL_SAVE:
                        results_dir = os.path.join(os.path.dirname(__file__), "tests", "results")
                        os.makedirs(results_dir, exist_ok=True)
                        results_path = os.path.join(results_dir, "results.json")
                        # Load existing
                        existing = []
                        if os.path.exists(results_path):
                            with open(results_path, "r", encoding="utf-8") as f:
                                try:
                                    existing = json.load(f)
                                except Exception:
                                    existing = []
                        # Build entries from session messages (user-assistant pairs)
                        entries = []
                        messages = st.session_state.messages
                        for i in range(0, len(messages)-1, 2):
                            user_msg = messages[i]
                            asst_msg = messages[i+1] if i+1 < len(messages) else None
                            if user_msg.get("role") != "user" or not asst_msg or asst_msg.get("role") != "assistant":
                                continue
                            entry = {
                                "query_index": len(existing) + len(entries),
                                "query": user_msg.get("content", ""),
                                "model": model_name,
                                "latency_s": asst_msg.get("latency_s"),
                                "end_to_end_latency_s": asst_msg.get("total_time_s"),
                                "response_text": asst_msg.get("content", ""),
                                "error": None,
                                "approx_input_tokens": asst_msg.get("approx_input_tokens"),
                                "approx_output_tokens": asst_msg.get("approx_output_tokens"),
                            }
                            entries.append(entry)
                        with open(results_path, "w", encoding="utf-8") as f:
                            json.dump(existing + entries, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            # Start new
            st.session_state.current_session_id = f"chat-{int(time.time())}"
            st.session_state.session_created_at = datetime.utcnow().isoformat()
            st.session_state.session_title = "New Chat"
            st.session_state.messages = []
            st.rerun()

    # Filter sessions by model or show all
    sessions = list_chat_sessions()
    # Minimal history list: each row has an Open button and a small X delete button
    session_list = list_chat_sessions()
    if not session_list:
        st.info("No saved chats yet.")
    else:
        for s in session_list:
            row = st.columns([6,1])
            # Display only the chat title (first words), not the model
            label = f"{s['title']}"
            with row[0]:
                if st.button(label, key=f"open_{s['id']}"):
                    load_chat(s['id'])
                    st.rerun()
            with row[1]:
                if st.button("x", key=f"del_{s['id']}"):
                    try:
                        was_current = st.session_state.get("current_session_id") == s['id']
                        if NO_LOCAL_SAVE:
                            st.session_state.archived_sessions = [a for a in st.session_state.get("archived_sessions", []) if a.get("id") != s['id']]
                        else:
                            os.remove(os.path.join(CHAT_DIR, f"{s['id']}.json"))
                        if was_current:
                            st.session_state.current_session_id = f"chat-{int(time.time())}"
                            st.session_state.session_created_at = datetime.utcnow().isoformat()
                            st.session_state.session_title = "New Chat"
                            st.session_state.messages = []
                    except Exception as e:
                        st.error(f"Failed to delete: {e}")
                    st.rerun()

    st.markdown("---")

# --- 1. Robust Initialization ---
# Update cache to depend on the selected embedding model
@st.cache_resource(show_spinner="Loading Knowledge Graph...")
def get_pipeline(embedding_model_name: str | None):
    """
    Initialize the RetrievalPipeline. 
    Arguments are hashed; changing the model_name will reload the pipeline.
    """
    # Fallback to a default embedding model if none is provided
    backend_model = embedding_model_name or "minilm"
    return RetrievalPipeline(model_name=backend_model)

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
            rating_label = f"Rating {float(score):.1f}"
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
st.title("Graph-RAG Travel Assistant")
st.markdown(f"Ask about hotels, visas, or reviews.  Retrieval: {retrieval_method}")

# Render Chat History
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Requirement: View KG-retrieved context, Cypher queries, & Visualization
        if "data" in msg:
            with st.expander(f"View Graph Reasoning (Found {len(msg['data'].get('combined',{}).get('hotels',[]))} nodes)"):
                
                # Tabbed view for cleaner UX
                tab1, tab2, tab3 = st.tabs(["KG Context", "Graph Visualization", "Cypher Query"])
                
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
if prompt := st.chat_input("Ask TAJR"):
    if not backend_ready:
        st.error("Backend is unavailable. Please check Neo4j connection.")
    else:
        # 1. Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process with Assistant
        with st.chat_message("assistant"):
            status_box = st.status("Processing...", expanded=True)
            
            try:
                start_time = time.perf_counter()
                
                # A. Retrieval Step
                status_box.write(f"Querying Graph (Retrieval: {retrieval_method})...")
                
                # Calling the cached pipeline - safe_retrieve handles errors gracefully
                retrieval_result = pipeline.safe_retrieve(
                    query=prompt,
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
                status_box.update(label="Complete", state="complete", expanded=False)
                
                # Display Result
                st.markdown(response_text)
                st.caption(f"Total: {total_time:.2f}s | LLM: {latency:.2f}s | Nodes Retrieved: {len(retrieval_result.get('combined',{}).get('hotels',[]))}")

                # 3. Append Assistant Message (with data for visualization)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "data": retrieval_result # Store full result for the expander view
                })
                # Attach timing metadata for results export
                st.session_state.messages[-1]["latency_s"] = latency
                st.session_state.messages[-1]["total_time_s"] = total_time
                # Optional token estimates if available from model output
                if "approx_input_tokens" in out:
                    st.session_state.messages[-1]["approx_input_tokens"] = out.get("approx_input_tokens")
                if "approx_output_tokens" in out:
                    st.session_state.messages[-1]["approx_output_tokens"] = out.get("approx_output_tokens")

                # Auto-save after assistant response for continuity
                try:
                    # Respect NO_LOCAL_SAVE when auto-saving
                    save_current_chat(st.session_state.current_session_id, model_name, st.session_state.messages)
                except Exception:
                    pass
                
                # Rerun to render the new message with the expander correctly
                st.rerun()

            except Exception as e:
                status_box.update(label="Error", state="error")
                st.error(f"An error occurred: {str(e)}")