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
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# --- Custom CSS for Professional UI ---
st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styling */
    * {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Dark Mode Variables */
    :root {{
        --bg-color: {'#1a1a1a' if st.session_state.dark_mode else '#ffffff'};
        --text-color: {'#e0e0e0' if st.session_state.dark_mode else '#1a1a1a'};
        --sidebar-bg: {'#1a1a1a' if st.session_state.dark_mode else '#ffffff'};
        --card-bg: {'#2d2d2d' if st.session_state.dark_mode else '#f8f9fa'};
        --input-bg: {'#2d2d2d' if st.session_state.dark_mode else '#f1f3f5'};
        --border-color: {'#404040' if st.session_state.dark_mode else '#dee2e6'};
        --accent-color: #667eea;
        --accent-secondary: #764ba2;
    }}
    
    /* Main Background */
    .stApp {{
        background-color: var(--bg-color);
        color: var(--text-color);
    }}
    
    /* Main container animation */
    .main {{
        animation: fadeIn 0.5s ease-in;
        background-color: var(--bg-color);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* Header styling */
    h1, h2, h3, h4, h5, h6 {{
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        animation: slideInDown 0.6s ease-out;
    }}
    
    @keyframes slideInDown {{
        from {{ transform: translateY(-20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    
    /* Paragraph and text styling */
    p, span, div, label, strong, em, b, i, code, pre, li, ul, ol, a {{
        color: var(--text-color) !important;
    }}
    
    /* Override any default dark text in light mode */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div, 
    .stMarkdown strong, .stMarkdown em, .stMarkdown li, .stMarkdown code {{
        color: var(--text-color) !important;
    }}
    
    /* Links styling */
    a, a:visited, a:hover {{
        color: var(--accent-color) !important;
    }}
    
    /* Chat message containers */
    .stChatMessage {{
        animation: messageSlideIn 0.4s ease-out;
        transition: all 0.3s ease;
        background-color: var(--bg-color) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .stChatMessage p, .stChatMessage span, .stChatMessage div, .stChatMessage li,
    .stChatMessage strong, .stChatMessage em, .stChatMessage code, .stChatMessage pre,
    .stChatMessage ul, .stChatMessage ol, .stChatMessage a {{
        color: var(--text-color) !important;
        background-color: transparent !important;
    }}
    
    /* All descendants of chat messages */
    .stChatMessage * {{
        color: var(--text-color) !important;
    }}
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {{
        background-color: var(--bg-color) !important;
    }}
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {{
        background-color: var(--bg-color) !important;
    }}
    
    .stChatMessage:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }}
    
    @keyframes messageSlideIn {{
        from {{ transform: translateX(-20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 3px solid var(--accent-color);
    }}
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }}
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {{
        color: var(--text-color) !important;
    }}
    
    /* Sidebar selectbox and radio styling */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stRadio label {{
        color: var(--text-color) !important;
        font-weight: 600;
    }}
    
    [data-testid="stSidebar"] [data-baseweb="select"] {{
        background-color: var(--card-bg);
    }}
    
    /* Sidebar metrics */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {{
        color: var(--text-color) !important;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {{
        color: var(--accent-color) !important;
    }}
    
    /* Button animations */
    .stButton button {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }}
    
    .stButton button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }}
    
    /* Input field styling */
    .stTextInput input, .stChatInput input {{
        border-radius: 10px;
        border: 2px solid var(--accent-color);
        transition: all 0.3s ease;
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
    }}
    
    .stTextInput input::placeholder, .stChatInput input::placeholder {{
        color: {'#888' if st.session_state.dark_mode else '#666'} !important;
    }}
    
    .stChatInput input:focus, .stTextInput input:focus {{
        border-color: var(--accent-secondary);
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* Text area styling - ensure proper contrast */
    .stTextArea textarea {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 2px solid var(--border-color) !important;
    }}
    
    .stTextArea textarea::placeholder {{
        color: {'#888' if st.session_state.dark_mode else '#666'} !important;
    }}
    
    /* Disabled text areas (like context display) */
    .stTextArea textarea:disabled {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        opacity: 1 !important;
        -webkit-text-fill-color: var(--text-color) !important;
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: var(--card-bg) !important;
        border-radius: 10px;
        transition: all 0.3s ease;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}
    
    /* All text inside expander content */
    .streamlit-expanderContent * {{
        color: var(--text-color) !important;
    }}
    
    /* Expander header text including bold */
    .streamlit-expanderHeader p, .streamlit-expanderHeader span, 
    .streamlit-expanderHeader strong, .streamlit-expanderHeader b {{
        color: var(--text-color) !important;
    }}
    
    /* Markdown headers inside expanders */
    .streamlit-expanderContent h1, .streamlit-expanderContent h2, 
    .streamlit-expanderContent h3, .streamlit-expanderContent h4 {{
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .streamlit-expanderHeader:hover {{
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }}
    
    /* Status box animation */
    .stStatus {{
        animation: pulse 2s infinite;
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* Status widget content */
    .stStatus * {{
        color: var(--text-color) !important;
    }}
    
    /* Status messages */
    [data-testid="stStatusWidget"] {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    [data-testid="stStatusWidget"] * {{
        color: var(--text-color) !important;
    }}
    
    /* Status widget details */
    .stStatus > div {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.8; }}
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
        background-color: var(--card-bg);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        transition: all 0.3s ease;
        color: var(--text-color) !important;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(102, 126, 234, 0.1);
    }}
    
    /* Card-like containers */
    .element-container {{
        transition: all 0.3s ease;
    }}
    
    /* Metric styling */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: var(--text-color) !important;
    }}
    
    /* Loading spinner */
    .stSpinner > div {{
        border-top-color: var(--accent-color) !important;
    }}
    
    /* Success/Info/Warning/Error boxes */
    .stAlert {{
        border-radius: 10px;
        animation: slideInRight 0.5s ease-out;
    }}
    
    @keyframes slideInRight {{
        from {{ transform: translateX(20px); opacity: 0; }}
        to {{ transform: translateX(0); opacity: 1; }}
    }}
    
    /* Plotly graph container */
    .js-plotly-plot {{
        border-radius: 15px;
        box-shadow: 0 8px 25px {'rgba(255,255,255,0.1)' if st.session_state.dark_mode else 'rgba(0,0,0,0.1)'};
        transition: all 0.3s ease;
        background-color: var(--card-bg) !important;
    }}
    
    .js-plotly-plot:hover {{
        box-shadow: 0 12px 35px {'rgba(255,255,255,0.15)' if st.session_state.dark_mode else 'rgba(0,0,0,0.15)'};
    }}
    
    /* Plotly text elements */
    .js-plotly-plot text, .js-plotly-plot .xtick text, .js-plotly-plot .ytick text {{
        fill: var(--text-color) !important;
    }}
    
    /* Caption styling */
    .caption {{
        font-size: 0.9rem;
        color: var(--text-color);
        font-style: italic;
    }}
    
    /* Code block styling */
    .stCodeBlock {{
        border-radius: 10px;
        border-left: 4px solid var(--accent-color);
        background-color: var(--card-bg) !important;
    }}
    
    .stCodeBlock code, .stCodeBlock pre {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* Inline code styling */
    code {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        padding: 2px 6px;
        border-radius: 4px;
    }}
    
    /* Dataframe styling */
    .stDataFrame, .stDataFrame * {{
        color: var(--text-color) !important;
    }}
    
    /* Alert/Info/Warning/Error content */
    .stAlert p, .stAlert span, .stAlert div, .stAlert strong {{
        color: var(--text-color) !important;
    }}
    
    /* JSON/Dict display */
    .stJson, .stJson * {{
        color: var(--text-color) !important;
        background-color: var(--card-bg) !important;
    }}
    
    /* Selectbox dropdown */
    [data-baseweb="select"] > div {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
    }}
    
    [data-baseweb="popover"] {{
        background-color: var(--card-bg) !important;
    }}
    
    /* Radio buttons */
    .stRadio > label {{
        color: var(--text-color) !important;
    }}
    
    .stRadio label {{
        color: var(--text-color) !important;
    }}
    
    /* Markdown content */
    .stMarkdown {{
        color: var(--text-color) !important;
    }}
    
    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* Status container */
    [data-testid="stStatusWidget"] {{
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* All divs within main content */
    .main .element-container div {{
        color: var(--text-color) !important;
    }}
    
    /* All containers */
    .stContainer, .element-container {{
        color: var(--text-color) !important;
    }}
    
    /* Tab content areas */
    .stTabs [data-baseweb="tab-panel"] {{
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }}
    
    /* All text within tabs */
    .stTabs [data-baseweb="tab-panel"] * {{
        color: var(--text-color) !important;
    }}
    
    /* Force all text elements in chat to follow theme */
    .stChatMessage * {{
        color: var(--text-color) !important;
    }}
    
    /* Chat input container - always black background */
    [data-testid="stChatInput"] {{
        background-color: #1a1a1a !important;
    }}
    
    [data-testid="stChatInput"] > div {{
        background-color: #1a1a1a !important;
    }}
    
    [data-testid="stChatInput"] input {{
        background-color: var(--input-bg) !important;
        color: var(--text-color) !important;
    }}
    
    /* Chat input form container */
    .stChatInputContainer {{
        background-color: #1a1a1a !important;
    }}
    
    /* Bottom container where chat input lives - always black */
    .stBottom {{
        background-color: #1a1a1a !important;
    }}
    
    section[data-testid="stBottom"] {{
        background-color: #1a1a1a !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- 1. Robust Initialization ---
@st.cache_resource(show_spinner="Connecting to Knowledge Graph...")
def get_pipeline():
    """
    Initialize the RetrievalPipeline once and cache it. 
    This prevents connection drops/stale states on subsequent queries.
    """
    return RetrievalPipeline()

try:
    pipeline = get_pipeline()
    backend_ready = True
except Exception as e:
    st.error(f"Failed to connect to Knowledge Graph: {e}")
    backend_ready = False

# --- 2. Visualization Helper ---
def visualize_subgraph(hotels: List[Dict]):
    """Creates a Plotly network graph from retrieved hotel nodes[cite: 101, 103]."""
    if not hotels:
        return None

    G = nx.Graph()
    
    for h in hotels:
        # Hotel Node
        h_name = h.get("name") or "Unknown Hotel"
        h_id = h.get("hotel_id") or h_name
        G.add_node(h_id, label=h_name, color='#FF6B6B', size=20, type='Hotel') # Red for Hotels
        
        # City Node (Connect Hotel -> City)
        city = h.get("city")
        if city:
            city_id = f"City_{city}"
            G.add_node(city_id, label=city, color='#4ECDC4', size=15, type='City') # Teal for Cities
            G.add_edge(h_id, city_id, label="LOCATED_IN")
            
        # Rating Node (Visual cue for high rating)
        score = h.get("avg_score_cleanliness") or h.get("average_reviews_score") or h.get("star_rating")
        if score and isinstance(score, (int, float)) and score > 8.0:
            rating_label = f"⭐ {score}"
            r_id = f"Rate_{h_id}"
            G.add_node(r_id, label=rating_label, color='#FFE66D', size=10, type='Rating') # Yellow
            G.add_edge(h_id, r_id)

    # Generate Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Edges trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Theme-aware edge color
    edge_color = '#888' if not st.session_state.dark_mode else '#aaa'
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color=edge_color),
        hoverinfo='none',
        mode='lines')

    # Nodes trace
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])
        node_color.append(G.nodes[node]['color'])
        node_size.append(G.nodes[node]['size'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(color=node_color, size=node_size, line_width=2),
        textfont=dict(
            color='#1a1a1a' if not st.session_state.dark_mode else '#e0e0e0',
            size=12
        ))

    # Theme-aware plot background
    plot_bgcolor = '#ffffff' if not st.session_state.dark_mode else '#1a1a1a'
    paper_bgcolor = '#f8f9fa' if not st.session_state.dark_mode else '#2d2d2d'
    
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0,l=0,r=0,t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor=plot_bgcolor,
                paper_bgcolor=paper_bgcolor,
                height=300
             ))
    return fig

# --- 3. Sidebar Configuration ---
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    
    # Dark Mode Toggle
    col_mode1, col_mode2 = st.columns([3, 1])
    with col_mode1:
        st.markdown("### 🌓 Theme")
    with col_mode2:
        if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="dark_mode_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    st.markdown("---")
    
    # [cite_start]Requirement: Compare at least 3 models [cite: 77, 113]
    st.markdown("### 🤖 AI Model")
    model_name = st.selectbox(
        "Choose your AI assistant",
        options=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.2"
        ],
        index=0,
        help="Select which language model to use for generating responses"
    )

    st.markdown("### 🔍 Retrieval Strategy")
    # [cite_start]Requirement: Retrieval Method Selection [cite: 116]
    retrieval_method = st.radio(
        "How to search the knowledge graph",
        options=["Hybrid (Baseline + Embeddings)", "Baseline Only", "Embeddings Only"],
        index=0,
        help="Hybrid combines traditional graph queries with semantic search"
    )
    
    # Logic mapping for the pipeline flags
    use_embeddings = True
    use_baseline = True
    if retrieval_method == "Baseline Only":
        use_embeddings = False
    elif retrieval_method == "Embeddings Only":
        use_baseline = False

    st.markdown("---")
    
    # Stats section
    if st.session_state.messages:
        st.markdown("### 📊 Session Stats")
        total_messages = len(st.session_state.messages)
        user_queries = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.metric("Total Messages", total_messages)
        st.metric("Your Questions", user_queries)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    
    about_bg_color = 'rgba(102, 126, 234, 0.1)' if not st.session_state.dark_mode else 'rgba(102, 126, 234, 0.2)'
    about_text_color = '#2c3e50' if not st.session_state.dark_mode else '#e0e0e0'
    
    st.markdown(f"""
    <div style='background: {about_bg_color}; 
                padding: 15px; border-radius: 10px; border-left: 3px solid #667eea;'>
    <p style='font-size: 0.85rem; margin: 0; color: {about_text_color};'>
    This assistant uses <strong>Graph-RAG</strong> technology, combining Neo4j knowledge graphs 
    with AI language models to provide accurate, grounded travel information.
    </p>
    </div>
    """, unsafe_allow_html=True)

# --- 4. Main Chat Interface ---
st.markdown("# 🌍 Graph-RAG Travel Assistant")

info_bg = 'rgba(102, 126, 234, 0.1)' if not st.session_state.dark_mode else 'rgba(102, 126, 234, 0.2)'
info_text = '#2c3e50' if not st.session_state.dark_mode else '#e0e0e0'

st.markdown(f"""
<div style='background: {info_bg}; 
            padding: 20px; border-radius: 15px; margin-bottom: 20px; border-left: 5px solid #667eea;'>
    <p style='font-size: 1.1rem; margin: 0; color: {info_text};'>
    Ask about <strong>hotels</strong>, <strong>visas</strong>, or <strong>reviews</strong>. 
    The system uses <strong>Neo4j Graph Retrieval</strong> to ground answers with real data.
    </p>
</div>
""", unsafe_allow_html=True)

# Render Chat History
if not st.session_state.messages:
    # Welcome screen
    welcome_text = '#667eea' if not st.session_state.dark_mode else '#8b9aee'
    
    st.markdown(f"""
    <div style='text-align: center; padding: 40px 20px;'>
        <h2 style='color: {welcome_text}; margin-bottom: 30px;'>👋 Welcome! How can I help you today?</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    feature_bg = 'rgba(102, 126, 234, 0.05)' if not st.session_state.dark_mode else 'rgba(102, 126, 234, 0.15)'
    feature_title = '#667eea' if not st.session_state.dark_mode else '#8b9aee'
    feature_text = '#2c3e50' if not st.session_state.dark_mode else '#e0e0e0'
    feature_subtext = '#666' if not st.session_state.dark_mode else '#b0b0b0'
    
    st.markdown(f"""
    <div style='background: {feature_bg}; padding: 25px; border-radius: 15px; margin-top: 30px;'>
        <h3 style='color: {feature_title}; margin-top: 0;'>✨ Features</h3>
        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;'>
            <div style='display: flex; align-items: start;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>🧠</span>
                <div>
                    <strong style='color: {feature_text};'>AI-Powered Answers</strong><br>
                    <span style='color: {feature_subtext}; font-size: 0.9rem;'>Multiple LLM models for optimal responses</span>
                </div>
            </div>
            <div style='display: flex; align-items: start;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>🔗</span>
                <div>
                    <strong style='color: {feature_text};'>Graph Knowledge Base</strong><br>
                    <span style='color: {feature_subtext}; font-size: 0.9rem;'>Neo4j-powered data retrieval</span>
                </div>
            </div>
            <div style='display: flex; align-items: start;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>🔍</span>
                <div>
                    <strong style='color: {feature_text};'>Hybrid Search</strong><br>
                    <span style='color: {feature_subtext}; font-size: 0.9rem;'>Combines semantic and structured queries</span>
                </div>
            </div>
            <div style='display: flex; align-items: start;'>
                <span style='font-size: 1.5rem; margin-right: 10px;'>📊</span>
                <div>
                    <strong style='color: {feature_text};'>Visual Insights</strong><br>
                    <span style='color: {feature_subtext}; font-size: 0.9rem;'>Interactive graph visualizations</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# FIX: Use enumerate to get a unique index 'i' for unique keys
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # [cite_start]Requirement: View KG-retrieved context, Cypher queries, & Visualization [cite: 91, 97, 101]
        if "data" in msg:
            with st.expander(f"🔍 **Graph Reasoning** • {len(msg['data'].get('combined',{}).get('hotels',[]))} nodes retrieved", expanded=False):
                
                # Tabbed view for cleaner UX
                tab1, tab2, tab3 = st.tabs(["📄 Context", "🕸️ Visualization", "⚡ Query"])
                
                with tab1:
                    st.markdown("#### Retrieved Knowledge Graph Context")
                    context_box = st.container()
                    with context_box:
                        st.text_area(
                            "Context",
                            msg["data"].get("context_text", "No context retrieved."),
                            height=200,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"context_{i}"
                        )
                
                with tab2:
                    st.markdown("#### Interactive Graph Visualization")
                    hotels = msg["data"].get("combined", {}).get("hotels", [])
                    if hotels:
                        fig = visualize_subgraph(hotels)
                        st.plotly_chart(fig, use_container_width=True, key=f"graph_{i}")
                    else:
                        st.info("💡 No hotel nodes found to visualize.")
                
                with tab3:
                    intent = msg["data"].get("intent", "Unknown")
                    st.markdown(f"**Classified Intent:** `{intent}`")
                    
                    # NEW: Retrieve the ACTUAL query string from the backend response
                    actual_cypher = msg["data"].get("cypher_query", "")
                    
                    st.markdown("**Executed Cypher Query:**")
                    if actual_cypher:
                        st.code(actual_cypher, language="cypher")
                    else:
                        st.info("💡 No Cypher query was executed (Embeddings Only mode or non-hotel query).")

# Input Handler
if prompt := st.chat_input("Ask the bot"):
    if not backend_ready:
        st.error("⚠️ Backend is unavailable. Please check Neo4j connection.")
    else:
        # 1. Append User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Process with Assistant
        with st.chat_message("assistant"):
            status_box = st.status("🔍 Processing your request...", expanded=True)
            
            try:
                start_time = time.perf_counter()
                
                # A. Retrieval Step
                status_box.write("🔎 Extracting entities & querying Knowledge Graph...")
                
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
                status_box.write(f"🤖 Generating answer with {model_name.split('/')[-1]}...")
                
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
                status_box.update(label="✅ Complete!", state="complete", expanded=False)
                
                # [cite_start]Display Result [cite: 94]
                st.markdown(response_text)
                
                # Create metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Total Time", f"{total_time:.2f}s")
                with col2:
                    st.metric("🤖 LLM Time", f"{latency:.2f}s")
                with col3:
                    st.metric("🔗 Nodes Retrieved", len(retrieval_result.get('combined',{}).get('hotels',[])))

                # 3. Append Assistant Message (with data for visualization)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "data": retrieval_result # Store full result for the expander view
                })
                
                # Rerun to render the new message with the expander correctly
                st.rerun()

            except Exception as e:
                status_box.update(label="❌ Error occurred", state="error")
                st.error(f"🚨 An error occurred: {str(e)}")