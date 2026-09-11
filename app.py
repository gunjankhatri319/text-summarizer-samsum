import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="SummarAI | Dialogue Abstractive Engine",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Modern / Glassmorphic CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient header container */
    .hero-container {
        padding: 2.2rem 2rem;
        background: linear-gradient(135deg, rgba(238, 242, 246, 0.08) 0%, rgba(200, 182, 175, 0.12) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        backdrop-filter: blur(12px);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(90deg, #f5f0eb, #d8b4a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #a8a29e;
        font-size: 0.98rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* Glass card panels */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.6rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    .output-box {
        background: linear-gradient(145deg, rgba(216, 180, 160, 0.08), rgba(255, 255, 255, 0.02));
        border: 1px solid rgba(216, 180, 160, 0.35);
        border-radius: 14px;
        padding: 1.4rem;
        color: #f3ede8;
        font-size: 1.05rem;
        line-height: 1.6;
        min-height: 120px;
    }

    /* Metric cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-top: 1.2rem;
    }

    .metric-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 0.9rem 0.6rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2d9d2;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #a8a29e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.2rem;
    }

    /* Primary button aesthetic override */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #c48b71 0%, #9e644b 100%);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 0.65rem 1.4rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 4px 14px rgba(196, 139, 113, 0.35);
        transition: all 0.2s ease;
    }

    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #d3997e 0%, #ab6e54 100%);
        box-shadow: 0 6px 20px rgba(196, 139, 113, 0.5);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">✨ Dialogue Abstractive Summarizer</h1>
    <p class="hero-subtitle">Sequence-to-Sequence Transformer architecture fine-tuned on the SAMSum conversational dataset.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Model Loader
# ---------------------------------------------------------
MODEL_NAME = "transformersbook/pegasus-samsum"

@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    return tokenizer, model, device

with st.spinner("Initializing transformer model weights..."):
    tokenizer, model, device = load_model_and_tokenizer()

# ---------------------------------------------------------
# Sample Data Loader
# ---------------------------------------------------------
default_dialogue = (
    "Hannah: Hey, do you have Betty's number?\n"
    "Amanda: Lemme check... Yeah, it's 555-1234.\n"
    "Hannah: Thanks! Need to ask her about the project notes."
)

samples_dict = {"Default Sample": default_dialogue}
try:
    test_df = pd.read_csv("data/samsum-test.csv")
    if "dialogue" in test_df.columns:
        for idx in range(min(5, len(test_df))):
            preview = test_df.iloc[idx]["dialogue"].split("\n")[0][:45] + "..."
            samples_dict[f"Test Case {idx+1}: {preview}"] = test_df.iloc[idx]["dialogue"]
except Exception:
    pass

# Quick Sample Picker
with st.expander("📂 Load a test case from the SAMSum dataset", expanded=False):
    selected_sample = st.selectbox("Choose a preloaded conversation:", list(samples_dict.keys()))
    if st.button("Load Conversation"):
        st.session_state["user_input_content"] = samples_dict[selected_sample]

if "user_input_content" not in st.session_state:
    st.session_state["user_input_content"] = default_dialogue

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    st.markdown("### 💬 Source Conversation")
    user_text = st.text_area(
        label="Dialogue input",
        label_visibility="collapsed",
        value=st.session_state["user_input_content"],
        height=270,
        placeholder="Paste your multi-turn chat transcript here..."
    )
    
    ctrl_a, ctrl_b = st.columns(2)
    with ctrl_a:
        min_tokens = st.slider("Minimum Length (Tokens)", 5, 40, 10)
    with ctrl_b:
        max_tokens = st.slider("Maximum Length (Tokens)", 20, 128, 60)
        
    generate_trigger = st.button("✨ Generate Summary", use_container_width=True)

with col2:
    st.markdown("### 📋 Generated Summary")
    
    if generate_trigger:
        if user_text.strip():
            with st.spinner("Decoding abstractive summary..."):
                inputs = tokenizer(user_text, truncation=True, padding="longest", return_tensors="pt").to(device)
                with torch.no_grad():
                    summary_ids = model.generate(
                        **inputs,
                        min_length=min_tokens,
                        max_length=max_tokens,
                        num_beams=3,
                        early_stopping=True
                    )
                summary_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                
                # Render styled output card
                st.markdown(f'<div class="output-box">{summary_output}</div>', unsafe_allow_html=True)
                
                # Calculate metrics
                orig_words = len(user_text.split())
                sum_words = len(summary_output.split())
                reduction = round((1 - (sum_words / max(orig_words, 1))) * 100, 1)
                
                # Render styled metric pills
                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-pill">
                        <div class="metric-value">{orig_words}</div>
                        <div class="metric-label">Original Words</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-value">{sum_words}</div>
                        <div class="metric-label">Summary Words</div>
                    </div>
                    <div class="metric-pill">
                        <div class="metric-value">{reduction}%</div>
                        <div class="metric-label">Reduction Ratio</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please paste some dialogue before running the summarizer.")
    else:
        st.markdown(
            '<div class="output-box" style="color: #78716c; font-style: italic;">The abstractive summary will be synthesized here upon clicking "Generate Summary".</div>',
            unsafe_allow_html=True
        )