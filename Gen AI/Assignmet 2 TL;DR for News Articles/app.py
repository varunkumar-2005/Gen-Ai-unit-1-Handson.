import streamlit as st
from transformers import pipeline

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="TL;DR AI News Summarizer",
    layout="centered"
)

# -----------------------------
# Advanced UI Styling
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #f9fafb, #eef2ff);
    font-family: 'Inter', sans-serif;
}

/* Header card */
.header {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    padding: 28px;
    border-radius: 14px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 32px;
    margin-bottom: 6px;
}

.header p {
    font-size: 15px;
    opacity: 0.9;
}

/* Main card */
.card {
    background: rgba(255, 255, 255, 0.9);
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Text area */
.stTextArea textarea {
    border-radius: 12px;
    padding: 14px;
    font-size: 15px;
}

/* Button */
.stButton button {
    background: linear-gradient(135deg, #2563eb, #4f46e5);
    color: white;
    border-radius: 12px;
    font-weight: 600;
    font-size: 16px;
    width: 100%;
    padding: 10px;
}

.stButton button:hover {
    transform: scale(1.01);
    transition: 0.2s ease-in-out;
}

/* Summary box */
.summary-box {
    background: linear-gradient(135deg, #ffffff, #f1f5f9);
    padding: 22px;
    border-radius: 14px;
    border-left: 5px solid #4f46e5;
    box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    color: #1e293b;
    line-height: 1.65;
    font-size: 15px;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 13px;
    color: #64748b;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load summarization model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

summarizer = load_model()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
    <h1>🧠 TL;DR News Summarizer</h1>
    <p>Turn long news articles into crisp, 3-sentence insights using AI</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input Card
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

article_text = st.text_area(
    "Paste your news article",
    height=220,
    placeholder="Paste a full news article here..."
)

word_count = len(article_text.split())
st.caption(f"📏 Word count: **{word_count}**")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Generate Summary
# -----------------------------
if st.button("✨ Generate Smart Summary"):
    if not article_text.strip():
        st.warning("Please paste a news article to summarize.")
    else:
        with st.spinner("🧠 AI is summarizing..."):
            summary = summarizer(
                article_text,
                max_length=90,
                min_length=45,
                do_sample=False
            )

            st.markdown(f"""
            <div class="summary-box">
            <b>📌 TL;DR Summary</b><br><br>
            {summary[0]["summary_text"]}
            </div>
            """, unsafe_allow_html=True)

            st.code(summary[0]["summary_text"], language="markdown")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    🚀 Built with Streamlit & Transformers | AI-powered summarization
</div>
""", unsafe_allow_html=True)
