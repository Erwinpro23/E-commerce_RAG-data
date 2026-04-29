import streamlit as st
import time
from dotenv import load_dotenv

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce RAG Analytics",
    page_icon="🛒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

load_dotenv()

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Hide sidebar toggle & default padding */
[data-testid="collapsedControl"] { display: none; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.block-container { padding-top: 2rem !important; max-width: 780px !important; }

/* ── Title ── */
.title-wrap {
    text-align: center;
    padding-bottom: 1.2rem;
}
.title-wrap h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
}
.title-wrap p {
    color: rgba(255,255,255,0.45);
    font-size: 0.88rem;
    margin: 0;
}

/* ── Chat bubbles ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.9rem;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #6d28d9, #4f46e5);
    color: #fff;
    padding: 0.7rem 1rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.92rem;
    line-height: 1.55;
    box-shadow: 0 4px 14px rgba(109,40,217,0.3);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 0.9rem;
    gap: 0.5rem;
    align-items: flex-start;
}
.msg-bot .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #34d399, #60a5fa);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0; margin-top: 2px;
}
.msg-bot .bubble {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.88);
    padding: 0.7rem 1rem;
    border-radius: 4px 18px 18px 18px;
    max-width: 85%;
    font-size: 0.92rem;
    line-height: 1.6;
    backdrop-filter: blur(8px);
}

/* ── Input ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #fff !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.92rem !important;
}
.stTextInput > div > div > input::placeholder {
    color: rgba(255,255,255,0.3) !important;
}
.stTextInput label { display: none !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    height: 2.6rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(109,40,217,0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; margin: 0.8rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Load RAG (cached) ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    from retriever import RAGRetriever
    from rag_llm import RAGWithLLM
    retriever = RAGRetriever(
        faiss_index_path="faiss_index.bin",
        metadata_path="metadata.pkl",
        model_name="all-MiniLM-L6-v2",
    )
    return RAGWithLLM(retriever=retriever, model_name="your_model")

# ─── Title ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-wrap">
  <h1>🛒 E-Commerce RAG Analytics</h1>
  <p>Hỏi bất cứ điều gì về dữ liệu bán hàng — AI sẽ tìm và phân tích cho bạn</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Load system ─────────────────────────────────────────────────────────────
with st.spinner("⏳ Đang tải hệ thống..."):
    try:
        rag = load_rag()
    except Exception as e:
        st.error(f"❌ Không thể tải RAG system: {e}")
        st.stop()

# ─── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="msg-bot">'
            f'<div class="avatar">🤖</div>'
            f'<div class="bubble">{msg["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ─── Input ───────────────────────────────────────────────────────────────────
st.markdown("---")
with st.form(key="chat_form", clear_on_submit=True):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "query",
            placeholder="Nhập câu hỏi của bạn... VD: Danh mục nào có lợi nhuận cao nhất?",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

# ─── Process query ────────────────────────────────────────────────────────────
if submitted and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("🔍 Đang phân tích..."):
        try:
            result = rag.query(query, top_k=5, use_fairness=True)
            answer = result["answer"]
        except Exception as e:
            answer = f"⚠️ Lỗi: {e}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
