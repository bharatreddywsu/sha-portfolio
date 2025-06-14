import os
import base64
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config must be the first Streamlit command
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHA — Bharat’s AI Assistant",
    page_icon="👩‍🚀",
    layout="centered",
)

# Pull your OpenAI key from Streamlit Cloud’s Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# ─────────────────────────────────────────────────────────────────────────────
# Now import your retrieval logic (chat.py) after config to avoid early st.* calls
# ─────────────────────────────────────────────────────────────────────────────
from chat import qa_chain, store  # Import RetrievalQA chain and FAISS store

# ─────────────────────────────────────────────────────────────────────────────
# Avatar display
# ─────────────────────────────────────────────────────────────────────────────
def show_sha_avatar():
    file_path = "shaavatar.png"
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <div style='text-align:center; margin-bottom:15px;'>
                <img src="data:image/png;base64,{encoded}" width="120"
                     style="border-radius:50%; box-shadow:0 0 15px #7F5AF0;">
                <h2 style='color:#E0E0E0; margin-top:10px;'>SHA — Bharat's Companion</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

show_sha_avatar()

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS for cosmic theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            background: linear-gradient(135deg, #0A0F2C 0%, #1B0033 100%);
            color: #E0E0E0;
            font-family: 'Poppins', sans-serif;
        }
        .stTextInput > div > div > input {
            background-color: #1B1B2F;
            color: #E0E0E0;
            border: 1px solid #7F5AF0;
            border-radius: 8px;
            padding: 12px;
        }
        .stButton>button, button[kind="primary"] {
            background-color: #7F5AF0 !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            padding: 8px 16px;
        }
        .stMarkdown, .stText {
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize miss_count for fallback logic
# ─────────────────────────────────────────────────────────────────────────────
if "miss_count" not in st.session_state:
    st.session_state["miss_count"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# Main Chat UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ask SHA anything about Bharat 👇")
user_input = st.text_input("Your Question:")

if user_input:
    with st.spinner("SHA is thinking..."):
        relevant_docs = store.as_retriever().get_relevant_documents(user_input)
        if not relevant_docs:
            st.session_state["miss_count"] += 1
            if st.session_state["miss_count"] == 1:
                msg = "Hmm, that’s not in my memory yet. Want to try asking something else?"
            elif st.session_state["miss_count"] == 2:
                msg = "Still not finding anything—maybe Bharat didn’t include it in his resume."
            else:
                msg = "Okay, here’s my best guess… but you might want to ask Bharat directly to confirm 😄"
            st.markdown(f"**SHA:** {msg}")
        else:
            st.session_state["miss_count"] = 0
            answer = qa_chain.run(user_input)
            st.markdown(f"**SHA:** {answer}")

    # Feedback buttons
    st.markdown("#### Was this helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👍 {user_input}\n")
    if col2.button("👎"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👎 {user_input}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown(
    "<center><small>🤖 Powered by SHA — Bharat’s AI Assistant (v3.0)</small></center>",
    unsafe_allow_html=True
)
