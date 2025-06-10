import os
import base64
import streamlit as st

# Pull your OpenAI key from Streamlit Cloud’s Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SHA — Bharat’s AI Assistant",
    page_icon="👩‍🚀",
    layout="centered",
)

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
                <h2 style='color:#E0E0E0; margin-top:10px;'>SHA — Your AI Companion</h2>
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
# Build or load the FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_dir = "sha_vector_store"

def build_vector_store():
    loader = PyPDFLoader("resume/bharat_resume.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    FAISS.from_documents(docs, embeddings).save_local(vector_dir)

try:
    store = FAISS.load_local(
        vector_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )
except Exception:
    build_vector_store()
    store = FAISS.load_local(
        vector_dir,
        embeddings,
        allow_dangerous_deserialization=True
    )

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1),
    chain_type="stuff",
    retriever=store.as_retriever()
)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize miss_count for fallback logic
# ─────────────────────────────────────────────────────────────────────────────
if "miss_count" not in st.session_state:
    st.session_state["miss_count"] = 0

# ─────────────────────────────────────────────────────────────────────────────
# Define all handlers (fun, recruiter, company, tech, education, projects,
# volunteer, behavioral) exactly as before...
# [handlers omitted here for brevity—they remain unchanged]
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Main interaction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ask SHA anything about Bharat 👇")
user_input = st.text_input("Your Question:")

if user_input:
    q_lower = user_input.lower()
    # iterate through your handlers in order
    for handler in [
        handle_fun,
        handle_recruiter,
        handle_company,
        handle_tech,
        handle_education,
        handle_projects,
        handle_volunteer,
        handle_behavioral,
    ]:
        resp = handler(q_lower)
        if resp:
            st.markdown(f"**SHA:** {resp}")
            break
    else:
        docs = store.as_retriever().get_relevant_documents(user_input)
        if not docs:
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
            with st.spinner("SHA is thinking..."):
                answer = qa_chain.run(user_input)
            st.markdown(f"**SHA:** {answer}")
