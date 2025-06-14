import os
import base64
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Path setup for knowledge base (relative to repo root)
# ─────────────────────────────────────────────────────────────────────────────
knowledge_path = "knowledge_base"

# Pull your OpenAI key from Streamlit Cloud’s Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

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
# Load all PDFs from knowledge_base folder and build FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_dir = "sha_vector_store"

if not os.path.exists(vector_dir):
    docs = []
    for file in os.listdir(knowledge_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(knowledge_path, file))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    FAISS.from_documents(split_docs, embeddings).save_local(vector_dir)

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

# (Any additional logic or UI input handling continues here)
