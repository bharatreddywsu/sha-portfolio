import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ─────────────────────────────────────────────────────────────────────────────
# Load env + secrets
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────────────────────────────────────
# Build or load the FAISS vector store
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
KNOWLEDGE    = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_STORE = os.path.join(BASE_DIR, "sha_vector_store")

emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
if not os.path.exists(VECTOR_STORE):
    docs = []
    for f in os.listdir(KNOWLEDGE):
        if f.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE, f))
            docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)
    FAISS.from_documents(chunks, emb).save_local(VECTOR_STORE)

store    = FAISS.load_local(VECTOR_STORE, emb, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1),
    chain_type="stuff",
    retriever=store.as_retriever()
)

# ─────────────────────────────────────────────────────────────────────────────
# Handlers for recruiter, company, tech, etc.
# ─────────────────────────────────────────────────────────────────────────────
def handle_fun(q):
    if any(w in q for w in ["girlfriend", "relationship", "single", "wife", "crush"]):
        return "Haha, that’s classified! Bharat is more in love with data pipelines than dating apps."
    if "favorite food" in q:
        return "He runs on JSON, chai, and weekend biryani—strictly in that order."
    if "age" in q:
        return "Age is just metadata—especially if there’s no timestamp 😉."
    if any(w in q for w in ["hobbies", "free time", "weekend"]):
        return "Debugging tricky pipelines, reading AI papers, and sharing memes with fellow engineers."
    # extra funny toss-ups:
    if "fruit" in q:
        return "I’d be a pineapple—tough exterior, sweet insights inside."
    if "island" in q:
        return "I’d build a coconut-powered server farm and live off solar CPU cycles."
    if "emoji" in q:
        return "🤖—because I’m building AI side-kicks for people."
    # catch-all fun fallback
    return "Beep boop… I’m still learning how to crack that one! Ask me something else."

def handle_recruiter(q):
    if any(w in q for w in ["sponsorship", "visa", "work authorization"]):
        return (
            "Bharat is on STEM OPT, authorized to work in the U.S., and married while waiting for his H4. "
            "Future sponsorship may be needed depending on the hire timeline."
        )
    if "notice period" in q:
        return "He can join with about a 2-week notice—flexible for the right opportunity."
    if any(w in q for w in ["salary expectation", "current salary", "expected salary"]):
        return "Bharat is open and flexible—happy to discuss compensation based on role, location, and growth potential."
    if any(w in q for w in ["relocation", "open to relocation"]):
        return "He’s open to relocation, hybrid, or remote roles—whatever best serves the team."
    return None

def handle_company(q):
    if any(t in q for t in ["current company", "working now"]):
        return "Bharat is currently at **KLA** as a Data Engineer since May 2024."
    if "dentsu" in q:
        return ("At Dentsu (May 2020–May 2022), he built pipelines using ADF, Spark, Kafka, "
                "and Power BI. RAG/LLM tech wasn’t in scope then.")
    if any(t in q for t in ["wichita state", "master’s", "mscs"]):
        return "He completed his Master’s in CS at Wichita State (Aug 2022–May 2024)."
    if "fagron" in q:
        return ("At Fagron (Dec 2022–Apr 2024), he built HIPAA-compliant ETL on AWS Glue/Redshift, "
                "and added a light-verification system post-prod.")
    return None

def handle_tech(q):
    if "rag" in q or "retrieval augmented generation" in q:
        return "He set up FAISS + OpenAI embeddings at KLA for wafer-defect Q&A."
    if "llm" in q or "large language model" in q:
        return "He uses LLMs in Databricks for code suggestions, debugging, and context-aware Q&A."
    if "airflow" in q:
        return "Built DAGs with sensors, retries, SLA alerts, and email notifications."
    if "kafka" in q or "streaming" in q:
        return "Wrote PySpark Structured Streaming consumers with exactly-once semantics."
    return None

def handle_education(q):
    if any(w in q for w in ["master", "wichita state"]):
        return "Master’s in CS from Wichita State University (Aug 2022–May 2024)."
    if "bachelor" in q:
        return "Bachelor’s in Engineering (CS) in 2018 with early Python/OpenCV projects."
    if "certification" in q:
        return ("Certs: AWS Solutions Architect, Databricks Data Engineer, Snowflake Data Engineer, "
                "and Python & SQL certifications.")
    return None

def handle_projects(q):
    if "sawyer" in q or "pybullet" in q:
        return "Built a Sawyer Arm sim in PyBullet (Jan 2023–Mar 2023) with IK and reward-based grasp tests."
    if "face recognition" in q:
        return "Developed a Raspberry Pi face-recognition system using OpenCV (2018)."
    return None

def handle_volunteer(q):
    if "guinness" in q or "wheelchair" in q:
        return "Coordinated a Guinness World Record wheelchair event at Vel Tech (May 2019)."
    return None

def handle_behavioral(q):
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return ("Sure—let me know whether you want a pipeline-optimization story, a leadership example, "
                "or something else.")
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main chat loop + UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ask SHA anything about Bharat 👇")
user_input = st.text_input("Your Question:")

if user_input:
    q = user_input.lower()
    # 1) Try all the handlers in order
    for fn in [handle_fun, handle_recruiter, handle_company,
               handle_tech, handle_education, handle_projects,
               handle_volunteer, handle_behavioral]:
        resp = fn(q)
        if resp:
            st.markdown(f"**SHA:** {resp}")
            break
    else:
        # 2) Fall back to RAG
        docs = store.as_retriever().get_relevant_documents(user_input)
        if docs:
            st.markdown("**SHA:** " + qa_chain.run(user_input))
        else:
            # 3) Bonus funny default if truly nothing matches
            st.markdown("**SHA:** My circuits are tickled—but I don’t have that one yet! Try another question 😊")

    # 4) Feedback logging
    st.markdown("#### Was this helpful?")
    c1, c2 = st.columns(2)
    if c1.button("👍"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👍 {user_input}\n")
    if c2.button("👎"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👎 {user_input}\n")
