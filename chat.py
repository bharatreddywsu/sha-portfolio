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
# Build or load the FAISS vector store
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

# ─────────────────────────────────────────────────────────────────────────────
# Handlers for different question types
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
    if any(w in q for w in ["calm", "composed", "personality"]):
        return "Bharat is calm, composed, and tackles challenges one data row at a time."
    return None

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
    if any(w in q for w in ["open to relocation", "relocation"]):
        return "He’s open to relocation, hybrid, or remote roles—whatever best serves the team."
    if "remote" in q:
        return "Absolutely—Bharat excels in both remote and in-office environments."
    if "experience" in q and "data engineer" in q:
        return "Bharat has 5+ years of data engineering experience across cloud, streaming, and analytics platforms."
    return None

def handle_company(q):
    if any(t in q for t in ["current company", "present company", "where are you working", "working now"]):
        return "Bharat is currently working at **KLA** as a Data Engineer since May 2024."
    if "dentsu" in q:
        return (
            "At Dentsu (May 2020–May 2022), Bharat built data pipelines using Azure Data Factory, Spark, Kafka, "
            "and Power BI dashboards. RAG and LLM tech weren't used then."
        )
    if any(t in q for t in ["wichita state", "master", "mscs"]):
        return (
            "Bharat completed his Master’s in Computer Science at Wichita State University (Aug 2022–May 2024), "
            "including BI Developer and Post-Production Lead internships."
        )
    if "fagron" in q:
        return (
            "At Fagron (Dec 2022–Apr 2024), he built ETL on AWS Glue, Redshift, and Snowflake, supported HIPAA/FDA compliance, "
            "and introduced a light-based verification system in post-production."
        )
    if "kla" in q:
        return (
            "At KLA (May 2024–Present), Bharat builds RAG pipelines using OpenAI and Azure Cognitive Search, "
            "implements LLM assistants in Databricks, and designs real-time analytics for wafer defect detection."
        )
    return None

def handle_tech(q):
    if "rag" in q or "retrieval augmented generation" in q:
        return (
            "Bharat has hands-on RAG experience at KLA, where he set up vector search with OpenAI embeddings "
            "and Azure Cognitive Search for semiconductor defect Q&A."
        )
    if "llm" in q or "large language model" in q:
        return (
            "He uses LLMs at KLA inside Databricks for code suggestions, pipeline debugging, "
            "and contextual Q&A since late 2023."
        )
    if "optimize" in q and "databricks" in q:
        return (
            "He optimized Databricks pipelines by tuning Spark partition sizes, caching hot tables, "
            "and refactoring PySpark jobs, reducing runtime by 40%."
        )
    if "kafka" in q or "streaming" in q:
        return (
            "At Dentsu and KLA, he used Kafka for real-time data ingestion, wrote consumers in "
            "PySpark Structured Streaming, and ensured exactly-once delivery semantics."
        )
    return None

def handle_education(q):
    if any(w in q for w in ["master", "wichita state"]):
        return "Bharat completed his Master’s in Computer Science at Wichita State University (Aug 2022–May 2024)."
    if "bachelor" in q:
        return "He earned his Bachelor’s in Engineering (Computer Science) in 2018 and built early projects like Raspberry Pi face recognition."
    if any(w in q for w in ["certification", "certified"]):
        return "He holds certifications: AWS Solutions Architect, Databricks Certified Data Engineer, Snowflake Data Engineer, and Python & SQL certifications."
    return None

def handle_projects(q):
    if any(w in q for w in ["sawyer", "pybullet", "grasping"]):
        return "He built a Sawyer Arm simulation in PyBullet (Jan 2023–Mar 2023), implementing inverse kinematics and reward-based grasp testing."
    if any(w in q for w in ["face recognition", "raspberry pi"]):
        return "He developed a Raspberry Pi face recognition system using Python and OpenCV for home security (2018)."
    if "expense tracker" in q or "hackathon" in q:
        return "He led a weekend hackathon building a Python expense tracker with Power BI visuals to alert on unusual spending (Mar 2023)."
    return None

def handle_volunteer(q):
    if any(w in q for w in ["guinness", "wheelchair", "coordinator"]):
        return "As a coordinator at Vel Tech (May 2019), he helped plan a Guinness World Record wheelchair event raising disability awareness."
    return None

def handle_behavioral(q):
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return "Sure—tell me which scenario you'd like, such as optimizing pipeline performance or leading a project, and I'll share the details."
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main interaction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ask SHA anything about Bharat 👇")
user_input = st.text_input("Your Question:")

if user_input:
    q_lower = user_input.lower()
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
            st.session_state["miss_count"] = st.session_state.get("miss_count", 0) + 1
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

    # Feedback buttons
    st.markdown("#### Was this helpful?")
    col1, col2 = st.columns(2)
    if col1.button("👍"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👍 {user_input}\n")
    if col2.button("👎"):
        with open("questions_log.txt", "a") as f:
            f.write(f"👎 {user_input}\n")
