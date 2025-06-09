import os
import base64
from dotenv import load_dotenv
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# ─────────────────────────────────────────────────────────────────────────────
# Load environment and set page
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
                <h2 style='color:#E0E0E0; margin-top:10px;'>SHA — Bharat's AI Companion</h2>
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
# Load SHA’s memory (vector store) and QA chain
# ─────────────────────────────────────────────────────────────────────────────
store = FAISS.load_local(
    "sha_vector_store",
    OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
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
# Handler: Fun / Personal Questions
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

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Recruiter Screening Questions
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Company Timeline & Roles
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Tech Deep-Dive Questions
# ─────────────────────────────────────────────────────────────────────────────
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
    if "airflow" in q or "etl" in q:
        return (
            "He built DAGs in Apache Airflow for orchestration, with sensors, retries, and SLA alerts "
            "across Fagron and Dentsu pipelines."
        )
    if any(w in q for w in ["jenkins", "terraform", "docker", "ci/cd"]):
        return (
            "He automated deployments using Jenkins, Terraform for infra-as-code, and containerized jobs with Docker."
        )
    if any(w in q for w in ["aws", "azure", "gcp"]):
        return (
            "He’s worked with AWS (Glue, S3, Lambda), Azure (Data Factory, Synapse), and GCP (BigQuery) "
            "depending on project needs."
        )
    if any(w in q for w in ["power bi", "tableau", "looker"]):
        return (
            "He’s proficient in Power BI, Tableau, and Looker—built many dashboards for KPIs, operations, and compliance."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Education & Certifications
# ─────────────────────────────────────────────────────────────────────────────
def handle_education(q):
    if any(w in q for w in ["master", "wichita state"]):
        return (
            "Bharat completed his Master’s in Computer Science at Wichita State University (Aug 2022–May 2024)."
        )
    if "bachelor" in q:
        return (
            "He earned his Bachelor’s in Engineering (Computer Science) in 2018 and built early projects like Raspberry Pi face recognition."
        )
    if any(w in q for w in ["certification", "certified"]):
        return (
            "He holds certifications: AWS Solutions Architect, Databricks Certified Data Engineer, "
            "Snowflake Data Engineer, and Python & SQL certifications."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Projects
# ─────────────────────────────────────────────────────────────────────────────
def handle_projects(q):
    if any(w in q for w in ["sawyer", "pybullet", "grasping"]):
        return (
            "He built a Sawyer Arm simulation in PyBullet (Jan 2023–Mar 2023), implementing inverse kinematics and reward-based grasp testing."
        )
    if any(w in q for w in ["face recognition", "raspberry pi"]):
        return (
            "He developed a Raspberry Pi face recognition system using Python and OpenCV for home security (2018)."
        )
    if "expense tracker" in q or "hackathon" in q:
        return (
            "He led a weekend hackathon building a Python expense tracker with Power BI visuals to alert on unusual spending (Mar 2023)."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Volunteer & Leadership
# ─────────────────────────────────────────────────────────────────────────────
def handle_volunteer(q):
    if any(w in q for w in ["guinness", "wheelchair", "coordinator"]):
        return (
            "As a coordinator at Vel Tech (May 2019), he helped plan a Guinness World Record wheelchair event raising disability awareness."
        )
    if any(w in q for w in ["uvh", "ethics", "mindfulness"]):
        return (
            "He volunteered in the UHV Cell (2018–19), organizing sessions on ethics, empathy, and mindfulness."
        )
    if "ieee" in q:
        return (
            "He was an IEEE member (2017–18), attending workshops and tech talks to stay connected beyond class."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Handler: Behavioral / Situational
# ─────────────────────────────────────────────────────────────────────────────
def handle_behavioral(q):
    if any(w in q for w in ["tell me about a time", "example of", "how did you"]):
        return (
            "Sure—tell me which scenario you'd like, such as optimizing pipeline performance or leading a project, and I'll share the details."
        )
    if any(w in q for w in ["leadership", "team", "collaborate"]):
        return (
            "He’s led cross-functional teams at Fagron and KLA, aligning product, QA, and engineering to deliver on time."
        )
    if any(w in q for w in ["challenge", "problem"]):
        return (
            "He faced a challenge optimizing Spark jobs at KLA—rewrote complex joins and used partition pruning to cut runtime by 60%."
        )
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Main interaction
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### Ask SHA anything about Bharat 👇")
user_input = st.text_input("Your Question:")

if user_input:
    q_lower = user_input.lower()
    # Try each handler
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
        # Fallback to vector QA
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
