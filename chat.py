import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Paths
BASE_DIR = os.path.dirname(__file__)
KNOWLEDGE_PATH = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DIR = os.path.join(BASE_DIR, "sha_vector_store")

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Build or load FAISS vector store
if not os.path.exists(VECTOR_DIR):
    all_docs = []
    for fname in os.listdir(KNOWLEDGE_PATH):
        if fname.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_PATH, fname))
            all_docs.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(all_docs)
    FAISS.from_documents(docs, embeddings).save_local(VECTOR_DIR)

# Load vector store
store = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

# Create the QA chain
dual_llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.1)
qa_chain = RetrievalQA.from_chain_type(
    llm=dual_llm,
    chain_type="stuff",
    retriever=store.as_retriever()
)

# Handler functions
def handle_fun(q: str) -> str:
    if any(w in q for w in ["girlfriend", "relationship", "single", "wife", "crush"]):
        return "Haha, that’s classified! Bharat is more in love with data pipelines than dating apps."
    if "favorite food" in q:
        return "He runs on JSON, chai, and weekend biryani—strictly in that order."
    if "age" in q:
        return "Age is just metadata—especially if there’s no timestamp 😉."
    if any(w in q for w in ["hobbies", "free time", "weekend"]):
        return "Debugging tricky pipelines, reading AI papers, and sharing memes with fellow engineers."
    if "fruit" in q:
        return "I’d be a pineapple—tough exterior, sweet insights inside."
    if "island" in q:
        return "I’d build a coconut-powered server farm and live off solar CPU cycles."
    if "emoji" in q:
        return "🤖—because I’m building AI side-kicks for people."
    return None

def handle_recruiter(q: str) -> str:
    if any(w in q for w in ["sponsorship", "visa", "work authorization"]):
        return (
            "Bharat is on STEM OPT, authorized to work in the U.S., married and awaiting H4. "
            "Future sponsorship can be discussed based on timelines."
        )
    if "notice period" in q:
        return "About a 2-week notice—flexible for the right opportunity."
    if any(w in q for w in ["salary expectation", "current salary", "expected salary"]):
        return "I’m open and flexible—happy to align on compensation based on role and impact."
    if any(w in q for w in ["relocation", "open to relocation"]):
        return "I’m open to remote, hybrid, or relocation—whatever works best for the team."
    return None

# Additional handlers omitted for brevity; include handle_company, handle_tech, etc.

def get_response(user_input: str) -> str:
    q = user_input.lower()
    for fn in [handle_fun, handle_recruiter]:  # extend list with other handlers
        resp = fn(q)
        if resp:
            return resp
    # Fallback to RAG
docs = store.as_retriever().get_relevant_documents(user_input)
    if docs:
        return qa_chain.run(user_input)
    return "My circuits are tickled—but I don’t have that one yet! Try another question 😊"
