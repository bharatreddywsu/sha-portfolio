import os
from dotenv import load_dotenv

# grab our secret
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# debugging info
print("cwd:", os.getcwd())
print("looking for:", os.path.abspath("resume/bharat_resume.pdf"))

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# read the PDF
loader = PyPDFLoader("resume/bharat_resume.pdf")
pages = loader.load()

# chop it into bite-sized bits
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(pages)

# turn text into numeric memory
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(docs, embeddings)

# save our vector index
db.save_local("sha_vector_store")

print("✅ done. resume is embedded and ready to use.")
