import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# load secret
load_dotenv()
key = os.getenv("OPENAI_API_KEY")

# init session state counter if missing
if "miss_count" not in st.session_state:
    st.session_state.miss_count = 0

# basic UI
st.title("SHA — Your AI Assistant")
prompt = st.text_input("Ask me anything about Bharat")

# load SHA’s memory
store = FAISS.load_local(
    "sha_vector_store",
    OpenAIEmbeddings(openai_api_key=key),
    allow_dangerous_deserialization=True
)
retriever = store.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=key, temperature=0),
    chain_type="stuff",
    retriever=retriever
)

if prompt:
    # first, find docs that match
    docs = retriever.get_relevant_documents(prompt)

    if not docs:
        # increment our “miss” counter
        st.session_state.miss_count += 1

        # decide how to reply
        if st.session_state.miss_count == 1:
            answer = (
                "I’m sorry, I don’t have info on that in my resume. "
                "Could you try asking something else?"
            )
        elif st.session_state.miss_count == 2:
            answer = (
                "Still not seeing that detail—maybe it’s not in my data. "
                "Feel free to ask another question."
            )
        else:
            answer = (
                "Okay, here’s my best guess based on similar experience: "
                "[Your inferred answer goes here]."
            )

    else:
        # reset counter on a hit
        st.session_state.miss_count = 0
        with st.spinner("SHA is thinking..."):
            answer = qa.run(prompt)

    st.markdown("**SHA:** " + answer)
