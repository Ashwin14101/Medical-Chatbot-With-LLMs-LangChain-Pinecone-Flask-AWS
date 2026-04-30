import streamlit as st
import os
from dotenv import load_dotenv

from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import system_prompt


# Load environment variables
load_dotenv()


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered"
)


# ---------------------------------------------------
# Load RAG Chain (cached — loads only once)
# ---------------------------------------------------
@st.cache_resource
def load_rag_chain():

    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")

    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["GOOGLE_API_KEY"]   = GOOGLE_API_KEY

    # Load embedding model
    embeddings = download_embedding()

    # Connect to existing Pinecone index
    docsearch = PineconeVectorStore.from_existing_index(
        index_name="medical-bot",
        embedding=embeddings
    )

    # Create retriever
    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Initialize Gemini LLM
    chatModel = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Build RAG chain
    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


# ---------------------------------------------------
# App Header
# ---------------------------------------------------
st.title("🏥 Medical Chatbot")
st.caption("Running with RAG • Gemini 2.5 Flash • Pinecone Vector DB")


# ---------------------------------------------------
# Chat History (session state)
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------
# Chat Input
# ---------------------------------------------------
if user_input := st.chat_input("What is on your mind?"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get RAG response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            rag_chain = load_rag_chain()
            response  = rag_chain.invoke({"input": user_input})
            answer    = response["answer"]
        st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
