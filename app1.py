# Import required libraries
from flask import Flask, render_template, jsonify, request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
import os


# Load environment variables from .env file
load_dotenv()


# -----------------------------
# Initialize Flask Application
# -----------------------------
app = Flask(__name__)   # Flask app initialization


# -----------------------------
# Load API Keys from .env file
# -----------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Set environment variables so LangChain and Pinecone can access them
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY 


# ---------------------------------------------------
# Load embedding model (HuggingFace embedding model)
# ---------------------------------------------------
embeddings = download_embedding()


# ---------------------------------------------------
# Connect to existing Pinecone index
# ---------------------------------------------------
index_name = "medical-bot"

# Load the already stored vectors from Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# ---------------------------------------------------
# Create Retriever from Pinecone vector store
# ---------------------------------------------------
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k":3}   # retrieve top 3 similar documents
)


# ---------------------------------------------------
# Initialize Gemini LLM
# ---------------------------------------------------
chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


# ---------------------------------------------------
# Create Prompt Template
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),   # system instructions
        ("human","{input}")          # user query placeholder
    ]
)


# ---------------------------------------------------
# Create Question Answer Chain
# ---------------------------------------------------
question_answer_chain = create_stuff_documents_chain(
    chatModel, 
    prompt
)


# ---------------------------------------------------
# Create Retrieval Augmented Generation (RAG) Chain
# ---------------------------------------------------
rag_chain = create_retrieval_chain(
    retriever, 
    question_answer_chain
)


# ---------------------------------------------------
# Default Route -> Open chatbot interface
# ---------------------------------------------------
@app.route("/")
def index():
    return render_template('chat.html')   # loads chatbot UI


# ---------------------------------------------------
# Chat API Route
# Receives user message from frontend
# Sends message to RAG chain
# Returns response
# ---------------------------------------------------
@app.route("/chat",methods=["POST"])
def chat():

    # Receive message sent from frontend
    data = request.get_json()
    msg = data["message"]

    input = msg

    print("User Question:", input)

    # Send question to RAG pipeline
    response = rag_chain.invoke({"input":msg})

    print("Response:", response["answer"])

    # Return response back to frontend
    return jsonify({
        "answer": response["answer"]
    })


# ---------------------------------------------------
# Run Flask Server
# ---------------------------------------------------
if __name__ == '__main__': 

    # Run server on port 8080
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )