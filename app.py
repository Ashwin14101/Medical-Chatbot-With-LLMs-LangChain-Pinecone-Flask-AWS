from flask import Flask, render_template,jsonify,request
from src.helper import download_embedding
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()

# Intialize app.py

app = Flask(__name__) # Flask intialization code

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY 


# Load embedding model and Try to load the index

embeddings = download_embedding()

index_name = "medical-bot"
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings
)

retriever = docsearch.as_retriever(search_type = "similarity", search_kwargs = {"k":3})


chatModel = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
# Basic route / Default route

@app.route("/")
def index():
    return render_template('chat.html') # Open Application render the chatbot.html

# route will be executed, he will get msg, whatever msg user sending i am receiving it here input and then i am executing in my rag chain 
@app.route("/chat",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input":msg})
    print("Response: ",response["answer"])
    return str(response["answer"])


# To execute the app this is the code
if __name__ == '__main__': 
    app.run(host="0.0.0.0",port=8080, debug = True) # Here I am running on local host to execute the app and here we are run on local host port 8080