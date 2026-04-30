from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_embedding
)

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")


# -----------------------------
# Load and Process Documents
# -----------------------------
print("Loading PDF files...")
extracted_data = load_pdf_file(data="data/")

print("Filtering metadata...")
filtered_data = filter_to_minimal_docs(extracted_data)

print("Splitting documents into chunks...")
text_chunks = text_split(filtered_data)


# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading embedding model...")
embeddings = download_embedding()


# -----------------------------
# Initialize Pinecone
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"  # must be lowercase + hyphen only


# -----------------------------
# Create Index (If Not Exists)
# -----------------------------
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match embedding model dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

print("Connecting to index...")
index = pc.Index(index_name)


# -----------------------------
# Store Vectors in Pinecone
# -----------------------------
print("Storing embeddings in Pinecone...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Indexing completed successfully ✅")