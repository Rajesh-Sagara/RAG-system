from dotenv import load_dotenv

import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# Ensure data directory and file exist
data_path = os.path.join(os.path.dirname(__file__), "data", "sample.txt")

if not os.path.exists(data_path):
    print(f"Error: {data_path} not found")
    exit(1)

loader = TextLoader(data_path)
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)

db.save_local("vectorstore")