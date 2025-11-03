import os
import json
import sys
import time
sys.path.append("../..")
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter, RecursiveJsonSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from config import BASE_DIR
# Define the persistent storage directory
persistent_directory = BASE_DIR / "db/chroma_hpv_data"

# Define the embedding model
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Initialize the text splitter
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=800, chunk_overlap=100)
json_splitter = RecursiveJsonSplitter(max_chunk_size=800)

# Directory paths for text and JSON files
text_files_dir = BASE_DIR / "data/example_articles"
json_files_dir = BASE_DIR / "data/example_articles"


# Detect repeated id in Chroma
def exsit_in_Chroma(collection, embedding_id):
    # Check if the ID already exists in the collection
    results = collection.get(ids=[embedding_id])

    # If the results contain the embedding ID, it means it already exists
    if results and embedding_id in results["ids"]:
        return True
    else:   
        return False


# Function to read and chunk text files
def read_and_chunk_text_files(directory):
    documents = []
     # Read the text content from each file and store it with metadata
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                filepath = os.path.join(root, filename)
                loader = TextLoader(filepath)
                book_docs = loader.load()
                for doc in book_docs:
                    # Add metadata to each document indicating its source
                    doc.metadata = {"source": filename, "type":"text"}
                    documents.append(doc)
    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=' ')
    docs = text_splitter.split_documents(documents)
    return docs

def read_and_chunk_json_files(directory):
    QA_documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator=' ')
    # Read the text content from each file and store it with metadata
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                # print("Processing file: ", filepath)
                with open(filepath, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    for qa_pair in data:
                        # Extract fields from JSON
                        question = qa_pair.get("question", "")
                        answer = qa_pair.get("answer", "").strip()
                        source = qa_pair.get("source", "")
                        doc_id = qa_pair.get("id", filename)
                        answer_chunked = text_splitter.split_text(answer)
                        for i in range(len(answer_chunked)):
                            question_document = Document(
                                page_content=f"[QUESTION] {question} [ANSWER] {answer_chunked[i]}",
                                metadata={"source": source,
                                        "id":f"{doc_id}_{i}"}
                            )
                        QA_documents.append(question_document)
    # Split the documents into chunks
    return QA_documents

qa_documents = read_and_chunk_json_files(json_files_dir)
text_documents = read_and_chunk_text_files(text_files_dir)

db = Chroma(
    persist_directory=persistent_directory, 
    embedding_function=embeddings
)

all_documents = qa_documents + text_documents

for i, doc in enumerate(all_documents):
    doc = [doc]
    print(f"Processing {i} documents")
    try:
        db.add_documents(doc)
    except ValueError as e:
        print(f"doc {doc[0].metadata['source']} is not good")
    time.sleep(0.3)
