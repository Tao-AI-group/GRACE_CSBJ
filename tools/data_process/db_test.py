# This file is for testing whether we successfully add knowledge to the vectore store
import sys
sys.path.append("../..")
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import BASE_DIR

# Define the persistent directory
persistent_directory = BASE_DIR / "db/chroma_hpv_data"

# Define the embedding model
embeddings_model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
query = "Who should take HPV vaccine?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.1},
)

relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata}\n")
    print(f"doc id: {doc.id}\n")

# Retrieve the answer using the retriever
answer_result = retriever.invoke(query)
print("\n--- Answer Result ---")
for i, doc in enumerate(answer_result, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata}\n")
    print(f"doc id: {doc.id}\n")