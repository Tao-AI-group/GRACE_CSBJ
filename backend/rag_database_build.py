import os
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import BASE_DIR

ROOT_DIRS = [
    BASE_DIR / "data/example_articles",
]

def load_qa_from_json(json_path):
    docs = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return docs  # Skip non-list structures

        for item in data:
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()
            source = item.get("source", "").strip()
            qa_id = item.get("id", "").strip()

            if not question or not answer:
                continue  # Skip empty entries

            content = f"Q: {question}\nA: {answer}"
            metadata = {
                "source": source or os.path.basename(json_path),
                "qa_id": qa_id or os.path.basename(json_path)
            }

            doc = Document(page_content=content, metadata=metadata)
            docs.append(doc)
    except Exception as e:
        print(f"⚠️ Failed to read: {json_path}, error: {e}")
    
    return docs

def collect_all_json_files(root_dirs):
    all_json_files = []
    for root_dir in root_dirs:
        for path in Path(root_dir).rglob("*.json"):
            all_json_files.append(str(path))
    return all_json_files

def build_vectorstore():
    all_docs = []
    ids = []

    json_files = collect_all_json_files(ROOT_DIRS)
    print(f"📄 Found {len(json_files)} JSON files")

    for json_path in json_files:
        docs = load_qa_from_json(json_path)
        all_docs.extend(docs)
        ids.extend([doc.metadata.get("qa_id", f"id_{i}") for i, doc in enumerate(docs)])

    print(f"✅ Prepared {len(all_docs)} documents")

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


    vector_store = Chroma(
        collection_name="hpv_chatbot",
        embedding_function=embeddings,
        persist_directory=BASE_DIR / "db",
    )

    vector_store.add_documents(documents=all_docs, ids=ids)

if __name__ == "__main__":
    build_vectorstore()
