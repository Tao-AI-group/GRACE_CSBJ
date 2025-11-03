from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from typing import List
import requests
import os

# Load environment variables from .env
load_dotenv()
# 设置API密钥和基础URL环境变量
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/embeddings"
# MODEL = "netease-youdao/bce-embedding-base_v1"
MODEL = "BAAI/bge-large-en-v1.5"

'''
class APIEmbedding(EmbeddingFunction):
    def __init__(self, model: str = "netease-youdao/bce-embedding-base_v1"):
        super().__init__()
        self.model = model
    def __call__(self, input: Documents) -> Embeddings:
        payload = {
            "model": self.model,
            "input": input,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }
        response = requests.request("POST", BASE_URL, json=payload, headers=headers).json()
        if "data" not in response:
            raise RuntimeError(response)
        embeddings = response["data"]
        # Sort resulting embeddings by index
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])

        # Return just the embeddings
        return cast(Embeddings, [result["embedding"] for result in sorted_embeddings])
''' 

class APIEmbedding(Embeddings):
    def __init__(self, model: str = "netease-youdao/bce-embedding-base_v1"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query into a vector."""
        return self._embed([text])[0]

    def _embed(self, input: List[str]) -> List[List[float]]:
        """Helper function to perform the embedding using the API."""
        payload = {
            "model": self.model,
            "input": input,
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}",
        }

        # Make the request to the embedding API
        response = requests.post(BASE_URL, json=payload, headers=headers).json()

        # Check for errors
        if "data" not in response:
            raise RuntimeError(f"Error from API: {response}")

        embeddings = response["data"]
        if embeddings is None:
            print(response)
            return None

        # Sort embeddings by index if necessary
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])

        # Extract and return only the embeddings
        return [result["embedding"] for result in sorted_embeddings]


if __name__ == "__main__":
    # print(API_KEY)
    embediing = APIEmbedding()
    result = embediing.embed_documents(texts=["nihao nihao", "buhao buhao"])
    print(result)

