import os
import requests
from dotenv import load_dotenv
from typing import List, Dict, Any
# Load environment variables from .env
load_dotenv()
# 设置API密钥和基础URL环境变量
API_KEY = os.getenv("SILICONFLOW_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1/rerank"
MODEL = "BAAI/bge-reranker-v2-m3"

class APIReranker:
    def __init__(self):

        self.url = BASE_URL
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

    def rerank(self, query: str, documents: List[str], top_n: int = 4, 
               return_documents: bool = True, max_chunks_per_doc: int = 123, overlap_tokens: int = 79) -> Dict[str, Any]:
        payload = {
            "model": MODEL,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens
        }

        response = requests.post(self.url, json=payload, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
if __name__ == "__main__":
    # Example usage
    reranker = APIReranker()
    result = reranker.rerank(
        query="Apple",
        documents=["苹果", "香蕉", "水果", "蔬菜"]
    )

    print(result)
