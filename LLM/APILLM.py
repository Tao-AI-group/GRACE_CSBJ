from langchain.llms.base import LLM
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.llms.utils import enforce_stop_tokens
import requests
import os

# Load environment variables from .env
load_dotenv()
# 设置API密钥和基础URL环境变量
# API_KEY = os.getenv("SILICONFLOW_API_KEY")
# BASE_URL = "https://api.siliconflow.cn/v1/chat/completions"
# MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "cognitivecomputations/dolphin3.0-mistral-24b:free"


class SiliconFlow(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def siliconflow_completions(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }
        

        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: list = None, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> str:
        response = self.siliconflow_completions(model=model, prompt=prompt)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response

class OpenRouter(LLM):

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def openrouter_completions(self, model: str, prompt: str) -> str:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=API_KEY,
            )

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="cognitivecomputations/dolphin3.0-mistral-24b:free",
            messages=[
                {
                "role": "user",
                "content": prompt
                }
            ]
        )
                    
        return completion.choices[0].message.content

    def _call(self, prompt: str, stop: list = None, model: str = "cognitivecomputations/dolphin3.0-mistral-24b:free") -> str:
        response = self.openrouter_completions(model=model, prompt=prompt)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response

class MayoOpenAI(LLM):
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def siliconflow_completions(self, model: str, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {API_KEY}"
        }
        

        response = requests.post(BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _call(self, prompt: str, stop: list = None, model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct") -> str:
        response = self.siliconflow_completions(model=model, prompt=prompt)
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        return response


if __name__ == "__main__":
    # print(API_KEY)
    llm = OpenRouter()
    response = llm._call(prompt="Which team did Jordan win NBA finals?")
    print(response)

