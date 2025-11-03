import os
import time
import requests
from urllib.parse import urljoin
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

load_dotenv()

# API Configuration
APIGEE_BASE_URL = 'https://internal.mcc.api.mayo.edu'
API_VERSION = '2024-02-01'
client_id = os.getenv('CLIENT_ID')
secret_id = os.getenv('SECRET_ID')

class TokenManager:
    def __init__(self, client_id: str, secret_id: str, token_url: str, expiry_buffer: int = 60):
        self.client_id = client_id
        self.secret_id = secret_id
        self.token_url = token_url
        self.expiry_buffer = expiry_buffer  # Buffer time before real expiration
        self.token: Optional[str] = None
        self.expires_at: Optional[float] = None

    def _fetch_new_token(self) -> str:
        payload = f'grant_type=client_credentials&client_id={self.client_id}&client_secret={self.secret_id}'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(self.token_url, headers=headers, data=payload)

        if response.status_code != 200:
            raise RuntimeError(f"Token request failed: {response.status_code}, {response.text}")
        
        data = response.json()
        self.token = data['access_token']
        self.expires_at = time.time() + data.get('expires_in', 3600) - self.expiry_buffer
        return self.token

    def get_token(self) -> str:
        if not self.token or not self.expires_at or time.time() >= self.expires_at:
            return self._fetch_new_token()
        return self.token

# Create token manager instance
token_manager = TokenManager(
    client_id=client_id,
    secret_id=secret_id,
    token_url="https://internal.mcc.api.mayo.edu/oauth/token"
)

def query_openai(prompt: str, engine: str = "gpt-4o", max_tokens: int = 1500, temperature: float = 0.7, 
                 top_p: float = 0.9, presence_penalty: float = 0.6, frequency_penalty: float = 0.2, 
                 timeout: int = 35) -> Dict[str, Any]:
    """
    Call OpenAI API to get response, retry once if token is expired.
    """
    def make_request(token: str) -> requests.Response:
        api_openai_url = urljoin(
            APIGEE_BASE_URL, 
            f"/ai-factory-product-build-azure-openai/openai/deployments/{engine}/chat/completions?api-version={API_VERSION}"
        )
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        endpoint_payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
        return requests.post(api_openai_url, headers=headers, json=endpoint_payload, timeout=timeout)

    token = token_manager.get_token()
    response = make_request(token)

    if response.status_code == 401:
        # Unauthorized, try refreshing token once
        token = token_manager._fetch_new_token()
        response = make_request(token)

    if response.status_code == 200:
        return response.json()
    else:
        raise RuntimeError(f"OpenAI API request failed: {response.status_code}, Response: {response.text}")

class MayoLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        response = query_openai(prompt, **kwargs)
        try:
            completion_text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Response data format error: {response}")

        if stop:
            for stop_token in stop:
                if stop_token in completion_text:
                    completion_text = completion_text.split(stop_token)[0]
        return completion_text

    @property
    def _llm_type(self) -> str:
        return "custom_openai_api_llm"
