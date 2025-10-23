import os
from urllib.parse import urljoin

import requests

# Build a full URL from base_url + endpoint and perform a GET with a 10s timeout.
def read_api_endpoint(endpoint="/", base_url="http://127.0.0.1:8000"):
    url = urljoin(base_url, endpoint)
    return requests.get(url, timeout=10)

#Build a full URL and POST a JSON payload (None -> {}), with a 10s timeout.
def post_api_endpoint(endpoint="/", payload=None, base_url="http://127.0.0.1:8000"):
    url = urljoin(base_url, endpoint)
    return requests.post(url, json=(payload or {}), timeout=10)

#Allow TAXIPRED_API_BASE to override the API base URL at runtime.
def get_base_url(default: str = "http://127.0.0.1:8000") -> str:
    return os.getenv("TAXIPRED_API_BASE", default)

#Convenience wrapper for the most common call: POST /predict with JSON payload.
def post_predict(payload: dict, endpoint: str = "/predict", base_url: str | None = None):
    return post_api_endpoint(
        endpoint=endpoint,
        payload=payload,
        base_url=(base_url or get_base_url()),
    )
