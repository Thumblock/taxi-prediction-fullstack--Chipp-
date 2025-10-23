import requests
from urllib.parse import urljoin

def read_api_endpoint(endpoint="/", base_url="http://127.0.0.1:8000"):
    url = urljoin(base_url, endpoint)
    return requests.get(url, timeout=10)

def post_api_endpoint(endpoint="/", payload=None, base_url="http://127.0.0.1:8000"):
    url = urljoin(base_url, endpoint)
    return requests.post(url, json=(payload or {}), timeout=10)
