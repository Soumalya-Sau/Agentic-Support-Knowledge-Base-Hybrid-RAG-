import requests

class MCPToolClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def rag_search(self, query: str):
        resp = requests.post(
            f"{self.base_url}/rag_search",
            json={"query": query},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()

    def best_source(self, results: list):
        resp = requests.post(
            f"{self.base_url}/best_source",
            json=results,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["sources"]

