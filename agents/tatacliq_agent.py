import os
from dotenv import load_dotenv
from utils.tavily_client import tavily_search
# Load API keys from .env
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def crawl_tatacliq(state: dict) -> dict:
    """
    Uses Tavily to search TataCliq.com for the user's product query.
    """
    query = state.get("query", "")
    search_query = f"{query} site:tatacliq.com"
    print(f"[TataCliq Agent] 🔍 Searching TataCliq.com for: {query}")

    links = tavily_search(search_query)
    print(f"[TataCliq Agent] ✅ Found {len(links)} results")
    return {**state, "tatacliq_results": links}
    