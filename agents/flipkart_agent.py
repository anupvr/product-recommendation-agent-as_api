import os
from dotenv import load_dotenv
from utils.tavily_client import tavily_search
# Load API keys from .env
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


def crawl_flipkart(state: dict) -> dict:
    """
    Uses Tavily to search Flipkart.com for the user's product query.
    """
    query = state.get("query", "")
    search_query = f"{query} site:flipkart.com"
    print(f"[Flipkart Agent] 🔍 Searching Flipkart.com for: {query}")

    links = tavily_search(search_query)
    print(f"[Flipkart Agent] ✅ Found {len(links)} results")
    return {**state, "flipkart_results": links}
