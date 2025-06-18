import os
from dotenv import load_dotenv
from utils.tavily_client import tavily_search

def crawl_amazon(state: dict) -> dict:
    """
    Uses Tavily to search Amazon.in for the user's product query.
    """
    query = state.get("query", "")
    search_query = f"{query} site:amazon.in"
    print(f"[Amazon Agent] 🔍 Searching Amazon.in for: {query}")

    links = tavily_search(search_query)
    print(f"[Amazon Agent] ✅ Found {len(links)} results")
    return {**state, "amazon_results": links}
