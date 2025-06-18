import os
from dotenv import load_dotenv
from tavily import TavilyClient
from typing import List

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Tavily client
tavily = TavilyClient(api_key=TAVILY_API_KEY)

def tavily_search(query: str, max_results: int = 5) -> List[str]:
    """
    Performs a Tavily web search and returns a list of result URLs.
    """
    try:
        results = tavily.search(query=query, max_results=max_results)
        return [item["url"] for item in results.get("results", [])]
    except Exception as e:
        print(f"❌ Tavily search failed: {e}")
        return []