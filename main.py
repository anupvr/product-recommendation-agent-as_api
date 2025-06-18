
import os
from dotenv import load_dotenv
from langchain.callbacks import LangChainTracer
from graph.workflow_graph import build_graph

# Load environment variables
load_dotenv()

# Enable LangSmith V2 Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"

def main():
    print("🛒 Product Recommendation System (LangGraph + Tavily + OpenAI + LangSmith)")
    user_query = input("🔍 Enter your product query: ")

    initial_state = {
        "query": user_query,
        "amazon_results": [],
        "flipkart_results": [],
        "tatacliq_results": [],
        "recommendations": [],
        "summary": ""
    }

    # Set up LangSmith tracer with dynamic tagging based on query
    tracer = LangChainTracer()
    tagified_query = user_query.lower().replace(" ", "-").replace(":", "").replace("'", "")

    config = {
        "callbacks": [tracer],
        "metadata": {"user_query": user_query},
        "run_name": f"Recommendation: {user_query}",
        "tags": ["multi-agent", tagified_query]  # <- dynamic query tag
    }

    graph = build_graph()
    result = graph.invoke(initial_state, config=config)

    print("\n✅ Final Summary:\n")
    print(result["summary"])

if __name__ == "__main__":
    main()