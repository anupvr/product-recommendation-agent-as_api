from langgraph.graph import StateGraph, END
from graph.state_definition import RecommendationState

from agents.input_agent import user_input_agent
from agents.amazon_agent import crawl_amazon
from agents.flipkart_agent import crawl_flipkart
from agents.tatacliq_agent import crawl_tatacliq
from agents.aggregator_agent import aggregate_agent
from agents.recommender_agent import recommend_top_product  # ✅
from agents.summarizer_agent import summarize_agent

def build_graph():
    workflow = StateGraph(RecommendationState)

    workflow.add_node("input", user_input_agent)
    workflow.add_node("amazon", crawl_amazon)
    workflow.add_node("flipkart", crawl_flipkart)
    workflow.add_node("tatacliq", crawl_tatacliq)
    workflow.add_node("aggregate", aggregate_agent)
    workflow.add_node("recommend", recommend_top_product)  # ✅ this must exist
    workflow.add_node("summarize", summarize_agent)

    # Execution path
    workflow.set_entry_point("input")
    workflow.add_edge("input", "amazon")
    workflow.add_edge("amazon", "flipkart")
    workflow.add_edge("flipkart", "tatacliq")
    workflow.add_edge("tatacliq", "aggregate")

    workflow.add_edge("aggregate", "recommend")     # ✅ must flow through recommender
    workflow.add_edge("recommend", "summarize")     # ✅ must pass data to summarizer

    workflow.add_edge("summarize", END)

    return workflow.compile()