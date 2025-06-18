from typing import TypedDict, List, Optional

class RecommendationState(TypedDict, total=False):
    query: str
    amazon_results: List[str]
    flipkart_results: List[str]
    tatacliq_results: List[str]
    recommendations: List[str]
    top_product: Optional[str]       # ✅ Add this
    reasoning: Optional[str]         # ✅ And this
    summary: str
