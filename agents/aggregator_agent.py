def aggregate_agent(state: dict) -> dict:
    print("[Aggregator Agent] 🔗 Aggregating results...")

    amazon = state.get("amazon_results", [])
    flipkart = state.get("flipkart_results", [])
    tatacliq = state.get("tatacliq_results", [])

    combined = amazon + flipkart + tatacliq
    deduped = list(dict.fromkeys(combined))

    print(f"[Aggregator Agent] ✅ Aggregated {len(deduped)} unique results")

    return {**state, "recommendations": deduped}  # Don't trim yet