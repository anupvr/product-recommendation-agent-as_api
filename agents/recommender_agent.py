import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

def recommend_top_product(state: dict) -> dict:
    print("[Recommender Agent] 🧠 Asking GPT to pick the best product...")

    query = state.get("query", "")
    recommendations = state.get("recommendations", [])

    if not recommendations:
        print("[Recommender Agent] ⚠️ No products available to evaluate.")
        return {**state, "top_product": None, "reasoning": "No products found."}

    # Prepare prompt
    links_text = "\n".join(f"- {url}" for url in recommendations)
    prompt = (
        f"You are an expert product recommender. A user is looking for: '{query}'\n"
        f"Here are some product links:\n{links_text}\n\n"
        "Based only on these links, pick the single best link and explain why.\n"
        "Return your answer in this format:\n"
        "RECOMMENDED: <url>\nREASON: <your reasoning>"
    )

    response = llm([HumanMessage(content=prompt)]).content.strip()
    print("[Recommender Agent] 📬 Raw GPT response:")
    print(response)

    # Improved robust parser
    top_product = None
    reasoning_lines = []

    for line in response.splitlines():
        if "RECOMMENDED:" in line.upper():
            top_product = line.split(":", 1)[1].strip()
        elif "REASON:" in line.upper():
            reasoning_line = line.split(":", 1)[1].strip()
            reasoning_lines.append(reasoning_line)
        elif top_product and line.strip():
            reasoning_lines.append(line.strip())

    reasoning = " ".join(reasoning_lines).strip()

    if top_product:
        print(f"[Recommender Agent] ✅ Chose: {top_product}")
    else:
        print("[Recommender Agent] ❌ GPT did not return a valid recommendation.")

    if reasoning:
        print("[Recommender Agent] ✅ Reasoning extracted.")
        print(reasoning)
    else:
        print("[Recommender Agent] ⚠️ No reasoning found in GPT output.")

    return {**state, "top_product": top_product, "reasoning": reasoning}