import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

def summarize_agent(state: dict) -> dict:
    print("[Summarizer Agent] 📝 Finalizing summary...")
    print("[Summarizer Agent] 📦 Incoming state:")
    print(state)

    query = state.get("query", "")
    top_product = state.get("top_product")
    reasoning = state.get("reasoning")

    if not top_product or not reasoning:
        print("[Summarizer Agent] ❌ Missing data.")
        summary = "No recommendation could be made for your query."
    else:
        # 🧠 Let GPT summarize/polish the explanation
        prompt = (
            f"A product was recommended for the query: '{query}'\n\n"
            f"Reason for recommendation:\n{reasoning}\n\n"
            f"Write a 2-line polished summary with a touch of persuasion."
        )
        summary_text = llm([HumanMessage(content=prompt)]).content.strip()

        summary = (
            f"{summary_text}\n\n"
            f"🔗 Product Link: {top_product}"
        )

    print("[Summarizer Agent] ✅ Summary generated")
    return {**state, "summary": summary}