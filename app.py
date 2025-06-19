import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import LangChainTracer
from graph.workflow_graph import build_graph

# Load environment variables (API keys, etc.)
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define input schema
class QueryRequest(BaseModel):
    query: str

# Initialize FastAPI app
app = FastAPI()

@app.post("/recommend")
async def recommend_route(request: QueryRequest):
    tracer = LangChainTracer()
    query = request.query

    initial_state = {
        "query": query,
        "amazon_results": [],
        "flipkart_results": [],
        "tatacliq_results": [],
        "recommendations": [],
        "summary": ""
    }

    config = {
        "callbacks": [tracer],
        "metadata": {"user_query": query},
        "run_name": f"FastAPI_Recommendation_{query}",
        "tags": ["api", "fastapi", query.replace(" ", "_")]
    }

    try:
        graph = build_graph()
        result = graph.invoke(initial_state, config=config)
        return {"summary": result["summary"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")