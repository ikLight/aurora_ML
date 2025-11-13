from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from agent import Agent
from functools import lru_cache
import os

load_dotenv('./.gitignore/.env')


##-----------------------------------------------------------------------------##

class QuestionRequest(BaseModel):
    question: str
    mode: str = "standard"  # Options: "standard", "verified"

##-----------------------------------------------------------------------------##

app = FastAPI(title="Agent Q&A API")

# Configure number of parallel workers based on CPU count or environment variable
max_workers = int(os.getenv("MAX_WORKERS", 8))
use_vector_search = os.getenv("USE_VECTOR_SEARCH", "true").lower() == "true"

agent = Agent(max_workers=max_workers, use_vector_search=use_vector_search)
API_ROUTE = "https://november7-730026606190.europe-west1.run.app/messages/?skip=0&limit=4000"
# Simple in-memory cache for frequently asked questions
response_cache = {}

@app.on_event("startup")
async def startup_event():
    """Load data when the app starts - processes in parallel and builds vector index"""
    print(f"Loading data with {max_workers} parallel workers...")
    print(f"Vector search enabled: {use_vector_search}")
    agent.load_data(API_ROUTE)
    print("Data loaded, cached, and indexed successfully!")

##-----------------------------------------------------------------------------##

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    POST endpoint to answer questions about user data
    Optimized with vector search, parallel processing, caching, and enhanced prompting
    
    Modes:
    - standard: Fast, enhanced prompts with reasoning (default)
    - verified: Includes verification step to reduce hallucinations (slower but more accurate)
    """
    # Create cache key with mode
    cache_key = f"{request.question}|{request.mode}"
    
    # Check cache first
    if cache_key in response_cache:
        return {"answer": response_cache[cache_key]}
    
    # Use appropriate method based on mode
    if request.mode == "verified":
        answer = agent.answer_question_with_verification(request.question, top_k=20)
    else:
        answer = agent.answer_question(request.question, top_k=20, use_reasoning=True)
    
    # Cache the response
    response_cache[cache_key] = answer
    
    return {"answer": answer}

##-----------------------------------------------------------------------------##
