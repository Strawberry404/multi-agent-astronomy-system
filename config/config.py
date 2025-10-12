import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Central configuration fot the astronomy agent"""
    # API Keys (loaded from .env)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    #Models
    EMBEDDING_MODEL="sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL="gemini-2.5-flash"
    DEVICE="cude"

    #Vector Store
    VECTOR_STORE_PATH="faiss_index"
    RAG_K = 10 # number of documents to retrieve

    #Search 
    TAVILY_MAX_RESULTS=5

    #PATHS
    PDF_PATH="Astronomy.pdf"



