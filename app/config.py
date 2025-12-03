"""
Configuration settings for the application
"""
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Application settings
API_TITLE = "AI-Powered Text Intelligence API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "NLP-based intelligent API service for sentiment analysis, text summarization, and semantic search"

# Model settings
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SUMMARIZATION_MODEL = "t5-small"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector store settings
FAISS_INDEX_PATH = "faiss_index.pkl"

