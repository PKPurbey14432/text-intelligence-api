# AI-Powered Text Intelligence API

A comprehensive NLP-based intelligent API service built with FastAPI, featuring sentiment analysis, text summarization, and semantic search capabilities.

## Features

- **Sentiment Analysis**: Analyze text sentiment (positive/negative/neutral) using transformer models
- **Keyword Extraction**: Extract top 5 keywords from text using TF-IDF and NLP techniques
- **Text Summarization**: Generate concise summaries using T5 transformer models
- **Semantic Search**: Perform semantic similarity search using FAISS vector store and embeddings

## Requirements

- Python 3.8+
- Docker (for containerized deployment)
- 4GB+ RAM (for ML models)

## Installation

### Option 1: Using Docker (Recommended)

1. **Build the Docker image:**
```bash
docker build -t text-intelligence-api .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 text-intelligence-api
```

3. **OR Run using Docker Compose:**
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

### Option 2: Local Installation

1. **Clone or download the project:**
```bash
cd text-intelligence-api
```

2. **Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
```

3. **Install dependencies:**
```bash
pip3 install -r requirements.txt
```

4. **Download SpaCy English model:**
```bash
python3 -m spacy download en_core_web_sm
```

5. **Run the application:**
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Once the server is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check

**GET** `/` or `/health`

Returns API status and available endpoints.

**Response:**
```json
{
  "message": "AI-Powered Text Intelligence API",
  "status": "running"
}
```

### 2. Text Analysis

**POST** `/analyze`

Analyzes text for sentiment and extracts top keywords.

**Request:**
```json
{
  "text": "I love working with AI! It makes everything efficient."
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "keywords": ["AI", "efficient", "love", "working", "everything"]
}
```

### 3. Text Summarization

**POST** `/summarize`

Summarizes input text using transformer models.

**Request:**
```json
{
  "text": "Your long text here..."
}
```

**Response:**
```json
{
  "summary": "Generated summary text...",
  "original_length": 500,
  "summary_length": 120
}
```

### 4. Semantic Search

**POST** `/semantic-search`

Performs semantic similarity search on stored texts.

**Request:**
```json
{
  "query": "artificial intelligence",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "artificial intelligence",
  "results": [
    {
      "text": "Machine learning is a subset of AI...",
      "similarity_score": 0.85
    }
  ]
}
```

### 5. Add Texts to Vector Store (Legacy)

**POST** `/semantic-search/add`

Adds texts to the vector store for semantic search (without chunking).

**Request:**
```json
{
  "texts": [
    {
      "text": "Artificial intelligence is transforming industries.",
      "id": "doc_1"
    },
    {
      "text": "Machine learning algorithms learn from data.",
      "id": "doc_2"
    }
  ]
}
```

**Response:**
```json
{
  "message": "Texts added successfully to vector store",
  "added_count": 2,
  "chunks_created": 2
}
```

### 6. Upload Documents (Text or PDF Files)

**POST** `/semantic-search/add-documents`

Upload text files (.txt) or PDF files (.pdf) to the vector store with automatic chunking.

**Features:**
- Automatic text chunking using LangChain's RecursiveCharacterTextSplitter
- Supports .txt and .pdf file uploads
- Automatically extracts text from PDFs
- Stores chunks in FAISS vector database
