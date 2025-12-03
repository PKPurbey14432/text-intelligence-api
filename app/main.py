"""
FastAPI Main Application
AI-Powered Text Intelligence API Service
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

from app.services.sentiment_analyzer import SentimentAnalyzer
from app.services.keyword_extractor import KeywordExtractor
from app.services.text_summarizer import TextSummarizer
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor

# Initialize FastAPI app
app = FastAPI(
    title="AI-Powered Text Intelligence API",
    description="NLP-based intelligent API service for sentiment analysis, text summarization, and semantic search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (lazy loading for better startup time)
sentiment_analyzer = None
keyword_extractor = None
text_summarizer = None
embedding_service = None
vector_store = None
document_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize ML models and services on startup"""
    global sentiment_analyzer, keyword_extractor, text_summarizer, embedding_service, vector_store, document_processor
    
    print("Initializing ML models and services...")
    sentiment_analyzer = SentimentAnalyzer()
    keyword_extractor = KeywordExtractor()
    text_summarizer = TextSummarizer()
    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)
    document_processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200
    )
    print("All services initialized successfully!")


class AnalyzeRequest(BaseModel):
    """Request model for text analysis"""
    text: str = Field(..., description="Input text to analyze", min_length=1)


class AnalyzeResponse(BaseModel):
    """Response model for text analysis"""
    sentiment: str = Field(..., description="Sentiment classification: positive, negative, or neutral")
    keywords: List[str] = Field(..., description="Top 5 keywords extracted from the text")


class SummarizeRequest(BaseModel):
    """Request model for text summarization"""
    text: str = Field(..., description="Input text to summarize", min_length=1)


class SummarizeResponse(BaseModel):
    """Response model for text summarization"""
    summary: str = Field(..., description="Generated summary of the input text")
    original_length: int = Field(..., description="Length of original text")
    summary_length: int = Field(..., description="Length of generated summary")


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query text", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of similar texts to return", ge=1, le=20)


class TextDocument(BaseModel):
    """Model for text document in semantic search"""
    text: str
    id: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result"""
    text: str = Field(..., description="The matching text chunk")
    similarity_score: float = Field(..., description="Similarity score (0-1)")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search"""
    query: str = Field(..., description="The search query")
    results: List[SearchResult] = Field(..., description="List of similar texts with similarity scores")


class AddTextRequest(BaseModel):
    """Request model to add text to vector store"""
    texts: List[TextDocument] = Field(..., description="List of texts to add to the vector store", min_items=1)


class AddTextResponse(BaseModel):
    """Response model for adding texts"""
    message: str
    added_count: int
    chunks_created: int = Field(..., description="Number of chunks created from documents")


class AddDocumentResponse(BaseModel):
    """Response model for adding documents (text/PDF)"""
    message: str
    files_processed: int


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "AI-Powered Text Intelligence API",
        "status": "running",
        "endpoints": {
            "analyze": "/analyze",
            "summarize": "/summarize",
            "semantic_search": "/semantic-search",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "text-intelligence-api"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """
    Analyze text for sentiment and extract top keywords
    
    - **sentiment**: Returns positive, negative, or neutral
    - **keywords**: Returns top 5 most important keywords
    """
    try:
        # Perform sentiment analysis
        sentiment = sentiment_analyzer.analyze(request.text)
        
        # Extract keywords
        keywords = keyword_extractor.extract_keywords(request.text, top_n=5)
        
        return AnalyzeResponse(
            sentiment=sentiment,
            keywords=keywords
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest):
    """
    Summarize input text using transformer models
    
    - Uses pre-trained T5 or BART models for summarization
    - Returns concise summary of the input text
    - Maximum summary length is set to 10,000 characters
    """
    try:
        # Fixed max_length of 10,000 characters in backend
        summary = text_summarizer.summarize(
            request.text,
            max_length=10000
        )
        
        return SummarizeResponse(
            summary=summary,
            original_length=len(request.text),
            summary_length=len(summary)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


@app.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """
    Perform semantic search using embeddings and vector similarity
    
    - Uses FAISS for efficient vector search
    - Returns most similar texts based on semantic meaning
    """
    try:
        results = vector_store.search(request.query, top_k=request.top_k)
        
        return SemanticSearchResponse(
            query=request.query,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing semantic search: {str(e)}")


@app.post("/semantic-search/add", response_model=AddTextResponse)
async def add_texts_to_store(request: AddTextRequest):
    """
    Add texts to the vector store for semantic search (legacy endpoint)
    
    - Stores text embeddings in FAISS index
    - Texts can be searched later using semantic-search endpoint
    - Note: For chunking support, use /semantic-search/add-documents instead
    """
    try:
        texts = [doc.text for doc in request.texts]
        ids = [doc.id if doc.id else f"doc_{i}" for i, doc in enumerate(request.texts)]
        
        vector_store.add_texts(texts, ids=ids)
        
        return AddTextResponse(
            message="Texts added successfully to vector store",
            added_count=len(texts),
            chunks_created=len(texts)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding texts: {str(e)}")


@app.post("/semantic-search/add-documents", response_model=AddDocumentResponse)
async def add_documents_to_store(
    files: List[UploadFile] = File(..., description="Text files (.txt) or PDF files (.pdf) to upload")
):
    """
    Upload and process text files or PDF files to the vector store
    
    - Accepts .txt and .pdf file uploads
    - Automatically extracts text and splits into chunks using LangChain's RecursiveCharacterTextSplitter
    - Stores chunks in FAISS vector database
    
    **Supported file types:**
    - Text files: .txt, .text
    - PDF files: .pdf
    """
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one file must be uploaded"
            )
        
        all_chunks = []
        processed_files = 0
        
        # Process each uploaded file
        for file in files:
            try:
                # Read file content
                file_bytes = await file.read()
                
                # Validate file type
                if not file.filename:
                    raise HTTPException(
                        status_code=400,
                        detail="File must have a filename"
                    )
                
                # Extract text from file (handles both PDF and text files)
                try:
                    text = document_processor.extract_text_from_file(file_bytes, file.filename)
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error processing {file.filename}: {str(e)}"
                    )
                
                # Split text into chunks (minimal metadata)
                chunks = document_processor.split_text(text)
                all_chunks.extend(chunks)
                processed_files += 1
                
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error processing file {file.filename}: {str(e)}"
                )
        
        if not all_chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the uploaded files"
            )
        
        # Extract texts and create simple IDs
        chunk_texts = [chunk["text"] for chunk in all_chunks]
        chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
        
        # Add to vector store (no metadata)
        vector_store.add_texts(
            texts=chunk_texts,
            ids=chunk_ids,
            metadata_list=[{}] * len(chunk_texts)  # Empty metadata
        )
        
        return AddDocumentResponse(
            message="Files processed and added to vector store successfully",
            files_processed=processed_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

