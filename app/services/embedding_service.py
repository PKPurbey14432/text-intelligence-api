"""
Embedding Service
Generates text embeddings using sentence transformers
"""
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Text embedding service using sentence transformers
    
    Uses sentence-transformers library which provides efficient
    models for generating semantic embeddings of text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        
        Args:
            model_name: Sentence transformer model identifier
                       "all-MiniLM-L6-v2" is fast and efficient
        """
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def encode(self, texts: list) -> list:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            list: List of embedding vectors (numpy arrays)
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings
        
        Returns:
            int: Embedding dimension
        """
        return self.embedding_dim

