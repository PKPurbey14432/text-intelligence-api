"""
Vector Store Service
Manages FAISS index for semantic search
"""
import faiss
import numpy as np
from typing import List, Dict
import logging
import os
import json

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store service using FAISS for efficient similarity search
    
    Stores text embeddings in a FAISS index and provides
    fast similarity search capabilities.
    """
    
    def __init__(self, embedding_service, index_path: str = "faiss_index.index"):
        """
        Initialize vector store
        
        Args:
            embedding_service: EmbeddingService instance for generating embeddings
            index_path: Path to save/load FAISS index (without extension)
        """
        self.embedding_service = embedding_service
        self.index_path = index_path
        self.metadata_path = f"{index_path}.metadata.json"
        self.dimension = embedding_service.get_embedding_dimension()

        self.index = faiss.IndexFlatIP(self.dimension)
        
        self.texts: List[str] = []
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        
        self._load_index()
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity
        
        Args:
            embeddings: Raw embedding vectors
            
        Returns:
            np.ndarray: Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    
    def add_texts(
        self,
        texts: List[str],
        ids: List[str] = None,
        metadata_list: List[Dict] = None
    ):
        """
        Add texts to the vector store
        
        Args:
            texts: List of text strings to add
            ids: Optional list of IDs for each text
            metadata_list: Optional list of metadata dictionaries for each text
        """
        try:
            if ids is None:
                ids = [f"doc_{len(self.texts) + i}" for i in range(len(texts))]
            
            if metadata_list is None:
                metadata_list = [{}] * len(texts)
            
            if len(metadata_list) != len(texts):
                logger.warning(
                    f"Metadata list length ({len(metadata_list)}) doesn't match texts length ({len(texts)}). "
                    "Padding with empty metadata."
                )
                metadata_list.extend([{}] * (len(texts) - len(metadata_list)))
            
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.embedding_service.encode(texts)
            embeddings = np.array(embeddings).astype('float32')
            
            embeddings = self._normalize_embeddings(embeddings)
            
            self.index.add(embeddings)
            
            self.texts.extend(texts)
            self.ids.extend(ids)
            self.metadata.extend(metadata_list)
            
            logger.info(f"Added {len(texts)} texts to vector store. Total: {len(self.texts)}")
            
            self._save_index()
            
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar texts using semantic similarity
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List[Dict]: List of similar texts with similarity scores
        """
        try:
            if len(self.texts) == 0:
                return []
            
            query_embedding = self.embedding_service.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            query_embedding = self._normalize_embeddings(query_embedding)
            
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.texts)))
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.texts):
                    results.append({
                        "text": self.texts[idx],
                        "similarity_score": float(distance)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    def _save_index(self):
        """Save FAISS index and metadata to disk using native FAISS methods"""
        try:
            faiss.write_index(self.index, self.index_path)
            
            metadata = {
                'texts': self.texts,
                'ids': self.ids,
                'metadata': self.metadata
            }
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Vector store saved: index to {self.index_path}, metadata to {self.metadata_path}")
        except Exception as e:
            logger.warning(f"Could not save vector store: {e}")
    
    def _load_index(self):
        """Load FAISS index and metadata from disk using native FAISS methods"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.texts = metadata.get('texts', [])
                    self.ids = metadata.get('ids', [])
                    self.metadata = metadata.get('metadata', [])
                    
                    if len(self.metadata) != len(self.texts):
                        logger.warning(
                            f"Metadata length mismatch. Padding with empty metadata."
                        )
                        self.metadata.extend([{}] * (len(self.texts) - len(self.metadata)))
                
                if self.index.ntotal != len(self.texts):
                    logger.warning(
                        f"Index size ({self.index.ntotal}) doesn't match metadata size ({len(self.texts)}). "
                        "This may indicate corrupted data."
                    )
                
                logger.info(
                    f"Loaded vector store from {self.index_path}. "
                    f"{len(self.texts)} texts found."
                )
            elif os.path.exists(self.index_path) or os.path.exists(self.metadata_path):
                logger.warning(
                    "Found partial vector store files. Both index and metadata are required. "
                    "Starting with empty index."
                )
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}. Starting with empty index.")
    
    def clear(self):
        """Clear all texts from the vector store"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.texts = []
        self.ids = []
        self.metadata = []
        
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        
        logger.info("Vector store cleared")

