"""
Keyword Extraction Service
Uses SpaCy and scikit-learn for keyword extraction
"""
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import logging

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """
    Keyword extraction service using TF-IDF and NLP techniques
    
    Combines SpaCy for text preprocessing and scikit-learn's
    TF-IDF vectorizer for keyword importance scoring.
    """
    
    def __init__(self):
        """Initialize keyword extractor with SpaCy model"""
        try:
            logger.info("Loading SpaCy model for keyword extraction...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            logger.info("Keyword extractor initialized")
        except Exception as e:
            logger.error(f"Error initializing keyword extractor: {e}")
            self.nlp = None
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for keyword extraction
        
        Args:
            text: Raw input text
            
        Returns:
            str: Preprocessed text
        """
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def _extract_nouns_and_phrases(self, text: str) -> list:
        """
        Extract nouns and important phrases using SpaCy
        
        Args:
            text: Input text
            
        Returns:
            list: List of extracted keywords/phrases
        """
        if not self.nlp:
            words = self._preprocess_text(text).split()
            return [w for w in words if len(w) > 3]
        
        doc = self.nlp(text)
        keywords = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2:
                keywords.append(chunk.text.lower().strip())
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                if len(token.text) > 2:
                    keywords.append(token.text.lower())
        
        return list(set(keywords))
    
    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        """
        Extract top N keywords from text
        
        Args:
            text: Input text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            list: List of top keywords
        """
        try:
            processed_text = self._preprocess_text(text)
            
            if not processed_text:
                return []
            
            candidates = self._extract_nouns_and_phrases(text)
            
            if not candidates:
                words = processed_text.split()
                candidates = [w for w in words if len(w) > 3]
            
            if not candidates:
                return []
            
            try:
                documents = [processed_text] + candidates
                tfidf_matrix = self.vectorizer.fit_transform(documents)
                
                feature_names = self.vectorizer.get_feature_names_out()
                
                scores = tfidf_matrix[0].toarray()[0]
                
                keyword_scores = list(zip(feature_names, scores))
                
                keyword_scores.sort(key=lambda x: x[1], reverse=True)
                top_keywords = [kw for kw, score in keyword_scores[:top_n] if score > 0]
                
                if len(top_keywords) < top_n:
                    for candidate in candidates:
                        if candidate not in top_keywords:
                            top_keywords.append(candidate)
                        if len(top_keywords) >= top_n:
                            break
                
                return top_keywords[:top_n]
                
            except Exception as e:
                logger.warning(f"TF-IDF extraction failed: {e}, using fallback")
                return candidates[:top_n]
                
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            # Final fallback: simple word extraction
            words = self._preprocess_text(text).split()
            return [w for w in words if len(w) > 3][:top_n]

