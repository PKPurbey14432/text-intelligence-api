"""
Text Summarization Service
Uses Hugging Face transformers (T5 or BART) for text summarization
"""
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class TextSummarizer:
    """
    Text summarization service using pre-trained transformer models
    
    Uses T5 or BART models fine-tuned for summarization tasks.
    T5 is efficient and works well for both short and long texts.
    """
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize text summarizer
        
        Args:
            model_name: Hugging Face model identifier for summarization
                       Options: "t5-small", "t5-base", "facebook/bart-large-cnn"
        """
        try:
            logger.info(f"Loading summarization model: {model_name}")
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name
            )
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            logger.info("Trying default summarization model...")
            try:
                self.summarizer = pipeline("summarization")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def summarize(self, text: str, max_length: int = 10000, min_length: int = 30) -> str:
        """
        Summarize input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            str: Generated summary
        """
        try:
            # Generate summary
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = result[0]['summary_text']
            
            summary = summary.strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            words = text.split()
            fallback_length = min(max_length, len(words))
            return " ".join(words[:fallback_length])

