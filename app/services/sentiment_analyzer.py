"""
Sentiment Analysis Service
Uses Hugging Face transformers for sentiment classification
"""
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analysis service using pre-trained transformer models
    
    Uses Hugging Face's sentiment-analysis pipeline which typically
    uses a RoBERTa-based model fine-tuned on sentiment analysis tasks.
    """
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer
        
        Args:
            model_name: Hugging Face model identifier for sentiment analysis
        """
        try:
            logger.info(f"Loading sentiment analysis model: {model_name}")
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                return_all_scores=False
            )
            logger.info("Sentiment analysis model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading sentiment model: {e}")
            logger.info("Trying fallback model...")
            self.classifier = pipeline("sentiment-analysis")
    
    def analyze(self, text: str) -> str:
        """
        Analyze sentiment of input text
        
        Args:
            text: Input text to analyze
            
        Returns:
            str: Sentiment label (positive, negative, or neutral)
        """
        try:
            result = self.classifier(text)[0]
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label:
                return "positive"
            elif 'negative' in label:
                return "negative"
            else:
                return "neutral"
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            # Return neutral as fallback
            return "neutral"

