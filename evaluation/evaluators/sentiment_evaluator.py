"""
Sentiment Evaluator
Evaluator implementation for sentiment analysis evaluation tasks.
Wraps SentimentDataset and adds evaluation-specific logic.
"""

from datasets import SentimentDataset
from .base_evaluator import BaseEvaluator


class SentimentEvaluator(BaseEvaluator):
    """
    Evaluator for sentiment analysis evaluation tasks.
    
    Wraps SentimentDataset and adds:
    - Ground truth retrieval
    - Parameter naming ('n_instance')
    
    All data operations (loading, slicing, formatting, questions) 
    are delegated to the SentimentDataset instance.
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize sentiment evaluator.
        
        Args:
            dataset_dir: Directory containing sentiment files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize SentimentDataset for data management
        self.dataset = SentimentDataset(dataset_dir, augment_approach=augment_approach)
