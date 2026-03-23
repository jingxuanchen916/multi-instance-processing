"""
Tweet Evaluator
Evaluator implementation for tweet-based evaluation tasks.
Wraps TweetDataset and adds evaluation-specific logic.
"""

from datasets import TweetDataset
from .base_evaluator import BaseEvaluator


class TweetEvaluator(BaseEvaluator):
    """
    Evaluator for tweet-based evaluation tasks.
    
    Wraps TweetDataset and adds:
    - Ground truth retrieval with column mapping
    - Parameter naming ('n_instance')
    
    All data operations (loading, slicing, formatting, questions) 
    are delegated to the TweetDataset instance.
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize tweet evaluator.
        
        Args:
            dataset_dir: Directory containing tweet files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize TweetDataset for data management
        self.dataset = TweetDataset(dataset_dir, augment_approach=augment_approach)
