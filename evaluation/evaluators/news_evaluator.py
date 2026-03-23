"""
News Evaluator
Evaluator implementation for news category classification evaluation tasks.
Wraps NewsDataset and adds evaluation-specific logic.
"""

from datasets import NewsDataset
from .base_evaluator import BaseEvaluator


class NewsEvaluator(BaseEvaluator):
    """
    Evaluator for news category classification evaluation tasks.
    
    Wraps NewsDataset and adds:
    - Ground truth retrieval
    - Parameter naming ('n_instance')
    
    All data operations (loading, slicing, formatting, questions) 
    are delegated to the NewsDataset instance.
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize news evaluator.
        
        Args:
            dataset_dir: Directory containing news files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize NewsDataset for data management
        self.dataset = NewsDataset(dataset_dir, augment_approach=augment_approach)
