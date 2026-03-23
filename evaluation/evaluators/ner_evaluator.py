"""
NER Evaluator
Evaluator implementation for Named Entity Recognition evaluation tasks.
Wraps NERDataset and adds evaluation-specific logic.
"""

from datasets import NERDataset
from .base_evaluator import BaseEvaluator


class NEREvaluator(BaseEvaluator):
    """
    Evaluator for NER-based evaluation tasks.
    
    Wraps NERDataset and adds:
    - Ground truth retrieval with column mapping
    - Parameter naming ('n_instance')
    
    All data operations (loading, slicing, formatting, questions) 
    are delegated to the NERDataset instance.
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize NER evaluator.
        
        Args:
            dataset_dir: Directory containing NER files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize NERDataset for data management
        self.dataset = NERDataset(dataset_dir, augment_approach=augment_approach)
