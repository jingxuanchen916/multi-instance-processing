"""
Parity Evaluator
Evaluator implementation for parity (odd/even) classification tasks.
Wraps ParityDataset and adds evaluation-specific logic.
"""

from datasets import ParityDataset
from .base_evaluator import BaseEvaluator


class ParityEvaluator(BaseEvaluator):
    """
    Evaluator for parity classification tasks.
    
    Wraps ParityDataset and adds:
    - Ground truth retrieval with column mapping
    - Parameter naming ('n_instance')
    
    All data operations (loading, slicing, formatting, questions) 
    are delegated to the ParityDataset instance.
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize Parity evaluator.
        
        Args:
            dataset_dir: Directory containing parity files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize ParityDataset for data management
        self.dataset = ParityDataset(dataset_dir, augment_approach=augment_approach)
