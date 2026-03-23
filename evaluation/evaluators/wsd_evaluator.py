"""
WSD Evaluator
Evaluation logic for Word Sense Disambiguation tasks.
"""

from datasets import WSDDataset
from .base_evaluator import BaseEvaluator


class WSDEvaluator(BaseEvaluator):
    """Evaluator for Word Sense Disambiguation tasks"""
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize WSD evaluator.
        
        Args:
            dataset_dir: Directory containing WSD data
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        self.dataset = WSDDataset(dataset_dir, augment_approach=augment_approach)
