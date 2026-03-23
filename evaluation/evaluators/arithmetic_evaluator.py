"""Evaluator for arithmetic question solving tasks."""

from datasets.arithmetic_dataset import ArithmeticDataset
from .base_evaluator import BaseEvaluator


class ArithmeticEvaluator(BaseEvaluator):
    """
    Evaluator for arithmetic question solving tasks.
    Handles questions about solving arithmetic problems (addition and subtraction).
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize the Arithmetic evaluator.
        
        Args:
            dataset_dir: Directory containing arithmetic dataset files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize ArithmeticDataset for data management
        self.dataset = ArithmeticDataset(dataset_dir, augment_approach=augment_approach)
