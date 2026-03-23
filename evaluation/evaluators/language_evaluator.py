"""Evaluator for language identification tasks."""

from datasets.language_dataset import LanguageDataset
from .base_evaluator import BaseEvaluator


class LanguageEvaluator(BaseEvaluator):
    """
    Evaluator for language identification tasks.
    Handles questions about identifying and classifying paragraphs by language (English, Chinese, Persian, Spanish).
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize the Language evaluator.
        
        Args:
            dataset_dir: Directory containing language dataset files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        super().__init__(dataset_dir, augment_approach)
        
        # Initialize LanguageDataset for data management
        self.dataset = LanguageDataset(dataset_dir, augment_approach=augment_approach)
