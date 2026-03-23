"""
Base Evaluator
Abstract base class for dataset-specific evaluation logic.

Evaluators wrap Dataset classes and add evaluation-specific functionality:
- Ground truth retrieval
- Parameter naming for results
"""

import logging
from abc import ABC
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Abstract base class for dataset-specific evaluators.
    
    Evaluators complement Dataset classes by providing:
    - Ground truth answer retrieval
    - Parameter naming for data size in results
    
    The dataset instance handles all data operations (loading, slicing, formatting, questions).
    """
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize evaluator with dataset directory and augmentation approach.
        
        Args:
            dataset_dir: Directory containing dataset files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        from pathlib import Path
        self.dataset_dir = Path(dataset_dir)
        self.augment_approach = augment_approach
        self.dataset = None  # Subclasses should initialize their dataset
        self.ground_truth_df = None  # Ground truth will be loaded dynamically
    
    def load_ground_truth(self, ground_truth_csv: str):
        """
        Load ground truth from CSV file.
        Common implementation for most evaluators.
        
        Args:
            ground_truth_csv: Path to ground truth CSV file
        """
        self.ground_truth_df = pd.read_csv(ground_truth_csv)
        logger.info(f"Loaded ground truth with {len(self.ground_truth_df)} rows")
    
    def get_ground_truth(self, question_id: str, selection_name: str) -> Any:
        """
        Get ground truth answer for a question and selection.
        Common implementation for most evaluators.
        
        Args:
            question_id: Question identifier (e.g., 'Q1')
            selection_name: Selection identifier (e.g., 'first_10', 'window_5_start0')
            
        Returns:
            Ground truth answer (type depends on question)
        """
        if self.ground_truth_df is None:
            raise ValueError("Ground truth not loaded. Call load_ground_truth() first.")
        
        # Find row with matching question_id and selection_name
        row = self.ground_truth_df[
            (self.ground_truth_df['question_id'] == question_id) &
            (self.ground_truth_df['selection_name'] == selection_name)
        ]
        
        if row.empty:
            logger.warning(f"No ground truth found for {question_id}/{selection_name}")
            return None
        
        ground_truth = row['ground_truth'].iloc[0]
        
        # Parse dict strings back to dict objects (for questions that return dicts)
        if isinstance(ground_truth, str) and ground_truth.strip().startswith('{'):
            import ast
            try:
                ground_truth = ast.literal_eval(ground_truth)
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse dict ground_truth: {e}")
        
        return ground_truth
    
    def get_data_size_param_name(self) -> str:
        """
        Get the parameter name for data size in results.
        
        Returns:
            Parameter name ('n_instance')
        """
        return 'n_instance'
