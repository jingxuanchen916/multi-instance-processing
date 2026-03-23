"""Arithmetic Dataset for arithmetic question solving tasks."""

import logging
import random
from typing import List

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ArithmeticDataset(BaseDataset):
    """
    Dataset for arithmetic question solving tasks.
    
    Contains 2500 arithmetic questions (addition and subtraction) with their numerical answers.
    """
    
    # Standard dataset sizes for arithmetic questions
    STANDARD_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]

    # Noise text used for augmentation approaches
    NOISE_TEXT = (
        " - IRRELEVANT CONTEXT START - The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
        "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. - IRRELEVANT CONTEXT END - "
    )

    AUGMENT_APPROACH = {
        "default": "",
        "head": NOISE_TEXT,
        "middle": NOISE_TEXT,
        "tail": NOISE_TEXT,
        "random": NOISE_TEXT,
    }
    
    def __init__(self, dataset_dir: str, augment_approach: str):
        """
        Initialize the Arithmetic dataset.
        
        Args:
            dataset_dir: Path to the directory containing arithmetic dataset files
            augment_approach: Augmentation approach ('default', 'head', 'middle', 'tail', 'random')
        """
        super().__init__(dataset_dir, augment_approach)
        self.text_file = self.data_dir / "2500_arithmetic_text.txt"
        self.labels_file = self.data_dir / "2500_arithmetic_label.txt"
        # Questions file is now in config/questions/
        self.questions_file = self.data_dir.parent.parent / "config" / "questions" / "arithmetic_question.yaml"
        
        # Load data and questions
        self.data = self.load_data()
        self.labels = self.load_labels()
        self.questions = self.load_questions()
    
    def load_data(self) -> List[str]:
        """
        Load arithmetic questions from file.
        
        Returns:
            List of arithmetic question strings
        """
        if not self.text_file.exists():
            raise FileNotFoundError(f"Text file not found: {self.text_file}")
        
        with open(self.text_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(questions)} arithmetic questions from {self.text_file}")
        return questions
    
    def load_labels(self) -> List[str]:
        """
        Load arithmetic answers from file.
        Note: Arithmetic labels are strings (numerical values), not integers.
        
        Returns:
            List of answer strings (numerical values as strings)
        """
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f.readlines()]
        
        logger.info(f"Loaded {len(labels)} answers from {self.labels_file}")
        return labels
    
    def format_for_prompt(self, data_slice: List[str]) -> str:
        """
        Format arithmetic questions for inclusion in LLM prompt.
        Each question is numbered for clarity, with optional noise injection based on augmentation approach.
        
        Args:
            data_slice: List of arithmetic question strings
            
        Returns:
            Formatted string for prompt
        """
        noise = self.AUGMENT_APPROACH[self.augment_approach]
        
        items = []
        for i, item in enumerate(data_slice):
            augmented = self._augment_item(item, self.augment_approach, noise, item_index=i)
            items.append(augmented)

        return self._format_numbered_block(
            items,
            "=== Here are the arithmetic questions ===",
            "Question",
            "=== End of arithmetic questions ==="
        )
    
    def _augment_item(self, item: str, approach: str, noise: str, item_index: int = 0) -> str:
        """
        Augment a single item based on approach.
        
        Args:
            item: The original item/instance
            approach: Augmentation approach ('default', 'head', 'middle', 'tail', 'random')
            noise: The noise text to inject
            item_index: Index of item (used for random seed)
            
        Returns:
            Augmented item
        """
        if approach == "default" or not noise:
            return item
        elif approach == "tail":
            return f"{item}{noise}"
        elif approach == "head":
            return f"{noise}{item}"
        elif approach == "middle":
            words = item.split()
            if len(words) <= 1:
                # If only one word or empty, treat as tail
                return f"{item}{noise}"
            mid_point = len(words) // 2
            first_half = " ".join(words[:mid_point])
            second_half = " ".join(words[mid_point:])
            return f"{first_half}{noise}{second_half}"
        elif approach == "random":
            words = item.split()
            if len(words) == 0:
                return f"{noise}{item}"
            
            # Use seed=0 + item_index for reproducibility
            random.seed(item_index)
            # x ranges from 0 to len(words)
            x = random.randint(0, len(words))
            
            if x == 0:
                return f"{noise}{item}"
            elif x == len(words):
                return f"{item}{noise}"
            else:
                first_part = " ".join(words[:x])
                second_part = " ".join(words[x:])
                return f"{first_part}{noise}{second_part}"
        else:
            raise ValueError(f"Unknown augmentation approach: {approach}")
    
    def get_labels_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get arithmetic answers by specific indices.
        Note: Overrides base method to return strings instead of ints.
        
        Args:
            indices: List of 0-based indices to retrieve
            
        Returns:
            List of answer strings at specified indices
        """
        if not indices:
            return []
        
        # Validate indices
        max_index = max(indices)
        if max_index >= len(self.labels):
            raise IndexError(
                f"Index {max_index} out of range for labels (max: {len(self.labels) - 1})"
            )
        
        return [self.labels[i] for i in indices]
