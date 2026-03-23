"""
WSD Dataset
Handles loading and formatting of Word Sense Disambiguation data (apple: company vs fruit).
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class WSDDataset(BaseDataset):
    """
    Dataset for Word Sense Disambiguation tasks.
    
    Each question can specify its own data_file to use different datasets.
    """

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
    
    def __init__(self, data_dir: str, augment_approach: str):
        """
        Initialize WSD dataset.
        
        Args:
            data_dir: Directory containing WSD data files
            augment_approach: Augmentation approach ('default', 'head', 'middle', 'tail', 'random')
        """
        super().__init__(data_dir, augment_approach)
        # Questions file is now in config/questions/
        self.questions_file = self.data_dir.parent.parent / "config" / "questions" / "wsd_question.yaml"
        
        # Load questions first to determine available data files
        self.questions = self.load_questions()
        
        # Default data file (used when question doesn't specify one)
        self.default_data_file = "2500_wsd_apple"
        
        # Cache for loaded data files: {data_file_name: (paragraphs, labels)}
        self.data_cache = {}
        
        # Load default dataset for backward compatibility
        self.paragraphs_file = self.data_dir / f"{self.default_data_file}_text.txt"
        self.labels_file = self.data_dir / f"{self.default_data_file}_label.txt"
        self.paragraphs = self._load_paragraphs_from_file(self.paragraphs_file)
        self.labels = self._load_labels_from_file(self.labels_file)
        
        # Cache the default dataset
        self.data_cache[self.default_data_file] = (self.paragraphs, self.labels)
        
        logger.info(f"Loaded {len(self.paragraphs)} paragraphs from default dataset: {self.default_data_file}")
        logger.info(f"Loaded {len(self.questions)} questions from {self.questions_file}")
    
    def _load_paragraphs_from_file(self, file_path: Path) -> List[str]:
        """Load paragraphs from a specific text file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            paragraphs = [line.strip() for line in f.readlines()]
        return paragraphs
    
    def _load_labels_from_file(self, file_path: Path) -> List[int]:
        """Load labels from a specific labels file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        return labels
    
    def _load_data_file(self, data_file: str) -> Tuple[List[str], List[int]]:
        """
        Load paragraphs and labels from a specific data file.
        
        Args:
            data_file: Base name of data file
            
        Returns:
            Tuple of (paragraphs, labels)
        """
        # Check cache first
        if data_file in self.data_cache:
            return self.data_cache[data_file]
        
        # Load from files
        text_file = self.data_dir / f"{data_file}_text.txt"
        labels_file = self.data_dir / f"{data_file}_label.txt"
        
        if not text_file.exists():
            raise FileNotFoundError(f"Data file not found: {text_file}")
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        paragraphs = self._load_paragraphs_from_file(text_file)
        labels = self._load_labels_from_file(labels_file)
        
        # Cache for future use
        self.data_cache[data_file] = (paragraphs, labels)
        
        logger.info(f"Loaded {len(paragraphs)} paragraphs from data file: {data_file}")
        
        return paragraphs, labels
    
    def get_data_for_question(self, question_id: str) -> Tuple[List[str], List[int]]:
        """
        Get the appropriate data (paragraphs and labels) for a specific question.
        
        Args:
            question_id: Question ID (e.g., "Q1", "Q2", "Q3", "Q4")
            
        Returns:
            Tuple of (paragraphs, labels) for this question
        """
        if question_id not in self.questions:
            raise ValueError(f"Question not found: {question_id}")
        
        question_config = self.questions[question_id]
        data_file = question_config.get('data_file', self.default_data_file)
        
        return self._load_data_file(data_file)
    
    def load_data(self) -> List[str]:
        """Load the dataset from files (implements abstract method)."""
        return self.paragraphs
    
    def load_questions(self) -> Dict[str, Any]:
        """Load questions (implements abstract method)."""
        return self.load_questions_from_file(self.questions_file)
    
    def get_data_slice(self, size: int) -> List[str]:
        """Get a slice of the data (implements abstract method)."""
        return self.paragraphs[:size]
    
    def get_questions(self) -> Dict[str, Any]:
        """Return questions dictionary."""
        return self.questions
    
    def get_standard_sizes(self) -> List[int]:
        """
        Return standard paragraph counts for evaluation.
        For WSD with multiple data files, return sizes valid for smallest dataset.
        """
        min_size = self._get_min_dataset_size()
        return [n for n in self.STANDARD_SIZES if n <= min_size]
    
    def get_data_by_indices(self, indices: List[int], question_id: str = None) -> List[str]:
        """
        Get paragraphs by specific indices.
        
        Args:
            indices: List of 0-based indices
            question_id: Optional question ID to get data for specific question
            
        Returns:
            List of paragraph strings
        """
        if question_id:
            paragraphs, _ = self.get_data_for_question(question_id)
        else:
            paragraphs = self.paragraphs
        
        return [paragraphs[i] for i in indices]
    
    def format_for_prompt(self, paragraphs: List[str]) -> str:
        """
        Format paragraphs for LLM prompt.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            Formatted string for prompt
        """
        noise = self.AUGMENT_APPROACH[self.augment_approach]
        
        items = []
        for i, item in enumerate(paragraphs):
            augmented = self._augment_item(item, self.augment_approach, noise, item_index=i)
            items.append(augmented)

        return self._format_numbered_block(
            items,
            "=== Here are the paragraphs ===",
            "Paragraph",
            "=== End of paragraphs ==="
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
    
    def get_total_size(self) -> int:
        """Return total number of paragraphs in dataset."""
        return len(self.paragraphs)
    
    def get_instance_selections(
        self,
        strategy: str = 'first_n',
        counts: List[int] = None,
        selection_config: str = None,
        window_end: int = None
    ) -> List[Tuple[str, List[int]]]:
        """
        Generate instance selections based on strategy.
        
        Note: For WSD with multiple data files, selections are based on the smallest dataset to ensure all selections are valid for all questions.
        Individual questions will access their specific data files when needed.
        
        Args:
            strategy: Selection strategy ('first_n', 'sliding_window', 'custom')
            counts: List of paragraph counts (for 'first_n' and 'sliding_window')
            selection_config: Path to custom selection config (for 'custom')
            window_end: Optional end index for sliding_window (default: use smallest dataset size)
            
        Returns:
            List of (selection_name, indices) tuples
        """
        if counts is None:
            counts = self.get_standard_sizes()
        
        # For selection generation, use the smallest dataset to ensure all selections are valid
        # Find the minimum size across all data files used by questions
        min_size = self._get_min_dataset_size()
        
        # Filter counts to not exceed the minimum dataset size
        valid_counts = [c for c in counts if c <= min_size]
        if len(valid_counts) < len(counts):
            logger.warning(
                f"Some counts exceed minimum dataset size ({min_size}). "
                f"Using only valid counts: {valid_counts}"
            )
        
        if strategy == 'first_n':
            return self._generate_first_n_selections(valid_counts, max_size=min_size)
        elif strategy == 'sliding_window':
            # If window_end is specified, use it; otherwise use min_size
            effective_end = min(window_end, min_size) if window_end else min_size
            return self._generate_sliding_window_selections(valid_counts, effective_end)
        elif strategy == 'custom':
            return self._load_custom_selections(selection_config, max_size=min_size)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def _get_min_dataset_size(self) -> int:
        """
        Get the minimum dataset size across all data files used by questions.
        
        Returns:
            Minimum number of paragraphs across all datasets
        """
        min_size = len(self.paragraphs)  # Start with default dataset
        
        for question_id, question_config in self.questions.items():
            data_file = question_config.get('data_file', self.default_data_file)
            paragraphs, _ = self._load_data_file(data_file)
            min_size = min(min_size, len(paragraphs))
        
        return min_size
    
    def _generate_first_n_selections(self, counts: List[int], max_size: int = None) -> List[Tuple[str, List[int]]]:
        """Generate first N paragraphs selections."""
        if max_size is None:
            max_size = len(self.paragraphs)
        
        selections = []
        for n in counts:
            if n > max_size:
                logger.warning(f"Requested {n} paragraphs but only {max_size} available")
                n = max_size
            
            indices = list(range(n))
            selection_name = f"first_{n}"
            selections.append((selection_name, indices))
        
        return selections
    
    def _generate_sliding_window_selections(
        self,
        window_sizes: List[int],
        window_end: int = None
    ) -> List[Tuple[str, List[int]]]:
        """Generate sliding window selections."""
        selections = []
        
        # Determine the range to slide over
        if window_end is None:
            window_end = len(self.paragraphs)
        else:
            window_end = min(window_end, len(self.paragraphs))
        
        for size in window_sizes:
            if size > window_end:
                logger.warning(f"Window size {size} exceeds window_end {window_end}, skipping")
                continue
            
            # Generate all possible windows of this size
            num_windows = window_end - size + 1
            for start_idx in range(num_windows):
                indices = list(range(start_idx, start_idx + size))
                selection_name = f"window_{size}_start{start_idx}"
                selections.append((selection_name, indices))
        
        return selections
    
    def _load_custom_selections(self, config_path: str, max_size: int = None) -> List[Tuple[str, List[int]]]:
        """Load custom selections from YAML config using base class method."""
        if config_path is None:
            raise ValueError("selection_config path is required for custom selection mode")
        
        if max_size is None:
            max_size = len(self.paragraphs)
        
        # Use base class method to load samples
        samples = self.load_custom_selection(config_path)
        
        selections = []
        for sample in samples:
            name = sample['name']
            indices = sample['indices']
            
            # Validate indices against min dataset size
            if any(idx < 0 or idx >= max_size for idx in indices):
                raise ValueError(
                    f"Invalid indices in selection '{name}': "
                    f"must be between 0 and {max_size - 1} (minimum dataset size)"
                )
            
            selections.append((name, indices))
        
        logger.info(f"Loaded {len(selections)} custom selections from {config_path}")
        return selections
