"""
Base Dataset Interface
Abstract class defining the interface for all datasets.
"""

import logging
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Abstract base class for datasets.
    Each dataset combines data + questions specific to that data.
    """
    
    # Default standard sizes - can be overridden by subclasses
    STANDARD_SIZES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    
    def __init__(self, data_dir: Path, augment_approach: str):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing dataset files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
        """
        self.data_dir = Path(data_dir)
        self.augment_approach = augment_approach
        self.data = None
        self.labels = None
        self.questions = None
    
    @abstractmethod
    def load_data(self) -> Any:
        """
        Load the dataset from files.
        
        Returns:
            Loaded data (format depends on dataset type)
        """
        pass
    
    def load_questions(self) -> Dict[str, Dict[str, Any]]:
        """
        Load questions/tasks for this dataset.
        Default implementation loads from self.questions_file using load_questions_from_file().
        Override if custom loading logic is needed.
        
        Returns:
            Dictionary mapping question IDs to question configs
        """
        if not hasattr(self, 'questions_file'):
            raise NotImplementedError("Subclass must define questions_file attribute or override load_questions()")
        
        questions = self.load_questions_from_file(self.questions_file)
        logger.info(f"Loaded {len(questions)} questions from {self.questions_file}")
        return questions
    
    @abstractmethod
    def format_for_prompt(self, data_slice: Any) -> str:
        """
        Format data slice for inclusion in LLM prompt.
        
        Args:
            data_slice: Slice of data to format
            
        Returns:
            Formatted string for prompt
        """
        pass
    
    def get_standard_sizes(self) -> List[int]:
        """
        Get standard data sizes for evaluation.
        Filters out sizes larger than available data.
        
        Returns:
            List of data sizes to test
        """
        max_size = len(self.data) if self.data else 0
        return [n for n in self.STANDARD_SIZES if n <= max_size]
    
    def load_questions_from_file(self, questions_file: Path) -> Dict[str, Dict[str, Any]]:
        """
        Shared loader for YAML question files.
        
        Args:
            questions_file: Path to YAML with a top-level 'questions' key
            
        Returns:
            Parsed questions mapping
        """
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'questions' not in config:
            raise ValueError(f"Questions file missing 'questions' key: {questions_file}")
        
        return config['questions']
    
    def _format_numbered_block(self, items: List[str], header: str, item_label: str, footer: str) -> str:
        """
        Utility to format a numbered list with consistent spacing.
        """
        lines = [header]
        for i, item in enumerate(items, 1):
            lines.append(f"{item_label} {i}: {item}")
        lines.append(footer)
        return "\n\n".join(lines)
    
    def load_labels(self) -> List[int]:
        """
        Load labels from file. 
        Common implementation for datasets with integer labels.
        Override if labels have different format.
        
        Returns:
            List of integer labels
        """
        if not hasattr(self, 'labels_file'):
            raise NotImplementedError("Subclass must define labels_file attribute")
        
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        with open(self.labels_file, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        
        logger.info(f"Loaded {len(labels)} labels from {self.labels_file}")
        return labels
    
    def get_labels_by_indices(self, indices: List[int]) -> List[int]:
        """
        Get labels by specific indices.
        Common implementation for datasets with labels.
        
        Args:
            indices: List of 0-based indices to retrieve
            
        Returns:
            List of labels at specified indices
        """
        if not indices:
            return []
        
        if self.labels is None:
            raise ValueError("Labels not loaded")
        
        # Validate indices
        max_index = max(indices)
        if max_index >= len(self.labels):
            raise IndexError(
                f"Index {max_index} out of range for labels (max: {len(self.labels) - 1})"
            )
        
        return [self.labels[i] for i in indices]
    
    def get_data_slice(self, size: int) -> List[str]:
        """
        Get the first n data items.
        Common implementation for list-based datasets.
        
        Args:
            size: Number of items to return
            
        Returns:
            List of data items
        """
        if size > len(self.data):
            logger.warning(
                f"Requested {size} items but only {len(self.data)} available. "
                f"Returning all items."
            )
            return self.data.copy()
        
        return self.data[:size]
    
    def get_questions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get questions for this dataset.
        
        Returns:
            Dictionary of questions
        """
        if self.questions is None:
            self.questions = self.load_questions()
        return self.questions
    
    def get_data(self) -> Any:
        """
        Get the loaded data.
        
        Returns:
            Dataset data
        """
        if self.data is None:
            self.data = self.load_data()
        return self.data
    
    def get_data_by_indices(self, indices: List[int]) -> List[str]:
        """
        Get data by specific indices.
        Common implementation for list-based datasets.
        
        Args:
            indices: List of 0-based indices to retrieve
            
        Returns:
            Data items at specified indices
        """
        if not indices:
            return []
        
        # Validate indices
        max_index = max(indices)
        if max_index >= len(self.data):
            raise IndexError(
                f"Index {max_index} out of range for data (max: {len(self.data) - 1})"
            )
        
        return [self.data[i] for i in indices]
    
    def load_custom_selection(self, config_path: str) -> List[Dict[str, Any]]:
        """
        Load custom instance selection configuration.
        
        Args:
            config_path: Path to selection config YAML file
            
        Returns:
            List of sample dictionaries with 'name', 'indices', and optional 'description'
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Selection config not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'samples' not in config:
            raise ValueError(f"Selection config must have 'samples' key: {config_path}")
        
        return config['samples']
    
    def get_instance_selections(
        self,
        strategy: str,
        counts: Optional[List[int]] = None,
        selection_config: Optional[str] = None,
        window_end: Optional[int] = None
    ) -> List[Tuple[str, List[int]]]:
        """
        Get instance selections based on strategy.
        
        Args:
            strategy: Selection strategy ('first_n', 'sliding_window', 'custom')
            counts: List of counts for first_n and sliding_window modes
            selection_config: Path to custom selection config (for custom mode)
            window_end: Optional end index for sliding_window (default: use all data)
            
        Returns:
            List of tuples: (selection_name, indices_list)
            selection_name is used for tracking in outputs
        """
        if strategy == 'first_n' or strategy is None:
            # Default behavior: use first N instances
            if counts is None:
                counts = self.get_standard_sizes()
            return [(f"first_{count}", list(range(count))) for count in counts]
        
        elif strategy == 'sliding_window':
            # Slide window across all data (or up to window_end if specified)
            if counts is None or len(counts) == 0:
                raise ValueError("sliding_window strategy requires 'counts' to specify window size")
            
            total_size = len(self.get_data())
            
            # Use window_end if specified, otherwise use all data
            effective_end = window_end if window_end is not None else total_size
            
            # Validate window_end
            if effective_end > total_size:
                raise ValueError(
                    f"window_end {effective_end} exceeds total data size {total_size}"
                )
            if effective_end < 1:
                raise ValueError(f"window_end must be at least 1, got {effective_end}")
            
            selections = []
            
            for window_size in counts:
                if window_size > effective_end:
                    raise ValueError(
                        f"Window size {window_size} exceeds window_end {effective_end}"
                    )
                
                # Generate sliding windows up to effective_end
                # Last window ends exactly at effective_end
                for start_idx in range(effective_end - window_size + 1):
                    end_idx = start_idx + window_size
                    indices = list(range(start_idx, end_idx))
                    # Name format: "window_{size}_start{start}"
                    selection_name = f"window_{window_size}_start{start_idx}"
                    selections.append((selection_name, indices))
            
            return selections
        
        elif strategy == 'custom':
            # Use custom selection config
            if selection_config is None:
                raise ValueError("custom strategy requires 'selection_config' path")
            
            samples = self.load_custom_selection(selection_config)
            selections = []
            
            for sample in samples:
                if 'name' not in sample or 'indices' not in sample:
                    raise ValueError("Each sample must have 'name' and 'indices' keys")
                
                name = sample['name']
                indices = sample['indices']
                
                # Validate indices
                total_size = len(self.get_data())
                if any(idx < 0 or idx >= total_size for idx in indices):
                    raise ValueError(
                        f"Sample '{name}' has invalid indices. "
                        f"Valid range: 0 to {total_size - 1}"
                    )
                
                selections.append((name, indices))
            
            return selections
        
        else:
            raise ValueError(
                f"Unknown instance_selection strategy: {strategy}. "
                f"Valid options: 'first_n', 'sliding_window', 'custom'"
            )
