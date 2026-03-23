"""
Configuration and Logging Setup
Centralized configuration loading and logging setup for the evaluation framework.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Dataset
    dataset_type: str
    dataset_path: Path
    
    # Evaluation
    models: Optional[List[str]]
    questions: Optional[List[str]]
    n_trials: int
    max_concurrent: int
    resume_from: Optional[str]
    
    # Dataset-specific (dynamic)
    dataset_config: Dict[str, Any]
    
    # Original config for snapshotting
    _raw_config: Dict[str, Any]


class ConfigLoader:
    """Loads experiment configuration from YAML files"""
    
    @staticmethod
    def _validate_dataset_config(dataset_type: str, config: Dict[str, Any]):
        """
        Validate dataset-specific configuration based on instance_selection mode.
        
        Args:
            dataset_type: Type of dataset ('tweets' or 'ner')
            config: Dataset-specific configuration dict
            
        Raises:
            ValueError: If configuration contains invalid or conflicting parameters
        """
        selection_mode = config.get('instance_selection', 'first_n')
        
        # Define which parameters are valid for each mode
        valid_params = {
            'first_n': {'instance_selection', 'counts'},
            'sliding_window': {'instance_selection', 'counts', 'window_end'},
            'custom': {'instance_selection', 'selection_config'}
        }
        
        # Define required parameters for each mode
        required_params = {
            'first_n': set(),
            'sliding_window': set(),
            'custom': {'selection_config'}
        }
        
        if selection_mode not in valid_params:
            raise ValueError(
                f"Invalid instance_selection mode '{selection_mode}' for {dataset_type}. "
                f"Valid options: {list(valid_params.keys())}"
            )
        
        # Get valid and required params for this mode
        valid_for_mode = valid_params[selection_mode]
        required_for_mode = required_params[selection_mode]
        
        # Check for invalid parameters
        provided_params = set(config.keys())
        invalid_params = provided_params - valid_for_mode
        
        if invalid_params:
            # Build helpful error message
            param_usage = {
                'counts': "only for 'first_n' or 'sliding_window' modes",
                'window_end': "only for 'sliding_window' mode",
                'selection_config': "only for 'custom' mode"
            }
            
            error_details = []
            for param in invalid_params:
                if param in param_usage:
                    error_details.append(f"  - '{param}' is {param_usage[param]}")
                else:
                    error_details.append(f"  - '{param}' is not a valid parameter")
            
            raise ValueError(
                f"\n{dataset_type} config error: Invalid parameters for instance_selection='{selection_mode}':\n"
                + "\n".join(error_details) +
                f"\n\nValid parameters for '{selection_mode}' mode: {sorted(valid_for_mode)}\n"
                "Please comment out or remove invalid parameters from config/experiment.yaml"
            )
        
        # Check for missing required parameters
        missing_params = required_for_mode - provided_params
        
        if missing_params:
            raise ValueError(
                f"\n{dataset_type} config error: Missing required parameters for instance_selection='{selection_mode}':\n"
                + "\n".join(f"  - '{param}'" for param in missing_params) +
                "\n\nPlease add these parameters to config/experiment.yaml"
            )
        
        logger.info(f"✓ {dataset_type} config validated: instance_selection={selection_mode}")
    
    @staticmethod
    def load(config_file: Path) -> ExperimentConfig:
        """
        Load experiment configuration.
        
        Args:
            config_file: Path to experiment.yaml
            
        Returns:
            ExperimentConfig object
        """
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        logger.info(f"Loaded config from {config_file}")
        
        # Extract dataset config
        dataset = raw_config.get('dataset', {})
        dataset_type = dataset.get('type')
        dataset_path = Path(dataset.get('path', f"data/{dataset_type}"))
        
        # Extract evaluation config
        evaluation = raw_config.get('evaluation', {})
        models = evaluation.get('models')
        questions = evaluation.get('questions')
        n_trials = evaluation.get('n_trials', 3)
        max_concurrent = evaluation.get('max_concurrent', 5)
        resume_from = evaluation.get('resume_from')
        
        # Extract dataset-specific config (e.g., tweets.counts)
        dataset_config = raw_config.get(dataset_type, {})
        
        # Validate dataset-specific configuration
        ConfigLoader._validate_dataset_config(dataset_type, dataset_config)
        
        config = ExperimentConfig(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
            models=models,
            questions=questions,
            n_trials=n_trials,
            max_concurrent=max_concurrent,
            resume_from=resume_from,
            dataset_config=dataset_config,
            _raw_config=raw_config
        )
        
        logger.info(f"Config: dataset={dataset_type}, models={models}, questions={questions}, "
                   f"trials={n_trials}, concurrent={max_concurrent}")
        
        return config
    
    @staticmethod
    def save_snapshot(config: ExperimentConfig, output_dir: Path):
        """
        Save configuration snapshot to output directory.
        
        Args:
            config: ExperimentConfig object
            output_dir: Output directory to save snapshot
        """
        snapshot_file = output_dir / "experiment_config.yaml"
        
        with open(snapshot_file, 'w') as f:
            yaml.dump(config._raw_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved config snapshot to {snapshot_file}")


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path, reset: bool = False) -> logging.Logger:
    """
    Configure logging to both console and file in the output directory.
    
    Args:
        output_dir: Directory where the log file will be saved
        reset: If True, removes all existing handlers before setting up new ones
        
    Returns:
        Logger instance for the caller
    """
    log_file = output_dir / "evaluation.log"
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # If reset requested, remove all existing handlers
    if reset:
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
    
    # Only configure if no handlers exist or if reset was requested
    if not root_logger.handlers or reset:
        root_logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    return logging.getLogger(__name__)
