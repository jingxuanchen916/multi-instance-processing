"""
Storage Management
Handles checkpoints, raw responses, errors, and summary results.
"""

import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

logger = logging.getLogger(__name__)


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """
    Manages checkpoint files for resuming evaluation runs.
    Uses experiment_config.yaml for validation instead of separate checkpoint_config.yaml.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize checkpoint manager.
        
        Args:
            run_dir: Directory for this evaluation run
        """
        self.run_dir = Path(run_dir)
        self.checkpoint_file = self.run_dir / "checkpoint.csv"
        self.config_file = self.run_dir / "experiment_config.yaml"
        self.selection_config_file = self.run_dir / "selection_config.yaml"
    
    def exists(self) -> bool:
        """Check if checkpoint file exists."""
        return self.checkpoint_file.exists()
    
    def has_config(self) -> bool:
        """Check if experiment config exists."""
        return self.config_file.exists()
    
    def save(self, results: List[Dict[str, Any]]):
        """
        Save checkpoint to CSV.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            logger.warning("No results to save to checkpoint")
            return
        
        df = pd.DataFrame(results)
        
        # Save with proper quoting to handle newlines in text fields
        df.to_csv(self.checkpoint_file, index=False, quoting=1, escapechar='\\')
        logger.debug(f"Saved checkpoint with {len(results)} results")
    
    def load(self) -> Tuple[List[Dict[str, Any]], Set[Tuple], float]:
        """
        Load checkpoint from CSV.
        
        Returns:
            Tuple of:
            - results: List of result dictionaries
            - completed_evals: Set of (model, question_id, data_size, trial, selection_name?) tuples
              Note: selection_name is included if present in checkpoint (instance selection feature)
            - cumulative_time: Total API time from previous sessions (seconds)
        """
        if not self.exists():
            return [], set(), 0.0
        
        df = pd.read_csv(self.checkpoint_file)
        results = df.to_dict('records')
        
        # Track completed evaluations
        # Detect data size column name (should be n_instance)
        data_size_col = None
        for col in df.columns:
            if col.startswith('n_'):
                data_size_col = col
                break
        
        if data_size_col is None:
            logger.error("No data size column (n_instance) found in checkpoint")
            return results, set(), 0.0
        
        completed_evals = set()
        for _, row in df.iterrows():
            # Include selection_name if available (for instance selection feature)
            if 'selection_name' in df.columns and pd.notna(row.get('selection_name')):
                eval_key = (row['model'], row['question_id'], row[data_size_col], row['trial'], row['selection_name'])
            else:
                # Backward compatibility: without selection_name
                eval_key = (row['model'], row['question_id'], row[data_size_col], row['trial'])
            completed_evals.add(eval_key)
        
        # Calculate cumulative time
        cumulative_time = 0.0
        if 'api_time_seconds' in df.columns:
            cumulative_time = df['api_time_seconds'].sum()
        
        logger.info(f"✓ Loaded checkpoint: {len(completed_evals)} evaluations completed")
        logger.info(f"✓ Cumulative API time: {cumulative_time:.1f}s")
        
        return results, completed_evals, cumulative_time
    
    def validate_config(
        self,
        models: List[str],
        questions: List[str],
        data_counts: List[int],
        n_trials: int,
        instance_selection: str = 'first_n',
        selection_config: str = None,
        window_end: int = None
    ) -> bool:
        """
        Validate that current configuration matches the saved experiment config.
        
        Args:
            models: List of model identifiers
            questions: List of question IDs
            data_counts: List of data counts (tweets, documents, etc.)
            n_trials: Number of trials per combination
            instance_selection: Selection strategy ('first_n', 'sliding_window', 'custom')
            selection_config: Path to custom selection config (for 'custom' mode)
            window_end: Optional end index for sliding_window mode
            
        Returns:
            True if configs match and can resume, False otherwise
        """
        if not self.has_config():
            logger.warning("No experiment config found in run directory")
            return False
        
        # Load saved config
        import yaml
        with open(self.config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        # Determine dataset type from saved config
        dataset_type = saved_config.get('dataset', {}).get('type', 'tweets')
        
        # Extract relevant fields from saved config
        saved_models = saved_config.get('evaluation', {}).get('models')
        saved_questions = saved_config.get('evaluation', {}).get('questions')
        # Get counts from dataset-specific section (tweets, ner, etc.)
        saved_counts = saved_config.get(dataset_type, {}).get('counts')
        saved_trials = saved_config.get('evaluation', {}).get('n_trials', 0)
        saved_instance_selection = saved_config.get(dataset_type, {}).get('instance_selection', 'first_n')
        saved_selection_config = saved_config.get(dataset_type, {}).get('selection_config')
        saved_window_end = saved_config.get(dataset_type, {}).get('window_end')
        
        # Compare (null in saved config means "all", so compare resolved lists)
        # If saved is None/null, it was resolved to all items, so compare the actual lists
        def lists_match(saved, current):
            """Check if lists match, treating None as 'use all' which gets resolved to current"""
            if saved is None:
                # None means "all", which was resolved to current - always matches
                return True
            return saved == current
        
        # Basic config match (always check these)
        configs_match = (
            lists_match(saved_models, models) and
            lists_match(saved_questions, questions) and
            saved_trials == n_trials and
            saved_instance_selection == instance_selection
        )
        
        # Mode-specific parameter validation
        if configs_match:
            if instance_selection in ['first_n', 'sliding_window']:
                # Both modes use 'counts' parameter
                if not lists_match(saved_counts, data_counts):
                    configs_match = False
                
                # Only sliding_window uses 'window_end'
                if instance_selection == 'sliding_window':
                    if saved_window_end != window_end:
                        configs_match = False
        
        # If using custom selection, validate the selection config file matches
        if configs_match and instance_selection == 'custom':
            if not self.selection_config_file.exists():
                logger.warning("Custom selection mode but no saved selection config found")
                configs_match = False
            elif selection_config is None:
                logger.warning("Custom selection mode requires selection_config path")
                configs_match = False
            else:
                # Compare the actual selection config files
                import yaml
                with open(self.selection_config_file, 'r') as f:
                    saved_selection = yaml.safe_load(f)
                
                from pathlib import Path
                current_selection_path = Path(selection_config)
                if not current_selection_path.exists():
                    logger.warning(f"Selection config not found: {selection_config}")
                    configs_match = False
                else:
                    with open(current_selection_path, 'r') as f:
                        current_selection = yaml.safe_load(f)
                    
                    if saved_selection != current_selection:
                        logger.warning("Custom selection config has changed")
                        configs_match = False
        
        if not configs_match:
            logger.warning("=" * 60)
            logger.warning("⚠️  CHECKPOINT CONFIG MISMATCH DETECTED")
            logger.warning("=" * 60)
            logger.warning("Current configuration differs from saved experiment config:")
            logger.warning(f"\nSaved experiment config ({dataset_type} dataset):")
            logger.warning(f"  Models: {saved_models}")
            logger.warning(f"  Questions: {saved_questions}")
            logger.warning(f"  Data counts: {saved_counts}")
            logger.warning(f"  Trials: {saved_trials}")
            logger.warning(f"  Instance selection: {saved_instance_selection}")
            if saved_instance_selection == 'custom':
                logger.warning(f"  Selection config: {saved_selection_config}")
            if saved_instance_selection == 'sliding_window' and saved_window_end is not None:
                logger.warning(f"  Window end: {saved_window_end}")
            logger.warning("\nCurrent config:")
            logger.warning(f"  Models: {models}")
            logger.warning(f"  Questions: {questions}")
            logger.warning(f"  Data counts: {data_counts}")
            logger.warning(f"  Trials: {n_trials}")
            logger.warning(f"  Instance selection: {instance_selection}")
            if instance_selection == 'custom':
                logger.warning(f"  Selection config: {selection_config}")
            if instance_selection == 'sliding_window' and window_end is not None:
                logger.warning(f"  Window end: {window_end}")
            logger.warning("\n⚠️  Will create NEW run with auto-generated timestamp")
            logger.warning("=" * 60)
            return False
        
        logger.info("✓ Configuration matches saved experiment config")
        return True
    
    def copy_selection_config(self, selection_config_path: str):
        """
        Copy the custom selection config file to the run directory.
        
        Args:
            selection_config_path: Path to the selection config YAML file
        """
        from pathlib import Path
        import shutil
        
        source = Path(selection_config_path)
        if not source.exists():
            logger.warning(f"Selection config not found: {selection_config_path}")
            return
        
        shutil.copy2(source, self.selection_config_file)
        logger.info("✓ Copied selection config to run directory")


# ============================================================================
# Result Store
# ============================================================================

class ResultStore:
    """
    Manages storage of evaluation results, raw responses, and errors.
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize result store.
        
        Args:
            run_dir: Directory for this evaluation run
        """
        self.run_dir = Path(run_dir)
        self.raw_dir = self.run_dir / "raw"
        self.errors_dir = self.run_dir / "errors"
        self.summary_file = self.run_dir / "summary.csv"
        
        # Create subdirectories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)
    
    def save_raw_response(
        self,
        model: str,
        question_id: str,
        n_instance: int,
        response: Dict[str, Any],
        trial: int = 1,
        attempt: int = 1,
        is_final_successful_attempt: bool = False,
        selection_name: str = None
    ):
        """
        Save raw API response to JSON file.
        
        Args:
            model: Model identifier
            question_id: Question ID
            n_instance: Number of instances
            response: Raw API response dictionary
            trial: Trial number
            attempt: Current attempt number (default=1)
            is_final_successful_attempt: If True and attempt==1, omit attempt number from filename
            selection_name: Optional selection name to include in filename
        """
        # Build filename with selection name if provided
        model_clean = model.replace('/', '_')
        base_name = f"{model_clean}_{question_id}_n{n_instance}"
        
        if selection_name:
            base_name += f"_{selection_name}"
        
        # Only omit attempt number if this is attempt 1 and it succeeded (no retries needed)
        if attempt == 1 and is_final_successful_attempt:
            filename = f"{base_name}_trial{trial}_raw.json"
        else:
            # Include attempt number for retries or if we're saving during execution
            filename = f"{base_name}_trial{trial}_attempt{attempt}_raw.json"
        
        filepath = self.raw_dir / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response, f, indent=2)
        
        logger.debug(f"Saved raw response: {filename}")

    
    def save_error(
        self,
        model: str,
        question_id: str,
        n_instance: int,
        error: Exception,
        trial: int = 1,
        selection_name: str = None
    ):
        """
        Save error log to text file.
        
        Args:
            model: Model identifier
            question_id: Question ID
            n_instance: Number of instances
            error: Exception that occurred
            trial: Trial number
            selection_name: Optional selection name to include in filename
        """
        model_clean = model.replace('/', '_')
        base_name = f"{model_clean}_{question_id}_n{n_instance}"
        
        if selection_name:
            base_name += f"_{selection_name}"
        
        filename = f"{base_name}_trial{trial}_error.txt"
        filepath = self.errors_dir / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Question: {question_id}\n")
            f.write(f"N instances: {n_instance}\n")
            if selection_name:
                f.write(f"Selection: {selection_name}\n")
            f.write(f"Trial: {trial}\n")
            f.write(f"Error: {str(error)}\n")
            f.write(f"Error type: {type(error).__name__}\n")
        
        logger.debug(f"Saved error log: {filename}")
    
    def save_summary(self, results: List[Dict[str, Any]]):
        """
        Save summary results to CSV.
        Sorts results by model, question_id, data_size, selection_name (if present), and trial.
        
        Args:
            results: List of result dictionaries
        """
        if not results:
            logger.warning("No results to save to summary")
            return
        
        df = pd.DataFrame(results)
        
        # Sort for readability (checkpoint can be unordered, but summary should be sorted)
        # Detect data size column (should be n_instance)
        data_size_col = None
        for col in df.columns:
            if col.startswith('n_'):
                data_size_col = col
                break
        
        # Build sort column list dynamically
        sort_columns = ['model', 'question_id']
        if data_size_col:
            sort_columns.append(data_size_col)
        if 'selection_name' in df.columns:
            sort_columns.append('selection_name')
        sort_columns.append('trial')
        
        # Filter to only existing columns
        available_sort_cols = [col for col in sort_columns if col in df.columns]
        if available_sort_cols:
            df = df.sort_values(available_sort_cols)
        
        # Save with proper quoting to handle newlines in text fields
        df.to_csv(self.summary_file, index=False, quoting=1, escapechar='\\')
        logger.info(f"Results saved to {self.summary_file}")
