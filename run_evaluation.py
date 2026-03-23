"""
Main Entry Point for LLM Evaluation
Run this script to start the evaluation.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

from evaluation import EvaluationRunner
from utils import ConfigLoader, setup_logging, CheckpointManager


def check_can_resume(output_dir: Path, timestamp: str, eval_config: dict) -> bool:
    """
    Check if we can resume from a checkpoint without initializing the runner.
    
    Args:
        output_dir: Base output directory
        timestamp: Timestamp of the run to potentially resume
        eval_config: Evaluation configuration dict
        
    Returns:
        True if can resume, False if config mismatch
    """
    run_dir = output_dir / timestamp
    checkpoint_manager = CheckpointManager(run_dir)
    
    if not checkpoint_manager.has_config():
        # No existing config, this is a fresh run
        return True
    
    # Validate configuration including instance selection strategy
    can_resume = checkpoint_manager.validate_config(
        eval_config['models'],
        eval_config['questions'],
        eval_config['data_sizes'],
        eval_config['n_trials'],
        eval_config['instance_selection'],
        eval_config['selection_config'],
        eval_config['window_end']
    )
    
    return can_resume


def create_runner(config, config_dir: Path, output_dir: Path, timestamp: str, augment_approach: str, models_file: str = 'models.yaml'):
    """
    Create an evaluation runner with the given configuration.
    
    Args:
        config: Loaded configuration object
        config_dir: Configuration directory
        output_dir: Base output directory
        timestamp: Timestamp for this run
        augment_approach: Data augmentation approach
        models_file: Models configuration file name (default: models.yaml)
        
    Returns:
        EvaluationRunner instance
        
    Raises:
        Exception: If runner initialization fails
    """
    dataset_dir = config.dataset_path
    
    try:
        runner = EvaluationRunner(
            config_dir=config_dir,
            dataset_dir=dataset_dir,
            dataset_type=config.dataset_type,
            augment_approach=augment_approach,
            output_dir=output_dir,
            timestamp=timestamp,
            max_concurrent=config.max_concurrent,
            models_file=models_file
        )
        return runner
    except Exception as e:
        raise Exception(f"Failed to initialize runner: {e}")


def main():
    """Main execution function."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run LLM Data Understanding Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiment.yaml',
        help='Experiment config file (default: experiment.yaml)'
    )
    parser.add_argument(
        '--models',
        type=str,
        default='models.yaml',
        help='Models config file (default: models.yaml)'
    )
    parser.add_argument(
        '--augment_approach',
        type=str,
        default='default',
        choices=['default', 'head', 'middle', 'tail', 'random'],
        help='Data augmentation approach (default: default)'
    )
    args = parser.parse_args()
    
    # Define paths
    BASE_DIR = Path(__file__).parent
    CONFIG_DIR = BASE_DIR / "config"
    OUTPUT_DIR = BASE_DIR / "outputs" / args.augment_approach
    
    # Load experiment configuration
    config_file = CONFIG_DIR / args.config
    config = ConfigLoader.load(config_file)
    
    # Build evaluation config
    dataset_cfg = config.dataset_config
    eval_config = {
        'models': config.models,
        'questions': config.questions,
        'data_sizes': dataset_cfg.get('counts'),
        'instance_selection': dataset_cfg.get('instance_selection', 'first_n'),
        'selection_config': dataset_cfg.get('selection_config'),
        'window_end': dataset_cfg.get('window_end'),
        'n_trials': config.n_trials
    }
    
    # Determine timestamp and check if we can resume
    if config.resume_from:
        timestamp = config.resume_from
        can_resume = check_can_resume(OUTPUT_DIR, timestamp, eval_config)
        
        if not can_resume:
            # Config mismatch - create new run
            logger_note = f"Cannot resume from {timestamp} (config mismatch)"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger_note += f" - creating new run: {timestamp}"
        else:
            logger_note = f"Attempting to resume from: {timestamp}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger_note = f"Creating new run: {timestamp}"
        can_resume = True  # Fresh run, nothing to resume
    
    # Create output directory
    run_output_dir = OUTPUT_DIR / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(run_output_dir)
    logger.info(logger_note)
    
    # Save config snapshot to output directory
    ConfigLoader.save_snapshot(config, run_output_dir)
    
    # If using custom selection, copy the selection config to the run directory
    if eval_config['instance_selection'] == 'custom' and eval_config['selection_config']:
        checkpoint_mgr = CheckpointManager(run_output_dir)
        checkpoint_mgr.copy_selection_config(eval_config['selection_config'])
    
    logger.info("=" * 60)
    logger.info("LLM Data Understanding - Evaluation Runner")
    logger.info("=" * 60)
    
    # Initialize runner
    try:
        runner = create_runner(config, CONFIG_DIR, OUTPUT_DIR, timestamp, args.augment_approach, args.models)
    except Exception as e:
        logger.error(f"Failed to initialize runner: {e}")
        logger.error("Make sure you have set up your .env file with API_KEY")
        sys.exit(1)
    
    # Run evaluation
    try:
        results, config_matched = runner.run_evaluation(
            **eval_config,
            resume=can_resume
        )
        
        # config_matched should always be True now since we pre-check
        if not config_matched:
            logger.error("Unexpected: config mismatch after pre-check")
            sys.exit(1)
        
        logger.info("\nEvaluation complete!")
        logger.info(f"Results saved to: {runner.run_dir}")
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
