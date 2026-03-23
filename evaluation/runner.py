"""
Evaluation Runner
Main orchestrator for running LLM evaluations.

Architecture:
- ClientManager: Handles API calls with concurrency control
- BaseEvaluator: Dataset-specific evaluation logic (pluggable)
- CheckpointManager: Handles checkpoint save/load/validation
- ResultStore: Handles file I/O for results
- ResponseParser: Parses LLM responses
- EvaluationReporter: Generates summaries and reports
- ResultBuilder: Constructs result dictionaries

Runner's role: Pure orchestration of evaluation flow
"""

import asyncio
import logging
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from tqdm import tqdm

from llm import ClientManager
from llm.model_utils import load_model_info, calculate_token_cost
from utils import CheckpointManager, ResultStore, ResponseParser, EvaluationReporter, ResultBuilder, GroundTruthGenerator
from .evaluators import BaseEvaluator, TweetEvaluator, NEREvaluator, WSDEvaluator, SentimentEvaluator, NewsEvaluator, LanguageEvaluator, ArithmeticEvaluator, ParityEvaluator
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Orchestrates LLM evaluation across models, questions, and data sizes.
    
    Three-step flow:
    1. Setup: Initialize clients, load dataset, configure storage
    2. Load: Get models, questions, and resume from checkpoint if applicable
    3. Execute: Run evaluation loop with async tasks, save results
    """
    
    # Keep third-party client logs concise unless debugging explicitly
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    def __init__(
        self,
        config_dir: str,
        dataset_dir: str,
        dataset_type: str,
        augment_approach: str,
        output_dir: str,
        api_key: Optional[str] = None,
        timestamp: Optional[str] = None,
        max_concurrent: int = 5,
        models_file: str = 'models.yaml'
    ):
        """
        Initialize evaluation runner.
        
        Args:
            config_dir: Directory containing models.yaml and model_info.csv
            dataset_dir: Directory containing dataset files
            dataset_type: Type of dataset ('tweets', etc.)
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
            output_dir: Directory to save outputs
            api_key: Optional API key (otherwise loads from .env)
            timestamp: Optional timestamp string (if None, creates new one)
            max_concurrent: Maximum concurrent API requests (default: 5)
            models_file: Models configuration file name (default: models.yaml)
        """
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.models_file = models_file
        
        # Load configurations
        self.models = self._load_models()
        self.model_info = load_model_info(self.config_dir)
        
        # Initialize dataset-specific evaluator
        self.evaluator = self._create_evaluator(dataset_type, dataset_dir, augment_approach)
        
        # Initialize components
        self.client_manager = ClientManager(
            api_key=api_key,
            max_concurrent=max_concurrent
        )
        self.parser = ResponseParser()
        self.reporter = EvaluationReporter()
        
        # Initialize result builder (stateless utility)
        self.result_builder = ResultBuilder()
        
        # Create timestamped output directory (use provided timestamp or create new)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage components
        self.checkpoint_manager = CheckpointManager(self.run_dir)
        self.result_store = ResultStore(self.run_dir)
        self.ground_truth_generator = GroundTruthGenerator(self.run_dir)
        
        logger.info(f"Output directory: {self.run_dir}")
    
    def _create_evaluator(self, dataset_type: str, dataset_dir: str, augment_approach: str) -> BaseEvaluator:
        """
        Create dataset-specific evaluator.
        
        Args:
            dataset_type: Type of dataset ('tweets', 'ner', 'wsd', 'sentiment', 'news', 'language', 'arithmetic', 'parity')
            dataset_dir: Directory containing dataset files
            augment_approach: Data augmentation approach (default, head, middle, tail, random)
            
        Returns:
            Evaluator instance
        """
        # Map dataset types to evaluator classes
        evaluator_map = {
            'tweets': TweetEvaluator,
            'ner': NEREvaluator,
            'wsd': WSDEvaluator,
            'sentiment': SentimentEvaluator,
            'news': NewsEvaluator,
            'language': LanguageEvaluator,
            'arithmetic': ArithmeticEvaluator,
            'parity': ParityEvaluator,
        }
        
        evaluator_class = evaluator_map.get(dataset_type)
        if evaluator_class is None:
            supported = ', '.join(f"'{k}'" for k in evaluator_map.keys())
            raise ValueError(f"Unsupported dataset type: {dataset_type}. Supported: {supported}")
        
        return evaluator_class(dataset_dir, augment_approach)
    
    def _load_models(self) -> List[str]:
        """Load model list from YAML config."""
        models_file = self.config_dir / self.models_file
        with open(models_file, 'r') as f:
            config = yaml.safe_load(f)
        models = config['models']
        logger.info(f"Loaded {len(models)} models from {self.models_file}")
        return models
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    @staticmethod
    def _extract_dict_summary(d: Any) -> Any:
        """
        Extract key summary value from dict for logging.
        
        Args:
            d: Dictionary (or other value)
            
        Returns:
            Summary value to display in logs
        """
        if not isinstance(d, dict):
            return d
        
        # Priority order for extracting summary value
        summary_keys = [
            'total',   # Q2 NER, Q2 tweets
            'company', # Q2 WSD apple
            'sum',     # Q2 arithmetic
            'odd',     # Q2 parity
        ]
        
        for key in summary_keys:
            if key in d:
                return d[key]
        
        # Binary classification questions - check for specific categories
        binary_keys = ['positive', 'tech', 'english']
        for key in binary_keys:
            if key in d:
                return d[key]
        
        return 'N/A'
    
    # =========================================================================
    # Storage Delegation (delegates to ResultStore)
    # =========================================================================
    
    def _save_raw_response(
        self,
        model: str,
        question_id: str,
        data_size: int,
        response: Dict[str, Any],
        trial: int = 1,
        attempt: Optional[int] = None,
        is_final_successful_attempt: bool = False,
        selection_name: Optional[str] = None
    ):
        """Save raw API response to file."""
        self.result_store.save_raw_response(
            model, question_id, data_size, response, trial, attempt, is_final_successful_attempt, selection_name
        )
    
    def _save_error_log(
        self,
        model: str,
        question_id: str,
        data_size: int,
        error: Exception,
        trial: int = 1,
        selection_name: Optional[str] = None
    ):
        """Save error log to file."""
        self.result_store.save_error(model, question_id, data_size, error, trial, selection_name)
    
    # =========================================================================
    # Core Evaluation
    # =========================================================================
    
    async def _evaluate_single(
        self,
        model: str,
        question_id: str,
        question_config: Dict,
        selection_name: str,
        indices: List[int],
        trial: int
    ) -> Dict[str, Any]:
        """
        Evaluate single model-question-selection-trial combination.
        
        Args:
            model: Model identifier
            question_id: Question ID
            question_config: Question configuration
            selection_name: Name/identifier for this data selection
            indices: List of data indices to use
            trial: Trial number (1, 2, or 3)
            
        Returns:
            Result dictionary with all evaluation details
        """
        # Get parameter name for data size
        data_size_param = self.evaluator.get_data_size_param_name()
        data_size = len(indices)
        
        # Helper for log formatting with selection info
        def format_log_header():
            # Only show selection name if it's not the default "first_N" format
            if selection_name and selection_name != f"first_{data_size}":
                return f"{question_id} | {model} | {data_size_param}={data_size} | selection={selection_name} | trial={trial}"
            else:
                return f"{question_id} | {model} | {data_size_param}={data_size} | trial={trial}"
        
        # Create initial result using ResultBuilder
        result = self.result_builder.create_initial_result(
            model, question_id, question_config['name'], data_size, trial, data_size_param, selection_name
        )
        
        try:
            # Get data by specific indices from dataset
            # For WSD dataset, pass question_id to get the correct data file
            if hasattr(self.evaluator.dataset, 'get_data_for_question'):
                # Dataset supports question-specific data files
                data = self.evaluator.dataset.get_data_by_indices(indices, question_id=question_id)
            else:
                # Dataset uses a single shared data file
                data = self.evaluator.dataset.get_data_by_indices(indices)
            
            # Build prompt using dataset's formatting
            data_text = self.evaluator.dataset.format_for_prompt(data)
            base_prompt = question_config['prompt_template']
            
            prompt = f"{base_prompt}\n\n{data_text}"
            
            # Retry logic for both API and parse errors (up to 3 attempts)
            max_retries = 3
            parse_error = None
            predicted_answer = None
            reasoning = None
            
            for attempt in range(1, max_retries + 1):
                try:
                    # Prepare API call parameters
                    api_params = {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    
                    # Call LLM asynchronously
                    logger.debug(f"Calling {format_log_header()} | attempt={attempt}")
                    start_time = time.time()
                    response = await self.client_manager.chat_completion(**api_params)
                    api_time = time.time() - start_time
                    
                    try:
                        # ALWAYS save raw response for this attempt (whether parse succeeds or fails)
                        # For attempt 1, we'll save later with clean filename if it succeeds
                        if attempt > 1:
                            self._save_raw_response(
                                model, question_id, data_size, response, trial, attempt,
                                is_final_successful_attempt=False,
                                selection_name=selection_name
                            )
                        
                        # Extract response text
                        response_text = response['choices'][0]['message']['content']
                        
                        # Parse response - let parser handle everything naturally
                        reasoning, predicted_answer, parse_error = self.parser.parse_response(
                            response_text,
                            question_config['answer_type']
                        )
                        
                        # Extract usage and calculate costs
                        usage = response.get('usage', {})
                        costs = calculate_token_cost(model, usage, self.model_info)
                        
                        # If parsing succeeded, update result and break
                        if predicted_answer is not None:
                            # Save raw response with appropriate filename
                            if attempt == 1:
                                # Attempt 1 succeeded - save with clean filename
                                self._save_raw_response(
                                    model, question_id, data_size, response, trial, attempt,
                                    is_final_successful_attempt=True,
                                    selection_name=selection_name
                                )
                            else:
                                # Attempt 2+ succeeded - already saved earlier with attempt number
                                pass
                            
                            # Update result with success data
                            self.result_builder.update_with_success(
                                result, api_time, costs, reasoning, predicted_answer
                            )
                            
                            if attempt > 1:
                                logger.info(
                                    f"SUCCESS_AFTER_RETRY | {question_id} | {model} | {data_size_param}={data_size} | trial={trial} | "
                                    f"attempt={attempt}/{max_retries} | Succeeded after {attempt-1} retries"
                                )
                            break
                        
                        # Parsing failed - log and retry
                        if attempt < max_retries:
                            # Save attempt 1 with attempt number since it failed
                            if attempt == 1:
                                self._save_raw_response(
                                    model, question_id, data_size, response, trial, attempt,
                                    is_final_successful_attempt=False,
                                    selection_name=selection_name
                                )
                            logger.warning(
                                f"PARSE_FAIL | {format_log_header()} | "
                                f"attempt={attempt}/{max_retries} | error={parse_error} | RETRYING..."
                            )
                        else:
                            # Final attempt - parse failed
                            logger.error(
                                f"PARSE_FAIL | {format_log_header()} | "
                                f"attempt={attempt}/{max_retries} | error={parse_error} | MAX_RETRIES_REACHED"
                            )
                            # Update result with last attempt's data (but no predicted answer)
                            self.result_builder.update_with_success(
                                result, api_time, costs, reasoning, None,
                                parse_error=f"After {max_retries} attempts: {parse_error}"
                            )
                    
                    except Exception as processing_error:
                        # Error during response processing/parsing - treat as API error and retry
                        logger.warning(
                            f"PROCESSING_ERROR | {format_log_header()} | "
                            f"attempt={attempt}/{max_retries} | {type(processing_error).__name__}: {str(processing_error)} | RETRYING..."
                        )
                        if attempt >= max_retries:
                            # Final attempt - processing error persists
                            result['api_error'] = f"After {max_retries} attempts: {str(processing_error)}"
                            self._save_error_log(model, question_id, data_size, processing_error, trial, selection_name)
                
                except Exception as api_error:
                    # API error occurred - log and retry
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"API_ERROR | {format_log_header()} | "
                            f"attempt={attempt}/{max_retries} | {type(api_error).__name__}: {str(api_error)} | RETRYING..."
                        )
                    else:
                        # Final attempt - API error persists
                        logger.error(
                            f"API_ERROR | {format_log_header()} | "
                            f"attempt={attempt}/{max_retries} | {type(api_error).__name__}: {str(api_error)} | MAX_RETRIES_REACHED"
                        )
                        result['api_error'] = f"After {max_retries} attempts: {str(api_error)}"
                        self._save_error_log(model, question_id, data_size, api_error, trial, selection_name)
            
            # Get ground truth
            ground_truth = self.evaluator.get_ground_truth(question_id, selection_name)
            result['ground_truth'] = ground_truth
            
            # Compare answers
            if predicted_answer is not None and ground_truth is not None:
                result['correct'] = self.parser.compare_answers(
                    predicted_answer,
                    ground_truth,
                    question_config['answer_type']
                )
            
            # Only log final result (not retries)
            if result['api_error']:
                logger.info(
                    f"FAILED | {format_log_header()} | "
                    f"error={result['api_error']}"
                )
            else:
                # For dict answers, show summary field(s) in logs (not the full dict with all IDs)
                if question_config['answer_type'] == 'dict':
                    predicted_display = self._extract_dict_summary(predicted_answer)
                    truth_display = self._extract_dict_summary(ground_truth)
                else:
                    predicted_display = predicted_answer
                    truth_display = ground_truth
                
                logger.info(
                    f"OK | {format_log_header()} | "
                    f"time={result['api_time_seconds']}s | correct={result['correct']} | "
                    f"predicted={predicted_display} | truth={truth_display}"
                )
            
        except Exception as e:
            logger.error(
                f"ERROR | {format_log_header()} | "
                f"{type(e).__name__}: {str(e)}"
            )
            result['api_error'] = str(e)
            self._save_error_log(model, question_id, data_size, e, trial)
        
        return result
    
    # =========================================================================
    # Main Orchestration
    # =========================================================================
    
    def run_evaluation(
        self,
        models: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        data_sizes: Optional[List[int]] = None,
        instance_selection: str = 'first_n',
        selection_config: Optional[str] = None,
        window_end: Optional[int] = None,
        n_trials: int = 3,
        resume: bool = True
    ) -> tuple[pd.DataFrame, bool]:
        """
        Run complete evaluation across all combinations.
        
        Args:
            models: List of models to test (None = all)
            questions: List of question IDs to test (None = all)
            data_sizes: List of data sizes to test (None = standard progression, ignored if custom selection)
            instance_selection: Selection strategy ('first_n', 'sliding_window', 'custom')
            selection_config: Path to selection config file (required for 'custom' mode)
            window_end: Optional end index for sliding_window (default: use all data)
            n_trials: Number of trials to run for each combination (default: 3)
            resume: Whether to resume from checkpoint if available (default: True)
            
        Returns:
            Tuple of (DataFrame with all results, bool indicating if config matched for resume)
            The bool is False if config mismatch detected (caller should create new run)
        """
        return asyncio.run(self._async_run_evaluation(
            models, questions, data_sizes, instance_selection, selection_config, window_end, n_trials, resume
        ))
    
    async def _async_run_evaluation(
        self,
        models: Optional[List[str]],
        questions: Optional[List[str]],
        data_sizes: Optional[List[int]],
        instance_selection: str,
        selection_config: Optional[str],
        window_end: Optional[int],
        n_trials: int,
        resume: bool
    ) -> tuple[pd.DataFrame, bool]:
        """Async implementation of run_evaluation with concurrent execution."""
        # Use defaults if not specified
        if models is None:
            models = self.models
        if questions is None:
            questions = list(self.evaluator.dataset.get_questions().keys())
        if data_sizes is None:
            data_sizes = self.evaluator.dataset.get_standard_sizes()
        
        # Validate checkpoint configuration BEFORE generating selections
        # (checkpoint system validates based on original counts and strategy)
        can_resume = False
        
        if resume and self.checkpoint_manager.has_config():
            can_resume_config = self.checkpoint_manager.validate_config(
                models, questions, data_sizes, n_trials, instance_selection, selection_config, window_end
            )
            if not can_resume_config:
                # Configuration mismatch detected
                logger.info("✗ Cannot resume - configuration mismatch (see warnings above)")
                logger.info("✗ Caller should create a new run with a new timestamp")
                return pd.DataFrame(), False
            
            # Load checkpoint if validation passed
            can_resume = True
        
        # Generate instance selections based on strategy
        selections = self.evaluator.dataset.get_instance_selections(
            instance_selection, data_sizes, selection_config, window_end
        )
        
        # Generate ground truth for all combinations upfront (only if starting fresh)
        if not can_resume or not self.ground_truth_generator.exists():
            dataset_type = type(self.evaluator.dataset).__name__
            
            logger.info("Generating ground truth for all question-selection combinations...")
            
            # Special datasets with question-specific data files
            if dataset_type == 'WSDDataset':
                self.ground_truth_generator.generate_wsd_with_dataset(
                    self.evaluator.dataset, questions, selections
                )
            else:
                # Standard datasets with single data/labels files
                data_file = str(self.evaluator.dataset.text_file)
                labels_file = str(self.evaluator.dataset.labels_file)
                
                self.ground_truth_generator.generate(
                    dataset_type, data_file, questions, selections, labels_file
                )
        else:
            logger.info("Using existing ground truth from previous run")
        
        # Load generated ground truth into evaluator
        self.evaluator.load_ground_truth(str(self.ground_truth_generator.get_file_path()))
        
        # Get parameter name for logging
        data_size_param = self.evaluator.get_data_size_param_name()
        
        # Format selection summary for logging
        selection_summary = [f"{name} (n={len(indices)})" for name, indices in selections]
        
        logger.info("Starting evaluation:")
        logger.info(f"  Models ({len(models)}): {models}")
        logger.info(f"  Questions ({len(questions)}): {questions}")
        logger.info(f"  Instance selection strategy: {instance_selection}")
        logger.info(f"  Selections ({len(selections)}): {selection_summary}")
        logger.info(f"  Trials per combination: {n_trials}")
        logger.info(f"  Max concurrent requests: {self.client_manager.get_concurrency_level()}")
        logger.info(f"  Total evaluations: {len(models) * len(questions) * len(selections) * n_trials}")
        
        # Load existing results if valid checkpoint found
        completed_evals = set()
        results = []
        
        if can_resume:
            results, completed_evals, _ = self.checkpoint_manager.load()
            logger.info(f"Resuming from checkpoint: {len(results)} existing results")
        else:
            logger.info("Starting fresh evaluation")
        
        # Track overall time for this session
        session_start_time = time.time()
        
        # Build all tasks in order (model → question → selection)
        # Use a generator to create tasks on-demand for efficient memory usage
        all_tasks = []
        for model in models:
            for question_id in questions:
                question_config = self.evaluator.dataset.get_questions()[question_id]
                
                for selection_name, indices in selections:
                    data_size = len(indices)
                    for trial in range(1, n_trials + 1):
                        # Check if already completed
                        # Include selection_name for proper tracking (esp. with sliding_window/custom modes)
                        eval_key = (model, question_id, data_size, trial, selection_name)
                        if eval_key in completed_evals:
                            continue
                        
                        # Store task info (create coroutine on-demand)
                        all_tasks.append((model, question_id, question_config, selection_name, indices, trial))
        
        # Execute tasks with progress bar
        total = len(models) * len(questions) * len(selections) * n_trials
        completed_count = len(completed_evals)
        pbar = tqdm(total=total, initial=completed_count, desc="Evaluating")
        
        # Update progress bar every 10 completions to reduce overhead
        pbar_update_interval = 10
        pending_pbar_updates = 0
        
        # Execute tasks in order, but don't wait - fill all available concurrency slots
        # This gives mostly-ordered output while maximizing throughput
        running_tasks = {}  # Map from asyncio.Task to task info
        task_index = 0
        
        # Start initial batch of tasks (up to max_concurrent)
        while task_index < len(all_tasks) and len(running_tasks) < self.client_manager.get_concurrency_level():
            model, question_id, question_config, selection_name, indices, trial = all_tasks[task_index]
            coro = self._evaluate_single(model, question_id, question_config, selection_name, indices, trial)
            task = asyncio.create_task(coro)
            running_tasks[task] = (model, question_id, len(indices), trial)
            task_index += 1
        
        # Process tasks as they complete, immediately starting next task
        while running_tasks:
            # Wait for any task to complete
            done, pending = await asyncio.wait(running_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
            
            for completed_task in done:
                # Get result and remove from running tasks
                result = await completed_task
                del running_tasks[completed_task]
                
                results.append(result)
                pending_pbar_updates += 1
                
                # Save checkpoint immediately after each completion
                self.checkpoint_manager.save(results)
                
                # Update progress bar periodically
                if pending_pbar_updates >= pbar_update_interval:
                    pbar.update(pending_pbar_updates)
                    pending_pbar_updates = 0
                
                # Start next task if available
                if task_index < len(all_tasks):
                    model, question_id, question_config, selection_name, indices, trial = all_tasks[task_index]
                    coro = self._evaluate_single(model, question_id, question_config, selection_name, indices, trial)
                    new_task = asyncio.create_task(coro)
                    running_tasks[new_task] = (model, question_id, len(indices), trial)
                    task_index += 1
        
        # Final progress bar update
        if pending_pbar_updates > 0:
            pbar.update(pending_pbar_updates)
        
        pbar.close()
        
        # Calculate session time and total evaluation time
        session_time = time.time() - session_start_time
        
        # Total API time is the sum of all results (both loaded and new)
        df_results = pd.DataFrame(results)
        total_eval_time = df_results['api_time_seconds'].sum() if 'api_time_seconds' in df_results.columns else 0.0
        
        # Print session summary
        self.reporter.print_session_summary(session_time, total_eval_time)
        
        # Save results
        self.result_store.save_summary(results)
        
        # Print comprehensive summary
        self.reporter.print_summary(df_results, data_size_param, total_eval_time)
        
        return df_results, True
