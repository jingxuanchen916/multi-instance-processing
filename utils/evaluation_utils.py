"""
Evaluation Utilities
Core utilities for parsing responses, building results, and reporting summaries.
"""

import json
import re
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Response Parser
# ============================================================================

class ResponseParser:
    """
    Parses LLM responses to extract reasoning and answers.
    """
    
    @staticmethod
    def parse_response(response_text: str, answer_type: str) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
        """
        Parse LLM response to extract reasoning and answer.
        
        Args:
            response_text: Raw response text from LLM
            answer_type: Expected answer type ('integer', 'float', 'string', 'dict')
            
        Returns:
            Tuple of (reasoning, answer, error_message)
            If parsing fails, answer will be None and error_message will explain why
        """
        try:
            # First, try to parse as JSON
            reasoning, answer = ResponseParser._parse_json_response(response_text, answer_type)
            if reasoning is not None and answer is not None:
                return reasoning, answer, None
            
            # If JSON parsing fails, try to extract from text
            logger.debug("JSON parsing failed, attempting text extraction")
            reasoning, answer = ResponseParser._parse_text_response(response_text, answer_type)
            if reasoning is not None and answer is not None:
                return reasoning, answer, None
            
            # If all parsing fails
            error_msg = "Could not parse response as JSON or extract answer from text"
            logger.warning(f"{error_msg}: {response_text[:200]}")
            return None, None, error_msg
            
        except Exception as e:
            error_msg = f"Parsing error: {str(e)}"
            logger.error(f"{error_msg} for response: {response_text[:200]}")
            return None, None, error_msg
    
    @staticmethod
    def _parse_json_response(response_text: str, answer_type: str) -> Tuple[Optional[str], Optional[Any]]:
        """
        Try to parse response as JSON with 'reasoning' and 'answer' fields.
        For dict type (WSD Q2), the entire JSON object is the answer.
        
        Returns:
            Tuple of (reasoning, answer) or (None, None) if parsing fails
        """
        try:
            # Try to find JSON in the response (may be wrapped in markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Try parsing the whole response
                    json_str = response_text.strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # For dict type (WSD Q2), the entire JSON object is the answer
            if answer_type == 'dict':
                # The whole JSON object is the answer
                reasoning = data.get('reasoning', '')
                answer = data  # Return the entire dict
                return reasoning, answer
            
            # For other types, extract from 'answer' field
            reasoning = data.get('reasoning', '')
            answer_raw = data.get('answer')
            
            if answer_raw is None:
                return None, None
            
            # Convert answer to the expected type
            answer = ResponseParser._convert_answer_type(answer_raw, answer_type)
            
            return reasoning, answer
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.debug(f"JSON parsing failed: {e}")
            return None, None
    
    @staticmethod
    def _parse_text_response(response_text: str, answer_type: str) -> Tuple[Optional[str], Optional[Any]]:
        """
        Try to extract answer from plain text response.
        Looks for common patterns like "Answer: X" or "The answer is X".
        
        Returns:
            Tuple of (reasoning, answer) or (None, None) if extraction fails
        """
        try:
            # Use the whole text as reasoning
            reasoning = response_text
            
            # Try different patterns to extract the answer
            patterns = [
                r'[Aa]nswer[:\s]+([^\n]+)',
                r'[Tt]he answer is[:\s]+([^\n]+)',
                r'[Rr]esult[:\s]+([^\n]+)',
                r'[Tt]otal[:\s]+([^\n]+)',
            ]
            
            answer_str = None
            for pattern in patterns:
                match = re.search(pattern, response_text)
                if match:
                    answer_str = match.group(1).strip()
                    break
            
            if answer_str is None:
                # As a last resort, try to extract based on answer type
                if answer_type in ['integer', 'float']:
                    # Look for numbers
                    numbers = re.findall(r'-?\d+\.?\d*', response_text)
                    if numbers:
                        answer_str = numbers[-1]  # Take the last number found
                
            if answer_str is None:
                return None, None
            
            # Convert to expected type
            answer = ResponseParser._convert_answer_type(answer_str, answer_type)
            
            return reasoning, answer
            
        except Exception as e:
            logger.debug(f"Text parsing failed: {e}")
            return None, None
    
    @staticmethod
    def _convert_answer_type(answer_raw: Any, answer_type: str) -> Any:
        """
        Convert answer to the expected type.
        
        Args:
            answer_raw: Raw answer value
            answer_type: Expected type ('integer', 'float', 'string', 'dict')
            
        Returns:
            Converted answer
            
        Raises:
            ValueError: If conversion fails
        """
        if answer_type == 'dict':
            # For dict type, return as-is if it's already a dict
            if isinstance(answer_raw, dict):
                return answer_raw
            else:
                raise ValueError(f"Expected dict but got {type(answer_raw)}")
        
        elif answer_type == 'integer':
            if isinstance(answer_raw, int):
                return answer_raw
            elif isinstance(answer_raw, float):
                return int(answer_raw)
            elif isinstance(answer_raw, str):
                # Remove any commas or extra whitespace
                cleaned = answer_raw.strip().replace(',', '')
                # Extract first number if there's text
                match = re.search(r'-?\d+', cleaned)
                if match:
                    return int(match.group(0))
                raise ValueError(f"Cannot convert '{answer_raw}' to integer")
            else:
                raise ValueError(f"Cannot convert type {type(answer_raw)} to integer")
        
        elif answer_type == 'float':
            if isinstance(answer_raw, (int, float)):
                return float(answer_raw)
            elif isinstance(answer_raw, str):
                # Remove any commas or extra whitespace
                cleaned = answer_raw.strip().replace(',', '')
                # Extract first number if there's text
                match = re.search(r'-?\d+\.?\d*', cleaned)
                if match:
                    return float(match.group(0))
                raise ValueError(f"Cannot convert '{answer_raw}' to float")
            else:
                raise ValueError(f"Cannot convert type {type(answer_raw)} to float")
        
        elif answer_type == 'string':
            # For strings, just convert to string and clean up
            result = str(answer_raw).strip()
            # Remove quotes if present
            if (result.startswith('"') and result.endswith('"')) or \
               (result.startswith("'") and result.endswith("'")):
                result = result[1:-1]
            return result
        
        else:
            raise ValueError(f"Unknown answer type: {answer_type}")
    
    @staticmethod
    def compare_answers(predicted: Any, ground_truth: Any, answer_type: str) -> bool:
        """
        Compare predicted answer with ground truth.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            answer_type: Type of answer
            
        Returns:
            True if answers match, False otherwise
        """
        if predicted is None:
            return False
        
        try:
            if answer_type == 'dict':
                # For dict type, compare all fields except 'reasoning'
                # Both predicted and ground_truth should be dicts now
                if not isinstance(predicted, dict):
                    logger.warning(f"Expected dict but got {type(predicted)}: {predicted}")
                    return False
                
                if not isinstance(ground_truth, dict):
                    logger.warning(f"Expected ground_truth to be dict but got {type(ground_truth)}: {ground_truth}")
                    return False
                
                # Get all keys from both dicts, excluding 'reasoning'
                pred_keys = set(k for k in predicted.keys() if k != 'reasoning')
                truth_keys = set(k for k in ground_truth.keys() if k != 'reasoning')
                
                # Check if keys match
                if pred_keys != truth_keys:
                    logger.warning(f"Dict keys mismatch. Predicted keys: {pred_keys}, Ground truth keys: {truth_keys}")
                    return False
                
                # Compare all values for matching keys
                for key in pred_keys:
                    try:
                        pred_val = predicted[key]
                        truth_val = ground_truth[key]
                        
                        # Compare as strings for exact match
                        # This is important for arithmetic tasks where "5" != "5.0"
                        pred_str = str(pred_val).strip()
                        truth_str = str(truth_val).strip()
                        
                        if pred_str != truth_str:
                            return False
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Error comparing dict field '{key}': {e}")
                        return False
                
                return True
            
            elif answer_type == 'integer':
                return int(predicted) == int(ground_truth)
            
            elif answer_type == 'float':
                # For floats, check if they're close (within small epsilon)
                return abs(float(predicted) - float(ground_truth)) < 0.01
            
            elif answer_type == 'string':
                # For strings, compare after lowercasing and stripping
                pred_str = str(predicted).strip().lower()
                truth_str = str(ground_truth).strip().lower()
                return pred_str == truth_str
            
            else:
                return False
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Comparison error: {e}")
            return False


# ============================================================================
# Result Builder
# ============================================================================

class ResultBuilder:
    """
    Builds result dictionaries for evaluation outputs.
    
    Responsibilities:
    - Create initial result structure
    - Update result with success data
    - Keep result structure consistent across evaluations
    """
    
    def create_initial_result(
        self,
        model: str,
        question_id: str,
        question_name: str,
        data_size: int,
        trial: int,
        data_size_param: str = 'n_instance',
        selection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create initial result dictionary with default values.
        
        Args:
            model: Model identifier
            question_id: Question ID
            question_name: Question name/description
            data_size: Size of data used
            trial: Trial number
            data_size_param: Parameter name for data size (default: 'n_instance')
            selection_name: Optional name/identifier for data selection
            
        Returns:
            Result dictionary with default values
        """
        result = {
            'model': model,
            'question_id': question_id,
            'question_name': question_name,
            data_size_param: data_size,
            'trial': trial,
            'timestamp': datetime.now().isoformat(),
            'api_time_seconds': None,
            'success': False,
            'correct': False,
            'reasoning': None,
            'predicted_answer': None,
            'ground_truth': None,
            'prompt_tokens': None,
            'completion_tokens': None,
            'total_tokens': None,
            'prompt_cost': None,
            'completion_cost': None,
            'total_cost': None,
            'parse_error': None,
            'api_error': None
        }
        
        # Add selection_name if provided
        if selection_name:
            result['selection_name'] = selection_name
            
        return result
    
    def update_with_success(
        self,
        result: Dict[str, Any],
        api_time: float,
        costs: Dict[str, Any],
        reasoning: Optional[str],
        predicted_answer: Any,
        parse_error: Optional[str] = None
    ) -> None:
        """
        Update result dictionary with successful API call data.
        
        Args:
            result: Result dictionary to update
            api_time: API call time in seconds
            costs: Cost dictionary from calculate_token_cost
            reasoning: Reasoning text from LLM
            predicted_answer: Predicted answer (or None if parsing failed)
            parse_error: Parse error message (if any)
        """
        result['api_time_seconds'] = round(api_time, 2)
        result['prompt_tokens'] = costs['prompt_tokens']
        result['completion_tokens'] = costs['completion_tokens']
        result['total_tokens'] = costs['total_tokens']
        result['prompt_cost'] = costs['prompt_cost']
        result['completion_cost'] = costs['completion_cost']
        result['total_cost'] = costs['total_cost']
        # Success is True only if API call succeeded AND parsing succeeded (predicted_answer is not None)
        result['success'] = predicted_answer is not None
        result['reasoning'] = reasoning
        result['predicted_answer'] = predicted_answer
        result['parse_error'] = parse_error


# ============================================================================
# Evaluation Reporter
# ============================================================================

class EvaluationReporter:
    """
    Generates and prints evaluation summaries and statistics.
    """
    
    @staticmethod
    def print_summary(
        df: pd.DataFrame,
        data_size_param: str = 'n_instance',
        total_eval_time: Optional[float] = None
    ):
        """
        Print comprehensive summary statistics.
        
        Args:
            df: DataFrame with all evaluation results
            data_size_param: Parameter name for data size (default: 'n_instance')
            total_eval_time: Total evaluation time across all sessions (seconds)
        """
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        
        total = len(df)
        successful = df['success'].sum()
        correct = df['correct'].sum()
        total_cost = df['total_cost'].sum()
        
        logger.info(f"Total evaluations: {total}")
        logger.info(f"Successful API calls: {successful} ({successful/total*100:.1f}%)")
        logger.info(f"Correct answers: {correct} ({correct/total*100:.1f}%)")
        logger.info(f"Total cost: ${total_cost:.6f}")
        
        # Show total evaluation time if provided
        if total_eval_time is not None:
            EvaluationReporter._print_time(total_eval_time, "Total evaluation time")
        
        logger.info("")
        
        # Per-model breakdown
        EvaluationReporter._print_model_breakdown(df, data_size_param)
        
        logger.info("\n" + "=" * 60)
    
    @staticmethod
    def _print_time(seconds: float, label: str = "Time"):
        """
        Print time in human-readable format.
        
        Args:
            seconds: Time in seconds
            label: Label for the time measurement
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        logger.info(f"{label}: {hours}h {minutes}m {secs}s ({seconds:.1f} seconds)")
    
    @staticmethod
    def _print_model_breakdown(df: pd.DataFrame, data_size_param: str):
        """
        Print per-model performance breakdown.
        
        Args:
            df: DataFrame with evaluation results
            data_size_param: Parameter name for data size
        """
        logger.info("=" * 60)
        logger.info("MODEL PERFORMANCE BY QUESTION AND DATA SIZE")
        logger.info("=" * 60)
        
        for model in sorted(df['model'].unique()):
            logger.info(f"\n{model}:")
            model_df = df[df['model'] == model]
            
            for question_id in sorted(model_df['question_id'].unique()):
                q_df = model_df[model_df['question_id'] == question_id]
                question_name = q_df['question_name'].iloc[0]
                logger.info(f"  {question_id} ({question_name}):")
                
                # Group by data size and calculate success rate
                success_by_size = {}
                for data_size in sorted(q_df[data_size_param].unique()):
                    size_df = q_df[q_df[data_size_param] == data_size]
                    n_correct = size_df['correct'].sum()
                    n_total = len(size_df)
                    success_rate = f"{n_correct}/{n_total}"
                    
                    if success_rate not in success_by_size:
                        success_by_size[success_rate] = []
                    success_by_size[success_rate].append(int(data_size))
                
                # Print grouped by success rate
                for success_rate in sorted(success_by_size.keys(), reverse=True):
                    data_sizes = success_by_size[success_rate]
                    logger.info(f"    {success_rate} success: {data_sizes}")
    
    @staticmethod
    def print_session_summary(session_time: float, total_eval_time: float):
        """
        Print session timing summary.
        
        Args:
            session_time: Time for current session (seconds)
            total_eval_time: Total evaluation time across all sessions (seconds)
        """
        logger.info("")
        EvaluationReporter._print_time(session_time, "Session time")
        EvaluationReporter._print_time(total_eval_time, "Total evaluation time (all sessions)")
